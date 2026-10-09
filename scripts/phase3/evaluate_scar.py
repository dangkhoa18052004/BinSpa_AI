"""Select the phase-3 scar detector on validation, lock it, then evaluate on test once.

Rule (fixed before any test result): best configuration per architecture by validation mAP50-95
(Ultralytics val at the run's own imgsz); between architectures the higher validation mAP50-95,
ties within 0.002 broken by validation hypertrophic_scar recall, then by latency.
Test (Ultralytics val, conf 0.001, IoU 0.7, max_det 500): overall and per-class P, R, AP50, AP50-95,
confusion matrices and PR/P/R/F1 curves (saved by Ultralytics), TP/FP/FN at the validation
max-F1 confidence, latency (median of 3 passes over the test images, batch 1).
Installs the chosen weights as models/phase3/best_scar_detector.pt with metadata.
Usage: venv/Scripts/python.exe scripts/phase3/evaluate_scar.py
"""
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import REPORTS, digest, read_json, write_csv, write_json  # noqa: E402
from train_scar import CONFIGS, DATASET, RUNS, WEIGHTS  # noqa: E402

OUT = REPORTS / 'phase3_evaluation'
CLASSES = ['atrophic_scar', 'hypertrophic_scar']
TIE = 0.002


def run_val(weights, imgsz, split, name):
    from ultralytics import YOLO
    m = YOLO(str(weights)).val(data=str(DATASET / 'data.yaml'), split=split, imgsz=imgsz, batch=4, conf=0.001, iou=0.7,
                               max_det=500, device='0', plots=True, project=str(OUT), name=name, exist_ok=True,
                               verbose=False)
    box = m.box
    per = {c: dict(precision=float(box.p[k]), recall=float(box.r[k]), ap50=float(box.ap50[k]), ap50_95=float(box.ap[k]))
           for k, c in enumerate(CLASSES)}
    # F1-optimal confidence and the confusion matrix at that point come from the curves Ultralytics computed
    f1 = box.f1_curve.mean(0)
    conf_op = float(np.linspace(0, 1, len(f1))[f1.argmax()])
    return dict(mAP50=float(box.map50), mAP50_95=float(box.map), precision=float(box.mp), recall=float(box.mr),
                per_class=per, f1_max_conf=conf_op, speed_ms=dict(m.speed))


def counts_at(weights, imgsz, split, conf):
    """TP/FP/FN per class at a fixed confidence, IoU 0.5, greedy by score within each class."""
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(weights))
    tp = np.zeros(2, int); fp = np.zeros(2, int); fn = np.zeros(2, int); times = []
    images = sorted((DATASET / 'images' / split).glob('*.jpg'))
    for rnd in range(3):  # 3 timing passes; counts from the first
        for p in images:
            img = cv2.imread(str(p))
            t = time.perf_counter()
            r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.7, max_det=500, device='0', verbose=False)[0]
            times.append((time.perf_counter() - t) * 1000)
            if rnd == 0:
                H, W = r.orig_shape
                gt = [list(map(float, l.split())) for l in (DATASET / 'labels' / split / f'{p.stem}.txt').read_text().splitlines()]
                for c in range(2):
                    g = np.array([[(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
                                  for k, x, y, w, h in gt if int(k) == c]).reshape(-1, 4)
                    keep = (r.boxes.cls.cpu().numpy() == c)
                    d = r.boxes.xyxy.cpu().numpy()[keep][np.argsort(-r.boxes.conf.cpu().numpy()[keep])]
                    used = np.zeros(len(g), bool)
                    for box in d:
                        if len(g):
                            x1 = np.maximum(box[0], g[:, 0]); y1 = np.maximum(box[1], g[:, 1])
                            x2 = np.minimum(box[2], g[:, 2]); y2 = np.minimum(box[3], g[:, 3])
                            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
                            iou = inter / ((box[2] - box[0]) * (box[3] - box[1]) + (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) - inter)
                            iou[used] = -1
                            k = int(iou.argmax())
                            if iou[k] >= 0.5:
                                used[k] = True; tp[c] += 1; continue
                        fp[c] += 1
                    fn[c] += int((~used).sum())
    return {c: dict(tp=int(tp[k]), fp=int(fp[k]), fn=int(fn[k])) for k, c in enumerate(CLASSES)}, float(np.median(times))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for arch in WEIGHTS:
        for cfg, settings in CONFIGS.items():
            run = RUNS / f'{arch}_{cfg}'
            if not (run / 'completed.json').exists():
                raise RuntimeError(f'{run} not completed')
            v = run_val(run / 'weights/best.pt', settings['imgsz'], 'val', f'val_{arch}_{cfg}')
            rows.append(dict(arch=arch, config=cfg, imgsz=settings['imgsz'], weights=str(run / 'weights/best.pt'),
                             val_mAP50_95=v['mAP50_95'], val_mAP50=v['mAP50'], val_recall=v['recall'],
                             val_hs_recall=v['per_class']['hypertrophic_scar']['recall'],
                             val_hs_ap50_95=v['per_class']['hypertrophic_scar']['ap50_95'], val_f1_conf=v['f1_max_conf'],
                             val_infer_ms=v['speed_ms']['inference'], best_epoch=read_json(run / 'completed.json')['best_epoch']['epoch']))
    write_csv(OUT / 'validation_runs.csv', rows, list(rows[0]))
    key = lambda r: (-round(r['val_mAP50_95'] / TIE), -r['val_hs_recall'], r['val_infer_ms'])
    reps = {a: sorted([r for r in rows if r['arch'] == a], key=key)[0] for a in WEIGHTS}
    chosen = sorted(reps.values(), key=key)[0]
    selection = dict(rule=f'val mAP50-95 (ties within {TIE}) -> val hypertrophic_scar recall -> latency',
                     representatives=reps, chosen=chosen, locked_before_test=True,
                     locked_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT / 'selection.json', selection)

    test = {}
    for arch, rep in reps.items():
        t = run_val(rep['weights'], rep['imgsz'], 'test', f'test_{arch}')
        t['counts_at_val_f1_conf'], t['latency_ms_median'] = counts_at(rep['weights'], rep['imgsz'], 'test', rep['val_f1_conf'])
        t.update(config=rep['config'], imgsz=rep['imgsz'], conf_op=rep['val_f1_conf'], weights_sha256=digest(rep['weights']),
                 model_mb=Path(rep['weights']).stat().st_size / 1024 ** 2)
        test[arch] = t
        print(arch, rep['config'], {k: round(v, 4) for k, v in t.items() if isinstance(v, float)}, t['per_class'], flush=True)
    write_json(OUT / 'test_metrics.json', dict(selection=selection, test=test))
    dst = ROOT / 'models/phase3'
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen['weights'], dst / 'best_scar_detector.pt')
    write_json(dst / 'model_metadata.json', dict(
        status='final', architecture=chosen['arch'], config=chosen['config'], imgsz=chosen['imgsz'],
        operating_conf=chosen['val_f1_conf'], classes=CLASSES, weights_sha256=digest(dst / 'best_scar_detector.pt'),
        validation=dict(mAP50_95=chosen['val_mAP50_95'], hypertrophic_recall=chosen['val_hs_recall']), test=test[chosen['arch']],
        dataset='ACNE-DET Roboflow mirror v1; master split datasets/acnedet_master_split.csv', report='reports/phase3_summary.md',
        disclaimer='Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da.'))
    print('chosen', chosen['arch'], chosen['config'])


if __name__ == '__main__':
    main()
