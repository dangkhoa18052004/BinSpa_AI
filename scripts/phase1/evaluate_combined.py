"""Decide whether the ACNE-DET fine-tuned detectors replace the deployed phase-1 detector.

Candidates: RT-DETR-L @1280 and YOLO26s @1280 before (base) and after (ft) fine-tuning on ACNE04-v2 +
ACNE-DET acne lesions. For each: common evaluator on ACNE04 val/test at imgsz 1280 (operating
confidence = ACNE04-val max-F1), and on ACNE-DET val/test the recall of the five acne classes at that
confidence (IoU >= 0.3 and 0.5, greedy one-to-one) plus the share of detections on any annotated lesion.
Rule fixed before results, validation only:
  1. eligible if ACNE04 val mAP50-95 >= (base of the same architecture) - 0.01
  2. among eligible, highest ACNE-DET val acne recall (IoU 0.3); within 0.01 -> higher ACNE04 val mAP50-95
The chosen model is installed as models/phase1/best_acne_detector.pt (previous file kept).
Usage: venv/Scripts/python.exe scripts/phase1/evaluate_combined.py
"""
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np

from common import ROOT, digest, read_json, write_csv, write_json
from evaluate_detector_common import ADD, evaluate_model, flat_row

sys.path.insert(0, str(ROOT / 'scripts/phase2'))
from common2 import ACNE_CLASSES, MASTER_SPLIT, SOURCE, load_coco  # noqa: E402
from eval_gd1_crops import greedy, iou  # noqa: E402

OUT = ROOT / 'reports/phase1_combined'
FT = ROOT / 'runs/phase1_combined'
CANDIDATES = {
    'rtdetr_base': dict(arch='rtdetr', weights=ROOT / 'models/phase1/rtdetr_1280_best.pt', summary=ADD / 'rtdetr_1280/common_eval_summary.json'),
    'yolo26s_base': dict(arch='yolo26s', weights=ROOT / 'models/phase1/yolo26s_1280_best.pt', summary=ADD / 'yolo26s_1280/common_eval_summary.json'),
    'rtdetr_ft': dict(arch='rtdetr', weights=FT / 'rtdetr_1280_ft_acnedet/weights/best.pt', summary=None),
    'yolo26s_ft': dict(arch='yolo26s', weights=FT / 'yolo26s_1280_ft_acnedet/weights/best.pt', summary=None),
}


def acnedet_recall(weights, conf, split):
    import torch
    from ultralytics import RTDETR, YOLO
    data, cats = load_coco()
    with MASTER_SPLIT.open(encoding='utf-8-sig') as f:
        ids = {int(r['image_id']) for r in csv.DictReader(f) if r['split'] == split}
    anns = defaultdict(list)
    for a in data['annotations']:
        anns[a['image_id']].append(a)
    ckpt = torch.load(str(weights), map_location='cpu', weights_only=False)
    detr = 'RTDETR' in type(ckpt.get('model') or ckpt.get('ema')).__name__
    del ckpt
    model = (RTDETR if detr else YOLO)(str(weights))
    n_gt, hit = defaultdict(int), {0.3: defaultdict(int), 0.5: defaultdict(int)}
    n_det = n_any = 0
    for img in (i for i in data['images'] if i['id'] in ids):
        r = model.predict(cv2.imread(str(SOURCE / img['file_name'])), imgsz=1280, conf=conf, max_det=500, verbose=False)[0]
        det = r.boxes.xyxy.cpu().numpy()[np.argsort(-r.boxes.conf.cpu().numpy())]
        xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]
        acne = [a for a in anns[img['id']] if cats[a['category_id']] in ACNE_CLASSES]
        gt = np.array([xyxy(a['bbox']) for a in acne]).reshape(-1, 4)
        every = np.array([xyxy(a['bbox']) for a in anns[img['id']]]).reshape(-1, 4)
        for a in acne:
            n_gt[cats[a['category_id']]] += 1
        n_det += len(det)
        n_any += sum(1 for d in det if len(every) and iou(d, every).max() >= 0.3)
        for t in hit:
            for j, k in greedy(det, gt, t).items():
                hit[t][cats[acne[k]['category_id']]] += 1
    total = sum(n_gt.values())
    return dict(detections=n_det, on_any_lesion=n_any / max(n_det, 1),
                **{f'recall_iou{int(t * 100)}': sum(hit[t].values()) / total for t in hit},
                **{f'recall_iou30_{c}': hit[0.3][c] / max(n_gt[c], 1) for c in ACNE_CLASSES})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, c in CANDIDATES.items():
        if not c['weights'].exists():
            raise RuntimeError(f'missing {c["weights"]}')
        summary = read_json(c['summary']) if c['summary'] else evaluate_model(name, c['weights'], OUT / name, imgsz=1280)
        conf = summary['val']['operating_point']['conf']
        v, t = flat_row(summary, 'val'), flat_row(summary, 'test')
        rows[name] = dict(name=name, arch=c['arch'], weights=str(c['weights']), weights_sha256=digest(c['weights']), conf=conf,
                          acne04_val_mAP50_95=v['mAP50_95'], acne04_val_recall=v['recall'],
                          acne04_test_mAP50_95=t['mAP50_95'], acne04_test_mAP50=t['mAP50'], acne04_test_recall=t['recall'],
                          acne04_test_precision=t['precision'], acne04_test_f1=t['f1'],
                          **{f'acnedet_val_{k}': x for k, x in acnedet_recall(c['weights'], conf, 'val').items()})
        print(name, {k: round(x, 4) for k, x in rows[name].items() if isinstance(x, float)}, flush=True)
    # ---- selection on validation only
    eligible = [r for r in rows.values()
                if r['acne04_val_mAP50_95'] >= rows[f'{r["arch"]}_base']['acne04_val_mAP50_95'] - 0.01]
    best = max(r['acnedet_val_recall_iou30'] for r in eligible)
    chosen = max((r for r in eligible if best - r['acnedet_val_recall_iou30'] < 0.01), key=lambda r: r['acne04_val_mAP50_95'])
    selection = dict(rule='eligible: ACNE04 val mAP50-95 >= base(same arch) - 0.01; then max ACNE-DET val acne recall@IoU0.3, '
                          'within 0.01 -> max ACNE04 val mAP50-95', eligible=[r['name'] for r in eligible],
                     chosen=chosen['name'], locked_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT / 'selection.json', selection)
    # ---- ACNE-DET test (after the choice is locked)
    for name, r in rows.items():
        r.update({f'acnedet_test_{k}': x for k, x in acnedet_recall(r['weights'], r['conf'], 'test').items()})
    write_csv(OUT / 'comparison.csv', list(rows.values()), list(next(iter(rows.values()))))
    write_json(OUT / 'comparison.json', dict(selection=selection, candidates=rows))
    # ---- install
    target, meta_path = ROOT / 'models/phase1/best_acne_detector.pt', ROOT / 'models/phase1/model_metadata.json'
    if digest(target) != chosen['weights_sha256']:
        meta = read_json(meta_path)
        backup = ROOT / f'models/phase1/best_acne_detector_{meta["architecture"]}_{meta["imgsz"]}.pt'
        if not backup.exists():
            shutil.copy2(target, backup)
            write_json(ROOT / f'models/phase1/model_metadata_{meta["architecture"]}_{meta["imgsz"]}.json', meta)
        shutil.copy2(chosen['weights'], target)
        meta.update(status='final', architecture=('rtdetr-l' if chosen['arch'] == 'rtdetr' else 'yolo26s') +
                    ('+acnedet' if chosen['name'].endswith('_ft') else ''), imgsz=1280, operating_conf=chosen['conf'],
                    selected_weights_sha256=digest(target), source_run=chosen['weights'],
                    combined_finetune=dict(comparison='reports/phase1_combined/comparison.json', selection=selection),
                    promoted_utc=datetime.now(timezone.utc).isoformat())
        write_json(meta_path, meta)
    print('chosen', chosen['name'], json.dumps(selection))


if __name__ == '__main__':
    main()
