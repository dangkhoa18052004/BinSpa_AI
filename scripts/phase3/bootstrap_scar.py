"""Paired bootstrap over test images for the two phase-3 representatives (after selection is locked).

Raw predictions (conf 0.001, IoU 0.7, max_det 500) at each model's imgsz; per class the same AP
routine as phase 1 (Ultralytics ap_per_class on one class); mAP50-95 = mean of the two class APs.
1000 resamples of the 45 test images, seed 42, the same resamples for both models.
Usage: venv/Scripts/python.exe scripts/phase3/bootstrap_scar.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import REPORTS, read_json, write_json  # noqa: E402
from evaluate_detector_common import image_stats, resample_metrics  # noqa: E402
from train_scar import DATASET  # noqa: E402

CLASSES = ['atrophic_scar', 'hypertrophic_scar']


def per_class_stats(weights, imgsz, conf_op):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    images = sorted((DATASET / 'images/test').glob('*.jpg'))
    recs = {c: [] for c in range(2)}; preds = {c: [] for c in range(2)}
    for p in images:
        r = model.predict(str(p), imgsz=imgsz, conf=0.001, iou=0.7, max_det=500, device='0', verbose=False)[0]
        H, W = r.orig_shape
        gt = [list(map(float, l.split())) for l in (DATASET / 'labels/test' / f'{p.stem}.txt').read_text().splitlines()]
        cls, box, score = r.boxes.cls.cpu().numpy(), r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        for c in range(2):
            g = np.array([[(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
                          for k, x, y, w, h in gt if int(k) == c]).reshape(-1, 4)
            keep = cls == c
            order = np.argsort(-score[keep])
            recs[c].append(dict(gt=g))
            preds[c].append(dict(boxes=box[keep][order].reshape(-1, 4), scores=score[keep][order]))
    return {c: image_stats(recs[c], preds[c], conf_op) for c in range(2)}


def main():
    sel = read_json(REPORTS / 'phase3_evaluation/selection.json')
    stats = {a: per_class_stats(r['weights'], r['imgsz'], r['val_f1_conf']) for a, r in sel['representatives'].items()}
    n = len(next(iter(stats.values()))[0])
    rng = np.random.default_rng(42)
    draws = {a: [] for a in stats}
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        for a, s in stats.items():
            m = [resample_metrics(s[c], idx) for c in range(2)]
            draws[a].append([np.mean([x['mAP50_95'] for x in m]), m[0]['mAP50_95'], m[1]['mAP50_95'],
                             np.mean([x['mAP50'] for x in m])])
    a, b = 'yolo26s', 'yolo11s'
    A, B = np.array(draws[a]), np.array(draws[b])
    names = ['mAP50_95', 'atrophic_AP50_95', 'hypertrophic_AP50_95', 'mAP50']
    out = {f'{a}_minus_{b}': {k: dict(mean=float((A - B)[:, i].mean()), ci=np.percentile((A - B)[:, i], [2.5, 97.5]).tolist(),
                                       p_better=float(((A - B)[:, i] > 0).mean())) for i, k in enumerate(names)},
           'ci': {m: {k: np.percentile(np.array(draws[m])[:, i], [2.5, 97.5]).tolist() for i, k in enumerate(names)} for m in draws},
           'n_boot': 1000, 'seed': 42, 'unit': 'test image'}
    write_json(REPORTS / 'phase3_evaluation/bootstrap_test.json', out)
    print(out)


if __name__ == '__main__':
    main()
