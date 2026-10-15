"""Phase-1 goal = "where is there acne", not "count every lesion": region-level metrics of the deployed detector.

Uses the saved ACNE04 test predictions of the deployed RT-DETR-L @1280 + ACNE-DET (reports/phase1_combined/
rtdetr_ft/eval_test/predictions.json) at its operating confidence. Three views, coarse to strict:
  * location hit: a detection is correct if its centre lies inside a labelled box enlarged 1.5x (or IoU >= 0.1);
    a labelled lesion is found if a detection centre lies inside it (enlarged 1.5x)
  * grid G x G (G = 8, 16) over the image: a cell is positive if it holds a labelled lesion centre, flagged if it
    holds a detection centre; precision/recall/F1 over cells, exact and with a 1-cell tolerance
  * the strict detection metric (IoU >= 0.5), for reference
95 % CIs: bootstrap over test images (1000x). Writes reports/phase1_combined/region_metrics.json.
Usage: venv/Scripts/python.exe scripts/phase1/evaluate_regions.py
"""
import json

import numpy as np

from common import REPORTS, ROOT, read_json, write_json
from evaluate_detector_common import load_split

PRED = REPORTS / 'phase1_combined/rtdetr_ft/eval_test/predictions.json'
GRIDS = (8, 16)


def centre(b):
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]).reshape(-1, 2)


def enlarge(b, f=1.5):
    c, half = centre(b), np.column_stack([(b[:, 2] - b[:, 0]), (b[:, 3] - b[:, 1])]) * f / 2
    return np.column_stack([c - half, c + half]).reshape(-1, 4)


def inside(points, boxes):
    """(n_points, n_boxes) bool."""
    if not len(points) or not len(boxes):
        return np.zeros((len(points), len(boxes)), bool)
    x, y = points[:, 0:1], points[:, 1:2]
    return (x >= boxes[:, 0]) & (x <= boxes[:, 2]) & (y >= boxes[:, 1]) & (y <= boxes[:, 3])


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None] - inter)


def per_image(rec, det):
    gt = rec['gt']
    s = {}
    hit_det = inside(centre(det), enlarge(gt)).any(1) | (iou(det, gt).max(1) >= 0.1 if len(gt) and len(det) else False)
    s['loc'] = dict(det_ok=int(np.sum(hit_det)), det=len(det), gt_found=int(inside(centre(det), enlarge(gt)).any(0).sum()), gt=len(gt))
    o = iou(det, gt)
    used, tp = set(), 0
    for j in range(len(det)):  # strict IoU 0.5, greedy (detections are in score order)
        if not len(gt):
            break
        row = o[j].copy(); row[list(used)] = -1
        k = int(row.argmax())
        if row[k] >= 0.5:
            used.add(k); tp += 1
    s['strict'] = dict(tp=tp, det=len(det), gt=len(gt))
    for g in GRIDS:
        cell = lambda p: set(map(tuple, np.minimum((p / [rec['width'] / g, rec['height'] / g]).astype(int), g - 1).tolist()))
        G, D = cell(centre(gt)), cell(centre(det))
        near = lambda c, S: any((c[0] + dx, c[1] + dy) in S for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        s[f'grid{g}'] = dict(tp=len(G & D), flagged=len(D), positive=len(G),
                             flagged_ok_tol=sum(near(c, G) for c in D), positive_found_tol=sum(near(c, D) for c in G))
    return s


def summarise(stats, idx):
    tot = lambda key, f: sum(stats[i][key][f] for i in idx)
    prf = lambda p, r: dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)
    out = dict(location=prf(tot('loc', 'det_ok') / max(tot('loc', 'det'), 1), tot('loc', 'gt_found') / max(tot('loc', 'gt'), 1)),
               strict_iou50=prf(tot('strict', 'tp') / max(tot('strict', 'det'), 1), tot('strict', 'tp') / max(tot('strict', 'gt'), 1)))
    for g in GRIDS:
        k = f'grid{g}'
        out[f'grid{g}x{g}'] = prf(tot(k, 'tp') / max(tot(k, 'flagged'), 1), tot(k, 'tp') / max(tot(k, 'positive'), 1))
        out[f'grid{g}x{g}_tolerance1'] = prf(tot(k, 'flagged_ok_tol') / max(tot(k, 'flagged'), 1),
                                             tot(k, 'positive_found_tol') / max(tot(k, 'positive'), 1))
    return out


def main():
    meta = read_json(ROOT / 'models/phase1/model_metadata.json')
    conf = meta['operating_conf']
    records = {r['file_name']: r for r in load_split('test')}
    preds = json.loads(PRED.read_text(encoding='utf-8'))
    stats = []
    for p in preds:
        b, sc = np.array(p['boxes']).reshape(-1, 4), np.array(p['scores'])
        stats.append(per_image(records[p['file_name']], b[sc >= conf]))
    point = summarise(stats, range(len(stats)))
    rng = np.random.default_rng(42)
    boots = [summarise(stats, rng.integers(0, len(stats), len(stats))) for _ in range(1000)]
    for view, m in point.items():
        for k in list(m):
            v = [b[view][k] for b in boots]
            m[f'{k}_ci'] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    write_json(REPORTS / 'phase1_combined/region_metrics.json', dict(
        model=meta['architecture'], conf=conf, images=len(stats), dataset='ACNE04-v2 test', metrics=point,
        note='location hit = detection centre inside a labelled box enlarged 1.5x or IoU >= 0.1'))
    for view, m in point.items():
        print(f'{view:24s} P {m["precision"]:.3f} R {m["recall"]:.3f} F1 {m["f1"]:.3f}  F1 CI {np.round(m["f1_ci"], 3).tolist()}')


if __name__ == '__main__':
    main()
