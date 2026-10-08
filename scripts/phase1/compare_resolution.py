"""Resolution study: YOLO26s at 640 vs 1280 on the test split.

Reads the saved common-evaluator predictions (no re-inference), runs the same paired
bootstrap as benchmark_additional_models.py (reference = 640) and breaks recall down by
lesion size (longest side, in pixels after resizing the image to 640) and lesions per image.
Usage: venv/Scripts/python.exe scripts/phase1/compare_resolution.py
"""
import json

import numpy as np

from common import write_json
from evaluate_detector_common import ADD, box_iou, bootstrap, image_stats, load_split

MODELS = {'yolo26s_640': ADD / 'common_eval/yolo26s', 'yolo26s_1280': ADD / 'yolo26s_1280'}
SIZE_BINS = [(0, 5), (5, 8), (8, 12), (12, 20), (20, 1e9)]
DENSITY_BINS = [(1, 10), (11, 30), (31, 60), (61, 1e9)]


def load(root, records):
    summary = json.loads((root / 'common_eval_summary.json').read_text(encoding='utf-8'))
    saved = json.loads((root / 'eval_test/predictions.json').read_text(encoding='utf-8'))
    if [p['file_name'] for p in saved] != [r['file_name'] for r in records]:
        raise RuntimeError(f'{root}: prediction order differs from the test manifest')
    preds = [dict(boxes=np.array(p['boxes'], dtype=np.float64).reshape(-1, 4), scores=np.array(p['scores'], dtype=np.float64))
             for p in saved]
    return summary, preds, summary['val']['operating_point']['conf']  # threshold chosen on validation


def matched_gt(rec, pred, conf):
    """Boolean per GT box: matched at IoU 0.5 by a prediction above the operating threshold (greedy by score)."""
    boxes = pred['boxes'][pred['scores'] >= conf]
    hit = np.zeros(len(rec['gt']), bool)
    if not len(boxes) or not len(rec['gt']):
        return hit
    iou = box_iou(rec['gt'], boxes)
    for j in range(iou.shape[1]):  # predictions are already sorted by score
        cand = np.where(~hit & (iou[:, j] >= .5))[0]
        if len(cand):
            hit[cand[np.argmax(iou[cand, j])]] = True
    return hit


def main():
    records = load_split('test')
    stats, by_size, by_density, rows = {}, {}, {}, {}
    for name, root in MODELS.items():
        summary, preds, conf = load(root, records)
        stats[name] = image_stats(records, preds, conf)
        sizes, hits, density = [], [], []
        for rec, pred in zip(records, preds):
            gt = np.asarray(rec['gt']).reshape(-1, 4)
            scale = 640 / max(rec['width'], rec['height'])
            sizes.append(np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]) * scale)
            hits.append(matched_gt(rec, pred, conf))
            density.append(np.full(len(gt), len(gt)))
        sizes, hits, density = map(np.concatenate, (sizes, hits, density))
        by_size[name] = {f'{lo:g}-{hi:g}px' if hi < 1e9 else f'>={lo:g}px': dict(n=int(m.sum()), recall=float(hits[m].mean()))
                         for lo, hi in SIZE_BINS for m in [(sizes >= lo) & (sizes < hi)]}
        by_density[name] = {f'{lo:g}-{hi:g}' if hi < 1e9 else f'>{lo - 1:g}': dict(lesions=int(m.sum()), recall=float(hits[m].mean()))
                            for lo, hi in DENSITY_BINS for m in [(density >= lo) & (density <= hi)]}
        t = summary['test']
        rows[name] = dict(imgsz=summary['imgsz'], conf=conf, mAP50=t['mAP50'], mAP50_95=t['mAP50_95'],
                          **{k: t['operating_point'][k] for k in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn')},
                          total_ms=t['speed_ms']['total'], weights_sha256=summary['weights_sha256'])
    ci = bootstrap(stats, 'yolo26s_640')
    result = dict(split='test', reference='yolo26s_640', n_boot=1000, seed=42,
                  method='paired percentile bootstrap over test images; thresholds chosen on validation',
                  test=rows, bootstrap=ci, recall_by_lesion_size_at_640=by_size, recall_by_lesions_per_image=by_density)
    write_json(ADD / 'yolo26s_1280' / 'resolution_comparison.json', result)
    print(json.dumps(dict(test=rows, by_size=by_size, by_density=by_density,
                          diff={k: {m: [round(ci['yolo26s_1280'][f'{m}_{s}'], 4) for s in ('diff_vs_ref', 'diff_ci_low', 'diff_ci_high')]
                                    for m in ('mAP50_95', 'mAP50', 'recall', 'precision', 'f1')} for k in ['yolo26s_1280']}), indent=1))


if __name__ == '__main__':
    main()
