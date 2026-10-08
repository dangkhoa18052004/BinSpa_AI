"""Final comparison of the three detectors retrained at imgsz 1280 (YOLO26s, YOLO11s, RT-DETR-L).

1. Validation ranking with the pre-registered rule: mAP50-95 (ties within 0.002 of the best)
   -> recall -> latency. The test set plays no part in this choice.
2. Paired bootstrap on test (1000 image resamples, seed 42), reported against the validation
   choice and against YOLO26s 1280; the current 640 detector is included for reference.
3. Recall by lesion size and by lesions per image at each model's validation threshold.
4. Controlled end-to-end latency (same protocol as benchmark_latency.py: 30 test images,
   5 interleaved rounds, batch 1, FP32, decode excluded).
Reads saved predictions; only step 4 runs the models.
Usage: venv/Scripts/python.exe -u scripts/phase1/compare_1280.py
"""
import json
import random

import cv2
import numpy as np

from common import REPORTS, write_csv, write_json
from compare_resolution import DENSITY_BINS, SIZE_BINS, load, matched_gt
from evaluate_detector_common import ADD, bootstrap, image_stats, load_split, make_predictor

MODELS = {  # name -> (eval root, imgsz, label)
    'yolo26s_1280': (ADD / 'yolo26s_1280', 1280, 'YOLO26s @1280'),
    'yolo11s_1280': (ADD / 'yolo11s_1280', 1280, 'YOLO11s @1280'),
    'rtdetr_1280': (ADD / 'rtdetr_1280', 1280, 'RT-DETR-L @1280'),
    'yolo26s_640': (ADD / 'common_eval/yolo26s', 640, 'YOLO26s @640 (current)'),
}
CANDIDATES = ['yolo26s_1280', 'yolo11s_1280', 'rtdetr_1280']
E2E_IMAGES, E2E_ROUNDS = 30, 5


def latency(records):
    sample = random.Random(42).sample(records, E2E_IMAGES)
    images = [cv2.imread(str(r['path'])) for r in sample]
    predictors = {}
    for name, (root, imgsz, _) in MODELS.items():
        weights = json.loads((root / 'common_eval_summary.json').read_text(encoding='utf-8'))['weights']
        predictors[name] = make_predictor(weights, imgsz)
    for p in predictors.values():
        for image in images[:10]:
            p(image)
    times = {k: [] for k in predictors}
    for r in range(E2E_ROUNDS):
        for k, p in predictors.items():
            for image in images:
                times[k].append(sum(p(image)[2].values()))
        print(f'latency round {r + 1}: ' + ', '.join(f'{k}={np.median(v[-E2E_IMAGES:]):.1f}ms' for k, v in times.items()), flush=True)
    return {k: dict(e2e_ms_median=float(np.median(v)), e2e_ms_p10=float(np.percentile(v, 10)),
                    e2e_ms_p90=float(np.percentile(v, 90))) for k, v in times.items()}


def main():
    val_records, records = load_split('val'), load_split('test')
    rows, stats, by_size, by_density = {}, {}, {}, {}
    for name, (root, imgsz, label) in MODELS.items():
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
        by_size[name] = {f'{lo:g}-{hi:g}px' if hi < 1e9 else f'>={lo:g}px': float(hits[(sizes >= lo) & (sizes < hi)].mean())
                         for lo, hi in SIZE_BINS}
        by_density[name] = {f'{lo:g}-{hi:g}' if hi < 1e9 else f'>{lo - 1:g}': float(hits[(density >= lo) & (density <= hi)].mean())
                            for lo, hi in DENSITY_BINS}
        v, t = summary['val'], summary['test']
        rows[name] = dict(model=name, label=label, imgsz=imgsz, conf=conf, model_mb=summary['model_mb'], params_m=summary['params_m'],
                          val_mAP50_95=v['mAP50_95'], val_recall=v['operating_point']['recall'],
                          **{f'test_{k}': t[k] for k in ('mAP50', 'mAP50_95')},
                          **{f'test_{k}': t['operating_point'][k] for k in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn')},
                          weights_sha256=summary['weights_sha256'])

    # 1. validation choice (test not consulted)
    best = max(rows[k]['val_mAP50_95'] for k in CANDIDATES)
    tied = [k for k in CANDIDATES if best - rows[k]['val_mAP50_95'] <= 0.002]
    chosen = max(tied, key=lambda k: rows[k]['val_recall'])
    print('validation tie group:', tied, '-> chosen by recall:', chosen, flush=True)

    # 2. paired bootstrap
    ci_chosen = bootstrap(stats, chosen)
    ci_26s = bootstrap(stats, 'yolo26s_1280')

    # 4. controlled latency
    lat = latency(records)
    for k in rows:
        rows[k].update(lat[k])
        for metric in ('mAP50_95', 'recall', 'precision', 'f1'):
            rows[k][f'{metric}_ci'] = [ci_26s[k][f'{metric}_ci_low'], ci_26s[k][f'{metric}_ci_high']]
            rows[k][f'{metric}_diff_vs_yolo26s_1280'] = [ci_26s[k][f'{metric}_{s}'] for s in ('diff_vs_ref', 'diff_ci_low', 'diff_ci_high')]
            rows[k][f'{metric}_diff_vs_{chosen}'] = [ci_chosen[k][f'{metric}_{s}'] for s in ('diff_vs_ref', 'diff_ci_low', 'diff_ci_high')]
    result = dict(selection_rule='validation mAP50-95 (ties within 0.002 of best) -> recall -> latency',
                  validation_tie_group=tied, chosen_by_validation=chosen, bootstrap=dict(n_boot=1000, seed=42, unit='test image'),
                  models=rows, recall_by_lesion_size_at_640=by_size, recall_by_lesions_per_image=by_density,
                  latency_protocol=f'{E2E_IMAGES} test images x {E2E_ROUNDS} interleaved rounds, batch 1, FP32, decode excluded')
    write_json(ADD / 'final_1280_comparison.json', result)
    flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows.values()]
    write_csv(REPORTS / 'metrics/phase1_1280_comparison.csv', flat, list(flat[0]))
    print(json.dumps(dict(chosen=chosen, by_size=by_size, by_density=by_density,
                          rows={k: {c: (np.round(v, 4).tolist() if isinstance(v, list) else (round(v, 4) if isinstance(v, float) else v))
                                    for c, v in r.items() if c not in ('weights_sha256', 'label')} for k, r in rows.items()}), indent=1))


if __name__ == '__main__':
    main()
