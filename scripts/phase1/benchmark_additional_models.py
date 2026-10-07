"""Final comparison of all detectors with one evaluator and one latency session.

Reads the frozen weights of YOLO26n/26s/11s (never retrained; SHA256 checked against
reports/metrics/phase1_test_benchmark.csv), the 3 additional models and the full-schedule
control runs (only after their selection.json was locked on validation).
1. Each model: val (fixes the operating confidence) -> test, common evaluator.
2. Paired bootstrap (1000 resamples of the 176 test images, same resamples for every
   model) -> 95% CIs and differences vs the model ranked first on VALIDATION.
3. Latency: 3 interleaved rounds over all models (batch 1, FP32), median reported.
Outputs: reports/phase1_additional_models/common_eval/, reports/metrics/phase1_*.csv.
Existing artifacts of the 3 earlier models are only read.
Usage: venv/Scripts/python.exe -u scripts/phase1/benchmark_additional_models.py [--tables-only]
"""
import csv
import json

import numpy as np

from common import REPORTS, ROOT, digest, read_json, write_csv, write_json
from evaluate_detector_common import ADD, MODELS_DIR, bootstrap, collect, evaluate_model, flat_row, image_stats, \
    load_split, make_predictor

EXISTING = {
    'yolo26n': ('YOLO26n', ROOT / 'runs/phase1/benchmark_yolo26n_e100_s42_b8_w0_cram/weights/best.pt'),
    'yolo26s': ('YOLO26s', ROOT / 'runs/phase1/benchmark_yolo26s_e100_s42_b8_w0_cram/weights/best.pt'),
    'yolo11s': ('YOLO11s', ROOT / 'runs/phase1/benchmark_yolo11s_e100_s42_b8_w0_cram/weights/best.pt'),
}
ADDITIONAL = {
    'yolov5s': ('YOLOv5s', MODELS_DIR / 'yolov5s_best.pt'),
    'faster_rcnn': ('Faster R-CNN R50-FPN', MODELS_DIR / 'faster_rcnn_best.pth'),
    'rtdetr': ('RT-DETR-L', MODELS_DIR / 'rtdetr_best.pt'),
}
# Control runs: same protocol, no early stopping (originals stopped before the end of their LR schedule).
CONTROL = {
    'yolo11s_full100': ('YOLO11s (100 ep)', ROOT / 'runs/phase1/benchmark_yolo11s_e100_s42_b8_w0_cram_p100/weights/best.pt'),
    'faster_rcnn_full50': ('Faster R-CNN (50 ep)', MODELS_DIR / 'faster_rcnn_full50_best.pth'),
    'rtdetr_full100': ('RT-DETR-L (100 ep)', MODELS_DIR / 'rtdetr_full100_best.pt'),
}
FRAMEWORK = {k: 'torchvision' if k.startswith('faster_rcnn') else 'Ultralytics' for k in (*EXISTING, *ADDITIONAL, *CONTROL)}
LATENCY_ROUNDS = 3


def existing_hashes():
    with (REPORTS / 'metrics/phase1_test_benchmark.csv').open(encoding='utf-8-sig') as f:
        return {r['architecture']: r for r in csv.DictReader(f)}


def gather_models():
    native = existing_hashes()
    models = []
    for key, (label, weights) in EXISTING.items():
        if digest(weights) != native[key]['weights_sha256']:
            raise RuntimeError(f'{weights} differs from the benchmarked {key} checkpoint')
        models.append((key, label, weights, 'existing', dict(native_test_mAP50_95=float(native[key]['mAP50_95']))))
    for group, table in (('additional', ADDITIONAL), ('control', CONTROL)):
        for key, (label, weights) in table.items():
            selection = ADD / key / 'selection.json'
            if not selection.exists():
                print(f'SKIP {key}: no locked validation selection (training not finished or failed)', flush=True)
                continue
            sel = read_json(selection)
            if sel['deploy_sha256'] != digest(weights):
                raise RuntimeError(f'{weights} differs from the checkpoint locked in {selection}')
            done = sel['completed']
            models.append((key, label, weights, group, dict(best_epoch=done['best_epoch']['epoch'],
                                                            actual_epochs=done['actual_epochs'],
                                                            chosen_config=sel['chosen_config'])))
    return models


ARCHITECTURE = {'yolo11s_full100': 'yolo11s', 'faster_rcnn_full50': 'faster_rcnn', 'rtdetr_full100': 'rtdetr'}


def write_tables(test_rows, val_rows):
    """All runs (incl. full-schedule controls) -> *_all_runs.csv; then one representative run per
    architecture, chosen on VALIDATION mAP50-95, -> the requested 6-model tables."""
    fields = list(dict.fromkeys(k for r in test_rows for k in r))
    val_fields = list(dict.fromkeys(k for r in val_rows for k in r))
    write_csv(REPORTS / 'metrics/phase1_all_models_comparison_all_runs.csv', test_rows, fields)
    write_csv(REPORTS / 'metrics/phase1_all_models_validation_all_runs.csv', val_rows, val_fields)
    best = {}
    for v in val_rows:
        arch = ARCHITECTURE.get(v['model'], v['model'])
        if arch not in best or v['mAP50_95'] > best[arch]['mAP50_95']:
            best[arch] = v
    chosen = {v['model']: arch for arch, v in best.items()}
    pick = lambda rows: [dict(r, architecture=chosen[r['model']], representative_rule='best validation mAP50-95 among runs of this architecture')
                         for r in rows if r['model'] in chosen]
    write_csv(REPORTS / 'metrics/phase1_all_models_comparison.csv', pick(test_rows), [*fields, 'architecture', 'representative_rule'])
    write_csv(REPORTS / 'metrics/phase1_all_models_validation.csv', pick(val_rows), [*val_fields, 'architecture', 'representative_rule'])
    additional = [r for r in pick(test_rows) if r['architecture'] in ADDITIONAL]
    if additional:
        write_csv(REPORTS / 'metrics/phase1_additional_models.csv', additional, [*fields, 'architecture', 'representative_rule'])


def read_rows(name):
    with (REPORTS / 'metrics' / name).open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def main():
    models = gather_models()
    test_rows, val_rows = [], []
    for key, label, weights, group, extra in models:
        print(f'== {label}', flush=True)
        summary = evaluate_model(key, weights, ADD / 'common_eval' / key)
        for split, rows in (('test', test_rows), ('val', val_rows)):
            rows.append(dict(flat_row(summary, split), label=label, framework=FRAMEWORK[key], group=group, **extra))  # flat_row carries 'model'
        write_tables(test_rows, val_rows)  # saved after every model

    # --- paired bootstrap on test; reference = best on VALIDATION (mAP50-95 -> recall), never test
    reference = sorted(val_rows, key=lambda r: (-r['mAP50_95'], -r['recall']))[0]['model']
    records = load_split('test')
    stats = {}
    for row in test_rows:
        saved = json.loads((ADD / 'common_eval' / row['model'] / 'eval_test/predictions.json').read_text(encoding='utf-8'))
        if [p['file_name'] for p in saved] != [r['file_name'] for r in records]:
            raise RuntimeError(f'{row["model"]}: prediction order differs from the test manifest')
        preds = [dict(boxes=np.array(p['boxes'], dtype=np.float64).reshape(-1, 4), scores=np.array(p['scores'], dtype=np.float64))
                 for p in saved]
        stats[row['model']] = image_stats(records, preds, row['conf_threshold'])
    print(f'Bootstrap (1000 paired resamples), reference by validation = {reference}', flush=True)
    ci = bootstrap(stats, reference)
    write_json(ADD / 'common_eval' / 'bootstrap_test.json', dict(reference_by_validation=reference, n_boot=1000, seed=42,
                                                                 method='paired percentile bootstrap over test images', results=ci))
    for row in test_rows:
        row.update({k: v for k, v in ci[row['model']].items() if k.startswith(('mAP50_95', 'recall', 'f1', 'mAP50_ci', 'precision_ci'))})
        row['reference_by_validation'] = reference

    # --- latency: interleaved rounds, all models in one session
    predictors = {row['model']: make_predictor(row['weights']) for row in test_rows}
    rounds = []
    for r in range(LATENCY_ROUNDS):
        for key, predictor in predictors.items():
            _, speed = collect(predictor, records)
            rounds.append(dict(round=r + 1, model=key, **speed))
            print(f'latency round {r + 1} {key}: {speed["total"]:.1f} ms', flush=True)
    write_csv(REPORTS / 'metrics/phase1_all_models_latency_rounds.csv', rounds, list(rounds[0]))
    for row in test_rows:
        mine = [x for x in rounds if x['model'] == row['model']]
        for k in ('preprocess', 'inference', 'postprocess', 'total', 'peak_vram_mb'):
            row[f'{k}_ms' if k != 'peak_vram_mb' else k] = float(np.median([x[k] for x in mine]))
        row['total_ms_min'] = min(x['total'] for x in mine)
        row['total_ms_max'] = max(x['total'] for x in mine)
        row['latency_note'] = f'median of {LATENCY_ROUNDS} interleaved rounds over 176 test images'
    write_tables(test_rows, val_rows)
    print('done', [(r['label'], round(r['mAP50_95'], 4)) for r in test_rows], flush=True)


if __name__ == '__main__':
    import sys
    if '--tables-only' in sys.argv:  # re-derive representative tables from the saved all-runs CSVs
        write_tables(read_rows('phase1_all_models_comparison_all_runs.csv'), read_rows('phase1_all_models_validation_all_runs.csv'))
    else:
        main()
