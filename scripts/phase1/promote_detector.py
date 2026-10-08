"""Make the model chosen in compare_1280.py the phase-1 deployment detector.

Copies every 1280 candidate to models/phase1/<name>_best.pt, keeps the previous detector as
models/phase1/best_acne_detector_<previous architecture>_<imgsz>.pt (never overwritten), installs
the chosen weights as best_acne_detector.pt and rewrites model_metadata.json with the imgsz and
validation-chosen confidence threshold that predict_acne.py reads as defaults.
Usage: venv/Scripts/python.exe scripts/phase1/promote_detector.py
"""
import json
import shutil
from datetime import datetime, timezone

from common import DATASET, ROOT, digest, read_json, write_json
from evaluate_detector_common import ADD

MODELS = ROOT / 'models/phase1'
ARCH = {'yolo26s_1280': 'yolo26s', 'yolo11s_1280': 'yolo11s', 'rtdetr_1280': 'rtdetr-l'}


def main():
    comparison = read_json(ADD / 'final_1280_comparison.json')
    chosen = comparison['chosen_by_validation']
    target, meta_path = MODELS / 'best_acne_detector.pt', MODELS / 'model_metadata.json'
    for name in ARCH:
        weights = json.loads((ADD / name / 'common_eval_summary.json').read_text(encoding='utf-8'))['weights']
        copy = MODELS / f'{name}_best.pt'
        if not copy.exists():
            shutil.copy2(weights, copy)
        if digest(copy) != comparison['models'][name]['weights_sha256']:
            raise RuntimeError(f'{copy} does not match the evaluated weights')
    source = MODELS / f'{chosen}_best.pt'
    previous = read_json(meta_path)
    if digest(target) != digest(source):
        backup = MODELS / f'best_acne_detector_{previous["architecture"]}_{previous["imgsz"]}.pt'
        if backup.exists() and digest(backup) != digest(target):
            raise RuntimeError(f'{backup} exists with different weights; refusing to overwrite')
        shutil.copy2(target, backup)
        write_json(MODELS / f'model_metadata_{previous["architecture"]}_{previous["imgsz"]}.json', previous)
        shutil.copy2(source, target)
    row = comparison['models'][chosen]
    write_json(meta_path, dict(
        status='final', architecture=ARCH[chosen], imgsz=row['imgsz'], operating_conf=row['conf'], max_det=500, nms_iou=0.7,
        selected_weights_sha256=digest(target), source_run=str(json.loads((ADD / chosen / 'selection.json').read_text(encoding='utf-8'))['final_run']),
        selection_rule=comparison['selection_rule'], validation_tie_group=comparison['validation_tie_group'],
        validation={k: row[k] for k in ('val_mAP50_95', 'val_recall')},
        test={k: row[k] for k in row if k.startswith('test_')}, e2e_latency_ms_median=row['e2e_ms_median'],
        comparison='reports/phase1_additional_models/final_1280_comparison.json', report='reports/phase1_resolution_study.md',
        alternatives={k: f'models/phase1/{k}_best.pt' for k in ARCH if k != chosen},
        previous_detector=dict(weights=f'models/phase1/best_acne_detector_{previous["architecture"]}_{previous["imgsz"]}.pt',
                               architecture=previous['architecture'], imgsz=previous['imgsz'],
                               sha256=previous['selected_weights_sha256']),
        dataset_manifest_sha256=digest(DATASET / 'manifest.json'), dataset_version=previous.get('dataset_version'),
        promoted_utc=datetime.now(timezone.utc).isoformat(),
        disclaimer='Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da.'))
    print('Deployed', chosen, '->', target)


if __name__ == '__main__':
    main()
