"""Retrain YOLO11s for the full 100 epochs (patience=100) — the original run early-stopped at
epoch 77 (best 57) with patience=20 and missed the low-LR and close_mosaic phases that
YOLO26n/26s (both ran 100/100) received.

Every other argument is copied from train_phase1.py (AdamW 1e-3, lrf 0.01, wd 5e-4, batch 8,
imgsz 640, seed 42, close_mosaic 10, cache RAM, workers 0, max_det 500). The original run
`runs/phase1/benchmark_yolo11s_e100_s42_b8_w0_cram` and models/phase1/best_acne_detector.pt
are never modified; results go to a new run and reports/phase1_additional_models/yolo11s_full100/.
Usage: venv/Scripts/python.exe -u scripts/phase1/train_yolo11s_full100.py
"""
import shutil
from datetime import datetime, timezone

from common import DATASET, REPORTS, ROOT, digest, read_json, verify_dataset, write_json
from evaluate_detector_common import ADD, evaluate_model, flat_row, portable_data_yaml
from train_ultralytics_common import best_epoch, environment

RUN_NAME = 'benchmark_yolo11s_e100_s42_b8_w0_cram_p100'
RUN = ROOT / 'runs/phase1' / RUN_NAME
OUT = ADD / 'yolo11s_full100'


def train():
    import ultralytics.engine.trainer as trainer_module
    from ultralytics import YOLO
    if (RUN / 'completed.json').exists():
        return read_json(RUN / 'completed.json')
    if RUN.exists():
        raise RuntimeError(f'Incomplete run exists: {RUN}; inspect it, it is never reset silently')
    cert = REPORTS / 'phase1_amp_certificate.json'
    if cert.is_file() and read_json(cert).get('environment') == environment() and read_json(cert).get('amp_enabled') is True:
        trainer_module.check_amp = lambda _: True  # same AMP policy as the original benchmark
    model = YOLO(str(ROOT / 'models/phase1/pretrained/yolo11s.pt'))
    model.add_callback('on_fit_epoch_end', lambda t: write_json(OUT / 'status.json', dict(
        updated_utc=datetime.now(timezone.utc).isoformat(), state='training', run=RUN_NAME, epoch=min(t.epoch + 1, 100),
        epochs=100, metrics={k: float(v) for k, v in t.metrics.items()})))
    model.train(data=str(portable_data_yaml()), epochs=100, imgsz=640, batch=8, device='0', seed=42, deterministic=True,
                patience=100, workers=0, amp=True, cache='ram', optimizer='AdamW', lr0=.001, lrf=.01, weight_decay=.0005,
                close_mosaic=10, project=str(ROOT / 'runs/phase1'), name=RUN_NAME, exist_ok=False, plots=True, max_det=500)
    best = RUN / 'weights/best.pt'
    saved = dict(model='yolo11s', epochs=100, patience=100, actual_epochs=min(model.trainer.epoch + 1, 100),
                 best_epoch=best_epoch(RUN), batch=int(model.trainer.batch_size), amp_enabled=bool(model.trainer.amp),
                 seed=42, manifest_sha256=digest(DATASET / 'manifest.json'), best_sha256=digest(best),
                 environment=environment(), completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(RUN / 'completed.json', saved)
    return saved


def main():
    verify_dataset(DATASET, require_review=True)
    OUT.mkdir(parents=True, exist_ok=True)
    saved = train()
    shutil.copy2(RUN / 'args.yaml', OUT / 'train_config.yaml')
    shutil.copy2(RUN / 'results.csv', OUT / 'train_log_results.csv')
    best = RUN / 'weights/best.pt'
    # best.pt is chosen by Ultralytics on validation mAP50-95; lock before test.
    write_json(OUT / 'selection.json', dict(model='yolo11s_full100', chosen_config='original protocol, patience=100',
                                           final_run=str(RUN), completed=saved, deploy_weights=str(best),
                                           deploy_sha256=digest(best), locked_before_test=True,
                                           locked_utc=datetime.now(timezone.utc).isoformat()))
    summary = evaluate_model('yolo11s_full100', best, OUT)
    write_json(OUT / 'test_metrics.json', flat_row(summary, 'test'))
    write_json(OUT / 'status.json', dict(updated_utc=datetime.now(timezone.utc).isoformat(), state='complete',
                                         test=flat_row(summary, 'test')))


if __name__ == '__main__':
    main()
