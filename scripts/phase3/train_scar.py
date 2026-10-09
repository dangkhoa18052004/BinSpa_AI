"""Train one scar-detector run (YOLO26s / YOLO11s, COCO-pretrained) on datasets/phase3_scar_detection.

Tuning budget: 3 configurations per architecture (scars are small: median ~50 px in 3456x5184 photos):
  c640       imgsz 640,  batch 8, AdamW lr0 1e-3
  c1280      imgsz 1280, batch 4, AdamW lr0 1e-3
  c1280_lr5  imgsz 1280, batch 4, AdamW lr0 5e-4
Common (phase-1 recipe): 100 epochs, patience 20, lrf 0.01, wd 5e-4, close_mosaic 10, cache RAM,
workers 0, seed 42, deterministic, AMP, max_det 500. nbs 64 keeps the effective batch equal.
Ultralytics keeps best.pt by validation fitness; the test split is not used here.
Usage: venv/Scripts/python.exe -u scripts/phase3/train_scar.py --model yolo26s --config c1280
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import REPORTS, digest, read_json, write_json  # noqa: E402

DATASET = ROOT / 'datasets/phase3_scar_detection'
RUNS = ROOT / 'runs/phase3'
WEIGHTS = {'yolo26s': 'yolo26s.pt', 'yolo11s': 'yolo11s.pt'}
CONFIGS = {'c640': dict(imgsz=640, batch=8, lr0=1e-3), 'c1280': dict(imgsz=1280, batch=4, lr0=1e-3),
           'c1280_lr5': dict(imgsz=1280, batch=4, lr0=5e-4)}
COMMON = dict(epochs=100, patience=20, optimizer='AdamW', lrf=.01, weight_decay=.0005, close_mosaic=10, cache='ram',
              workers=0, seed=42, deterministic=True, amp=True, max_det=500)


def main():
    import torch
    import ultralytics.engine.trainer as trainer_module
    from ultralytics import YOLO
    from train_ultralytics_common import best_epoch, environment
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=list(WEIGHTS), required=True)
    p.add_argument('--config', choices=list(CONFIGS), required=True)
    a = p.parse_args()
    name = f'{a.model}_{a.config}'
    run = RUNS / name
    if (run / 'completed.json').exists():
        print('Already completed:', run)
        return
    if run.exists():
        raise RuntimeError(f'Incomplete run exists: {run}; inspect it, it is never reset silently')
    cert = REPORTS / 'phase1_amp_certificate.json'
    if cert.is_file() and read_json(cert).get('environment') == environment() and read_json(cert).get('amp_enabled') is True:
        trainer_module.check_amp = lambda _: True  # same AMP policy as phase 1 (the self-test stalls on this GPU)
    cfg = dict(COMMON, **CONFIGS[a.config])
    model = YOLO(str(ROOT / 'models/phase1/pretrained' / WEIGHTS[a.model]))
    model.add_callback('on_fit_epoch_end', lambda t: write_json(RUNS / f'{name}_status.json', dict(
        updated_utc=datetime.now(timezone.utc).isoformat(), epoch=t.epoch + 1, metrics={k: float(v) for k, v in t.metrics.items()})))
    model.train(data=str(DATASET / 'data.yaml'), device='0', project=str(RUNS), name=name, exist_ok=False, plots=True, **cfg)
    best = run / 'weights/best.pt'
    write_json(run / 'completed.json', dict(model=a.model, config=a.config, settings=cfg, best_epoch=best_epoch(run),
                                            actual_epochs=model.trainer.epoch + 1, batch=int(model.trainer.batch_size),
                                            amp_enabled=bool(model.trainer.amp), best_sha256=digest(best),
                                            peak_train_vram_gb=torch.cuda.max_memory_reserved() / 1024 ** 3,
                                            dataset_labels_sha256=digest(DATASET / 'data.yaml'), environment=environment(),
                                            completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
