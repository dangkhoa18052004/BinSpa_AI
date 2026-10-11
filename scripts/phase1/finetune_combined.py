"""Fine-tune the two phase-1 detectors at 1280 on ACNE04-v2 + ACNE-DET acne lesions (1 class).

Starts from the evaluated 1280 checkpoints (models/phase1/{rtdetr,yolo26s}_1280_best.pt), keeps each
architecture's recipe, lowers the learning rate for fine-tuning and shortens the schedule:
  rtdetr : AdamW lr0 1e-4, wd 1e-4, warmup_bias_lr 0, batch 2, max_det 300
  yolo26s: AdamW lr0 5e-4, wd 5e-4, batch 4, max_det 500
Common: imgsz 1280, 40 epochs, patience 10, warmup 1 epoch, close_mosaic 5, lrf 0.01, cache RAM,
workers 0, seed 42, deterministic, AMP. best.pt = best combined-val mAP50-95 (Ultralytics fitness).
Usage: venv/Scripts/python.exe -u scripts/phase1/finetune_combined.py --model rtdetr
"""
import argparse
from datetime import datetime, timezone

from common import REPORTS, ROOT, digest, read_json, write_json
from train_ultralytics_common import best_epoch, environment

DATA = ROOT / 'datasets/phase1_combined_acnedet/data.yaml'
RUNS = ROOT / 'runs/phase1_combined'
COMMON = dict(imgsz=1280, epochs=40, patience=10, warmup_epochs=1, close_mosaic=5, lrf=.01, optimizer='AdamW',
              cache='ram', workers=0, seed=42, deterministic=True, amp=True)
MODELS = {'rtdetr': dict(start='rtdetr_1280_best.pt', detr=True, cfg=dict(batch=2, lr0=1e-4, weight_decay=1e-4,
                                                                           warmup_bias_lr=0.0, max_det=300)),
          'yolo26s': dict(start='yolo26s_1280_best.pt', detr=False, cfg=dict(batch=4, lr0=5e-4, weight_decay=5e-4, max_det=500))}


def main():
    import torch
    import ultralytics.engine.trainer as trainer_module
    from ultralytics import RTDETR, YOLO
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=list(MODELS), required=True)
    a = p.parse_args()
    spec = MODELS[a.model]
    name = f'{a.model}_1280_ft_acnedet'
    run = RUNS / name
    if (run / 'completed.json').exists():
        print('Already completed:', run)
        return
    if run.exists():
        raise RuntimeError(f'Incomplete run exists: {run}; inspect it, it is never reset silently')
    cert = REPORTS / 'phase1_amp_certificate.json'
    if cert.is_file() and read_json(cert).get('environment') == environment() and read_json(cert).get('amp_enabled') is True:
        trainer_module.check_amp = lambda _: True  # same AMP policy as the original benchmark
    start = ROOT / 'models/phase1' / spec['start']
    model = (RTDETR if spec['detr'] else YOLO)(str(start))
    cfg = dict(COMMON, **spec['cfg'])
    model.add_callback('on_fit_epoch_end', lambda t: write_json(RUNS / f'{name}_status.json', dict(
        updated_utc=datetime.now(timezone.utc).isoformat(), epoch=t.epoch + 1, epochs=cfg['epochs'],
        metrics={k: float(v) for k, v in t.metrics.items()})))
    model.train(data=str(DATA), device='0', project=str(RUNS), name=name, exist_ok=False, plots=True, **cfg)
    best = run / 'weights/best.pt'
    write_json(run / 'completed.json', dict(model=a.model, start_weights=str(start), start_sha256=digest(start), settings=cfg,
                                            best_epoch=best_epoch(run), actual_epochs=model.trainer.epoch + 1,
                                            best_sha256=digest(best), peak_train_vram_gb=torch.cuda.max_memory_reserved() / 1024 ** 3,
                                            environment=environment(), completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
