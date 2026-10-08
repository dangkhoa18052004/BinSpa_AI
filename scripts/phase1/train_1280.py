"""Resolution study: retrain the three statistically tied detectors at imgsz 1280.

Each model keeps the configuration that represented it at 640 (chosen on validation); only imgsz
changes, and batch drops to fit 8 GB VRAM (nbs=64 gradient accumulation keeps the effective batch
for the YOLO models). Patience is 20 for all three, as in the original YOLO26s / RT-DETR runs.
  yolo26s: AdamW 1e-3, lrf 0.01, wd 5e-4 (train_phase1.py)
  yolo11s: same recipe as yolo26s
  rtdetr : cfg_b of train_rtdetr.py (AdamW 2e-4, wd 1e-4, warmup_bias_lr 0, max_det 300)
Runs never overwrite the 640 models; evaluation uses the common evaluator at imgsz 1280,
validation first (operating threshold), then test once.

Usage:
  venv/Scripts/python.exe -u scripts/phase1/train_1280.py --model yolo11s --smoke   # VRAM/time check
  venv/Scripts/python.exe -u scripts/phase1/train_1280.py --model yolo11s --batch 4
"""
import argparse
import shutil
import time
from datetime import datetime, timezone

from common import DATASET, REPORTS, ROOT, digest, read_json, verify_dataset, write_json
from evaluate_detector_common import ADD, evaluate_model, flat_row, portable_data_yaml
from train_ultralytics_common import best_epoch, environment

IMGSZ = 1280
COMMON = dict(epochs=100, patience=20, close_mosaic=10, cache='ram', workers=0, seed=42, deterministic=True, amp=True, lrf=.01)
YOLO_RECIPE = dict(COMMON, optimizer='AdamW', lr0=.001, weight_decay=.0005, max_det=500)
MODELS = {
    'yolo26s': dict(weights='yolo26s.pt', detr=False, protocol=YOLO_RECIPE),
    'yolo11s': dict(weights='yolo11s.pt', detr=False, protocol=YOLO_RECIPE),
    'rtdetr': dict(weights='rtdetr-l.pt', detr=True,
                   protocol=dict(COMMON, optimizer='AdamW', lr0=.0002, weight_decay=.0001, warmup_bias_lr=0., max_det=300)),
}


def out_dir(key):
    return ADD / f'{key}_1280'


def status(key, **data):
    write_json(out_dir(key) / 'status.json', dict(updated_utc=datetime.now(timezone.utc).isoformat(), **data))


def make_model(key):
    import ultralytics.engine.trainer as trainer_module
    from ultralytics import RTDETR, YOLO
    cert = REPORTS / 'phase1_amp_certificate.json'
    if cert.is_file() and read_json(cert).get('environment') == environment() and read_json(cert).get('amp_enabled') is True:
        trainer_module.check_amp = lambda _: True  # same AMP policy as the original benchmark
    spec = MODELS[key]
    return (RTDETR if spec['detr'] else YOLO)(str(ROOT / 'models/phase1/pretrained' / spec['weights']))


def smoke(key, batch):
    """One epoch on 25% of train to measure VRAM and speed; results go to a throwaway run."""
    import torch
    model = make_model(key)
    start = time.perf_counter()
    model.train(data=str(portable_data_yaml()), imgsz=IMGSZ, batch=batch, device='0', fraction=.25, val=False, plots=False,
                project=str(ROOT / 'runs/phase1_smoke_1280'), name=f'{key}_b{batch}', exist_ok=True,
                **dict(MODELS[key]['protocol'], epochs=1, cache=False))
    result = dict(model=key, batch=batch, imgsz=IMGSZ, seconds_quarter_epoch=time.perf_counter() - start,
                  peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024 ** 3,
                  total_gb=torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
    write_json(out_dir(key) / f'smoke_b{batch}.json', result)
    print(result)


def run_name(key, batch):
    # yolo26s keeps the name used when it was trained first.
    return f'benchmark_{key}_e100_s42_b{batch}_w0_cram_img{IMGSZ}'


def train(key, batch):
    import torch
    protocol = MODELS[key]['protocol']
    name = run_name(key, batch)
    run = ROOT / 'runs/phase1' / name
    if (run / 'completed.json').exists():
        return run, read_json(run / 'completed.json')
    if run.exists():
        raise RuntimeError(f'Incomplete run exists: {run}; inspect it, it is never reset silently')
    model = make_model(key)
    model.add_callback('on_fit_epoch_end', lambda t: status(
        key, state='training', run=name, epoch=min(t.epoch + 1, protocol['epochs']), epochs=protocol['epochs'],
        metrics={k: float(v) for k, v in t.metrics.items()}, gpu_mem_gb=torch.cuda.max_memory_reserved() / 1024 ** 3))
    model.train(data=str(portable_data_yaml()), imgsz=IMGSZ, batch=batch, device='0', project=str(ROOT / 'runs/phase1'),
                name=name, exist_ok=False, plots=True, **protocol)
    best = run / 'weights/best.pt'
    saved = dict(model=key, imgsz=IMGSZ, config=protocol, batch=int(model.trainer.batch_size),
                 actual_epochs=min(model.trainer.epoch + 1, protocol['epochs']), best_epoch=best_epoch(run),
                 amp_enabled=bool(model.trainer.amp), peak_train_vram_gb=torch.cuda.max_memory_reserved() / 1024 ** 3,
                 manifest_sha256=digest(DATASET / 'manifest.json'), best_sha256=digest(best), environment=environment(),
                 completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(run / 'completed.json', saved)
    return run, saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=list(MODELS), required=True)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    verify_dataset(DATASET, require_review=True)
    out = out_dir(a.model)
    out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        return smoke(a.model, a.batch)
    run, saved = train(a.model, a.batch)
    best = run / 'weights/best.pt'
    shutil.copy2(run / 'args.yaml', out / 'train_config.yaml')
    shutil.copy2(run / 'results.csv', out / 'train_log_results.csv')
    if (out / 'test_metrics.json').exists():
        print('Already evaluated:', out)
        return
    # best.pt is chosen by Ultralytics on validation mAP50-95; lock before test.
    write_json(out / 'selection.json', dict(model=f'{a.model}_1280', chosen_config=f'640 representative config, imgsz {IMGSZ}',
                                           final_run=str(run), completed=saved, deploy_weights=str(best),
                                           deploy_sha256=digest(best), locked_before_test=True,
                                           locked_utc=datetime.now(timezone.utc).isoformat()))
    status(a.model, state='evaluating')
    summary = evaluate_model(f'{a.model}_1280', best, out, imgsz=IMGSZ)
    write_json(out / 'val_metrics.json', flat_row(summary, 'val'))
    write_json(out / 'test_metrics.json', flat_row(summary, 'test'))
    status(a.model, state='complete', test=flat_row(summary, 'test'))


if __name__ == '__main__':
    main()
