"""Fine-tune the two best phase-1 detectors as 5-class acne-type detectors on ACNE-DET (one-stage).

Starts from the ACNE-DET fine-tuned 1-class checkpoints (runs/phase1_combined/*_1280_ft_acnedet), whose
backbone already finds ACNE-DET comedones; the 1-class head is replaced by a 5-class head (Ultralytics
transfers every layer whose shape matches). Data: datasets/phase2_det5_acnedet (master split 190/40/45).
Recipe per architecture as in phase 1 fine-tuning, longer schedule because the training set is small:
  rtdetr : AdamW lr0 1e-4, wd 1e-4, warmup_bias_lr 0, batch 2, max_det 300
  yolo26s: AdamW lr0 5e-4, wd 5e-4, batch 4, max_det 500
Common: imgsz 1280, 100 epochs, patience 25, warmup 3, close_mosaic 10, lrf 0.01, cache RAM, workers 0,
seed 42, deterministic, AMP. best.pt = best ACNE-DET val fitness; the test split is never read here.
Cross-validation (--fold k --task det5|det1, RT-DETR only): data datasets/phase2_cv/fold{k}/{task}; starts
from the ACNE04-only checkpoint models/phase1/rtdetr_1280_best.pt because the ACNE-DET fine-tuned ones have
seen the master-split training images, which are test images in some folds. det1 = the pipeline's 1-class
detector trained on exactly the same fold data. Same recipe as above.
Usage: venv/Scripts/python.exe -u scripts/phase2/train_det5.py --model rtdetr
       venv/Scripts/python.exe -u scripts/phase2/train_det5.py --model rtdetr --fold 0 --task det1
"""
import argparse
import sys
from datetime import datetime, timezone

from common2 import REPORTS, ROOT, digest, read_json, write_json

sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from train_ultralytics_common import best_epoch, environment  # noqa: E402

DATA = ROOT / 'datasets/phase2_det5_acnedet/data.yaml'
RUNS = ROOT / 'runs/phase2_det5'
START = ROOT / 'runs/phase1_combined'
COMMON = dict(imgsz=1280, epochs=100, patience=25, warmup_epochs=3, close_mosaic=10, lrf=.01, optimizer='AdamW',
              cache='ram', workers=0, seed=42, deterministic=True, amp=True)
MODELS = {'rtdetr': dict(start='rtdetr_1280_ft_acnedet', detr=True, cfg=dict(batch=2, lr0=1e-4, weight_decay=1e-4,
                                                                              warmup_bias_lr=0.0, max_det=300)),
          'yolo26s': dict(start='yolo26s_1280_ft_acnedet', detr=False, cfg=dict(batch=4, lr0=5e-4, weight_decay=5e-4, max_det=500))}


def main():
    import torch
    import ultralytics.engine.trainer as trainer_module
    from ultralytics import RTDETR, YOLO
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=list(MODELS), required=True)
    p.add_argument('--fold', type=int)
    p.add_argument('--task', choices=['det5', 'det1'], default='det5')
    p.add_argument('--resume', action='store_true', help='continue an interrupted run from its weights/last.pt')
    a = p.parse_args()
    spec = MODELS[a.model]
    if a.fold is None:
        runs, name, data = RUNS, f'{a.model}_1280_det5', DATA
        start = START / spec['start'] / 'weights/best.pt'
    else:
        if a.model != 'rtdetr':
            p.error('cross-validation is defined for rtdetr only')
        runs, name = ROOT / f'runs/phase2_cv/fold{a.fold}', a.task
        data = ROOT / f'datasets/phase2_cv/fold{a.fold}/{a.task}/data.yaml'
        start = ROOT / 'models/phase1/rtdetr_1280_best.pt'
    run = runs / name
    if (run / 'completed.json').exists():
        print('Already completed:', run)
        return
    last = run / 'weights/last.pt'
    resume = run.exists() and a.resume and last.exists()
    if run.exists() and not resume:
        raise RuntimeError(f'Incomplete run exists: {run}; inspect it, it is never reset silently (--resume continues it)')
    cert = REPORTS / 'phase1_amp_certificate.json'
    if cert.is_file() and read_json(cert).get('environment') == environment() and read_json(cert).get('amp_enabled') is True:
        trainer_module.check_amp = lambda _: True  # same AMP policy as phase 1
    model = (RTDETR if spec['detr'] else YOLO)(str(last if resume else start))
    cfg = dict(COMMON, **spec['cfg'])
    model.add_callback('on_fit_epoch_end', lambda t: write_json(runs / f'{name}_status.json', dict(
        updated_utc=datetime.now(timezone.utc).isoformat(), epoch=t.epoch + 1, epochs=cfg['epochs'],
        metrics={k: float(v) for k, v in t.metrics.items()})))
    if resume:  # Ultralytics restores args, optimizer, EMA and epoch from last.pt
        print('Resuming', last, flush=True)
        model.train(resume=True)
    else:
        model.train(data=str(data), device='0', project=str(runs), name=name, exist_ok=False, plots=True, **cfg)
    best = run / 'weights/best.pt'
    write_json(run / 'completed.json', dict(model=a.model, data=str(data), start_weights=str(start), start_sha256=digest(start), settings=cfg,
                                            best_epoch=best_epoch(run), actual_epochs=model.trainer.epoch + 1,
                                            best_sha256=digest(best), peak_train_vram_gb=torch.cuda.max_memory_reserved() / 1024 ** 3,
                                            resumed_after_interruption=resume,
                                            environment=environment(), completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
