"""Shared screening + final training loop for Ultralytics detectors (YOLOv5s, RT-DETR).

Protocol (equal budget for every additional model):
1. Screen at most 3 configurations for `screen_epochs` each; rank them on the
   VALIDATION split with the common evaluator (mAP50-95, then recall, then latency).
2. Train the selected configuration for the full schedule; Ultralytics keeps the
   best epoch by validation fitness.
3. Lock the selection (selection.json) BEFORE any test-set evaluation, then
   evaluate val -> test with evaluate_detector_common and save artifacts at once.
Runs are never overwritten; a completed run is reused, an incomplete one stops.
"""
import gc
import importlib.metadata
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import DATASET, REPORTS, ROOT, digest, read_json, verify_dataset, write_csv, write_json
from evaluate_detector_common import ADD, MODELS_DIR, RUNS_ADD, compute_metrics, collect, evaluate_model, \
    flat_row, load_split, make_predictor, portable_data_yaml

PRETRAINED = MODELS_DIR / 'pretrained'


def environment():
    import torch
    return dict(python=sys.version, packages={p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'ultralytics')},
                cuda_available=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                torch_cuda=torch.version.cuda)


def status(model, **data):
    write_json(ADD / model / 'status.json', dict(updated_utc=datetime.now(timezone.utc).isoformat(), **data))


def train_run(model_key, weights_name, run_name, cfg, epochs, model_cls):
    """One Ultralytics training run with the OOM fallback used in train_phase1.py."""
    import torch
    import ultralytics.engine.trainer as trainer_module
    manifest_sha = digest(DATASET / 'manifest.json')
    source = PRETRAINED / weights_name  # downloads stay inside the workspace
    PRETRAINED.mkdir(parents=True, exist_ok=True)
    batch = cfg['batch']
    for candidate in dict.fromkeys([batch, min(batch, 4), min(batch, 2), 1]):
        name = f'{run_name}_b{candidate}'
        run = RUNS_ADD / name
        if (run / 'completed.json').exists():
            saved = read_json(run / 'completed.json')
            if saved['manifest_sha256'] != manifest_sha or saved['config'] != cfg or saved['epochs'] != epochs:
                raise RuntimeError(f'{run} was trained with a different dataset/config; choose a new run name')
            if digest(run / 'weights/best.pt') != saved['best_sha256']:
                raise RuntimeError(f'{run}: best.pt changed after completion')
            return run, saved
        if (run / 'oom.json').exists():
            continue
        if run.exists():
            raise RuntimeError(f'Incomplete run exists: {run}. Inspect/resume it explicitly; it is never reset silently.')
        write_json(ADD / model_key / 'run_contracts' / f'{name}.json',
                   dict(manifest_sha256=manifest_sha, epochs=epochs, config=cfg, batch=candidate, weights=weights_name))
        original_amp_check = trainer_module.check_amp
        model = None
        try:
            # Same policy as the 3 existing models: the Ultralytics AMP self-test stalled on this GPU;
            # reuse the smoke-run certificate only when GPU + library versions are identical.
            certificate = REPORTS / 'phase1_amp_certificate.json'
            if certificate.is_file():
                cert = read_json(certificate)
                if cert.get('environment') == environment() and cert.get('amp_enabled') is True:
                    trainer_module.check_amp = lambda _: True
            model = model_cls(str(source))

            def on_epoch_end(trainer):
                values = {k: float(v) for k, v in trainer.metrics.items()}
                if not all(math.isfinite(v) for v in values.values()):
                    raise RuntimeError('Non-finite validation metrics/losses; aborting run')
                status(model_key, state='training', run=name, epoch=min(trainer.epoch + 1, epochs), epochs=epochs,
                       batch=int(trainer.batch_size), metrics=values,
                       gpu_mem_gb=torch.cuda.max_memory_reserved() / 1024 ** 3)
            model.add_callback('on_fit_epoch_end', on_epoch_end)
            args = {k: v for k, v in cfg.items() if k != 'batch'}
            model.train(data=str(portable_data_yaml()), epochs=epochs, batch=candidate, imgsz=640, device='0', seed=42,
                        deterministic=True, workers=0, amp=True, project=str(RUNS_ADD), name=name, exist_ok=False,
                        plots=True, **args)
            run = Path(model.trainer.save_dir)
            best = run / 'weights/best.pt'
            if not best.is_file():
                raise RuntimeError('Training produced no best checkpoint')
            saved = dict(model=model_key, config=cfg, epochs=epochs, actual_epochs=min(model.trainer.epoch + 1, epochs),
                         best_epoch=best_epoch(run), batch=int(model.trainer.batch_size), requested_batch=candidate,
                         amp_enabled=bool(model.trainer.amp), peak_train_vram_gb=torch.cuda.max_memory_reserved() / 1024 ** 3,
                         seed=42, manifest_sha256=manifest_sha, best_sha256=digest(best), environment=environment(),
                         completed_utc=datetime.now(timezone.utc).isoformat())
            write_json(run / 'completed.json', saved)
            return run, saved
        except Exception as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            run.mkdir(parents=True, exist_ok=True)
            write_json(run / 'oom.json', dict(batch=candidate, error=str(exc)))
            print(f'OOM at batch {candidate}; retrying with a smaller batch (dataset unchanged)', flush=True)
        finally:
            trainer_module.check_amp = original_amp_check
            del model
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError('OOM even at batch 1')


def best_epoch(run):
    """Epoch with the highest Ultralytics fitness in results.csv (the one saved as best.pt)."""
    import pandas as pd
    df = pd.read_csv(run / 'results.csv')
    df.columns = [c.strip() for c in df.columns]
    row = df.loc[df['metrics/mAP50-95(B)'].idxmax()]  # Ultralytics 8.4 fitness = mAP50-95 only
    return dict(epoch=int(row['epoch']), val_mAP50=float(row['metrics/mAP50(B)']),
                val_mAP50_95=float(row['metrics/mAP50-95(B)']), val_precision=float(row['metrics/precision(B)']),
                val_recall=float(row['metrics/recall(B)']))


def val_score(weights):
    """Validation-only ranking metrics with the common evaluator (no test access)."""
    records = load_split('val')
    preds, speed = collect(make_predictor(weights), records)
    metrics, _ = compute_metrics(records, preds)
    return dict(mAP50_95=metrics['mAP50_95'], mAP50=metrics['mAP50'], recall=metrics['operating_point']['recall'],
                precision=metrics['operating_point']['precision'], total_ms=speed['total'])


def rank_key(row, tolerance=0.002):
    # mAP50-95 first; within `tolerance` treat as tie -> recall, then latency.
    return (-round(row['mAP50_95'] / tolerance), -row['recall'], row['total_ms'])


def run_model(model_key, weights_name, model_cls, configs, screen_epochs, final_epochs, deploy_name, full_schedule=False):
    """full_schedule=True: reuse the (completed) screening, retrain the chosen config with
    patience=final_epochs as `<model_key>_full<final_epochs>` (no early stopping); originals untouched."""
    if len(configs) > 3:
        raise ValueError('Tuning budget is at most 3 configurations per model')
    verify_dataset(DATASET, require_review=True)
    out = ADD / model_key
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'environment.json', environment())
    screening = []
    for key, cfg in configs.items():
        status(model_key, state='screening', config=key)
        run, saved = train_run(model_key, weights_name, f'{model_key}_screen_{key}_e{screen_epochs}', cfg, screen_epochs, model_cls)
        score_file = run / 'val_score_common.json'
        score = read_json(score_file) if score_file.exists() else val_score(run / 'weights/best.pt')
        write_json(score_file, score)
        screening.append(dict(config=key, run=str(run), **score, best_epoch=saved['best_epoch']['epoch'],
                              actual_epochs=saved['actual_epochs'], batch=saved['batch']))
        write_csv(out / 'screening_val.csv', screening, list(screening[0]))
    chosen = sorted(screening, key=rank_key)[0]
    write_json(out / 'screening_selection.json', dict(rule='validation mAP50-95 (ties within 0.002) -> recall -> total latency',
                                                      screen_epochs=screen_epochs, ranking=sorted(screening, key=rank_key),
                                                      chosen=chosen['config']))
    cfg = configs[chosen['config']]
    run_name = f'{model_key}_final_{chosen["config"]}_e{final_epochs}'
    if full_schedule:
        model_key, cfg, run_name = f'{model_key}_full{final_epochs}', dict(cfg, patience=final_epochs), run_name + '_full'
        deploy_name = f'{model_key}_best.pt'
        out = ADD / model_key
        out.mkdir(parents=True, exist_ok=True)
    status(model_key, state='final_training', config=chosen['config'])
    run, saved = train_run(model_key, weights_name, run_name, cfg, final_epochs, model_cls)
    best = run / 'weights/best.pt'
    deploy = MODELS_DIR / deploy_name
    if deploy.exists() and digest(deploy) != digest(best):
        raise RuntimeError(f'{deploy} already exists with different weights; refusing to overwrite')
    shutil.copy2(best, deploy)
    shutil.copy2(run / 'args.yaml', out / 'train_config.yaml')
    shutil.copy2(run / 'results.csv', out / 'train_log_results.csv')
    for plot in ('results.png', 'labels.jpg', 'val_batch0_pred.jpg', 'val_batch0_labels.jpg'):
        if (run / plot).exists():
            shutil.copy2(run / plot, out / f'train_{plot}')
    # Lock selection before test evaluation.
    write_json(out / 'selection.json', dict(model=model_key, chosen_config=chosen['config'], config=cfg, final_run=str(run),
                                            completed=saved, deploy_weights=str(deploy), deploy_sha256=digest(deploy),
                                            locked_before_test=True, locked_utc=datetime.now(timezone.utc).isoformat()))
    status(model_key, state='evaluating')
    summary = evaluate_model(model_key, deploy, out)
    write_json(out / 'test_metrics.json', flat_row(summary, 'test'))
    write_json(out / 'val_metrics.json', flat_row(summary, 'val'))
    status(model_key, state='complete', test=flat_row(summary, 'test'))
    return summary
