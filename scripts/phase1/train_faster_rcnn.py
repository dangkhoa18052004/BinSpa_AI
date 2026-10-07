"""Faster R-CNN ResNet50-FPN (torchvision, COCO_V1 pretrained) on the fixed phase-1 split.

- Dataset adapter reads datasets/phase1_acne_detection/{images,labels}/<split> using the
  manifest's file list (no re-split); YOLO `cls cx cy w h` (normalized) is converted in
  memory to `xmin ymin xmax ymax` pixels. Label files are opened read-only.
- Head replaced by FastRCNNPredictor(2): background + acne.
- Input resolution matches the YOLO models: long side 640 (min_size=max_size=640).
- Lesions are tiny (median ~8 px at 640), so anchor sizes are part of the screened
  configurations (torchvision default 32-512 cannot match most lesions).
- SGD + warm-up + cosine, AMP, grad-norm clipping 10, best checkpoint by validation
  mAP50-95 (common evaluator), early stopping on the same metric.
Usage: venv/Scripts/python.exe -u scripts/phase1/train_faster_rcnn.py
"""
import gc
import math
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from common import DATASET, digest, read_json, verify_dataset, write_csv, write_json
from evaluate_detector_common import ADD, MODELS_DIR, RUNS_ADD, compute_metrics, evaluate_model, flat_row, load_split

MODEL_KEY = 'faster_rcnn'
CACHE_LONG_SIDE = 1024  # in-RAM copy; random crops are then resized to 640 by the model transform
BASE = dict(imgsz=640, batch=4, momentum=0.9, weight_decay=1e-4, warmup_iters=300, min_lr_ratio=0.01,
            clip_grad=10.0, amp=True, trainable_layers=3, hflip=0.5, crop_p=0.5, crop_scale=(0.6, 1.0),
            color_jitter=True, patience=10)
CONFIGS = {
    'cfg_a': dict(BASE, anchors=(16, 32, 64, 128, 256), lr=0.005),
    'cfg_b': dict(BASE, anchors=(8, 16, 32, 64, 128), lr=0.005),
    'cfg_c': dict(BASE, anchors=(8, 16, 32, 64, 128), lr=0.01),
}
SCREEN_EPOCHS, FINAL_EPOCHS = 15, 50


def seed_all(seed=42):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_model(cfg, pretrained=True):
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.rpn import AnchorGenerator
    anchors = AnchorGenerator(sizes=tuple((int(s),) for s in cfg['anchors']), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None, weights_backbone=None,
        trainable_backbone_layers=cfg['trainable_layers'], min_size=cfg['imgsz'], max_size=cfg['imgsz'],
        rpn_anchor_generator=anchors, box_detections_per_img=500, box_score_thresh=0.001,
        rpn_pre_nms_top_n_test=2000, rpn_post_nms_top_n_test=1000, box_batch_size_per_image=512)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    return model


class AcneDataset:
    """Images resized once (long side 1024) and kept in RAM; boxes in that pixel space."""

    def __init__(self, split, cfg=None, train=False, seed=42):
        self.records = load_split(split)
        self.cfg, self.train = cfg, train
        self.rng = np.random.default_rng(seed)
        self.items = []
        for rec in self.records:
            image = cv2.cvtColor(cv2.imread(str(rec['path'])), cv2.COLOR_BGR2RGB)
            s = CACHE_LONG_SIDE / max(image.shape[:2])
            if s < 1:
                image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            else:
                s = 1.0
            self.items.append((image, rec['gt'] * s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        image, boxes = self.items[i]
        boxes = boxes.copy()
        if self.train:
            c, rng = self.cfg, self.rng
            if rng.random() < c['crop_p']:
                h, w = image.shape[:2]
                f = rng.uniform(*c['crop_scale'])
                ch, cw = int(h * f), int(w * f)
                y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                keep = (centers[:, 0] >= x0) & (centers[:, 0] < x0 + cw) & (centers[:, 1] >= y0) & (centers[:, 1] < y0 + ch)
                if keep.any():  # never produce an image without supervision
                    image = image[y0:y0 + ch, x0:x0 + cw]
                    boxes = boxes[keep] - [x0, y0, x0, y0]
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, cw)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, ch)
            if rng.random() < c['hflip']:
                image = image[:, ::-1]
                w = image.shape[1]
                boxes = np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1)
            if c['color_jitter']:
                image = np.clip(image.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255)
            valid = ((boxes[:, 2] - boxes[:, 0]) > 0.5) & ((boxes[:, 3] - boxes[:, 1]) > 0.5)
            boxes = boxes[valid]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255)
        target = dict(boxes=torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                      labels=torch.ones(len(boxes), dtype=torch.int64))
        return tensor, target


def validate(model, dataset, device):
    """Validation metrics with the common evaluator (boxes in cached pixel space)."""
    import torch
    model.eval()
    preds, records = [], []
    with torch.inference_mode():
        for i in range(len(dataset)):
            image, target = dataset[i]
            out = model([image.to(device)])[0]
            scores = out['scores'].float().cpu().numpy().astype(np.float64)
            order = np.argsort(-scores, kind='stable')
            preds.append(dict(boxes=out['boxes'].float().cpu().numpy().astype(np.float64)[order], scores=scores[order]))
            records.append(dict(file_name=dataset.records[i]['file_name'], gt=target['boxes'].numpy().astype(np.float64)))
    metrics, _ = compute_metrics(records, preds)
    return dict(val_mAP50=metrics['mAP50'], val_mAP50_95=metrics['mAP50_95'],
                val_precision=metrics['operating_point']['precision'], val_recall=metrics['operating_point']['recall'])


def train_run(name, cfg, epochs, datasets):
    """Train one config; resumes from last.pth if interrupted; OOM -> smaller batch (new run dir)."""
    import torch
    manifest_sha = digest(DATASET / 'manifest.json')
    for batch in dict.fromkeys([cfg['batch'], min(cfg['batch'], 2), 1]):
        run = RUNS_ADD / f'{name}_b{batch}'
        if (run / 'completed.json').exists():
            saved = read_json(run / 'completed.json')
            if saved['manifest_sha256'] != manifest_sha or saved['config'] != jsonable(cfg) or saved['epochs'] != epochs:
                raise RuntimeError(f'{run} was trained with a different dataset/config')
            return run, saved
        if (run / 'oom.json').exists():
            continue
        try:
            return run, fit(run, dict(cfg, batch=batch, lr=cfg['lr'] * batch / cfg['batch']), cfg, epochs, datasets, manifest_sha)
        except torch.OutOfMemoryError as exc:
            write_json(run / 'oom.json', dict(batch=batch, error=str(exc)))
            print(f'OOM at batch {batch}; retrying with a smaller batch (dataset unchanged)', flush=True)
            gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError('OOM even at batch 1')


def jsonable(cfg):
    return {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}


def fit(run, cfg, requested_cfg, epochs, datasets, manifest_sha):
    import torch
    seed_all(42)
    device = 'cuda'
    run.mkdir(parents=True, exist_ok=True)
    train_set, val_set = datasets
    train_set.cfg, train_set.rng = cfg, np.random.default_rng(42)  # same augmentation stream for every config
    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['batch'], shuffle=True, num_workers=0,
                                         collate_fn=lambda b: tuple(zip(*b)), generator=torch.Generator().manual_seed(42))
    total_iters = epochs * len(loader)

    def lr_lambda(it):
        if it < cfg['warmup_iters']:
            return 0.001 + (1 - 0.001) * it / cfg['warmup_iters']
        progress = (it - cfg['warmup_iters']) / max(1, total_iters - cfg['warmup_iters'])
        return cfg['min_lr_ratio'] + (1 - cfg['min_lr_ratio']) * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    history, start, best, stale = [], 0, -1.0, 0
    if (run / 'last.pth').exists():  # resume an interrupted run of the same config
        state = torch.load(run / 'last.pth', map_location='cpu', weights_only=False)
        if state['config'] != jsonable(cfg):
            raise RuntimeError(f'{run}/last.pth has a different config')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        history, start, best, stale = state['history'], state['epoch'], state['best'], state['stale']
        print(f'Resuming {run.name} from epoch {start}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start, epochs):
        model.train()
        t0, sums, n = time.time(), {}, 0
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast('cuda', enabled=cfg['amp']):
                losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch + 1}: { {k: float(v) for k, v in losses.items()} }')
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg['clip_grad'])
            scaler.step(optimizer); scaler.update(); scheduler.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n += 1
        row = dict(epoch=epoch + 1, **{f'train/{k}': v / n for k, v in sums.items()}, lr=optimizer.param_groups[0]['lr'])
        row.update(validate(model, val_set, device))
        row.update(epoch_seconds=time.time() - t0, peak_vram_gb=torch.cuda.max_memory_reserved() / 1024 ** 3)
        history.append(row)
        print(row, flush=True)
        improved = row['val_mAP50_95'] > best
        if improved:
            best, stale = row['val_mAP50_95'], 0
            torch.save(dict(config=jsonable(cfg), model=model.state_dict(), epoch=epoch + 1, val=row), run / 'best.pth')
        else:
            stale += 1
        write_csv(run / 'results.csv', history, list(history[-1]))
        torch.save(dict(config=jsonable(cfg), model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch + 1, best=best,
                        stale=stale, history=history), run / 'last.pth')
        status_dir = ADD / (f'{MODEL_KEY}_full{FINAL_EPOCHS}' if '_full_b' in run.name else MODEL_KEY)
        write_json(status_dir / 'status.json', dict(updated_utc=datetime.now(timezone.utc).isoformat(), state='training',
                                                       run=run.name, epoch=epoch + 1, epochs=epochs, last=row, best_val_mAP50_95=best))
        if stale >= cfg['patience']:
            print(f'Early stopping at epoch {epoch + 1} (no val mAP50-95 gain for {stale} epochs)', flush=True)
            break
    best_row = max(history, key=lambda r: r['val_mAP50_95'])
    saved = dict(model=MODEL_KEY, config=jsonable(requested_cfg), effective_config=jsonable(cfg), epochs=epochs,
                 actual_epochs=len(history), best_epoch=dict(epoch=best_row['epoch'], **{k: v for k, v in best_row.items() if k.startswith('val_')}),
                 batch=cfg['batch'], amp_enabled=cfg['amp'], peak_train_vram_gb=max(r['peak_vram_gb'] for r in history),
                 seed=42, manifest_sha256=manifest_sha, best_sha256=digest(run / 'best.pth'),
                 completed_utc=datetime.now(timezone.utc).isoformat())
    (run / 'last.pth').unlink()  # optimizer state is not needed after completion (saves ~300 MB)
    write_json(run / 'completed.json', saved)
    plot_history(history, run / 'train_curves.png')
    del model, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return saved


def plot_history(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in [k for k in history[0] if k.startswith('train/')]:
        axes[0].plot(epochs, [r[k] for r in history], label=k.removeprefix('train/'))
    axes[0].set(title='Training losses', xlabel='epoch'); axes[0].legend(fontsize=8)
    for k in ('val_mAP50', 'val_mAP50_95', 'val_recall', 'val_precision'):
        axes[1].plot(epochs, [r[k] for r in history], label=k.removeprefix('val_'))
    axes[1].set(title='Validation (common evaluator)', xlabel='epoch'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def finalize(key, run, saved, cfg, chosen, out, deploy, env):
    """Deploy FP16 copy, lock selection (before test), evaluate val -> test with the common evaluator."""
    import torch
    previous = read_json(out / 'selection.json') if (out / 'selection.json').exists() else None
    if previous and previous['final_run'] != str(run):
        raise RuntimeError(f'{out} already locked another run ({previous["final_run"]}); refusing to overwrite')
    if not (previous and deploy.exists() and previous['deploy_sha256'] == digest(deploy)):
        # Deployment artifact: FP16 weights + config (like Ultralytics' stripped .pt); loaded back as FP32.
        state = torch.load(run / 'best.pth', map_location='cpu', weights_only=False)
        if deploy.exists():
            raise RuntimeError(f'{deploy} exists but is not the locked checkpoint; refusing to overwrite')
        tmp = deploy.with_suffix('.tmp')
        torch.save(dict(config=state['config'], model={k: v.half() if v.is_floating_point() else v for k, v in state['model'].items()},
                        epoch=state['epoch'], val=state['val'], framework=env), tmp)
        tmp.replace(deploy)
    shutil.copy2(run / 'results.csv', out / 'train_log_results.csv')
    shutil.copy2(run / 'train_curves.png', out / 'train_curves.png')
    write_json(out / 'train_config.json', dict(config=jsonable(cfg), screen_epochs=SCREEN_EPOCHS, final_epochs=FINAL_EPOCHS,
                                               pretrained='torchvision FasterRCNN_ResNet50_FPN_Weights.COCO_V1',
                                               cache_long_side=CACHE_LONG_SIDE, optimizer='SGD', scheduler='linear warm-up + cosine'))
    write_json(out / 'selection.json', dict(model=key, chosen_config=chosen, config=jsonable(cfg), final_run=str(run),
                                            completed=saved, deploy_weights=str(deploy), deploy_sha256=digest(deploy),
                                            locked_before_test=True, locked_utc=datetime.now(timezone.utc).isoformat()))
    # Evaluations made by the first predictor version (non-antialiased resize) are kept for audit, not reused.
    for split in ('val', 'test'):
        old, stale = out / f'eval_{split}', out / f'eval_{split}_stale_bilinear_predictor'
        if old.exists() and not (old / 'predictor_v2.flag').exists() and not stale.exists():
            old.rename(stale)
    summary = evaluate_model(key, deploy, out)
    for split in ('val', 'test'):
        (out / f'eval_{split}' / 'predictor_v2.flag').write_text('INTER_AREA pre-resize (matches training)\n')
    write_json(out / 'test_metrics.json', flat_row(summary, 'test'))
    write_json(out / 'val_metrics.json', flat_row(summary, 'val'))
    write_json(out / 'status.json', dict(updated_utc=datetime.now(timezone.utc).isoformat(), state='complete', test=flat_row(summary, 'test')))


def main():
    import argparse
    from train_ultralytics_common import environment, rank_key, val_score
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full-schedule', action='store_true',
                    help='Retrain the validation-chosen config for all FINAL_EPOCHS (patience=FINAL_EPOCHS) as faster_rcnn_full50')
    args = ap.parse_args()
    verify_dataset(DATASET, require_review=True)
    out = ADD / MODEL_KEY
    out.mkdir(parents=True, exist_ok=True)
    env = environment()
    write_json(out / 'environment.json', env)
    print(env, flush=True)
    datasets = (AcneDataset('train', train=True), AcneDataset('val'))
    print(f'Loaded train={len(datasets[0])} val={len(datasets[1])} images into RAM', flush=True)
    screening = []
    for key, cfg in CONFIGS.items():
        run, saved = train_run(f'{MODEL_KEY}_screen_{key}_e{SCREEN_EPOCHS}', cfg, SCREEN_EPOCHS, datasets)
        # v2 = predictor with INTER_AREA pre-resize; v1 files (non-antialiased resize) are left untouched for audit.
        score_file = run / 'val_score_common_v2.json'
        score = read_json(score_file) if score_file.exists() else val_score(run / 'best.pth')
        write_json(score_file, score)
        screening.append(dict(config=key, run=str(run), **score, best_epoch=saved['best_epoch']['epoch'],
                              actual_epochs=saved['actual_epochs'], batch=saved['batch']))
        write_csv(out / 'screening_val.csv', screening, list(screening[0]))
    chosen = sorted(screening, key=rank_key)[0]
    previous = read_json(out / 'screening_selection.json') if (out / 'screening_selection.json').exists() else None
    if previous and previous['chosen'] != chosen['config']:
        raise RuntimeError(f'Re-scored validation now prefers {chosen["config"]} over {previous["chosen"]}; '
                           'decide explicitly before training another final run')
    write_json(out / 'screening_selection.json', dict(rule='validation mAP50-95 (ties within 0.002) -> recall -> total latency',
                                                      screen_epochs=SCREEN_EPOCHS, ranking=sorted(screening, key=rank_key),
                                                      chosen=chosen['config'], scorer='common evaluator, predictor v2'))
    cfg = CONFIGS[chosen['config']]
    if args.full_schedule:
        key, cfg = f'{MODEL_KEY}_full{FINAL_EPOCHS}', dict(cfg, patience=FINAL_EPOCHS)
        run, saved = train_run(f'{MODEL_KEY}_final_{chosen["config"]}_e{FINAL_EPOCHS}_full', cfg, FINAL_EPOCHS, datasets)
        out = ADD / key
        out.mkdir(parents=True, exist_ok=True)
        deploy = MODELS_DIR / f'{key}_best.pth'
    else:
        key = MODEL_KEY
        run, saved = train_run(f'{MODEL_KEY}_final_{chosen["config"]}_e{FINAL_EPOCHS}', cfg, FINAL_EPOCHS, datasets)
        deploy = MODELS_DIR / 'faster_rcnn_best.pth'
    del datasets
    gc.collect()
    finalize(key, run, saved, cfg, chosen['config'], out, deploy, env)


if __name__ == '__main__':
    sys.exit(main())
