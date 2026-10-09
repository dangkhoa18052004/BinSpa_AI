"""Train one acne-type classifier run (ConvNeXt-Tiny or EfficientNet-B0, ImageNet-pretrained, timm).

Tuning budget: 3 configurations per architecture, chosen on validation macro-F1 only:
  cfg_a  base lr, cross-entropy (label smoothing 0.1), natural class frequencies
  cfg_b  base lr, class-balanced sampler (weights ~ 1/sqrt(class frequency))
  cfg_c  half lr, class-weighted cross-entropy (weights ~ 1/sqrt(class frequency))
Common: input 224, AdamW (wd 0.05), 1 warm-up epoch + cosine decay, AMP, batch 32, max 30 epochs,
early stopping after 6 epochs without val macro-F1 gain, seed 42. Augmentation (train only): random
resized crop (scale 0.75-1), horizontal flip, rotation <= 15 deg, brightness/contrast/saturation <= 0.1.
The test split is never read here.
Usage: venv/Scripts/python.exe -u scripts/phase2/train_classifier.py --model convnext_tiny --config cfg_a
"""
import argparse
import csv
import math
import random
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import torch

from common2 import ACNE_CLASSES, CROPS, RUNS, digest, write_csv, write_json

ARCH = {'convnext_tiny': ('convnext_tiny.fb_in1k', 1e-4), 'efficientnet_b0': ('efficientnet_b0.ra_in1k', 3e-4)}
CONFIGS = {'cfg_a': dict(lr_scale=1.0, sampler=False, class_weights=False),
           'cfg_b': dict(lr_scale=1.0, sampler=True, class_weights=False),
           'cfg_c': dict(lr_scale=0.5, sampler=False, class_weights=True)}
CACHE_SIDE, INPUT, BATCH, EPOCHS, PATIENCE, SEED = 256, 224, 32, 30, 6, 42


def load_split(split):
    """Crops of one split, resized to CACHE_SIDE, cached as an uint8 .npy (RGB) that workers memory-map."""
    cache = CROPS / f'cache_{split}_{CACHE_SIDE}.npy'
    index = CROPS / f'cache_{split}_{CACHE_SIDE}_paths.txt'
    with (CROPS / 'crop_manifest.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == split]
    paths = [r['crop_path'] for r in rows]
    if not (cache.exists() and index.exists() and index.read_text(encoding='utf-8').split('\n') == paths):
        x = np.stack([cv2.resize(cv2.imread(str(CROPS / p)), (CACHE_SIDE, CACHE_SIDE),
                                 interpolation=cv2.INTER_AREA)[:, :, ::-1] for p in paths])
        np.save(cache, x)
        index.write_text('\n'.join(paths), encoding='utf-8')
    y = np.array([ACNE_CLASSES.index(r['cls']) for r in rows])
    return cache, y, rows


def transforms(train, mean, std):
    from torchvision.transforms import v2
    ops = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if train:
        ops += [v2.RandomResizedCrop(INPUT, scale=(0.75, 1.0), antialias=True), v2.RandomHorizontalFlip(),
                v2.RandomRotation(15), v2.ColorJitter(brightness=.1, contrast=.1, saturation=.1)]
    else:
        ops += [v2.Resize(INPUT, antialias=True)]
    return v2.Compose(ops + [v2.Normalize(mean, std)])


class CropSet(torch.utils.data.Dataset):
    """Opens the cache lazily with mmap so DataLoader workers share it instead of copying it."""

    def __init__(self, cache, y, tf):
        self.cache, self.y, self.tf, self.x = cache, y, tf, None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.x is None:
            self.x = np.load(self.cache, mmap_mode='r')
        return self.tf(np.array(self.x[i])), int(self.y[i])  # copy: mmap rows are read-only


def macro_f1(y, p, k=len(ACNE_CLASSES)):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average='macro', labels=list(range(k)), zero_division=0))


def predict(model, loader):
    model.eval()
    probs = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for xb, _ in loader:
            probs.append(torch.softmax(model(xb.cuda(non_blocking=True)).float(), 1).cpu())
    return torch.cat(probs).numpy()


def main():
    import timm
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=list(ARCH), required=True)
    p.add_argument('--config', choices=list(CONFIGS), required=True)
    a = p.parse_args()
    name, base_lr = ARCH[a.model]
    cfg = CONFIGS[a.config]
    run = RUNS / f'{a.model}_{a.config}'
    if (run / 'completed.json').exists():
        print('Already completed:', run)
        return
    run.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False

    xtr, ytr, _ = load_split('train')
    xva, yva, _ = load_split('val')
    model = timm.create_model(name, pretrained=True, num_classes=len(ACNE_CLASSES)).cuda()
    dcfg = timm.data.resolve_data_config({}, model=model)
    freq = np.bincount(ytr, minlength=len(ACNE_CLASSES))
    inv = 1 / np.sqrt(freq)
    g = torch.Generator().manual_seed(SEED)
    sampler = (torch.utils.data.WeightedRandomSampler(inv[ytr], len(ytr), replacement=True, generator=g)
               if cfg['sampler'] else None)
    train_loader = torch.utils.data.DataLoader(CropSet(xtr, ytr, transforms(True, dcfg['mean'], dcfg['std'])), BATCH,
                                               shuffle=sampler is None, sampler=sampler, generator=g, num_workers=4,
                                               persistent_workers=True, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(CropSet(xva, yva, transforms(False, dcfg['mean'], dcfg['std'])), 64,
                                             num_workers=2, persistent_workers=True, pin_memory=True)
    weight = torch.tensor(inv / inv.mean(), dtype=torch.float32).cuda() if cfg['class_weights'] else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
    lr = base_lr * cfg['lr_scale']
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = EPOCHS * len(train_loader)
    warm = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    scaler = torch.amp.GradScaler()
    history, best, stale = [], -1.0, 0
    write_json(run / 'config.json', dict(model=a.model, timm_name=name, config=a.config, **cfg, lr=lr, weight_decay=0.05,
                                         batch=BATCH, input=INPUT, max_epochs=EPOCHS, patience=PATIENCE, seed=SEED,
                                         class_freq_train=freq.tolist(), timm=timm.__version__, torch=torch.__version__,
                                         crop_manifest_sha256=digest(CROPS / 'crop_manifest.csv')))
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(yb); n += len(yb)
        pred = predict(model, val_loader).argmax(1)
        f1 = macro_f1(yva, pred)
        history.append(dict(epoch=epoch, train_loss=total / n, val_macro_f1=f1, val_acc=float((pred == yva).mean()),
                            lr=opt.param_groups[0]['lr'], seconds=time.perf_counter() - start))
        write_csv(run / 'history.csv', history, list(history[0]))
        print(f'{a.model} {a.config} epoch {epoch}: loss {total / n:.4f} val macro-F1 {f1:.4f} acc {history[-1]["val_acc"]:.4f}', flush=True)
        if f1 > best:
            best, stale = f1, 0
            torch.save(dict(model=model.state_dict(), timm_name=name, classes=ACNE_CLASSES, input=INPUT,
                            mean=dcfg['mean'], std=dcfg['std'], epoch=epoch, val_macro_f1=f1), run / 'best.pth')
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    best_row = max(history, key=lambda r: r['val_macro_f1'])
    write_json(run / 'completed.json', dict(best_epoch=best_row['epoch'], val_macro_f1=best_row['val_macro_f1'],
                                            val_acc=best_row['val_acc'], epochs_run=len(history),
                                            train_seconds=time.perf_counter() - start, best_sha256=digest(run / 'best.pth'),
                                            peak_vram_gb=torch.cuda.max_memory_allocated() / 1024 ** 3,
                                            completed_utc=datetime.now(timezone.utc).isoformat()))
    print('best', best_row)


if __name__ == '__main__':
    main()
