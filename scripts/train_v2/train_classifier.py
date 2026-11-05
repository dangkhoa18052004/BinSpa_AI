"""P07: train one acne-type classifier experiment from configs/selfie_v1/<name>.yaml.

Reuses the phase-2 recipe (scripts/phase2/train_classifier.py: augmentation, phone degradations, AMP, cosine LR)
on a crops_v1-style manifest; picks the epoch with the best mean per-source val macro-F1 and records every source's
macro-F1 and the deployed classifier's on the same val crops. Writes runs/selfie_v1/<name>/{best.pth, history.csv,
config.json, completed.json}. The T1/T2 photos are not in the crops (lineage roles train/val only).
Usage: venv/Scripts/python.exe -u scripts/train_v2/train_classifier.py --config configs/selfie_v1/cls_c1.yaml
"""
import argparse
import csv
import importlib.util
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
from common2 import ACNE_CLASSES, digest, write_csv, write_json  # noqa: E402

_spec = importlib.util.spec_from_file_location('phase2_train_classifier', ROOT / 'scripts/phase2/train_classifier.py')
_p2 = importlib.util.module_from_spec(_spec)  # same file name as this script, so load the phase-2 one by path
sys.modules[_spec.name] = _p2  # DataLoader workers (spawn) unpickle CropSet by this module name
_spec.loader.exec_module(_p2)
CACHE_SIDE, INPUT, CropSet, Degrade = _p2.CACHE_SIDE, _p2.INPUT, _p2.CropSet, _p2.Degrade
macro_f1, predict, transforms = _p2.macro_f1, _p2.predict, _p2.transforms

RUNS = ROOT / 'runs/selfie_v1'
DEPLOYED = ROOT / 'models/phase2/best_acne_type_robust.pth'


def load_split(root, split):
    """Crops of one split resized to CACHE_SIDE, cached as uint8 RGB .npy next to the manifest."""
    with (root / 'crop_manifest.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == split]
    cache, index = root / f'cache_{split}_{CACHE_SIDE}.npy', root / f'cache_{split}_{CACHE_SIDE}_paths.txt'
    paths = [r['crop_path'] for r in rows]
    if not (cache.exists() and index.exists() and index.read_text(encoding='utf-8').split('\n') == paths):
        x = np.stack([cv2.resize(cv2.imread(str(root / p)), (CACHE_SIDE, CACHE_SIDE), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                      for p in paths])
        np.save(cache, x)
        index.write_text('\n'.join(paths), encoding='utf-8')
    return cache, np.array([ACNE_CLASSES.index(r['cls']) for r in rows]), rows


def per_source(y, pred, sources):
    out = {s: macro_f1(y[sources == s], pred[sources == s]) for s in sorted(set(sources))}
    return out, float(np.mean(list(out.values())))


def load_model(path, timm):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    m = timm.create_model(ckpt['timm_name'], pretrained=False, num_classes=len(ckpt['classes']))
    m.load_state_dict(ckpt['model'])
    return m.cuda()


def main():
    import timm
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    cfg_path = ROOT / a.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    run = RUNS / cfg['name']
    if (run / 'completed.json').exists():
        print('Already completed:', run)
        return
    run.mkdir(parents=True, exist_ok=True)
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    root = ROOT / cfg['crops']
    xtr, ytr, rtr = load_split(root, 'train')
    xva, yva, rva = load_split(root, 'val')
    src_va = np.array([r['source'] for r in rva])
    model = timm.create_model(cfg['timm_name'], pretrained=True, num_classes=len(ACNE_CLASSES)).cuda()
    dcfg = timm.data.resolve_data_config({}, model=model)
    freq = np.bincount(ytr, minlength=len(ACNE_CLASSES))
    g = torch.Generator().manual_seed(seed)
    sampler = (torch.utils.data.WeightedRandomSampler((1 / np.sqrt(freq))[ytr], len(ytr), replacement=True, generator=g)
               if cfg['sampler'] == 'class_sqrt' else None)
    train_set = CropSet(xtr, ytr, transforms(True, dcfg['mean'], dcfg['std']),
                        Degrade(cfg['degrade']) if cfg['degrade'] else None, [float(r['crop_side']) for r in rtr])
    train_loader = torch.utils.data.DataLoader(train_set, cfg['batch'], shuffle=sampler is None, sampler=sampler, generator=g,
                                               num_workers=4, persistent_workers=True, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(CropSet(xva, yva, transforms(False, dcfg['mean'], dcfg['std'])), 64,
                                             num_workers=2, persistent_workers=True, pin_memory=True)
    deployed = load_model(DEPLOYED, timm)
    base_src, base_mean = per_source(yva, predict(deployed, val_loader).argmax(1), src_va)
    del deployed
    print('deployed classifier val macro-F1', base_src, round(base_mean, 4), flush=True)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    steps, warm = cfg['epochs'] * len(train_loader), len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    scaler = torch.amp.GradScaler()
    write_json(run / 'config.json', dict(cfg, config_sha256=digest(cfg_path), class_freq_train=freq.tolist(),
                                         manifest_sha256=digest(root / 'crop_manifest.csv'), timm=timm.__version__,
                                         torch=torch.__version__, deployed_val_macro_f1=dict(per_source=base_src, mean=base_mean)))
    history, best, stale, start = [], -1.0, 0, time.perf_counter()
    for epoch in range(1, cfg['epochs'] + 1):
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
        src, mean = per_source(yva, pred, src_va)
        history.append(dict(epoch=epoch, train_loss=total / n, val_mean_source_macro_f1=mean, val_macro_f1=macro_f1(yva, pred),
                            **{f'val_f1_{s}': v for s, v in src.items()}, seconds=time.perf_counter() - start))
        write_csv(run / 'history.csv', history, list(history[0]))
        print(f"{cfg['name']} epoch {epoch}: loss {total / n:.4f} mean-source macro-F1 {mean:.4f} {src}", flush=True)
        if mean > best:
            best, stale = mean, 0
            torch.save(dict(model=model.state_dict(), timm_name=cfg['timm_name'], classes=ACNE_CLASSES, input=INPUT,
                            mean=dcfg['mean'], std=dcfg['std'], epoch=epoch, val_mean_source_macro_f1=mean), run / 'best.pth')
        else:
            stale += 1
            if stale >= cfg['patience']:
                break
    b = max(history, key=lambda r: r['val_mean_source_macro_f1'])
    write_json(run / 'completed.json', dict(best=b, deployed_val_macro_f1=dict(per_source=base_src, mean=base_mean),
                                            epochs_run=len(history), best_sha256=digest(run / 'best.pth'),
                                            train_seconds=time.perf_counter() - start,
                                            completed_utc=datetime.now(timezone.utc).isoformat()))
    print('best', b)


if __name__ == '__main__':
    main()
