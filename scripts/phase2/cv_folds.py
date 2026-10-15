"""Grouped 5-fold cross-validation over all 275 ACNE-DET images (one-stage vs two-stage comparison).

The 50 patient groups of the master split are dealt into 10 sub-folds of 5 groups each, balanced on image
count and per-class lesion counts (best of 20,000 seeded random assignments). Fold k uses sub-folds
{2k, 2k+1} as test, sub-fold (2k+2) mod 10 as validation and the other 7 as training, so every image is
tested exactly once and a patient group never spans two roles.
For each fold builds, with hard-linked images:
  datasets/phase2_cv/fold{k}/det5   5 acne classes (one-stage detector)
  datasets/phase2_cv/fold{k}/det1   the same boxes as one class 'acne' (detector of the pipeline)
  datasets/phase2_cv/fold{k}/crop_manifest.csv  the phase-2 crops with the fold's split (classifier)
Usage: venv/Scripts/python.exe scripts/phase2/cv_folds.py
"""
import csv
import os
import shutil
from collections import defaultdict

import numpy as np

from common2 import ACNE_CLASSES, CROPS, MASTER_SPLIT, REPORTS, ROOT, SOURCE, digest, load_coco, write_csv, write_json

OUT = ROOT / 'datasets/phase2_cv'
K, SUB, SEED, TRIES = 5, 10, 42, 20000


def roles(k):
    return {2 * k: 'test', 2 * k + 1: 'test', (2 * k + 2) % SUB: 'val'}


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def assign(rows):
    groups = sorted({r['group'] for r in rows}, key=int)
    feat = {g: np.zeros(1 + len(ACNE_CLASSES)) for g in groups}
    for r in rows:
        feat[r['group']] += [1] + [int(r[c]) for c in ACNE_CLASSES]
    F = np.stack([feat[g] for g in groups])
    target = F.sum(0) / SUB
    rng, best, best_cost = np.random.default_rng(SEED), None, np.inf
    for _ in range(TRIES):
        perm = rng.permutation(len(groups))
        sub = np.empty(len(groups), int); sub[perm] = np.arange(len(groups)) % SUB
        tot = np.stack([F[sub == s].sum(0) for s in range(SUB)])
        cost = (np.abs(tot - target) / target).max()
        if cost < best_cost:
            best, best_cost = sub.copy(), cost
    return dict(zip(groups, best.tolist())), float(best_cost)


def main():
    if OUT.exists():
        raise RuntimeError(f'{OUT} exists; delete it explicitly to rebuild')
    with MASTER_SPLIT.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    sub_of, cost = assign(rows)
    data, cats = load_coco()
    boxes = defaultdict(list)
    for a in data['annotations']:
        if cats[a['category_id']] in ACNE_CLASSES:
            boxes[a['image_id']].append((ACNE_CLASSES.index(cats[a['category_id']]), a['bbox']))
    images = {i['id']: i for i in data['images']}
    with (CROPS / 'crop_manifest.csv').open(encoding='utf-8-sig') as f:
        crops = list(csv.DictReader(f))
    table = {}
    for k in range(K):
        role = roles(k)
        split_of = {r['source_id']: role.get(sub_of[r['group']], 'train') for r in rows}
        for task in ('det5', 'det1'):
            for s in ('train', 'val', 'test'):
                for d in ('images', 'labels'):
                    (OUT / f'fold{k}' / task / d / s).mkdir(parents=True, exist_ok=True)
        for r in rows:
            img, s = images[int(r['image_id'])], split_of[r['source_id']]
            stem = f'acnedet_{int(r["source_id"]):03d}'
            W, H, lines = img['width'], img['height'], []
            for c, (x, y, w, h) in boxes[img['id']]:
                x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
                if x2 - x1 >= 1 and y2 - y1 >= 1:
                    lines.append((c, f'{(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}'))
            for task in ('det5', 'det1'):
                d = OUT / f'fold{k}' / task
                link(SOURCE / img['file_name'], d / 'images' / s / f'{stem}.jpg')
                (d / 'labels' / s / f'{stem}.txt').write_text(
                    '\n'.join(f'{c if task == "det5" else 0} {t}' for c, t in lines) + '\n', encoding='utf-8')
        for task, names in (('det5', ACNE_CLASSES), ('det1', ['acne'])):
            d = OUT / f'fold{k}' / task
            (d / 'data.yaml').write_text(f'path: "{d.as_posix()}"\ntrain: images/train\nval: images/val\ntest: images/test\n'
                                         'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(names)), encoding='utf-8')
        fold_crops = [dict(r, split=split_of[r['source_id']]) for r in crops]
        write_csv(OUT / f'fold{k}' / 'crop_manifest.csv', fold_crops, list(crops[0]))
        table[k] = {s: dict(images=sum(1 for r in rows if split_of[r['source_id']] == s),
                            **{c: sum(int(r[c]) for r in rows if split_of[r['source_id']] == s) for c in ACNE_CLASSES})
                    for s in ('train', 'val', 'test')}
        print(k, table[k], flush=True)
    write_csv(OUT / 'folds.csv', [dict(source_id=r['source_id'], image_id=r['image_id'], group=r['group'],
                                       subfold=sub_of[r['group']], test_fold=[k for k in range(K) if sub_of[r['group']] in
                                                                              (2 * k, 2 * k + 1)][0]) for r in rows],
              ['source_id', 'image_id', 'group', 'subfold', 'test_fold'])
    write_json(REPORTS / 'phase2_cv/folds.json', dict(k=K, subfolds=SUB, seed=SEED, tries=TRIES, max_relative_deviation=cost,
                                                     group_to_subfold=sub_of, table=table, master_split_sha256=digest(MASTER_SPLIT)))


if __name__ == '__main__':
    main()
