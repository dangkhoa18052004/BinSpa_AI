"""Combined phase-1 dataset: ACNE04-v2 (unchanged split) + ACNE-DET acne lesions (master split), 1 class.

Goal: teach the generic acne detector the comedones it misses (ACNE-DET recall ~11% closed / ~5%
open comedo). ACNE-DET's five acne classes (closed_comedo, open_comedo, papule, pustule, nodule)
become class 0 'acne'; scars, melasma, nevus and other are not labelled. Each ACNE-DET image keeps
the split of the master split, so ACNE-DET test images never reach training, and the ACNE04 test
split is untouched. Images are hard-linked. ACNE04 boxes come from centre+radius (square), ACNE-DET
boxes from polygons; the mix is reported as a limitation.
Usage: venv/Scripts/python.exe scripts/phase1/build_combined_dataset.py
"""
import csv
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import DATASET, ROOT, digest, verify_dataset, write_json

sys.path.insert(0, str(ROOT / 'scripts/phase2'))
from common2 import ACNE_CLASSES, MASTER_SPLIT, SOURCE, load_coco  # noqa: E402

OUT = ROOT / 'datasets/phase1_combined_acnedet'


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    if OUT.exists():
        raise RuntimeError(f'{OUT} exists; delete it explicitly to rebuild')
    verify_dataset(DATASET, require_review=True)
    counts = Counter()
    for split in ('train', 'val', 'test'):
        for sub in ('images', 'labels'):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)
        for img in sorted((DATASET / 'images' / split).glob('*.jpg')):
            link(img, OUT / 'images' / split / img.name)
            shutil.copy2(DATASET / 'labels' / split / f'{img.stem}.txt', OUT / 'labels' / split / f'{img.stem}.txt')
            counts[(split, 'acne04_images')] += 1
            counts[(split, 'acne04_boxes')] += len((DATASET / 'labels' / split / f'{img.stem}.txt').read_text().split('\n')) - 1
    data, cats = load_coco()
    with MASTER_SPLIT.open(encoding='utf-8-sig') as f:
        meta = {int(r['image_id']): r for r in csv.DictReader(f)}
    boxes = defaultdict(list)
    for a in data['annotations']:
        if cats[a['category_id']] in ACNE_CLASSES:
            boxes[a['image_id']].append(a['bbox'])
    for img in data['images']:
        m = meta[img['id']]
        stem = f'acnedet_{int(m["source_id"]):03d}'
        link(SOURCE / img['file_name'], OUT / 'images' / m['split'] / f'{stem}.jpg')
        W, H, lines = img['width'], img['height'], []
        for x, y, w, h in boxes[img['id']]:
            x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                lines.append(f'0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
        (OUT / 'labels' / m['split'] / f'{stem}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        counts[(m['split'], 'acnedet_images')] += 1
        counts[(m['split'], 'acnedet_boxes')] += len(lines)
    (OUT / 'data.yaml').write_text(f'path: "{OUT.as_posix()}"\ntrain: images/train\nval: images/val\ntest: images/test\n'
                                   'names:\n  0: acne\n', encoding='utf-8')
    table = {s: {k: counts[(s, k)] for k in ('acne04_images', 'acne04_boxes', 'acnedet_images', 'acnedet_boxes')}
             for s in ('train', 'val', 'test')}
    write_json(ROOT / 'reports/phase1_combined/dataset.json', dict(
        table=table, acnedet_classes_merged=ACNE_CLASSES, acne04_manifest_sha256=digest(DATASET / 'manifest.json'),
        master_split_sha256=digest(MASTER_SPLIT), note='test splits of both sources are never used for training or selection'))
    print(table)


if __name__ == '__main__':
    main()
