"""5-class acne detection dataset from ACNE-DET (one-stage alternative to detector -> classifier).

Same images, same master split (190/40/45) and same five classes as the phase-2 classifier, so the
one-stage detectors and the current pipeline are compared on identical test lesions. Class ids follow
ACNE_CLASSES; scars, melasma, nevus and other are not labelled. Boxes = annotated bbox clipped to the
image. Images are hard-linked.
Usage: venv/Scripts/python.exe scripts/phase2/build_det5_dataset.py
"""
import csv
import os
import shutil
from collections import Counter, defaultdict

from common2 import ACNE_CLASSES, MASTER_SPLIT, REPORTS, ROOT, SOURCE, digest, load_coco, write_json

OUT = ROOT / 'datasets/phase2_det5_acnedet'


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    if OUT.exists():
        raise RuntimeError(f'{OUT} exists; delete it explicitly to rebuild')
    data, cats = load_coco()
    with MASTER_SPLIT.open(encoding='utf-8-sig') as f:
        meta = {int(r['image_id']): r for r in csv.DictReader(f)}
    boxes = defaultdict(list)
    for a in data['annotations']:
        if cats[a['category_id']] in ACNE_CLASSES:
            boxes[a['image_id']].append((ACNE_CLASSES.index(cats[a['category_id']]), a['bbox']))
    counts = Counter()
    for split in ('train', 'val', 'test'):
        for sub in ('images', 'labels'):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)
    for img in data['images']:
        m = meta[img['id']]
        stem = f'acnedet_{int(m["source_id"]):03d}'
        link(SOURCE / img['file_name'], OUT / 'images' / m['split'] / f'{stem}.jpg')
        W, H, lines = img['width'], img['height'], []
        for c, (x, y, w, h) in boxes[img['id']]:
            x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                lines.append(f'{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
                counts[(m['split'], ACNE_CLASSES[c])] += 1
        (OUT / 'labels' / m['split'] / f'{stem}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        counts[(m['split'], 'images')] += 1
    names = ''.join(f'  {k}: {c}\n' for k, c in enumerate(ACNE_CLASSES))
    (OUT / 'data.yaml').write_text(f'path: "{OUT.as_posix()}"\ntrain: images/train\nval: images/val\ntest: images/test\n'
                                   f'names:\n{names}', encoding='utf-8')
    table = {s: {k: counts[(s, k)] for k in ['images'] + ACNE_CLASSES} for s in ('train', 'val', 'test')}
    write_json(REPORTS / 'phase2_det5/dataset.json', dict(table=table, classes=ACNE_CLASSES,
                                                          master_split_sha256=digest(MASTER_SPLIT)))
    print(table)


if __name__ == '__main__':
    main()
