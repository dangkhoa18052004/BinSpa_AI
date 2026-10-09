"""Build the 2-class scar detection dataset (YOLO format) from the ACNE-DET Roboflow mirror.

Classes: 0 atrophic_scar, 1 hypertrophic_scar (names as in ACNE-DET; no rolling/icepick/boxcar/keloid).
Split: the master source-image split shared with phase 2 (datasets/acnedet_master_split.csv); no new
split. All 275 images are kept: other lesion classes are simply not labelled here, as in any
2-class detector; they are not relabelled as scars. Boxes are the COCO boxes clipped to the image.
Images are hard-linked (same drive) instead of copied. Also draws a label check sheet of >= 30 images.
Usage: venv/Scripts/python.exe scripts/phase3/build_scar_dataset.py
"""
import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
from common2 import MASTER_SPLIT, REPORTS, SCAR_CLASSES, SOURCE, digest, load_coco, write_json  # noqa: E402

DATASET = ROOT / 'datasets/phase3_scar_detection'
CHECK = REPORTS / 'phase3_label_check'


def main():
    if DATASET.exists():
        raise RuntimeError(f'{DATASET} exists; delete it explicitly to rebuild')
    data, cats = load_coco()
    with MASTER_SPLIT.open(encoding='utf-8-sig') as f:
        meta = {int(r['image_id']): r for r in csv.DictReader(f)}
    boxes = defaultdict(list)
    for a in data['annotations']:
        name = cats[a['category_id']]
        if name in SCAR_CLASSES:
            boxes[a['image_id']].append((SCAR_CLASSES.index(name), a['bbox']))
    counts, clipped = Counter(), 0
    for img in data['images']:
        m = meta[img['id']]
        split, stem = m['split'], f'acnedet_{int(m["source_id"]):03d}'
        W, H = img['width'], img['height']
        for sub in ('images', 'labels'):
            (DATASET / sub / split).mkdir(parents=True, exist_ok=True)
        dst = DATASET / 'images' / split / f'{stem}.jpg'
        try:
            os.link(SOURCE / img['file_name'], dst)
        except OSError:
            shutil.copy2(SOURCE / img['file_name'], dst)
        lines = []
        for cls, (x, y, w, h) in boxes[img['id']]:
            x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
            clipped += (x1, y1, x2, y2) != (x, y, x + w, y + h)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(f'{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
            counts[(split, SCAR_CLASSES[cls])] += 1
        (DATASET / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    (DATASET / 'data.yaml').write_text(
        f'path: "{DATASET.as_posix()}"\ntrain: images/train\nval: images/val\ntest: images/test\n'
        'names:\n  0: atrophic_scar\n  1: hypertrophic_scar\n', encoding='utf-8')
    table = {s: {c: counts[(s, c)] for c in SCAR_CLASSES} | {'images': len(list((DATASET / 'images' / s).glob('*.jpg')))}
             for s in ('train', 'val', 'test')}

    # label check: 30 random images with at least one hypertrophic scar or atrophic scar
    CHECK.mkdir(parents=True, exist_ok=True)
    with_scar = sorted(p for p in (DATASET / 'labels').rglob('*.txt') if p.read_text().strip())
    sample = random.Random(42).sample(with_scar, min(30, len(with_scar)))
    tiles = []
    for lab in sample:
        img = cv2.imread(str(DATASET / 'images' / lab.parent.name / f'{lab.stem}.jpg'), cv2.IMREAD_REDUCED_COLOR_4)
        h, w = img.shape[:2]
        for line in lab.read_text().splitlines():
            c, cx, cy, bw, bh = map(float, line.split())
            p1 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h)); p2 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
            cv2.rectangle(img, p1, p2, (0, 255, 255) if c == 0 else (255, 0, 255), 2)
        cv2.imwrite(str(CHECK / f'{lab.parent.name}_{lab.stem}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        t = cv2.resize(img, (240, 360)); cv2.putText(t, lab.stem[-3:], (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
        tiles.append(t)
    while len(tiles) % 10:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(CHECK / 'contact_sheet.jpg'), np.vstack([np.hstack(tiles[k:k + 10]) for k in range(0, len(tiles), 10)]))
    write_json(REPORTS / 'phase3_data.json', dict(classes=SCAR_CLASSES, table=table, clipped_boxes=clipped,
                                                  master_split_sha256=digest(MASTER_SPLIT), label_check=len(sample),
                                                  legend='yellow = atrophic_scar, magenta = hypertrophic_scar'))
    print(table, 'clipped', clipped)


if __name__ == '__main__':
    main()
