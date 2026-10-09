"""Cut one crop per acne lesion (5 ACNE-DET classes) from the full-resolution images.

Each crop is a square centred on the lesion box with side = CONTEXT x max(w, h) (minimum MIN_SIDE px),
shifted to stay inside the image, saved at native resolution (resizing to 224 happens at train time).
The split of every crop is the split of its source image in the master split (no crop-level split).
Writes datasets/phase2_acne_type/{train,val,test}/{class}/ and crop_manifest.csv.
Usage: venv/Scripts/python.exe scripts/phase2/make_crops.py
"""
import csv
from collections import Counter, defaultdict

import cv2

from common2 import ACNE_CLASSES, CROPS, DATA_REPORT, MASTER_SPLIT, SOURCE, digest, load_coco, write_csv, write_json

CONTEXT, MIN_SIDE, MIN_LESION = 2.0, 32, 4


def crop_box(bbox, width, height):
    x, y, w, h = bbox
    side = max(MIN_SIDE, CONTEXT * max(w, h))
    side = min(side, width, height)
    x0 = min(max(x + w / 2 - side / 2, 0), width - side)
    y0 = min(max(y + h / 2 - side / 2, 0), height - side)
    return int(round(x0)), int(round(y0)), int(round(side))


def main():
    data, cats = load_coco()
    with MASTER_SPLIT.open(encoding='utf-8-sig') as f:
        split_of = {int(r['image_id']): r for r in csv.DictReader(f)}
    if CROPS.exists() and any(CROPS.rglob('*.jpg')):
        raise RuntimeError(f'{CROPS} already has crops; delete it explicitly to rebuild')
    by_image = defaultdict(list)
    for a in data['annotations']:
        if cats[a['category_id']] in ACNE_CLASSES:
            by_image[a['image_id']].append(a)
    rows, skipped = [], []
    for img in data['images']:
        meta = split_of[img['id']]
        image = cv2.imread(str(SOURCE / img['file_name']))
        if image is None or image.shape[:2] != (img['height'], img['width']):
            raise RuntimeError(f'Cannot read {img["file_name"]} at the annotated size')
        for a in by_image[img['id']]:
            cls = cats[a['category_id']]
            if max(a['bbox'][2], a['bbox'][3]) < MIN_LESION:
                skipped.append(dict(source_id=meta['source_id'], annotation_id=a['id'], cls=cls, reason=f'lesion < {MIN_LESION}px'))
                continue
            x0, y0, side = crop_box(a['bbox'], img['width'], img['height'])
            out = CROPS / meta['split'] / cls / f'{meta["source_id"]}_{a["id"]}.jpg'
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), image[y0:y0 + side, x0:x0 + side], [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(dict(source_id=meta['source_id'], source_image=img['file_name'], annotation_id=a['id'], cls=cls,
                             bbox=[round(v, 2) for v in a['bbox']], crop_xywh=[x0, y0, side, side], crop_side=side,
                             crop_path=str(out.relative_to(CROPS)).replace('\\', '/'), split=meta['split']))
    write_csv(CROPS / 'crop_manifest.csv', rows, list(rows[0]))
    counts = Counter((r['split'], r['cls']) for r in rows)
    table = {s: {c: counts[(s, c)] for c in ACNE_CLASSES} for s in ('train', 'val', 'test')}
    write_json(DATA_REPORT / 'crops.json', dict(context=CONTEXT, min_side=MIN_SIDE, min_lesion=MIN_LESION, crops=len(rows),
                                                skipped=skipped, table=table, master_split_sha256=digest(MASTER_SPLIT),
                                                manifest_sha256=digest(CROPS / 'crop_manifest.csv')))
    print(len(rows), 'crops; skipped', len(skipped))
    for s, t in table.items():
        print(s, t)


if __name__ == '__main__':
    main()
