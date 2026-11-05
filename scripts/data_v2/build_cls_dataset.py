"""P07: acne-type crop dataset on mixed sources, in the layout of scripts/phase2/train_classifier.py.

Typed boxes (five classes) of photos whose lineage role is train (crops `train`) or val (crops `val`):
  - ACNE-DET author release (dermatologist labels);
  - ACNE04 photos with the kbm0q types mapped onto them (datasets/selfie_v1/kbm0q_on_acne04.jsonl, status mapped;
    owner review 30/30 "right place"); one kbm0q copy per ACNE04 photo, the one with most boxes;
  - DetectDataSkin-2, one export copy per source image (the evaluator's choice); its xnbio relabel is left out because
    the two label sets disagree on the same photos.
Labels from Roboflow sources are `unverified` (plan section 2). Crop = scripts/phase2/make_crops.crop_box (square,
2 x the box side, min 32 px, shifted inside the photo). Train boxes also get one jittered crop (centre shifted up to
25 % of the box side, side x0.75-1.5, seeded by photo and box) so the classifier sees detector-like boxes.
Crops are written at their native side capped at 512 px.
Usage: venv/Scripts/python.exe scripts/data_v2/build_cls_dataset.py [--out datasets/selfie_v1/crops_v1]
"""
import argparse
import csv
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
from common import digest, write_json  # noqa: E402
from evaluate_pipeline import CLASSES, DATA, eval_items  # noqa: E402
from make_crops import crop_box  # noqa: E402

CAP = 512
FIELDS = ['source_id', 'source_image', 'annotation_id', 'cls', 'bbox', 'crop_xywh', 'crop_side', 'crop_path', 'split',
          'source', 'kind', 'lineage_group_id']


def photos(role):
    """-> [(source, path, width, height, lineage_group_id, typed boxes xyxy)] of one lineage role."""
    out = [(it['source'], it['path'], it['width'], it['height'], it['lineage_group_id'],
            [b for b in it['boxes'] if b['label'] in CLASSES])
           for it in eval_items(role, ['acnedet_author', 'detectdataskin2'])]
    lineage = {r['path']: r for r in (json.loads(x) for x in (DATA / 'lineage.jsonl').read_text(encoding='utf-8').splitlines())}
    size = {}
    for line in (DATA / 'annotations/acne04v2.jsonl').read_text(encoding='utf-8').splitlines():
        a = json.loads(line)
        size[a['path']] = (a['width'], a['height'])
    best = {}
    for line in (DATA / 'kbm0q_on_acne04.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r['status'] != 'mapped':
            continue
        path = f"raw/ACNE04_images/{r['acne04']}"
        L = lineage.get(path)
        boxes = [b for b in r['boxes'] if b['label'] in CLASSES]
        if L is None or L['role'] != role or L['quarantined'] or not boxes:
            continue
        if path not in best or len(boxes) > len(best[path][5]):
            best[path] = ('acne04', path, *size.get(path, (None, None)), L['lineage_group_id'], boxes)  # not in v2: size read later
    return [p for p in out if p[5]] + sorted(best.values(), key=lambda p: p[1])


def jitter(xyxy, key):
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    x1, y1, x2, y2 = xyxy
    s = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2 + rng.uniform(-.25, .25) * s, (y1 + y2) / 2 + rng.uniform(-.25, .25) * s
    w, h = (x2 - x1) * rng.uniform(.75, 1.5), (y2 - y1) * rng.uniform(.75, 1.5)
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(DATA / 'crops_v1'))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists; remove it to rebuild')
    rows, stats = [], Counter()
    for split in ('train', 'val'):
        for c in CLASSES:
            (out / split / c).mkdir(parents=True)
        for src, path, w, h, gid, boxes in photos(split):
            img = cv2.imread(str(ROOT / path))
            if w is None:
                h, w = img.shape[:2]
            if img.shape[:2] != (h, w):
                img = cv2.imread(str(ROOT / path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if img.shape[:2] != (h, w):
                raise SystemExit(f'{path}: image {img.shape[1]}x{img.shape[0]} != annotation {w}x{h}')
            stem = f'{src}_{Path(path).stem}'
            for k, b in enumerate(boxes):
                kinds = [('gt', b['bbox'])] + ([('jitter', jitter(b['bbox'], f'{path}|{k}'))] if split == 'train' else [])
                for kind, (x1, y1, x2, y2) in kinds:
                    x0, y0, side = crop_box([x1, y1, x2 - x1, y2 - y1], w, h)
                    crop = img[y0:y0 + side, x0:x0 + side]
                    if crop.size == 0:
                        continue
                    if side > CAP:
                        crop = cv2.resize(crop, (CAP, CAP), interpolation=cv2.INTER_AREA)
                    rel = f"{split}/{b['label']}/{stem}_{k}{'' if kind == 'gt' else '_j'}.jpg"
                    cv2.imwrite(str(out / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append(dict(source_id=stem, source_image=path, annotation_id=k, cls=b['label'],
                                     bbox=json.dumps([round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]),
                                     crop_xywh=json.dumps([x0, y0, side, side]), crop_side=side, crop_path=rel, split=split,
                                     source=src, kind=kind, lineage_group_id=gid))
                    stats[f'{split}:{src}:{b["label"]}' + ('' if kind == 'gt' else ':jitter')] += 1
    with (out / 'crop_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    write_json(out / 'build.json', dict(script=str(Path(__file__).relative_to(ROOT)), script_sha256=digest(Path(__file__)),
                                        lineage_sha256=digest(DATA / 'lineage.jsonl'),
                                        mapping_sha256=digest(DATA / 'kbm0q_on_acne04.jsonl'),
                                        manifest_sha256=digest(out / 'crop_manifest.csv'), stats=dict(sorted(stats.items()))))
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == '__main__':
    main()
