"""P06: one-class detector dataset on mixed sources (clinical + phone-like), Ultralytics YOLO layout.

train (role train of datasets/selfie_v1/lineage.jsonl only):
  - ACNE04-v2 and ACNE-DET author photos, plus one phone-style degraded copy of each (random downscale, JPEG,
    sensor noise, white balance, exposure; seeded by the photo path);
  - ACNE-DET author photos twice more (the only dermatologist-labelled source is small);
  - every DetectDataSkin-2 export copy (Roboflow copies of one source image share its lineage group, so they stay in
    train), and xnbio photos whose source image is not already in DetectDataSkin-2.
val: the evaluator's val photos (one per source image), excluding photos seen by the deployed detector.
Labels: every lesion box (five types and acne_untyped) is class 0 `acne`. `ignore` boxes (scars, pigmentation, pores,
Cysts, nevus) are left out, so they count as background: Ultralytics has no ignore-region loss, and the display must
not mark them. Images are written with the long side <= 1280 (the training size), boxes rescaled.
Usage: venv/Scripts/python.exe scripts/data_v2/build_det_dataset.py [--out datasets/selfie_v1/det_v1]
"""
import argparse
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
from common import digest, write_json  # noqa: E402
from evaluate_pipeline import DATA, eval_items, source_key  # noqa: E402

SIDE = 1280
CLINICAL = ('acne04', 'acnedet_author')
ANN = dict(acne04='acne04v2', acnedet_author='acnedet_author', detectdataskin2='detectdataskin2', xnbio='xnbio')
EXTRA_ACNEDET = 2


def phone_degrade(img, key):
    """Random phone-style degradation; geometry unchanged except a uniform downscale (boxes scale with it)."""
    rng = np.random.default_rng(zlib.crc32(f'phone|{key}'.encode()))
    h, w = img.shape[:2]
    s = rng.uniform(0.35, 0.8)
    img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img *= np.array([rng.uniform(0.85, 1.05), 1.0, rng.uniform(0.95, 1.15)], np.float32)  # BGR white balance
    img *= rng.uniform(0.65, 1.2)
    img += rng.normal(0, rng.uniform(2, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 80))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), s


def read(path, w, h):
    img = cv2.imread(str(ROOT / path))
    if img is None:
        raise SystemExit(f'cannot read {path}')
    if img.shape[:2] != (h, w):  # annotation pixels follow the stored pixels; undo an EXIF rotation if one applied
        img = cv2.imread(str(ROOT / path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img.shape[:2] != (h, w):
        raise SystemExit(f'{path}: image {img.shape[1]}x{img.shape[0]} != annotation {w}x{h}')
    return img


def write(out, split, name, img, boxes, scale):
    s = min(1.0, SIDE / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    lines = []
    for b in boxes:
        if b['label'] == 'ignore':
            continue
        x1, y1, x2, y2 = [v * scale * s for v in b['bbox']]
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f'0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}')
    cv2.imwrite(str(out / 'images' / split / f'{name}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (out / 'labels' / split / f'{name}.txt').write_text('\n'.join(lines), encoding='utf-8')
    return len(lines)


def train_records():
    lineage = [json.loads(line) for line in (DATA / 'lineage.jsonl').read_text(encoding='utf-8').splitlines()]
    train = {r['path']: r for r in lineage if r['role'] == 'train' and not r['quarantined'] and not r['unlabelled']}
    ann = {}
    for src in ANN.values():
        for line in (DATA / 'annotations' / f'{src}.jsonl').read_text(encoding='utf-8').splitlines():
            a = json.loads(line)
            if a['path'] in train and train[a['path']]['source'] in ANN:
                ann[a['path']] = a
    dds2_keys = {source_key(train[p]) for p in ann if train[p]['source'] == 'detectdataskin2'}
    recs = []
    for p in sorted(ann):
        r = train[p]
        if r['source'] == 'xnbio' and source_key(r) in dds2_keys:
            continue
        recs.append((r, ann[p]))
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(DATA / 'det_v1'))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists; remove it to rebuild')
    for d in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        (out / d).mkdir(parents=True)
    stats = Counter()
    for i, (r, an) in enumerate(train_records()):
        img = read(an['path'], an['width'], an['height'])
        base = f"{r['source']}_{Path(an['path']).stem}"
        stats[f"boxes:{r['source']}"] += write(out, 'train', base, img, an['boxes'], 1.0)
        stats[f"photos:{r['source']}"] += 1
        if r['source'] in CLINICAL:
            deg, s = phone_degrade(img, an['path'])
            write(out, 'train', base + '_phone', deg, an['boxes'], s)
            stats[f"photos:{r['source']}_phone"] += 1
        if r['source'] == 'acnedet_author':
            for k in range(EXTRA_ACNEDET):
                write(out, 'train', f'{base}_rep{k}', img, an['boxes'], 1.0)
                stats['photos:acnedet_author_rep'] += 1
        if i % 500 == 0:
            print(i, flush=True)
    val_seen = {r['path'] for r in (json.loads(x) for x in (DATA / 'lineage.jsonl').read_text(encoding='utf-8').splitlines())
                if r['role'] == 'val' and r['source'] == 'acne04' and r['own_locked_split'] == 'train'}
    for it in eval_items('val'):
        if it['path'] in val_seen:
            continue
        img = read(it['path'], it['width'], it['height'])
        write(out, 'val', f"{it['source']}_{Path(it['path']).stem}", img, it['boxes'], 1.0)
        stats[f"val:{it['source']}"] += 1
    (out / 'data.yaml').write_text(f'path: "{out.as_posix()}"\ntrain: images/train\nval: images/val\nnames:\n  0: acne\n',
                                   encoding='utf-8')
    write_json(out / 'build.json', dict(script=str(Path(__file__).relative_to(ROOT)), script_sha256=digest(Path(__file__)),
                                        lineage_sha256=digest(DATA / 'lineage.jsonl'), side=SIDE, stats=dict(sorted(stats.items())),
                                        train_images=len(list((out / 'images/train').iterdir())),
                                        val_images=len(list((out / 'images/val').iterdir()))))
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == '__main__':
    main()
