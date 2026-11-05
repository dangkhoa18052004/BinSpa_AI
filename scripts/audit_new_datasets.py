"""Audit the datasets added in November 2026 against the ones already used.

Read-only over data_raw/ and raw/. For each source it reports image/label statistics,
Roboflow augmentation groups (copies of one source image) and split leakage, then
near-duplicates (pHash, 8 flip/rotation variants, Hamming <= THRESHOLD) between every
pair of sources. It also matches the ACNE-DET author release to the Roboflow mirror and
summarises the SCIN acne cases. Writes reports/new_datasets_audit/{audit.json,
duplicates.csv, sheet_*.jpg}. pHash pairs need a visual check; crops and heavy shear
are not caught.
Usage: venv/Scripts/python.exe -I scripts/audit_new_datasets.py
"""
import ast
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data_raw'
OUT = ROOT / 'reports/new_datasets_audit'
THRESHOLD = 10
POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

ROBOFLOW = {
    'kbm0q': RAW / 'acne-kbm0q_roboflow/extracted',
    'xnbio': RAW / 'acne-xnbio_roboflow/extracted',
    'acnedet_mirror': RAW / 'acnedet_roboflow/acnedet.v1i.coco',
    'detectdataskin2': RAW / 'detectdataskin2',
}


def phash_variants(path):
    g = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if g is None:
        return None
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    out = []
    for flip in (False, True):
        a = g[:, ::-1] if flip else g
        for k in range(4):
            d = cv2.dct(np.ascontiguousarray(np.rot90(a, k)))[:8, :8].flatten()
            out.append(np.packbits(d > np.median(d[1:])))
    return np.array(out)  # (8, 8) uint8


def source_of(name):
    return name.split('.rf.')[0] if '.rf.' in name else Path(name).stem


def load_coco_source(root):
    """Images, labels and augmentation groups of a Roboflow COCO export."""
    images, groups, splits_of = [], defaultdict(list), defaultdict(set)
    per_split, cls_all, cls_unique, boxes, lesions = {}, Counter(), Counter(), [], []
    seen_src = set()
    for ann_path in sorted(root.glob('*/_annotations.coco.json')):
        split = ann_path.parent.name
        d = json.loads(ann_path.read_text(encoding='utf-8'))
        cats = {c['id']: c['name'] for c in d['categories']}
        by_img = defaultdict(list)
        for a in d['annotations']:
            by_img[a['image_id']].append(a)
        per_split[split] = {'images': len(d['images']), 'annotations': len(d['annotations'])}
        for im in d['images']:
            src = source_of(im['file_name'])
            path = ann_path.parent / im['file_name']
            images.append({'path': path, 'split': split, 'source': src})
            groups[src].append(path)
            splits_of[src].add(split)
            anns = by_img[im['id']]
            cls_all.update(cats[a['category_id']] for a in anns)
            if src not in seen_src:  # count each source image once
                seen_src.add(src)
                cls_unique.update(cats[a['category_id']] for a in anns)
                lesions.append(len(anns))
                boxes += [max(a['bbox'][2], a['bbox'][3]) for a in anns]
    sizes = Counter((cv2.imread(str(i['path'])).shape[1::-1]) for i in images[:50])
    return images, {
        'files': len(images), 'source_images': len(groups), 'per_split': per_split,
        'copies_per_source': dict(Counter(len(v) for v in groups.values())),
        'sources_in_several_splits': sum(len(s) > 1 for s in splits_of.values()),
        'classes_all_copies': dict(cls_all.most_common()),
        'classes_unique_sources': dict(cls_unique.most_common()),
        'lesions_per_image_median': statistics.median(lesions) if lesions else 0,
        'box_long_side_px': {'median': statistics.median(boxes), 'p10': float(np.percentile(boxes, 10))} if boxes else {},
        'image_size_sample': {f'{w}x{h}': n for (w, h), n in sizes.most_common(5)},
        'leaked_sources_example': [s for s, v in splits_of.items() if len(v) > 1][:10],
    }


def author_vs_mirror(hashes):
    """Label summary of the ACNE-DET author release; pairs it with the mirror by pHash."""
    labels, shapes, per_img = Counter(), Counter(), {}
    for p in sorted((RAW / 'acnedet_author/labelme').glob('*.json')):
        d = json.loads(p.read_text(encoding='utf-8'))
        labels.update(s['label'] for s in d['shapes'])
        shapes.update(s['shape_type'] for s in d['shapes'])
        per_img[p.stem] = len(d['shapes'])
    mirror = json.loads((ROBOFLOW['acnedet_mirror'] / 'train/_annotations.coco.json').read_text(encoding='utf-8'))
    mirror_count = Counter(a['image_id'] for a in mirror['annotations'])
    mirror_by_src = {source_of(i['file_name']).removesuffix('_jpg'): mirror_count[i['id']] for i in mirror['images']}
    a_names, a_h = hashes['acnedet_author']
    m_names, m_h = hashes['acnedet_mirror']
    dist = pair_dist(a_h, m_h)
    best = dist.argmin(1)
    matched = [(a_names[i], m_names[best[i]], int(dist[i, best[i]])) for i in range(len(a_names))]
    same_name = sum(Path(a).stem == source_of(Path(m).name).removesuffix('_jpg') for a, m, d in matched if d <= THRESHOLD)
    count_diff = [per_img[Path(a).stem] - mirror_by_src.get(Path(a).stem, -1) for a, _, _ in matched]
    return {
        'author_images': len(a_names), 'author_label_files': len(per_img),
        'author_labels': dict(labels.most_common()), 'author_shape_types': dict(shapes),
        'author_shapes_total': sum(per_img.values()), 'mirror_annotations_total': len(mirror['annotations']),
        'matched_to_mirror': sum(d <= THRESHOLD for _, _, d in matched), 'matched_same_name': same_name,
        'unmatched_author': [a for a, _, d in matched if d > THRESHOLD],
        'shape_count_minus_mirror': dict(Counter(count_diff).most_common(8)),
    }


def scin_summary():
    """Case-level acne summary of SCIN (labels are per case, no lesion boxes)."""
    base = RAW / 'scin'
    cases = {r['case_id']: r for r in csv.DictReader(open(base / 'scin_cases.csv', encoding='utf-8'))}
    labels = {r['case_id']: r for r in csv.DictReader(open(base / 'scin_labels.csv', encoding='utf-8'))}
    acne = {}
    for cid, r in labels.items():
        w = ast.literal_eval(r['weighted_skin_condition_label'] or '{}')
        a = sum(v for k, v in w.items() if re.search(r'acne', k, re.I))
        if a > 0:
            acne[cid] = (a, w)
    rows = []
    for cid, (a, w) in acne.items():
        c = cases.get(cid, {})
        imgs = [c.get(f'image_{i}_path') for i in (1, 2, 3) if c.get(f'image_{i}_path')]
        rows.append({'case': cid, 'acne_weight': a, 'top': max(w, key=w.get), 'head_neck': c.get('body_parts_head_or_neck') == 'YES',
                     'fst': labels[cid]['dermatologist_fitzpatrick_skin_type_label_1'] or c.get('fitzpatrick_skin_type') or 'NA',
                     'images': imgs, 'shots': [c.get(f'image_{i}_shot_type') for i in (1, 2, 3) if c.get(f'image_{i}_path')]})
    strong = [r for r in rows if r['acne_weight'] >= 0.5]
    face = [r for r in strong if r['head_neck']]
    have = {p.name for p in (base / 'images').iterdir()}
    return {
        'cases_total': len(cases), 'images_on_disk': len(have),
        'cases_any_acne_label': len(rows), 'cases_acne_weight_ge_0.5': len(strong),
        'cases_acne_ge_0.5_head_neck': len(face),
        'images_acne_ge_0.5_head_neck': sum(len(r['images']) for r in face),
        'images_present_on_disk': sum(Path(p).name in have for r in face for p in r['images']),
        'fitzpatrick_acne_ge_0.5_head_neck': dict(Counter(r['fst'] for r in face).most_common()),
        'shot_types': dict(Counter(s for r in face for s in r['shots']).most_common()),
        'top_labels_any_acne': dict(Counter(r['top'] for r in rows).most_common(10)),
        '_face_images': [base / 'images' / Path(p).name for r in face for p in r['images']],
    }


def pair_dist(a, b):
    """Min Hamming distance over a's 8 variants vs b's identity hash: (len(a), len(b))."""
    out = np.empty((len(a), len(b)), dtype=np.uint8)
    bb = b[:, 0]
    for i in range(0, len(a), 128):
        x = a[i:i + 128][:, :, None, :] ^ bb[None, None]
        out[i:i + 128] = POP[x].sum(-1).min(1)
    return out


def contact_sheet(paths, dest, n=24, tile=180):
    paths = [p for p in paths[::max(1, len(paths) // n)]][:n]
    tiles = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        s = tile / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        canvas = np.full((tile, tile, 3), 255, np.uint8)
        canvas[:im.shape[0], :im.shape[1]] = im
        tiles.append(canvas)
    while len(tiles) % 6:
        tiles.append(np.full((tile, tile, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(str(dest), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report, sets = {}, {}
    for name, root in ROBOFLOW.items():
        images, report[name] = load_coco_source(root)
        first = {}
        for im in images:
            first.setdefault(im['source'], im['path'])
        sets[name] = list(first.values())  # one file per source image
    sets['acne04'] = sorted((ROOT / 'raw/ACNE04_images').glob('*.jpg'))
    sets['acnedet_author'] = sorted((RAW / 'acnedet_author/images').glob('*.jpg'))
    scin = scin_summary()
    sets['scin_acne_face'] = [p for p in scin.pop('_face_images') if p.exists()]
    report['scin'] = scin

    hashes = {}
    for name, paths in sets.items():
        names, hs = [], []
        for p in paths:
            h = phash_variants(p)
            if h is not None:
                names.append(str(p.relative_to(ROOT)))
                hs.append(h)
        hashes[name] = (names, np.array(hs))
        print(f'hashed {name}: {len(names)}', flush=True)

    report['acnedet_author_vs_mirror'] = author_vs_mirror(hashes)

    dup_rows, pair_summary = [], {}
    keys = list(hashes)
    for i, a in enumerate(keys):
        for b in keys[i:]:
            an, ah = hashes[a]
            bn, bh = hashes[b]
            d = pair_dist(ah, bh)
            if a == b:
                d[np.tril_indices(len(an))] = 255
            hits = np.argwhere(d <= THRESHOLD)
            pair_summary[f'{a} ~ {b}'] = {'pairs': len(hits), 'images_a': len({int(x) for x, _ in hits}),
                                         'images_b': len({int(y) for _, y in hits})}
            dup_rows += [(a, an[x], b, bn[y], int(d[x, y])) for x, y in hits]
    report['near_duplicates'] = pair_summary

    with open(OUT / 'duplicates.csv', 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows([('set_a', 'image_a', 'set_b', 'image_b', 'hamming')] + dup_rows)
    for name in ('kbm0q', 'xnbio', 'scin_acne_face', 'detectdataskin2'):
        contact_sheet(sets[name], OUT / f'sheet_{name}.jpg')
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
