"""Audit the ACNE-DET Roboflow mirror and build the master source-image split shared by GĐ2 and GĐ3.

Audit: classes, instances and images per class, image sizes, exact duplicates (SHA256), invalid
boxes, lesion size per class, and similarity between images with consecutive ACNE-DET numbers.

Grouping (no patient ID in the mirror): the paper says one patient may be imaged several times, and
visual review showed repeat visits carry consecutive numbers. Images are therefore grouped in
blocks of 5 consecutive numbers, and blocks are merged when the two images at their boundary have
ImageNet-embedding cosine similarity >= 0.95. Every lesion crop or box inherits its image's split.

Split: groups -> train/val/test ~70/15/15. Among 5000 seeded shuffles, keep the one whose
per-class instance shares (5 acne + 2 scar classes) and image share are closest to the targets.
Usage: venv/Scripts/python.exe scripts/phase2/audit_and_split.py
"""
import hashlib
from collections import Counter, defaultdict

import cv2
import numpy as np

from common2 import (ACNE_CLASSES, DATA_REPORT, MASTER_SPLIT, SCAR_CLASSES, SOURCE, digest, load_coco,
                     source_id, write_csv, write_json)

BLOCK, SIM_MERGE, SEED, TRIALS = 5, 0.95, 42, 5000
TARGET = dict(train=.70, val=.15, test=.15)


def embeddings(images):
    import timm
    import torch
    model = timm.create_model('convnext_tiny.fb_in22k', pretrained=True, num_classes=0).eval().cuda()
    cfg = timm.data.resolve_data_config({}, model=model)
    mean, std = np.array(cfg['mean']), np.array(cfg['std'])
    out = []
    with torch.inference_mode():
        for img in images:
            x = cv2.imread(str(SOURCE / img['file_name']), cv2.IMREAD_REDUCED_COLOR_8)[:, :, ::-1]
            x = (cv2.resize(x, (288, 384)).astype(np.float32) / 255 - mean) / std
            f = model(torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float32).cuda())
            out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy()[0])
    return np.array(out)


def groups_for(images, emb):
    parent = list(range(len(images)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for k in range(len(images)):
        if k % BLOCK:
            parent[find(k)] = find(k - 1)
    merged = []
    for k in range(1, len(images)):
        sim = float(emb[k] @ emb[k - 1])
        if k % BLOCK == 0 and sim >= SIM_MERGE and source_id(images[k]) - source_id(images[k - 1]) == 1:
            parent[find(k)] = find(k - 1)
            merged.append((source_id(images[k - 1]), source_id(images[k]), round(sim, 4)))
    roots = {}
    return [roots.setdefault(find(k), len(roots)) for k in range(len(images))], merged


def choose_split(group_of, per_image):
    classes = ACNE_CLASSES + SCAR_CLASSES
    n_groups = max(group_of) + 1
    g_counts = np.zeros((n_groups, len(classes) + 1))
    for k, g in enumerate(group_of):
        g_counts[g, :-1] += [per_image[k][c] for c in classes]
        g_counts[g, -1] += 1
    total = g_counts.sum(0)
    rng = np.random.default_rng(SEED)
    best = None
    for _ in range(TRIALS):
        order = rng.permutation(n_groups)
        cum = np.cumsum(g_counts[order, -1]) / total[-1]
        assign = np.where(cum <= TARGET['train'] + 1e-9, 0, np.where(cum <= TARGET['train'] + TARGET['val'] + 1e-9, 1, 2))
        split = np.empty(n_groups, int)
        split[order] = assign
        shares = np.stack([g_counts[split == s].sum(0) / total for s in range(3)])
        err = np.abs(shares - np.array(list(TARGET.values()))[:, None]).max()
        if best is None or err < best[0]:
            best = (err, split.copy(), shares)
    return best


def main():
    data, cats = load_coco()
    images = sorted(data['images'], key=source_id)
    by_image = defaultdict(list)
    for a in data['annotations']:
        by_image[a['image_id']].append(a)

    # --- audit
    hashes, invalid = defaultdict(list), []
    sizes = defaultdict(list)
    per_image = []
    for img in images:
        hashes[hashlib.sha256((SOURCE / img['file_name']).read_bytes()).hexdigest()].append(img['extra']['name'])
        c = Counter()
        for a in by_image[img['id']]:
            name = cats[a['category_id']]
            x, y, w, h = a['bbox']
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > img['width'] + 1 or y + h > img['height'] + 1:
                invalid.append(dict(image=img['extra']['name'], annotation_id=a['id'], cls=name, bbox=a['bbox']))
            sizes[name].append(max(w, h))
            c[name] += 1
        per_image.append(c)
    class_rows = []
    for name in sorted(set(cats.values()) - {'ACNE-Nprb'}):
        s = np.array(sizes[name]) if sizes[name] else np.zeros(1)
        class_rows.append(dict(cls=name, task='phase2' if name in ACNE_CLASSES else 'phase3' if name in SCAR_CLASSES else 'unused',
                               instances=len(sizes[name]), images=sum(1 for c in per_image if c[name]),
                               size_p5=float(np.percentile(s, 5)), size_median=float(np.median(s)), size_p95=float(np.percentile(s, 95))))
    duplicates = [v for v in hashes.values() if len(v) > 1]

    # --- grouping and split
    emb = embeddings(images)
    group_of, merged = groups_for(images, emb)
    consecutive = [float(emb[k] @ emb[k - 1]) for k in range(1, len(images))]
    err, split_of_group, shares = choose_split(group_of, per_image)
    names = ['train', 'val', 'test']
    rows = []
    for k, img in enumerate(images):
        rows.append(dict(source_id=source_id(img), original_name=img['extra']['name'], file_name=img['file_name'],
                         image_id=img['id'], width=img['width'], height=img['height'], group=group_of[k],
                         split=names[split_of_group[group_of[k]]],
                         **{c: per_image[k][c] for c in ACNE_CLASSES + SCAR_CLASSES}))
    write_csv(MASTER_SPLIT, rows, list(rows[0]))
    split_table = {s: dict(images=sum(r['split'] == s for r in rows), groups=len({r['group'] for r in rows if r['split'] == s}),
                           **{c: sum(r[c] for r in rows if r['split'] == s) for c in ACNE_CLASSES + SCAR_CLASSES},
                           **{f'{c}_images': sum(r['split'] == s and r[c] > 0 for r in rows) for c in ACNE_CLASSES + SCAR_CLASSES})
                   for s in names}
    write_csv(DATA_REPORT / 'class_audit.csv', class_rows, list(class_rows[0]))
    write_csv(DATA_REPORT / 'invalid_annotations.csv', invalid or [dict(image='', annotation_id='', cls='', bbox='')],
              ['image', 'annotation_id', 'cls', 'bbox'])
    write_json(DATA_REPORT / 'audit.json', dict(
        source='https://universe.roboflow.com/detect-4fwbi/acnedet (version v1, COCO export, no augmentation)',
        declared_license='MIT (as declared by the Roboflow uploader; original ACNE-DET authors state no license)',
        annotation_sha256=digest(SOURCE / '_annotations.coco.json'), images=len(images), annotations=len(data['annotations']),
        image_sizes=Counter(f"{i['width']}x{i['height']}" for i in images), exact_duplicate_groups=duplicates,
        invalid_annotations=len(invalid), classes=class_rows,
        grouping=dict(rule=f'blocks of {BLOCK} consecutive ACNE-DET numbers; boundary merged if cosine >= {SIM_MERGE}',
                      groups=max(group_of) + 1, merged_boundaries=merged,
                      consecutive_similarity_pct={p: float(np.percentile(consecutive, p)) for p in (10, 50, 90, 99)}),
        split=dict(seed=SEED, trials=TRIALS, target=TARGET, max_share_error=float(err), table=split_table,
                   manifest=str(MASTER_SPLIT.relative_to(MASTER_SPLIT.parents[1])), manifest_sha256=digest(MASTER_SPLIT))))
    print('duplicates', duplicates, 'invalid', len(invalid), 'groups', max(group_of) + 1, 'max share error', round(float(err), 3))
    for s in names:
        print(s, split_table[s])


if __name__ == '__main__':
    main()
