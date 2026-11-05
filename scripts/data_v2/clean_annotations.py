"""P02 (selfie_v1): geometry cleaning into one derived annotation format. Sources stay untouched.

Per source, one JSON line per image:
  {source, image, path, sha256, width, height (real decoded size), lineage_key, boxes: [{bbox [x1,y1,x2,y2] in real
   pixels, raw_label, label, fixes}], dropped: {reason: n}}
Rules (logged per box in `fixes` or per image in `dropped`):
  - polygon (ACNE-DET author) -> min/max box; circle (Acne04-v2) -> square box centre +- radius
  - declared COCO size != real size: boxes scaled by real/declared ('rescaled')
  - clipped to the image ('clipped'); dropped if less than half of the box remains inside ('mostly_outside')
  - width or height <= 0, or < 1 px after clipping ('degenerate')
  - labels through configs/selfie_v1/label_mapping.yaml; 'drop' labels are skipped silently
Also compares the ACNE-DET author polygons with the Roboflow mirror boxes (same ID and class, one-to-one) and
lists every box whose corners differ by more than 1 px, with a crop sheet for a visual check.
Writes datasets/selfie_v1/annotations/<source>.jsonl, reports/selfie_v1/geometry_audit.json and
reports/selfie_v1/review/geometry/ (local).
Usage: venv/Scripts/python.exe scripts/data_v2/clean_annotations.py
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import digest, write_json  # noqa: E402

MAPPING = ROOT / 'configs/selfie_v1/label_mapping.yaml'
OUT = ROOT / 'datasets/selfie_v1/annotations'
REPORT = ROOT / 'reports/selfie_v1/geometry_audit.json'
REVIEW = ROOT / 'reports/selfie_v1/review/geometry'
COCO_SOURCES = {
    'kbm0q': ROOT / 'data_raw/acne-kbm0q_roboflow/extracted',
    'xnbio': ROOT / 'data_raw/acne-xnbio_roboflow/extracted',
    'detectdataskin2': ROOT / 'data_raw/detectdataskin2',
    'acnedet_mirror': ROOT / 'data_raw/acnedet_roboflow/acnedet.v1i.coco',
}


def real_size(path):
    with Image.open(path) as im:  # header only; no EXIF transpose (all sources are stored upright, orientation 1)
        return im.size


def fix_box(x1, y1, x2, y2, W, H, fixes):
    """-> clipped box or (None, reason)."""
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None, 'degenerate'
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if cx2 - cx1 < 1 or cy2 - cy1 < 1:
        return None, 'degenerate'
    if (cx2 - cx1) < 0.5 * (x2 - x1) or (cy2 - cy1) < 0.5 * (y2 - y1):
        return None, 'mostly_outside'
    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
        fixes.append('clipped')
    return [round(float(v), 2) for v in (cx1, cy1, cx2, cy2)], None


def coco_source(name, root, label_map):
    rows = []
    for ann in sorted(root.glob('*/_annotations.coco.json')):
        d = json.loads(ann.read_text(encoding='utf-8'))
        cats = {c['id']: c['name'] for c in d['categories']}
        by_img = defaultdict(list)
        for a in d['annotations']:
            by_img[a['image_id']].append(a)
        for im in d['images']:
            path = ann.parent / im['file_name']
            W, H = real_size(path)
            fx, fy = W / im['width'], H / im['height']
            rescaled = (W, H) != (im['width'], im['height'])
            boxes, dropped = [], Counter()
            for a in by_img[im['id']]:
                raw = cats[a['category_id']]
                label = label_map[raw]
                if label == 'drop':
                    continue
                x, y, w, h = a['bbox']
                fixes = ['rescaled'] if rescaled else []
                box, why = fix_box(x * fx, y * fy, (x + w) * fx, (y + h) * fy, W, H, fixes)
                if box is None:
                    dropped[why] += 1
                    continue
                boxes.append(dict(bbox=box, raw_label=raw, label=label, fixes=fixes))
            rows.append(dict(source=name, image=im['file_name'], split_export=ann.parent.name,
                             path=str(path.relative_to(ROOT)).replace('\\', '/'), width=W, height=H,
                             declared=[im['width'], im['height']], lineage_key=im['file_name'].split('.rf.')[0],
                             extra_name=(im.get('extra') or {}).get('name'), boxes=boxes, dropped=dict(dropped)))
    return rows


def acnedet_author(label_map):
    base = ROOT / 'data_raw/acnedet_author'
    rows = []
    for p in sorted((base / 'labelme').glob('*.json'), key=lambda q: int(q.stem)):
        d = json.loads(p.read_text(encoding='utf-8'))
        img = base / 'images' / f'{p.stem}.jpg'
        W, H = real_size(img)
        declared = (d['imageWidth'], d['imageHeight'])
        fx, fy = W / declared[0], H / declared[1]
        boxes, dropped = [], Counter()
        for s in d['shapes']:
            label = label_map[s['label']]
            if label == 'drop':
                continue
            pts = np.array(s['points'], float)
            fixes = ['polygon_to_box'] + (['rescaled'] if (W, H) != declared else [])
            (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
            box, why = fix_box(x1 * fx, y1 * fy, x2 * fx, y2 * fy, W, H, fixes)
            if box is None:
                dropped[why] += 1
                continue
            boxes.append(dict(bbox=box, raw_label=s['label'], label=label, fixes=fixes))
        rows.append(dict(source='acnedet_author', image=img.name, path=str(img.relative_to(ROOT)).replace('\\', '/'),
                         width=W, height=H, declared=list(declared), lineage_key=f'acnedet_{p.stem}',
                         boxes=boxes, dropped=dict(dropped)))
    return rows


def acne04v2(label_map):
    d = json.loads((ROOT / 'acne04v2/Acne04-v2_annotations.json').read_text(encoding='utf-8'))
    by_img = defaultdict(list)
    for a in d['annotations']:
        by_img[a['image_id']].append(a)
    rows = []
    for im in d['images']:
        path = ROOT / 'raw/ACNE04_images' / im['file_name']
        W, H = real_size(path)
        fx, fy = W / im['width'], H / im['height']
        boxes, dropped = [], Counter()
        for a in by_img[im['id']]:
            (cx, cy), r = a['coordinates'], a['radius']
            fixes = ['circle_to_box'] + (['rescaled'] if (W, H) != (im['width'], im['height']) else [])
            box, why = fix_box((cx - r) * fx, (cy - r) * fy, (cx + r) * fx, (cy + r) * fy, W, H, fixes)
            if box is None:
                dropped[why] += 1
                continue
            boxes.append(dict(bbox=box, raw_label=a['class_name'], label=label_map[a['class_name']], fixes=fixes))
        rows.append(dict(source='acne04v2', image=im['file_name'], path=str(path.relative_to(ROOT)).replace('\\', '/'),
                         width=W, height=H, declared=[im['width'], im['height']], lineage_key=f"acne04_{im['file_name']}",
                         boxes=boxes, dropped=dict(dropped)))
    return rows


def author_vs_mirror(author, mirror):
    """One-to-one match per image and class by max corner difference; boxes differing by > 1 px are listed."""
    by_id = {r['lineage_key'].split('_')[1]: r for r in author}
    diffs, matched, unmatched = [], 0, Counter()
    for m in mirror:
        sid = re.match(r'\d+', m['extra_name'] or m['image']).group()
        a = by_id.get(sid)
        if a is None:
            unmatched['mirror_image_without_author'] += 1
            continue
        for lab in {b['raw_label'] for b in m['boxes']} | {b['raw_label'] for b in a['boxes']}:
            A = np.array([b['bbox'] for b in a['boxes'] if b['raw_label'] == lab]).reshape(-1, 4)
            M = np.array([b['bbox'] for b in m['boxes'] if b['raw_label'] == lab]).reshape(-1, 4)
            if not len(A) or not len(M):
                unmatched['class_missing_on_one_side'] += max(len(A), len(M))
                continue
            cost = np.abs(A[:, None, :] - M[None, :, :]).max(-1)
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(cost)
            matched += len(ri)
            unmatched['author_extra'] += len(A) - len(ri)
            unmatched['mirror_extra'] += len(M) - len(ci)
            for i, j in zip(ri, ci):
                if cost[i, j] > 1:
                    diffs.append(dict(id=sid, label=lab, author=A[i].tolist(), mirror=M[j].tolist(),
                                      max_corner_diff_px=round(float(cost[i, j]), 2), path=a['path']))
    return dict(matched=matched, over_1px=len(diffs), over_2px=sum(d['max_corner_diff_px'] > 2 for d in diffs),
                unmatched=dict(unmatched), boxes=sorted(diffs, key=lambda d: -d['max_corner_diff_px']))


def diff_sheet(diffs, dest):
    tiles = []
    for d in diffs:
        img = cv2.imread(str(ROOT / d['path']))
        A, M = np.array(d['author']), np.array(d['mirror'])
        x1, y1 = np.minimum(A[:2], M[:2]).astype(int) - 40
        x2, y2 = np.maximum(A[2:], M[2:]).astype(int) + 40
        x1, y1 = max(x1, 0), max(y1, 0)
        crop = img[y1:y2, x1:x2].copy()
        for box, col in ((A, (0, 255, 0)), (M, (0, 0, 255))):
            cv2.rectangle(crop, tuple((box[:2] - [x1, y1]).astype(int)), tuple((box[2:] - [x1, y1]).astype(int)), col, 1)
        s = 200 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        canvas = np.full((220, 200, 3), 255, np.uint8)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(canvas, f"{d['id']} {d['label'][:12]} {d['max_corner_diff_px']}", (2, 215), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1)
        tiles.append(canvas)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.full((220, 200, 3), 255, np.uint8))
        cv2.imwrite(str(dest), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


def main():
    t0 = time.time()
    mapping = yaml.safe_load(MAPPING.read_text(encoding='utf-8'))['sources']
    out = {name: coco_source(name, root, mapping[name]['map']) for name, root in COCO_SOURCES.items()}
    out['acnedet_author'] = acnedet_author(mapping['acnedet_author']['map'])
    out['acne04v2'] = acne04v2(mapping['acne04v2']['map'])
    summary = {}
    for name, rows in out.items():
        for r in rows:
            r['sha256'] = digest(ROOT / r['path'])
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        fixes = Counter(fx for r in rows for b in r['boxes'] for fx in b['fixes'])
        summary[name] = dict(images=len(rows), boxes_kept=sum(len(r['boxes']) for r in rows),
                             dropped=dict(sum((Counter(r['dropped']) for r in rows), Counter())), fixes=dict(fixes),
                             images_rescaled=sum((r['width'], r['height']) != tuple(r['declared']) for r in rows),
                             max_size_diff_px=max(max(abs(r['width'] - r['declared'][0]), abs(r['height'] - r['declared'][1])) for r in rows),
                             labels=dict(Counter(b['label'] for r in rows for b in r['boxes']).most_common()))
        print(name, summary[name], flush=True)
    avm = author_vs_mirror(out['acnedet_author'], out['acnedet_mirror'])
    REVIEW.mkdir(parents=True, exist_ok=True)
    diff_sheet(avm['boxes'], REVIEW / 'author_vs_mirror_over_1px.jpg')
    write_json(REPORT, dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), elapsed_s=round(time.time() - t0),
                            mapping_version=yaml.safe_load(MAPPING.read_text(encoding='utf-8'))['version'],
                            sources=summary, acnedet_author_vs_mirror=avm,
                            decision='acnedet_author is the label source; mirror kept for the historical split only'))
    print(json.dumps(dict(author_vs_mirror={k: v for k, v in avm.items() if k != 'boxes'}), ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
