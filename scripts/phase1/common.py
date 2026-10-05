"""Shared paths, deterministic manifests and geometry for phase 1."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Keep library settings and caches with the project, including worker processes.
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.cache/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('TORCH_HOME', str(ROOT/'.cache/torch'))
RAW = ROOT / 'raw' / 'ACNE04_images'
ANNOTATIONS = ROOT / 'acne04v2' / 'Acne04-v2_annotations.json'
DATASET = ROOT / 'datasets' / 'phase1_acne_detection'
REPORTS = ROOT / 'reports'
AUDIT = REPORTS / 'phase1_data'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def load_annotations(path=ANNOTATIONS):
    data = read_json(path)
    images = data['images']
    if len({i['id'] for i in images}) != len(images):
        raise ValueError('Duplicate image IDs in annotation source')
    if len({i['file_name'] for i in images}) != len(images):
        raise ValueError('Duplicate filenames in annotation source')
    for image in images:
        name = image['file_name']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError(f'Unsafe image filename: {name}')
    ids = {i['id'] for i in images}
    if len({a['id'] for a in data['annotations']}) != len(data['annotations']):
        raise ValueError('Duplicate annotation IDs')
    if any(a['image_id'] not in ids for a in data['annotations']):
        raise ValueError('Annotation refers to unknown image ID')
    return data


def yolo_box(annotation, width, height):
    """Return normalized clipped bbox, reason, clipped; never guess bad coordinates."""
    try:
        cx, cy = map(float, annotation['coordinates'])
        radius = float(annotation['radius'])
        if not all(math.isfinite(v) for v in (cx, cy, radius, width, height)):
            return None, 'non_finite', False
        if radius <= 0 or width <= 0 or height <= 0:
            return None, 'non_positive_radius_or_size', False
        original = (cx-radius, cy-radius, cx+radius, cy+radius)
        x1, y1 = max(0, original[0]), max(0, original[1])
        x2, y2 = min(width, original[2]), min(height, original[3])
        if x2 <= x1 or y2 <= y1:
            return None, 'outside_image_or_empty', False
        box = ((x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height)
        return box, '', original != (x1, y1, x2, y2)
    except (KeyError, TypeError, ValueError):
        return None, 'malformed_annotation', False


def verify_dataset(dataset=DATASET, require_review=False):
    dataset = Path(dataset)
    manifest = read_json(dataset / 'manifest.json')
    seen = set()
    split_hashes = {'train': set(), 'val': set(), 'test': set()}
    for row in manifest['images']:
        name, split = row['file_name'], row['split']
        if name in seen:
            raise RuntimeError('Image occurs in multiple splits')
        seen.add(name)
        image = dataset / 'images' / split / name
        label = dataset / 'labels' / split / (Path(name).stem + '.txt')
        if digest(image) != row['sha256'] or digest(label) != row['label_sha256']:
            raise RuntimeError(f'Dataset modified: {name}; rebuild/review explicitly')
        split_hashes[split].add(row['sha256'])
        lines = label.read_text().splitlines()
        if len(lines) != row['lesions'] or not lines:
            raise RuntimeError(f'Invalid label count for {name}')
        for line in lines:
            values = line.split()
            if len(values) != 5 or values[0] != '0':
                raise RuntimeError(f'Invalid class/label: {name}')
            cx, cy, w, h = map(float, values[1:])
            if not all(math.isfinite(v) for v in (cx, cy, w, h)) or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise RuntimeError(f'Invalid normalized box: {name}')
            if min(cx-w/2, cy-h/2) < -1e-6 or max(cx+w/2, cy+h/2) > 1+1e-6:
                raise RuntimeError(f'BBox outside image: {name}')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if split_hashes[a] & split_hashes[b]:
            raise RuntimeError(f'Content leakage between {a} and {b}')
    for split in split_hashes:
        expected_images = {r['file_name'] for r in manifest['images'] if r['split'] == split}
        actual_images = {p.name for p in (dataset/'images'/split).iterdir() if p.is_file()}
        expected_labels = {Path(n).stem+'.txt' for n in expected_images}
        actual_labels = {p.name for p in (dataset/'labels'/split).glob('*.txt')}
        if actual_images != expected_images or actual_labels != expected_labels:
            raise RuntimeError(f'Extra/missing files in {split}')
    if require_review:
        review = read_json(dataset / 'label_review.json')
        if review.get('manifest_sha256') != digest(dataset / 'manifest.json') or review.get('status') != 'reviewed':
            raise RuntimeError('Label visualization review is missing or stale')
    return manifest
