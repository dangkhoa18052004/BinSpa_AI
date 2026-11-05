"""P02: owner review items for the kbm0q -> ACNE04 mapping (scripts/data_v2/review_server.py --task kbm0q_mapping).

Picks N mapped kbm0q sources at random (seed 0, one copy per source). Each item shows the kbm0q photo with its own
boxes (left) and the matching region of the full-resolution ACNE04 photo with the mapped boxes (right), same
colours per type. Question for a non-expert: do the right boxes sit on the same spots as the left ones?
Writes reports/selfie_v1/review/kbm0q_mapping/ (local).
Usage: venv/Scripts/python.exe scripts/data_v2/make_kbm0q_review.py [--n 30]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import write_json  # noqa: E402

KBM = ROOT / 'data_raw/acne-kbm0q_roboflow/extracted'
MAPPED = ROOT / 'datasets/selfie_v1/kbm0q_on_acne04.jsonl'
OUT = ROOT / 'reports/selfie_v1/review/kbm0q_mapping'
COLOURS = dict(closed_comedo=(255, 255, 255), open_comedo=(40, 40, 40), papule=(0, 0, 255), pustule=(0, 220, 255),
               nodule=(255, 0, 255), ignore=(160, 160, 160))
HEIGHT = 560


def draw(img, boxes, width):
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b['bbox']]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), width + 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOURS[b['label']], width)
    return img


def fit(img, h=HEIGHT):
    s = h / img.shape[0]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def item(r, label_map):
    kimg = cv2.imread(str(KBM / r['file']))
    (dw, dh), (rw, rh) = r['declared_size'], r['real_size']
    native = []  # the source boxes, drawn on the kbm0q photo as delivered
    ann = json.loads((KBM / r['kbm0q_split'] / '_annotations.coco.json').read_text(encoding='utf-8'))
    cats = {c['id']: c['name'] for c in ann['categories']}
    im = next(i for i in ann['images'] if f"{r['kbm0q_split']}/{i['file_name']}" == r['file'])
    for a in ann['annotations']:
        if a['image_id'] == im['id'] and label_map.get(cats[a['category_id']]) not in (None, 'drop'):
            x, y, w, h = a['bbox']
            native.append(dict(bbox=[x * rw / dw, y * rh / dh, (x + w) * rw / dw, (y + h) * rh / dh],
                               label=label_map[cats[a['category_id']]]))
    left = fit(draw(cv2.resize(kimg, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                    [dict(b, bbox=[v * 2 for v in b['bbox']]) for b in native], 2))
    full = cv2.imread(str(ROOT / 'raw/ACNE04_images' / r['acne04']))
    # region of the ACNE04 photo covered by the kbm0q frame
    M = np.array(r['transform_kbm0q_to_small'])
    corners = cv2.transform(np.float32([[[0, 0], [rw, 0], [0, rh], [rw, rh]]]), M)[0] * r['small_to_full']
    x1, y1 = np.maximum(corners.min(0), 0).astype(int)
    x2, y2 = np.minimum(corners.max(0), [full.shape[1], full.shape[0]]).astype(int)
    lw = max(2, int((x2 - x1) / 400))
    region = draw(full, r['boxes'], lw)[y1:y2, x1:x2]
    if r['flipped']:
        region = region[:, ::-1]
    right = fit(region)
    gap = np.full((HEIGHT, 12, 3), 255, np.uint8)
    return np.hstack([left, gap, right])


def main():
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=30)
    a = p.parse_args()
    label_map = yaml.safe_load((ROOT / 'configs/selfie_v1/label_mapping.yaml').read_text(encoding='utf-8'))['sources']['kbm0q']['map']
    rows = [json.loads(line) for line in MAPPED.read_text(encoding='utf-8').splitlines()]
    one = {}
    for r in rows:
        if r['status'] == 'mapped' and any(b['label'] != 'ignore' for b in r['boxes']):
            one.setdefault(r['source'], r)
    picks = random.Random(0).sample(sorted(one), min(a.n, len(one)))
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for k, src in enumerate(picks):
        r = one[src]
        name = f'{k:03d}.jpg'
        cv2.imwrite(str(OUT / name), item(r, label_map), [cv2.IMWRITE_JPEG_QUALITY, 90])
        items.append(dict(id=r['file'], image=name, note=f"{r['acne04']} · lật: {'có' if r['flipped'] else 'không'} · {len(r['boxes'])} khung"))
    write_json(OUT / 'items.json', dict(
        title='Kiểm tra ghép nhãn kbm0q → ACNE04',
        question='Các khung bên PHẢI có nằm đúng trên cùng các nốt như khung bên TRÁI không?',
        help=('Trái: ảnh kbm0q nhỏ với nhãn gốc. Phải: cùng vùng trên ảnh ACNE04 gốc với nhãn đã ghép (đã lật lại nếu cần). '
              'Chỉ cần so vị trí, không cần biết loại mụn. Màu: trắng = đầu trắng, đen = đầu đen, đỏ = sẩn, vàng = mủ, '
              'tím = nốt, xám = thâm. Chọn "Sai" nếu nhiều khung lệch rõ khỏi nốt.'),
        items=items))
    print(f'{len(items)} items -> {OUT}')


if __name__ == '__main__':
    main()
