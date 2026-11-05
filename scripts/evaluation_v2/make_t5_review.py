"""P11 T5: owner review task of the boxes the released policy shows on the T5 photos (non-expert question).

One item per shown box (all of them, or a seeded sample of MAX_BOXES when there are more): left, the area around the
box enlarged with the box in red; right, the whole photo with the box marked. Question: is the box on skin, on a
spot / bump / dot? "Sai" = clearly not (eye, nostril, hair, eyebrow, background, jewellery, clothes). The answers
give the T5 share of boxes clearly off skin or spot (final_evaluation.py --score). No answer is used as a label.
Writes reports/selfie_v1/review/t5_boxes/{items.json, *.jpg}; review with
  venv/Scripts/python.exe scripts/data_v2/review_server.py --task t5_boxes
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
from display_policy import apply_policy  # noqa: E402

RUN = ROOT / 'runs/selfie_v1/final'
OUT = ROOT / 'reports/selfie_v1/review/t5_boxes'
RELEASED = ROOT / 'reports/selfie_v1/final_evaluation/released_policy.json'
MAX_BOXES, SEED, SIDE = 300, 0, 420
RED = (40, 40, 230)


def panel(img, box):
    x1, y1, x2, y2 = box
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 12) * 4
    h, w = img.shape[:2]
    a, b = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    c, d = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    crop = img[b:d, a:c].copy()
    k = SIDE / max(crop.shape[:2])
    crop = cv2.resize(crop, (round(crop.shape[1] * k), round(crop.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(crop, (round((x1 - a) * k), round((y1 - b) * k)), (round((x2 - a) * k), round((y2 - b) * k)), RED, 2)
    k2 = SIDE / h
    full = cv2.resize(img, (round(w * k2), SIDE), interpolation=cv2.INTER_AREA)
    m = 6
    cv2.rectangle(full, (round(x1 * k2) - m, round(y1 * k2) - m), (round(x2 * k2) + m, round(y2 * k2) + m), RED, 2)
    crop = np.vstack([crop, np.full((SIDE - crop.shape[0], crop.shape[1], 3), 255, np.uint8)])
    return np.hstack([crop, np.full((SIDE, 12, 3), 255, np.uint8), full])


def main():
    policy = json.loads(RELEASED.read_text(encoding='utf-8'))
    recs = [json.loads(line) for line in (RUN / 'T5.jsonl').read_text(encoding='utf-8').splitlines()]
    boxes = [(r['key'], j, b['bbox']) for r in recs if 'error' not in r for j, b in enumerate(apply_policy(r, policy)['boxes'])]
    pick = sorted(np.random.default_rng(SEED).permutation(len(boxes))[:MAX_BOXES]) if len(boxes) > MAX_BOXES else range(len(boxes))
    OUT.mkdir(parents=True, exist_ok=True)
    items, imgs = [], {}
    for n, i in enumerate(pick):
        key, j, bbox = boxes[i]
        if key not in imgs:
            imgs = {key: cv2.imread(str(ROOT / key))}
        name = f'{n:04d}.jpg'
        cv2.imwrite(str(OUT / name), panel(imgs[key], bbox), [cv2.IMWRITE_JPEG_QUALITY, 90])
        items.append(dict(id=f'{key}|{j}', image=name, note=f'{"selfie" if key.startswith("test_img") else "SCIN"} · {Path(key).name} · box {j + 1}'))
    task = dict(title='T5 — box AI hiển thị có nằm đúng chỗ không',
                question='Khung đỏ có nằm trên da, đúng chỗ có một nốt / đốm / chấm không?',
                help='Đúng: khung nằm trên da và có nốt, mụn, đốm hoặc chấm (không cần biết là loại gì). '
                     'Sai: rõ ràng không phải, ví dụ mắt, lỗ mũi, tóc, lông mày, nền, quần áo, trang sức, hoặc da trơn không có gì. '
                     'Không chắc: ảnh quá mờ hoặc không nhìn ra. Bên trái là vùng phóng to, bên phải là cả ảnh.',
                items=items, boxes_shown_total=len(boxes), sampled=len(items) < len(boxes), photos=len(recs),
                policy_sha256=__import__('hashlib').sha256(RELEASED.read_bytes()).hexdigest())
    (OUT / 'items.json').write_text(json.dumps(task, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{len(items)} boxes ({len(boxes)} shown on {len(recs)} photos) -> {OUT}')


if __name__ == '__main__':
    main()
