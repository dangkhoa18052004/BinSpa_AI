"""Before / after sheet of the test_img selfies: the boxes each display policy shows (raw caches, no GPU).

Left: base policy, right: candidate. Tier A box coloured by type with its name, tier B by group, tier C grey.
No labels exist for these photos: the sheet shows what a customer would see, not whether it is right.
Writes reports/selfie_v1/compare/<cand>_vs_<base>_selfies/<photo>.jpg.
Usage: venv/Scripts/python.exe scripts/evaluation_v2/selfie_sheet.py --cand cand_d1c1 [--base baseline]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
from display_policy import apply_policy  # noqa: E402

TYPE_BGR = dict(closed_comedo=(240, 240, 240), open_comedo=(60, 60, 60), papule=(46, 87, 228), pustule=(0, 200, 255), nodule=(160, 40, 160))
GROUP_BGR = dict(inflammatory=(46, 87, 228), non_inflammatory=(78, 193, 242))
SHORT = dict(closed_comedo='dau trang', open_comedo='dau den', papule='san', pustule='mu', nodule='not')
WIDTH = 900


def load(tag):
    path = ROOT / 'runs/selfie_v1/raw_cache' / tag / 'check.jsonl'
    return {r['key']: r for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())}


def panel(img, rec, policy, title):
    d = apply_policy(rec, policy)
    s = WIDTH / img.shape[1]
    out = cv2.resize(img, (WIDTH, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    for b in sorted(d['boxes'], key=lambda b: 'CBA'.index(b['tier'])):
        x1, y1, x2, y2 = (round(v * s) for v in b['bbox'])
        if b['tier'] == 'C':
            for x in range(x1, x2, 6):  # dashed grey
                cv2.line(out, (x, y1), (min(x + 3, x2), y1), (171, 165, 154), 1)
                cv2.line(out, (x, y2), (min(x + 3, x2), y2), (171, 165, 154), 1)
            for y in range(y1, y2, 6):
                cv2.line(out, (x1, y), (x1, min(y + 3, y2)), (171, 165, 154), 1)
                cv2.line(out, (x2, y), (x2, min(y + 3, y2)), (171, 165, 154), 1)
            continue
        color = TYPE_BGR[b['type']] if b['tier'] == 'A' else GROUP_BGR[b['group']]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if b['tier'] == 'A':
            cv2.putText(out, SHORT[b['type']], (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    n = Counter(b['tier'] for b in d['boxes'])
    head = np.full((44, WIDTH, 3), 255, np.uint8)
    cv2.putText(head, f"{title} | risk {d['risk']} | A {n['A']}  B {n['B']}  C {n['C']}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([head, out])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='baseline')
    ap.add_argument('--cand', required=True)
    a = ap.parse_args()
    tags = (a.base, a.cand)
    policies = [json.loads((ROOT / f'models/candidates/selfie_v1/{t}_display_policy.json').read_text(encoding='utf-8')) for t in tags]
    caches = [load(t) for t in tags]
    out = ROOT / 'reports/selfie_v1/compare' / f'{a.cand}_vs_{a.base}_selfies'
    out.mkdir(parents=True, exist_ok=True)
    for key in sorted(k for k in caches[1] if k.startswith('test_img') and k in caches[0] and (ROOT / k).exists()):
        img = cv2.imread(str(ROOT / key))
        sides = [panel(img, caches[k][key], policies[k], t) for k, t in enumerate(tags)]
        h = max(p.shape[0] for p in sides)
        sides = [np.vstack([p, np.full((h - p.shape[0], WIDTH, 3), 255, np.uint8)]) for p in sides]
        cv2.imwrite(str(out / Path(key).name), np.hstack([sides[0], np.full((h, 12, 3), 255, np.uint8), sides[1]]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f'-> {out}')


if __name__ == '__main__':
    main()
