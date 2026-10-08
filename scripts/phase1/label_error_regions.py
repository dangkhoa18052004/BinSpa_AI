"""Pre-fill the region column of false_positive_region_review.csv from face landmarks.

Regions follow summarize_error_regions.py: eyes, lips, mole, hair_beard, other.
Rules, in order: eye/eyelid polygon -> eyes; lip polygon -> lips; eyebrow polygon or
hair segmentation -> hair_beard; dark, non-red spot versus surrounding skin -> mole;
everything else -> other (notes say whether it sits on face skin or off the face).
Every row is marked "auto" in notes; contact sheets per region are written for visual
verification, and rows already filled by hand are kept unless --overwrite is given.
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from common import DATASET, REPORTS, ROOT, digest, write_csv, write_json

TOOLS = ROOT/'models/phase1/face_tools'
FIELDS = ['file_name','detection_index','confidence','x1','y1','x2','y2','region','notes']
# MediaPipe face mesh contours.
EYES = [[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
        [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]]
BROWS = [[70,63,105,66,107,55,65,52,53,46],[300,293,334,296,336,285,295,282,283,276]]
LIPS = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
# selfie_multiclass_256x256 classes.
HAIR, FACE_SKIN = 1, 3
WORK = 1280


def polygon_mask(shape, points, dilate=0):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    if dilate > 0:
        k = 2*int(dilate)+1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def build_masks(rgb, landmarker, segmenter):
    """Region masks at working resolution; landmarks may be missing on tight crops."""
    h, w = rgb.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    seg = segmenter.segment(image).category_mask.numpy_view()
    seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
    masks = dict(hair=(seg == HAIR).astype(np.uint8), face_skin=(seg == FACE_SKIN).astype(np.uint8))
    faces = landmarker.detect(image).face_landmarks
    masks['landmarks'] = bool(faces)
    zero = np.zeros((h, w), np.uint8)
    masks.update(eyes=zero.copy(), lips=zero.copy(), brows=zero.copy(), oval=zero.copy())
    for face in faces:
        pts = np.array([[p.x*w, p.y*h] for p in face])
        for eye in EYES:
            e = pts[eye]; span = np.ptp(e[:, 0])
            masks['eyes'] |= polygon_mask((h, w), e, dilate=.35*span)  # lids and lashes
        for brow in BROWS:
            b = pts[brow]
            masks['brows'] |= polygon_mask((h, w), cv2.convexHull(b.astype(np.float32))[:, 0], dilate=.08*np.ptp(b[:, 0]))
        lip = pts[LIPS]
        masks['lips'] |= polygon_mask((h, w), lip, dilate=.15*np.ptp(lip[:, 1]))
        masks['oval'] |= polygon_mask((h, w), pts[OVAL])
    return masks


def is_mole(lab, box):
    """Dark, not reddish centre versus the ring of skin around the box."""
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    H, W = lab.shape[:2]
    pad = max(bw, bh)*.6
    ox1, oy1, ox2, oy2 = [int(v) for v in (max(0, x1-pad), max(0, y1-pad), min(W, x2+pad), min(H, y2+pad))]
    patch = lab[oy1:oy2, ox1:ox2].astype(np.float32)
    if patch.size == 0 or bw < 2 or bh < 2:
        return False, ''
    yy, xx = np.mgrid[oy1:oy2, ox1:ox2]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    r = ((xx-cx)/(bw/2))**2+((yy-cy)/(bh/2))**2
    inner, ring = patch[r <= .3], patch[(r > 1.5)]
    if len(inner) < 4 or len(ring) < 16:
        return False, ''
    L_in = np.percentile(inner[:, 0], 30); L_ring = np.median(ring[:, 0])
    a_in = np.median(inner[:, 1]); a_ring = np.median(ring[:, 1])
    dl, da = L_ring-L_in, a_in-a_ring
    return bool(dl > 28 and da < 6), f'dL={dl:.0f},da={da:.0f}'


def fraction(mask, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    sub = mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    return float(sub.mean()) if sub.size else 0.


def classify(masks, lab, box):
    cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
    centre = lambda m: bool(m[min(cy, m.shape[0]-1), min(cx, m.shape[1]-1)])
    f = {k: fraction(masks[k], box) for k in ('eyes', 'lips', 'brows', 'hair', 'face_skin', 'oval')}
    if centre(masks['eyes']) or f['eyes'] > .3:
        return 'eyes', f"eye={f['eyes']:.2f}"
    if centre(masks['lips']) or f['lips'] > .3:
        return 'lips', f"lip={f['lips']:.2f}"
    if centre(masks['brows']) or f['brows'] > .3:
        return 'hair_beard', f"eyebrow={f['brows']:.2f}"
    if f['hair'] > .35:
        return 'hair_beard', f"hair={f['hair']:.2f}"
    mole, why = is_mole(lab, box)
    if mole:
        return 'mole', why
    on_face = f['face_skin'] > .3 or (masks['landmarks'] and centre(masks['oval']))
    return 'other', ('face skin (possible unannotated lesion/texture)' if on_face else 'off face (neck/ear/background/clothes)')+f";skin={f['face_skin']:.2f}"


def contact_sheets(rows, images, out, per_sheet=80, cell=96):
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row['region']].append(row)
    cache = {}
    for region, items in groups.items():
        items = sorted(items, key=lambda r: (r['file_name'], int(r['detection_index'])))
        for s in range(0, len(items), per_sheet):
            chunk = items[s:s+per_sheet]; cols = 10
            sheet = np.full(((len(chunk)+cols-1)//cols*(cell+14), cols*cell, 3), 255, np.uint8)
            for i, row in enumerate(chunk):
                name = row['file_name']
                if name not in cache:
                    cache.clear(); cache[name] = cv2.imread(str(images/name))
                img = cache[name]
                x1, y1, x2, y2 = (float(row[k]) for k in ('x1', 'y1', 'x2', 'y2'))
                pad = max(x2-x1, y2-y1)*1.5
                c = img[int(max(0, y1-pad)):int(y2+pad), int(max(0, x1-pad)):int(x2+pad)].copy()
                ox, oy = max(0, x1-pad), max(0, y1-pad)
                cv2.rectangle(c, (int(x1-ox), int(y1-oy)), (int(x2-ox), int(y2-oy)), (0, 0, 255), max(1, c.shape[1]//60))
                c = cv2.resize(c, (cell, cell))
                r, k = divmod(i, cols)
                sheet[r*(cell+14):r*(cell+14)+cell, k*cell:(k+1)*cell] = c
                cv2.putText(sheet, str(s+i), (k*cell+2, r*(cell+14)+cell+11), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1)
            cv2.imwrite(str(out/f'{region}_{s//per_sheet:02d}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {k: len(v) for k, v in groups.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, default=REPORTS/'phase1_predictions/false_positive_region_review.csv')
    p.add_argument('--images', type=Path, default=DATASET/'images/test')
    p.add_argument('--overwrite', action='store_true', help='replace regions already filled by hand')
    a = p.parse_args()
    with a.csv.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    backup = a.csv.with_suffix('.before_auto.csv')
    if not backup.exists():
        shutil.copyfile(a.csv, backup)
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TOOLS/'face_landmarker.task')), num_faces=1, min_face_detection_confidence=.3))
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(TOOLS/'selfie_multiclass_256x256.tflite')), output_category_mask=True))
    by_image = defaultdict(list)
    for row in rows:
        by_image[row['file_name']].append(row)
    no_landmarks = []
    for name, items in sorted(by_image.items()):
        bgr = cv2.imread(str(a.images/name))
        scale = WORK/max(bgr.shape[:2])
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        masks = build_masks(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), landmarker, segmenter)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        if not masks['landmarks']:
            no_landmarks.append(name)
        for row in items:
            if row['region'].strip() and not row['notes'].startswith('auto') and not a.overwrite:
                continue
            box = [float(row[k])*scale for k in ('x1', 'y1', 'x2', 'y2')]
            region, why = classify(masks, lab, box)
            row['region'] = region
            row['notes'] = f"auto:{why}" + ('' if masks['landmarks'] else ';no face landmarks')
    write_csv(a.csv, rows, FIELDS)
    counts = contact_sheets(rows, a.images, a.csv.parent/'region_review')
    write_json(a.csv.parent/'region_autolabel_meta.json', dict(
        rows=len(rows), images=len(by_image), counts=counts, images_without_landmarks=no_landmarks,
        tools={p.name: digest(p) for p in sorted(TOOLS.iterdir())}, mediapipe=mp.__version__, work_resolution=WORK))
    print(counts, 'no landmarks:', len(no_landmarks))


if __name__ == '__main__':
    main()
