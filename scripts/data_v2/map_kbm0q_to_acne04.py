"""P02 (selfie_v1): map the kbm0q type labels back onto the full-resolution ACNE04 photos.

kbm0q v21 is mostly ACNE04 resized to fit 416 px, every train copy flipped horizontally with p = 0.5. For every
kbm0q file:
  1. candidates: the 3 ACNE04 photos with the smallest pHash distance over 8 flip/rotation variants (<= PHASH_MAX)
  2. SIFT + ratio test against each candidate, kbm0q as is and mirrored; RANSAC similarity transform
     (cv2.estimateAffinePartial2D); keep the candidate/orientation with most inliers
  3. accept when inliers >= MIN_INLIERS, inlier share >= MIN_SHARE, |rotation| <= MAX_ROT_DEG and the ACNE04
     photo warped into the kbm0q frame correlates with it (NCC >= MIN_NCC)
  4. boxes: COCO xywh scaled from the declared to the real kbm0q size (<= 2 px), mirrored if needed, mapped through
     the transform to ACNE04 pixels, clipped; labels through configs/selfie_v1/label_mapping.yaml
Checks written to the report: agreement of mapped boxes with the independent Acne04-v2 lesions, and agreement
between two copies of the same kbm0q source mapped separately. Each mapped photo inherits the locked phase-1
split of its ACNE04 photo (by SHA256; byte duplicates inherit the kept photo's split).
Writes datasets/selfie_v1/kbm0q_on_acne04.jsonl, reports/selfie_v1/kbm0q_mapping.json.
Usage: venv/Scripts/python.exe scripts/data_v2/map_kbm0q_to_acne04.py
"""
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import digest, write_json  # noqa: E402

KBM = ROOT / 'data_raw/acne-kbm0q_roboflow/extracted'
ACNE04 = ROOT / 'raw/ACNE04_images'
V2 = ROOT / 'acne04v2/Acne04-v2_annotations.json'
SPLIT = ROOT / 'datasets/phase1_acne_detection/manifest.json'
MAPPING = ROOT / 'configs/selfie_v1/label_mapping.yaml'
CACHE = ROOT / 'datasets/selfie_v1/cache'
OUT = ROOT / 'datasets/selfie_v1/kbm0q_on_acne04.jsonl'
REPORT = ROOT / 'reports/selfie_v1/kbm0q_mapping.json'
SMALL = 1100          # ACNE04 longest side for feature matching
PHASH_MAX, TOP_K = 16, 3
MIN_INLIERS, MIN_SHARE, MAX_ROT_DEG, MIN_NCC = 25, 0.25, 3.0, 0.85
POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def phash_variants(gray):
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    out = []
    for flip in (False, True):
        a = g[:, ::-1] if flip else g
        for k in range(4):
            d = cv2.dct(np.ascontiguousarray(np.rot90(a, k)))[:8, :8].flatten()
            out.append(np.packbits(d > np.median(d[1:])))
    return np.array(out)


def acne04_index():
    """-> names, pHash (n, 8, 8), sha256 per file; cached by file list + sizes."""
    files = sorted(ACNE04.glob('*.jpg'))
    key = digest_list(files)
    cache = CACHE / f'acne04_index_{key}.npz'
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return [str(x) for x in d['names']], d['hashes'], [str(x) for x in d['sha']]
    names, hashes, sha = [], [], []
    for i, p in enumerate(files):
        g = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        names.append(p.name)
        hashes.append(phash_variants(g))
        sha.append(digest(p))
        if i % 200 == 0:
            print(f'  ACNE04 index {i}/{len(files)}', flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, names=np.array(names), hashes=np.array(hashes), sha=np.array(sha))
    return names, np.array(hashes), sha


def digest_list(files):
    import hashlib
    h = hashlib.sha256()
    for p in files:
        h.update(f'{p.name}:{p.stat().st_size};'.encode())
    return h.hexdigest()[:12]


def locked_splits(names, sha):
    """ACNE04 file name -> locked phase-1 split; byte duplicates inherit; others 'unsplit'."""
    manifest = json.loads(SPLIT.read_text(encoding='utf-8'))['images']
    by_sha = {m['sha256']: m['split'] for m in manifest}
    return {n: by_sha.get(s, 'unsplit') for n, s in zip(names, sha)}


class Acne04Features:
    def __init__(self, sift):
        self.sift, self.cache = sift, {}

    def get(self, name):
        if name not in self.cache:
            full = cv2.imread(str(ACNE04 / name), cv2.IMREAD_GRAYSCALE)
            H, W = full.shape
            s = SMALL / max(H, W)
            small = cv2.resize(full, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
            kp, des = self.sift.detectAndCompute(small, None)
            self.cache[name] = dict(small=small, kp=kp, des=des, scale=(W / small.shape[1], H / small.shape[0]), size=(W, H))
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[name]


def estimate(gk, kpk, desk, ref, matcher):
    """kbm0q (gray, keypoints, descriptors) vs ACNE04 ref -> (M kbm0q->small, inliers, share) or None."""
    if desk is None or ref['des'] is None or len(kpk) < 8:
        return None
    pairs = matcher.knnMatch(desk, ref['des'], k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None
    src = np.float32([kpk[m.queryIdx].pt for m in good])
    dst = np.float32([ref['kp'][m.trainIdx].pt for m in good])
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000)
    if M is None:
        return None
    inl = int(mask.sum())
    return M, inl, inl / len(good)


def ncc(gk, ref, M):
    """Correlation of the kbm0q photo with the ACNE04 photo warped into the kbm0q frame (valid area only)."""
    h, w = gk.shape
    Minv = cv2.invertAffineTransform(M)
    warped = cv2.warpAffine(ref['small'], Minv, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
    valid = cv2.warpAffine(np.ones_like(ref['small'], np.uint8), Minv, (w, h), flags=cv2.INTER_NEAREST) > 0
    if valid.sum() < 0.3 * h * w:
        return 0.0
    a, b = gk.astype(np.float32)[valid], warped[valid]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def kbm0q_files():
    """-> list of dict(file, split, source, declared size, boxes [(x, y, w, h, raw label)])."""
    out = []
    for ann in sorted(KBM.glob('*/_annotations.coco.json')):
        d = json.loads(ann.read_text(encoding='utf-8'))
        cats = {c['id']: c['name'] for c in d['categories']}
        boxes = defaultdict(list)
        for a in d['annotations']:
            boxes[a['image_id']].append((*a['bbox'], cats[a['category_id']]))
        for im in d['images']:
            out.append(dict(file=f"{ann.parent.name}/{im['file_name']}", split=ann.parent.name,
                            source=im['file_name'].split('.rf.')[0], declared=(im['width'], im['height']),
                            boxes=boxes[im['id']]))
    return out


def map_boxes(rec, real_wh, flip, M, ref, label_map):
    """kbm0q COCO boxes -> ACNE04 pixel boxes (clipped). Returns kept boxes and drop reasons."""
    (dw, dh), (rw, rh) = rec['declared'], real_wh
    fx, fy = rw / dw, rh / dh
    sx, sy = ref['scale']
    W, H = ref['size']
    kept, dropped = [], Counter()
    for x, y, w, h, raw in rec['boxes']:
        label = label_map.get(raw, 'UNMAPPED')
        if label == 'drop':
            continue
        if w <= 0 or h <= 0:
            dropped['degenerate'] += 1
            continue
        x1, y1, x2, y2 = x * fx, y * fy, (x + w) * fx, (y + h) * fy
        if flip:
            x1, x2 = rw - x2, rw - x1
        pts = cv2.transform(np.float32([[[x1, y1], [x2, y2], [x1, y2], [x2, y1]]]), M)[0]
        X1, Y1 = pts.min(0) * (sx, sy)
        X2, Y2 = pts.max(0) * (sx, sy)
        cx1, cy1, cx2, cy2 = max(X1, 0), max(Y1, 0), min(X2, W), min(Y2, H)
        if cx2 - cx1 < 0.5 * (X2 - X1) or cy2 - cy1 < 0.5 * (Y2 - Y1):
            dropped['mostly_outside'] += 1
            continue
        kept.append(dict(bbox=[round(float(v), 1) for v in (cx1, cy1, cx2, cy2)], raw_label=raw, label=label,
                         clipped=bool((cx1, cy1, cx2, cy2) != (X1, Y1, X2, Y2))))
    return kept, dropped


def iou_matrix(A, B):
    A, B = np.asarray(A, float).reshape(-1, 4), np.asarray(B, float).reshape(-1, 4)
    x1 = np.maximum(A[:, None, 0], B[None, :, 0]); y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2]); y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])  # noqa: E731
    return inter / (area(A)[:, None] + area(B)[None, :] - inter + 1e-9)


def greedy(iou, thr):
    pairs, ui, uj = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
        if iou[i, j] < thr:
            break
        if i not in ui and j not in uj:
            pairs.append((int(i), int(j), float(iou[i, j])))
            ui.add(i); uj.add(j)
    return pairs


def v2_boxes():
    d = json.loads(V2.read_text(encoding='utf-8'))
    names = {i['id']: i['file_name'] for i in d['images']}
    out = defaultdict(list)
    for a in d['annotations']:
        (cx, cy), r = a['coordinates'], a['radius']
        out[names[a['image_id']]].append([cx - r, cy - r, cx + r, cy + r])
    return out


def main():
    t0 = time.time()
    label_map = yaml.safe_load(MAPPING.read_text(encoding='utf-8'))['sources']['kbm0q']['map']
    names, hashes, sha = acne04_index()
    split_of = locked_splits(names, sha)
    base = hashes[:, 0]  # identity-variant hash of every ACNE04 photo
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    feats = Acne04Features(sift)
    records, status = [], Counter()
    files = kbm0q_files()
    for n, rec in enumerate(files):
        img = cv2.imread(str(KBM / rec['file']), cv2.IMREAD_GRAYSCALE)
        real = (img.shape[1], img.shape[0])
        hv = phash_variants(img)
        dist = POP[hv[:, None, :] ^ base[None, :, :]].sum(-1).min(0)
        cands = [int(i) for i in np.argsort(dist)[:TOP_K] if dist[i] <= PHASH_MAX]
        out = dict(file=rec['file'], kbm0q_split=rec['split'], source=rec['source'], real_size=list(real),
                   declared_size=list(rec['declared']), raw_boxes=len(rec['boxes']),
                   candidates=[[names[i], int(dist[i])] for i in cands])
        best = None
        for flip in (False, True):
            g = img[:, ::-1].copy() if flip else img
            kp, des = sift.detectAndCompute(g, None)
            for i in cands:
                ref = feats.get(names[i])
                r = estimate(g, kp, des, ref, matcher)
                if r and (best is None or r[1] > best[2]):
                    best = (names[i], flip, r[1], r[2], r[0], g)
        if best is None:
            out['status'] = 'no_candidate' if not cands else 'no_transform'
        else:
            name, flip, inl, share, M, g = best
            ref = feats.get(name)
            rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
            scale = math.hypot(M[0, 0], M[1, 0])
            c = ncc(g, ref, M)
            ok = inl >= MIN_INLIERS and share >= MIN_SHARE and abs(rot) <= MAX_ROT_DEG and c >= MIN_NCC
            out.update(acne04=name, acne04_split=split_of[name], flipped=flip, inliers=inl, inlier_share=round(share, 3),
                       rotation_deg=round(rot, 3), scale_to_full=round(scale * ref['scale'][0], 4), ncc=round(c, 4),
                       transform_kbm0q_to_small=M.round(6).tolist(), small_to_full=list(ref['scale']), acne04_size=list(ref['size']))
            if ok:
                kept, dropped = map_boxes(rec, real, flip, M, ref, label_map)
                out.update(status='mapped', boxes=kept, dropped=dict(dropped))
            else:
                out['status'] = 'rejected'
                out['reject'] = [k for k, bad in (('inliers', inl < MIN_INLIERS), ('share', share < MIN_SHARE),
                                                  ('rotation', abs(rot) > MAX_ROT_DEG), ('ncc', c < MIN_NCC)) if bad]
        status[out['status']] += 1
        records.append(out)
        if n % 100 == 0:
            print(f'{n}/{len(files)} {dict(status)} {time.time() - t0:.0f}s', flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Check 1: mapped boxes vs the independent Acne04-v2 lesions (location only; v2 has no type).
    v2 = v2_boxes()
    agree = Counter()
    for r in records:
        if r['status'] != 'mapped' or r['acne04'] not in v2:
            continue
        typed = [b['bbox'] for b in r['boxes'] if b['label'] not in ('ignore',)]
        if not typed:
            continue
        iou = iou_matrix(typed, v2[r['acne04']])
        p3 = greedy(iou, 0.3)
        A = np.asarray(typed, float)
        centres = (A[:, :2] + A[:, 2:]) / 2
        V = np.asarray(v2[r['acne04']], float)
        inside = ((centres[:, None, 0] >= V[None, :, 0]) & (centres[:, None, 0] <= V[None, :, 2]) &
                  (centres[:, None, 1] >= V[None, :, 1]) & (centres[:, None, 1] <= V[None, :, 3])).any(1)
        agree['kbm0q_boxes'] += len(typed); agree['v2_lesions'] += len(V)
        agree['matched_iou30'] += len(p3); agree['kbm0q_centre_in_v2_box'] += int(inside.sum())
        agree['photos'] += 1
    # Check 2: two copies of the same kbm0q source mapped separately must land on the same photo and boxes.
    by_src = defaultdict(list)
    for r in records:
        if r['status'] == 'mapped':
            by_src[r['source']].append(r)
    copy = Counter(); copy_iou = []
    for src, rs in by_src.items():
        if len(rs) < 2:
            continue
        a, b = rs[0], rs[1]
        copy['pairs'] += 1
        if a['acne04'] != b['acne04']:
            copy['different_photo'] += 1
            continue
        if a['boxes'] and b['boxes']:
            pairs = greedy(iou_matrix([x['bbox'] for x in a['boxes']], [x['bbox'] for x in b['boxes']]), 0.1)
            copy_iou += [p[2] for p in pairs]
            copy['boxes_a'] += len(a['boxes']); copy['boxes_b'] += len(b['boxes']); copy['matched'] += len(pairs)
            copy['same_label'] += sum(a['boxes'][i]['label'] == b['boxes'][j]['label'] for i, j, _ in pairs)
    mapped = [r for r in records if r['status'] == 'mapped']
    roles = Counter(r['acne04_split'] for r in mapped)
    labels = Counter(b['label'] for r in mapped for b in r['boxes'])
    box_sizes = [max(b['bbox'][2] - b['bbox'][0], b['bbox'][3] - b['bbox'][1]) for r in mapped for b in r['boxes'] if b['label'] != 'ignore']
    unique = {}
    for r in mapped:
        unique.setdefault(r['source'], r)
    report = dict(
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), elapsed_s=round(time.time() - t0),
        params=dict(phash_max=PHASH_MAX, top_k=TOP_K, min_inliers=MIN_INLIERS, min_share=MIN_SHARE,
                    max_rotation_deg=MAX_ROT_DEG, min_ncc=MIN_NCC, small_side=SMALL),
        files=len(records), status=dict(status),
        reject_reasons=dict(Counter(k for r in records if r['status'] == 'rejected' for k in r['reject'])),
        mapped_sources=len(unique), mapped_acne04_photos=len({r['acne04'] for r in mapped}),
        mapped_files_by_locked_split=dict(roles),
        mapped_sources_by_locked_split=dict(Counter(r['acne04_split'] for r in unique.values())),
        flipped_share=round(sum(r['flipped'] for r in mapped) / max(len(mapped), 1), 3),
        ncc_median=round(float(np.median([r['ncc'] for r in mapped])), 4) if mapped else None,
        scale_to_full_median=round(float(np.median([r['scale_to_full'] for r in mapped])), 3) if mapped else None,
        labels_all_files=dict(labels.most_common()),
        typed_box_long_side_px_full_res=dict(median=float(np.median(box_sizes)), p10=float(np.percentile(box_sizes, 10)),
                                             p90=float(np.percentile(box_sizes, 90))) if box_sizes else None,
        dropped=dict(sum((Counter(r.get('dropped', {})) for r in mapped), Counter())),
        check_vs_acne04v2=dict(agree, note='location only; v2 is an independent expert annotation of the same photos'),
        check_copies=dict(copy, iou_median=round(float(np.median(copy_iou)), 3) if copy_iou else None,
                          iou_p10=round(float(np.percentile(copy_iou, 10)), 3) if copy_iou else None),
    )
    write_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    random.seed(0)
    main()
