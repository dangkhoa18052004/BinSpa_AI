"""T3: simulated phone-photo degradations of T1 photos (plan section 4, T3), boxes moved with the image.

Fixed before any result was seen (P05); do not tune them on results. Each keeps the lesions visible (the detector
already sees ACNE-DET photos at 1280 px on the long side, about 0.28x, so a 1080 px wide photo is not smaller than
what it sees today). Seeded per photo and variant.
"""
import zlib

import cv2
import numpy as np

VARIANTS = {
    'downscale_1080': 'resize to 1080 px wide (front camera size)',
    'jpeg_q35': 'JPEG quality 35',
    'noise_s8': 'Gaussian noise sigma 8 (of 255)',
    'dark_x0.6': 'brightness x0.6',
    'warm_wb': 'warm white balance: R x1.12, B x0.85',
    'rotate_5': 'rotate 5 degrees about the centre, black corners',
    'phone_combo': '1080 px wide + warm white balance + noise sigma 4 + JPEG quality 60',
}


def _jpeg(img, q):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _noise(img, sigma, rng):
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _warm(img):
    out = img.astype(np.float32)
    out[..., 2] *= 1.12
    out[..., 0] *= 0.85
    return np.clip(out, 0, 255).astype(np.uint8)


def _affine_boxes(boxes, M, w, h):
    """xyxy boxes through a 2x3 affine -> axis-aligned box of the moved corners, clipped; boxes whose centre leaves
    the image are dropped."""
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b['bbox']
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], float) @ M.T
        cx, cy = pts.mean(0)
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        nb = [max(pts[:, 0].min(), 0), max(pts[:, 1].min(), 0), min(pts[:, 0].max(), w), min(pts[:, 1].max(), h)]
        out.append(dict(b, bbox=[round(float(v), 2) for v in nb]))
    return out


def geometry(variant, w, h):
    """-> (2x3 affine from the original to the degraded photo, new width, new height)."""
    if variant in ('downscale_1080', 'phone_combo'):
        s = 1080 / w
        return np.array([[s, 0, 0], [0, s, 0]], float), 1080, int(round(h * s))
    if variant == 'rotate_5':
        return cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1.0), w, h
    if variant in VARIANTS:
        return np.array([[1, 0, 0], [0, 1, 0]], float), w, h
    raise ValueError(variant)


def degrade_boxes(boxes, variant, w, h):
    M, nw, nh = geometry(variant, w, h)
    return _affine_boxes(boxes, M, nw, nh)


def degrade(img, boxes, variant, key):
    """img BGR, boxes [{bbox, label}] -> (degraded img, boxes in its pixels)."""
    rng = np.random.default_rng(zlib.crc32(f'{key}|{variant}'.encode()))
    h, w = img.shape[:2]
    M, nw, nh = geometry(variant, w, h)
    if variant in ('downscale_1080', 'phone_combo'):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if variant == 'jpeg_q35':
        img = _jpeg(img, 35)
    elif variant == 'noise_s8':
        img = _noise(img, 8, rng)
    elif variant == 'dark_x0.6':
        img = np.clip(img.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    elif variant == 'warm_wb':
        img = _warm(img)
    elif variant == 'rotate_5':
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    elif variant == 'phone_combo':
        img = _jpeg(_noise(_warm(img), 4, rng), 60)
    return img, _affine_boxes(boxes, M, nw, nh)
