"""valP: one phone-style copy of every val photo, for choosing display thresholds that hold on phone photos.

The degradation is build_det_dataset.phone_degrade (seeded per photo: downscale 0.35-0.8, white balance, exposure,
noise, optional blur, JPEG 35-80), not the T3 variants, so T3 stays an independent check. Key '<path>#phone';
source '<source>_phone' (scored against the T2 targets in select_display_policy.py --phone-val); same lineage group
as the original photo, so cross-validation folds keep the pair together.
"""
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/data_v2'))
from build_det_dataset import phone_degrade  # noqa: E402

SUFFIX = '#phone'


def phone_key(path):
    return f'{path}{SUFFIX}'


def phone_image(img, path):
    return phone_degrade(img, phone_key(path))[0]


def phone_scale(path, w, h):
    """(sx, sy) of phone_image for a w x h photo: the first draw of phone_degrade's generator, rounded as it rounds."""
    s = np.random.default_rng(zlib.crc32(f'phone|{phone_key(path)}'.encode())).uniform(0.35, 0.8)
    return max(1, round(w * s)) / w, max(1, round(h * s)) / h


def phone_items(items):
    """Evaluator items -> their phone copies (boxes scaled; bbox values alternate x, y)."""
    out = []
    for it in items:
        sx, sy = phone_scale(it['path'], it['width'], it['height'])
        boxes = [dict(b, bbox=[v * (sx if i % 2 == 0 else sy) for i, v in enumerate(b['bbox'])]) for b in it['boxes']]
        out.append(dict(it, path=phone_key(it['path']), source=f"{it['source']}_phone", boxes=boxes,
                        width=max(1, round(it['width'] * sx)), height=max(1, round(it['height'] * sy))))
    return out
