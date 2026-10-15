"""Near-duplicate check between the ACNE-DET master split and the data the detectors were trained on.

Exact SHA256 duplicates were already removed; this looks for re-encoded, resized, flipped or rotated
copies. Each image gets a 64-bit DCT perceptual hash (pHash, 32x32 greyscale, top-left 8x8 DCT vs its
median) for itself and its 7 flip/rotation variants. A pair is flagged when the minimum Hamming
distance over variants is <= THRESHOLD (out of 64).
Pairs checked (all that can inflate a test number):
  ACNE-DET test vs ACNE-DET train/val, ACNE-DET val vs train,
  ACNE-DET test/val vs ACNE04-v2 (all splits; ACNE04 train was used by the phase-1 detectors).
Writes reports/phase2_det5/duplicates.{json,csv}. Flagged pairs need a visual check: consecutive
visits of the same patient are expected to look alike (that is what the grouped split handles).
Usage: venv/Scripts/python.exe scripts/phase2/check_duplicates.py
"""
import csv
import sys

import cv2
import numpy as np

from common2 import MASTER_SPLIT, REPORTS, ROOT, SOURCE, write_csv, write_json

sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import DATASET  # noqa: E402

THRESHOLD = 10
OUT = REPORTS / 'phase2_det5'


def phash_variants(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f'cannot read {path}')
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    out = []
    for flip in (False, True):
        a = g[:, ::-1] if flip else g
        for k in range(4):
            d = cv2.dct(np.ascontiguousarray(np.rot90(a, k)))[:8, :8].flatten()
            out.append(d > np.median(d[1:]))
    return np.array(out)  # (8, 64) bool


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with MASTER_SPLIT.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    sets = {f'acnedet_{s}': [(r['original_name'], SOURCE / r['file_name']) for r in rows if r['split'] == s]
            for s in ('train', 'val', 'test')}
    acne04 = sorted((DATASET / 'images').glob('*/*.jpg'))
    sets['acne04_all'] = [(f'{p.parent.name}/{p.name}', p) for p in acne04]
    hashes = {}
    for name, items in sets.items():
        hashes[name] = np.stack([phash_variants(p) for _, p in items])
        print(name, len(items), flush=True)
    pairs = [('acnedet_test', 'acnedet_train'), ('acnedet_test', 'acnedet_val'), ('acnedet_val', 'acnedet_train'),
             ('acnedet_test', 'acne04_all'), ('acnedet_val', 'acne04_all')]
    flagged, summary = [], {}
    for a, b in pairs:
        A = hashes[a][:, 0]                            # identity variant of the query set
        B = hashes[b]                                  # all 8 variants of the reference set
        dist = (A[:, None, None, :] != B[None]).sum(-1).min(-1)   # (nA, nB)
        nearest = dist.min(1)
        summary[f'{a}_vs_{b}'] = dict(n_query=len(A), n_ref=len(B), flagged=int((nearest <= THRESHOLD).sum()),
                                      min_distance=int(nearest.min()), median_nearest=float(np.median(nearest)))
        for i, j in zip(*np.where(dist <= THRESHOLD)):
            flagged.append(dict(query_set=a, query=sets[a][i][0], ref_set=b, ref=sets[b][j][0], hamming=int(dist[i, j])))
    write_csv(OUT / 'duplicates.csv', flagged or [dict(query_set='', query='', ref_set='', ref='', hamming='')],
              ['query_set', 'query', 'ref_set', 'ref', 'hamming'])
    write_json(OUT / 'duplicates.json', dict(method='pHash 64-bit, min over 8 flip/rotation variants', threshold=THRESHOLD,
                                             summary=summary, flagged=flagged))
    for k, v in summary.items():
        print(k, v)


if __name__ == '__main__':
    main()
