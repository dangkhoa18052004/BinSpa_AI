"""P09/P10 end-to-end check: the served TieredPipeline gives the boxes the evaluation scored.

For each photo of the raw cache `check` set (test_img selfies + 10 T1 photos), TieredPipeline.analyze (real models,
GPU) is compared with present() of the cached raw record under the same policy: same boxes, tiers, types and groups
(bbox within 1 px, probabilities within 0.01: GPU kernels differ slightly between batch layouts).
Usage: venv/Scripts/python.exe scripts/evaluation_v2/check_serving.py --tag hyb_d1   (or baseline)
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
from tiered_pipeline import TieredPipeline, present  # noqa: E402


def same(a, b):
    if len(a) != len(b):
        return False
    return all(p['tier'] == q['tier'] and p.get('type') == q.get('type') and p.get('group') == q.get('group')
               and max(abs(u - v) for u, v in zip(p['bbox'], q['bbox'])) <= 1.0
               and abs((p.get('probability') or 0) - (q.get('probability') or 0)) <= 0.01 for p, q in zip(a, b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='baseline')
    a = ap.parse_args()
    policy = f'models/candidates/selfie_v1/{a.tag}_display_policy.json'
    with tempfile.TemporaryDirectory() as d:
        srv = Path(d) / 'serving.json'
        srv.write_text(json.dumps(dict(policy=policy, published={})), encoding='utf-8')
        pipe = TieredPipeline(srv)
    cache = [json.loads(line) for line in (ROOT / 'runs/selfie_v1/raw_cache' / a.tag / 'check.jsonl').read_text(encoding='utf-8').splitlines()]
    bad, n = [], 0
    for rec in cache:
        if not (ROOT / rec['key']).exists():
            continue
        n += 1
        live = pipe.analyze(cv2.imread(str(ROOT / rec['key'])))
        ref = present(rec, pipe.policy, {})
        if live['risk'] != ref['risk'] or not same(live['predictions'], ref['predictions']):
            bad.append(dict(photo=rec['key'], live=[(p['tier'], p.get('type')) for p in live['predictions']],
                            cached=[(p['tier'], p.get('type')) for p in ref['predictions']], risk=(live['risk'], ref['risk'])))
    print(f'{a.tag}: {n - len(bad)}/{n} photos identical', flush=True)
    for b in bad:
        print(' ', b)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
