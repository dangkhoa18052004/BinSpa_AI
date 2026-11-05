"""P05: one GPU pass of the deployed pipeline, raw outputs cached for every later display policy.

For each photo: the input-check metrics and failed checks, and every detector box above DEMO_CONF (0.30) with its
detector score, classifier logits (5 classes, order in meta) and Mahalanobis crop distance. Normal and demo mode,
and every P08a policy, are recomputed from this cache (baseline_eval.py); the `check` set also stores the real
AcnePipeline.analyze output of both modes so the recomputation can be compared with the deployed code.
Sets: check (test_img + 10 T1 photos), T1, val, T3 (T1 x t3_degrade.VARIANTS, key '<path>#<variant>'),
T4 (SCIN head/neck cases with a dermatologist label: acne weight >= 0.5 or 0), valP (one phone-style copy of each
val photo, val_phone.py, key '<path>#phone'). T2 is never run here.
Writes runs/selfie_v1/raw_cache/<tag>/<set>.jsonl (append, resumable) and meta.json.
Candidates (P08b, --tag other than baseline): --detector and --classifier replace the deployed checkpoints; the
classifier runs at temperature 1 (the display calibrators absorb the scale) and the domain statistics are refitted
on its features of the train GT crops of datasets/selfie_v1/crops_v1 (domain_stats.npz in the cache folder); the
deployed pipeline output of the check set is stored for the baseline only.
Usage: venv/Scripts/python.exe scripts/evaluation_v2/run_raw_cache.py [--sets check T1 val T3 T4]
       ... --tag cand_d1c1 --detector runs/selfie_v1/det_d1/weights/best.pt --classifier runs/selfie_v1/cls_c1/best.pth
"""
import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
from evaluate_pipeline import eval_items  # noqa: E402
from t3_degrade import VARIANTS, degrade  # noqa: E402
from val_phone import phone_image, phone_key  # noqa: E402
from tiered_pipeline import raw_analyze  # noqa: E402  (same steps as AcnePipeline.analyze, nothing withheld)

CACHES = ROOT / 'runs/selfie_v1/raw_cache'
CROPS_V1 = ROOT / 'datasets/selfie_v1/crops_v1'
SCIN = ROOT / 'data_raw/scin'


def scin_items():
    """SCIN head/neck cases: label 1 = acne weight >= 0.5, 0 = dermatologist label without acne; cases without a
    dermatologist label or with 0 < weight < 0.5 are left out."""
    label = {}
    for r in csv.DictReader(open(SCIN / 'scin_labels.csv', encoding='utf-8')):
        w = ast.literal_eval(r['weighted_skin_condition_label'] or '{}')
        a = sum(v for k, v in w.items() if 'acne' in k.lower())
        if w and (a >= 0.5 or a == 0):
            label[r['case_id']] = int(a >= 0.5)
    items = []
    for r in csv.DictReader(open(SCIN / 'scin_cases.csv', encoding='utf-8')):
        if r['body_parts_head_or_neck'] == 'YES' and r['case_id'] in label:
            for i in (1, 2, 3):
                p = r[f'image_{i}_path']
                if p:
                    items.append(dict(key=f"data_raw/scin/images/{Path(p).name}", case_id=r['case_id'],
                                      acne=label[r['case_id']], shot=r[f'image_{i}_shot_type']))
    return items


def items_of(name):
    if name == 'check':
        imgs = sorted((ROOT / 'test_img').glob('*.jpg'))
        return [dict(key=str(p.relative_to(ROOT)).replace('\\', '/')) for p in imgs] + \
               [dict(key=it['path']) for it in eval_items('T1')[:10]]
    if name in ('T1', 'val'):
        return [dict(key=it['path']) for it in eval_items(name)]
    if name == 'T3':
        return [dict(key=f"{it['path']}#{v}", path=it['path'], variant=v, boxes=it['boxes'])
                for it in eval_items('T1') for v in VARIANTS]
    if name == 'T4':
        return scin_items()
    if name == 'valP':
        return [dict(key=phone_key(it['path']), path=it['path'], phone=True) for it in eval_items('val')]
    raise ValueError(name)


def load(item):
    path = item.get('path', item['key'])
    img = cv2.imread(str(ROOT / path))
    if img is None:
        raise RuntimeError(f'cannot read {path}')
    if 'variant' in item:
        img, _ = degrade(img, item['boxes'], item['variant'], item['path'])
    if item.get('phone'):
        img = phone_image(img, item['path'])
    return img


def fit_domain(clf, path):
    """Domain statistics of a candidate classifier: its features of the crops_v1 train GT crops."""
    from input_gate import DomainScore
    with (CROPS_V1 / 'crop_manifest.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'train' and r['kind'] == 'gt']
    feats = []
    for i in range(0, len(rows), 2048):
        feats.append(clf.embed([cv2.imread(str(CROPS_V1 / r['crop_path'])) for r in rows[i:i + 2048]])[0])
    y = np.array([clf.classes.index(r['cls']) for r in rows])
    ds = DomainScore.fit(np.concatenate(feats), y, len(clf.classes))
    ds.save(path)
    return ds


def main():
    from predict_acne_type import DEMO_CONF, AcnePipeline, TypeClassifier
    from common2 import digest
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sets', nargs='+', default=['check', 'T1', 'val', 'T3', 'T4'])
    p.add_argument('--tag', default='baseline')
    p.add_argument('--detector', help='candidate detector weights (default: deployed)')
    p.add_argument('--classifier', help='candidate classifier .pth (default: deployed)')
    a = p.parse_args()
    OUT = CACHES / a.tag
    baseline = a.tag == 'baseline'
    if baseline != (a.detector is None and a.classifier is None):
        p.error('the baseline tag uses the deployed checkpoints; candidates need their own tag')
    OUT.mkdir(parents=True, exist_ok=True)
    demo = AcnePipeline(demo=True, **(dict(weights=ROOT / a.detector) if a.detector else {}))
    assert demo.conf == DEMO_CONF and demo.face and demo.domain
    if a.classifier:
        demo.clf = TypeClassifier(ROOT / a.classifier, temperature=1.0)
    if not baseline:
        stats = OUT / 'domain_stats.npz'
        from input_gate import DomainScore
        demo.domain = DomainScore.load(stats) if stats.exists() else fit_domain(demo.clf, stats)
    meta = dict(versions=demo.versions(), classes=demo.clf.classes, temperature=demo.clf.temperature, tau=demo.tau,
                gate=demo.gate['thresholds'], gate_sha256=digest(ROOT / 'models/phase2/input_gate.json'),
                normal_conf=json.loads((ROOT / 'models/phase1/model_metadata.json').read_text(encoding='utf-8'))['operating_conf'],
                demo_conf=DEMO_CONF, min_lesions_domain=3, t3_variants=VARIANTS)
    if not baseline:
        meta.update(detector=a.detector or 'models/phase1/best_acne_detector.pt',
                    classifier=a.classifier or 'models/phase2/best_acne_type_robust.pth',
                    domain_stats=str((OUT / 'domain_stats.npz').relative_to(ROOT)), domain_stats_sha256=digest(OUT / 'domain_stats.npz'))
        meta['versions']['type_model'] = demo.clf.version
    old = OUT / 'meta.json'
    if old.exists() and json.loads(old.read_text(encoding='utf-8'))['versions'] != meta['versions']:
        raise SystemExit(f'cache was made with other model versions; move {OUT} away first')
    old.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    normal = AcnePipeline(demo=False) if 'check' in a.sets and baseline else None
    for name in a.sets:
        items = items_of(name)
        f = OUT / f'{name}.jsonl'
        done = {json.loads(line)['key'] for line in f.read_text(encoding='utf-8').splitlines()} if f.exists() else set()
        todo = [it for it in items if it['key'] not in done]
        print(f'{name}: {len(items)} photos, {len(todo)} to run', flush=True)
        with f.open('a', encoding='utf-8') as out:
            for k, it in enumerate(todo):
                img = load(it)
                t = time.perf_counter()
                rec = raw_analyze(demo, img)
                rec['seconds'] = round(time.perf_counter() - t, 3)
                rec.update({k2: v for k2, v in it.items() if k2 not in ('boxes', 'path')}, set=name)
                if name == 'check' and baseline:
                    rec['deployed'] = dict(normal=normal.analyze(img, it['key']), demo=demo.analyze(img, it['key']))
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
                if (k + 1) % 50 == 0:
                    print(f'  {name} {k + 1}/{len(todo)}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
