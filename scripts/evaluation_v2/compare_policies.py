"""Decision point 2: paired comparison of two display policies (each on its own raw cache) on the same photos.

For val, valP (phone copies of val, when both caches have them), T1 and T3 (all variants pooled): per-photo
counts at IoU 0.3 for both policies, then one bootstrap over lineage groups with the SAME draws for both, so the CI
is on the difference (candidate - base), per source and overall. The test_img selfies (no labels): boxes shown per tier for both policies.
Rule (task file, decision point 2): the candidate is a clear improvement only if its CI on the difference excludes 0
for the coverage it adds AND the A+B coverage gain on the phone source (detectdataskin2) is >= 5 points, with the
tier precisions still at their targets (the candidate's own policy is selected under the same rule as the base).
Writes reports/selfie_v1/compare/<cand>_vs_<base>.{json,md}.
Usage: venv/Scripts/python.exe scripts/evaluation_v2/compare_policies.py --cand cand_d1c1 [--base baseline]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import write_json  # noqa: E402
from display_policy import apply_policy  # noqa: E402
from evaluate_pipeline import eval_items, photo_counts, summarize  # noqa: E402
from t3_degrade import VARIANTS, degrade_boxes  # noqa: E402
from val_phone import phone_items  # noqa: E402

RESAMPLES, SEED, IOU = 1000, 0, 0.3
METRICS = ['location_precision', 'location_recall', 'tier_A_precision', 'tier_B_precision', 'coverage_A_on_found',
           'coverage_A_or_B_on_found', 'a_shown_per_photo', 'ab_correct_per_photo']


def metrics(c):
    m = summarize(c)
    n = c.get('photos', 0) or None
    m['a_shown_per_photo'] = None if not n else c.get('a_shown', 0) / n
    m['ab_correct_per_photo'] = None if not n else (c.get('a_correct', 0) + c.get('b_correct', 0)) / n
    return {k: m[k] for k in METRICS} | dict(photos=m['photos'], tier_A_shown=m['tier_A_shown'], tier_B_shown=m['tier_B_shown'])


def load(tag, name):
    path = ROOT / 'runs/selfie_v1/raw_cache' / tag / f'{name}.jsonl'
    return {r['key']: r for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())}


def paired(items, caches, policies):
    """-> {scope: {base, cand, diff, diff_ci95}} for 'overall' and each source."""
    counts = [defaultdict(Counter), defaultdict(Counter)]  # per policy: (source, group) -> Counter
    for it in items:
        for k in (0, 1):
            shown = apply_policy(caches[k][it['path']], policies[k])['boxes']
            counts[k][(it['source'], it['lineage_group_id'])].update(photo_counts(it, shown, IOU))
    out = {}
    for scope in ['overall'] + sorted({it['source'] for it in items}):
        keys = sorted(g for g in counts[0] if scope == 'overall' or g[0] == scope)
        groups = sorted({g[1] for g in keys})
        per = [defaultdict(Counter), defaultdict(Counter)]
        for k in (0, 1):
            for g in keys:
                per[k][g[1]].update(counts[k][g])
        tot = [metrics(sum(per[k].values(), Counter())) for k in (0, 1)]
        ckeys = sorted({c for k in (0, 1) for cnt in per[k].values() for c in cnt})
        M = [np.array([[per[k][g].get(c, 0) for c in ckeys] for g in groups], float) for k in (0, 1)]
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(groups), size=(RESAMPLES, len(groups)))
        W = np.stack([np.bincount(d, minlength=len(groups)) for d in draws])
        S = [W @ M[k] for k in (0, 1)]
        diffs = defaultdict(list)
        for i in range(RESAMPLES):
            ms = [metrics(dict(zip(ckeys, S[k][i]))) for k in (0, 1)]
            for name in METRICS:
                if ms[0][name] is not None and ms[1][name] is not None:
                    diffs[name].append(ms[1][name] - ms[0][name])
        out[scope] = dict(base=tot[0], cand=tot[1], groups=len(groups),
                          diff={n: None if tot[0][n] is None or tot[1][n] is None else tot[1][n] - tot[0][n] for n in METRICS},
                          diff_ci95={n: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                     for n, v in diffs.items() if len(v) >= RESAMPLES / 2})
    return out


def pct(v):
    return '–' if v is None else f'{v * 100:.1f}'


def table(res, per_photo=('a_shown_per_photo', 'ab_correct_per_photo')):
    L = ['| Phạm vi | Chỉ số | Chặng 1 | Candidate | Chênh (CI 95%) |', '| --- | --- | ---: | ---: | --- |']
    for scope, r in res.items():
        for n in METRICS:
            f = (lambda v: '–' if v is None else f'{v:.2f}') if n in per_photo else pct
            ci = r['diff_ci95'].get(n)
            d = r['diff'][n]
            L.append(f"| {scope} ({r['base']['photos']} ảnh) | {n} | {f(r['base'][n])} | {f(r['cand'][n])} | "
                     f"{'–' if d is None else f(d)}" + (f" [{f(ci[0])}; {f(ci[1])}]" if ci else '') + ' |')
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='baseline')
    ap.add_argument('--cand', required=True)
    a = ap.parse_args()
    tags = (a.base, a.cand)
    policies = [json.loads((ROOT / f'models/candidates/selfie_v1/{t}_display_policy.json').read_text(encoding='utf-8')) for t in tags]
    lineage = {r['path']: r for r in map(json.loads, (ROOT / 'datasets/selfie_v1/lineage.jsonl').read_text(encoding='utf-8').splitlines())}
    val = [it for it in eval_items('val') if lineage[it['path']]['own_locked_split'] != 'train']
    t1 = eval_items('T1')
    t3 = [dict(it, path=f"{it['path']}#{v}", boxes=degrade_boxes(it['boxes'], v, it['width'], it['height'])) for v in VARIANTS for it in t1]
    res = dict(base=a.base, cand=a.cand, policies=[p['version'] for p in policies], iou=IOU, resamples=RESAMPLES)
    sets = [('val', val, 'val'), ('T1', t1, 'T1'), ('T3', t3, 'T3')]
    if all((ROOT / 'runs/selfie_v1/raw_cache' / t / 'valP.jsonl').exists() for t in tags):
        sets.insert(1, ('valP', phone_items(val), 'valP'))
    for name, items, cache in sets:
        res[name] = paired(items, [load(t, cache) for t in tags], policies)
        print(name, 'done', flush=True)
    checks = [load(t, 'check') for t in tags]
    res['selfies_test_img'] = []
    for key in sorted(k for k in checks[0] if k.startswith('test_img') and k in checks[1]):  # photos in both runs
        ds = [apply_policy(checks[k][key], policies[k]) for k in (0, 1)]
        res['selfies_test_img'].append(dict(photo=key, base=dict(risk=ds[0]['risk'], shown=dict(Counter(b['tier'] for b in ds[0]['boxes']))),
                                            cand=dict(risk=ds[1]['risk'], shown=dict(Counter(b['tier'] for b in ds[1]['boxes'])))))
    phone = res['val'].get('detectdataskin2')
    gain, gain_ci = (phone['diff']['coverage_A_or_B_on_found'], phone['diff_ci95'].get('coverage_A_or_B_on_found')) if phone else (None, None)
    res['decision_rule'] = dict(phone_val_coverage_gain=gain, ci95=gain_ci,
                                clear=bool(gain is not None and gain >= 0.05 and gain_ci and gain_ci[0] > 0))
    out = ROOT / 'reports/selfie_v1/compare' / f'{a.cand}_vs_{a.base}'
    write_json(out.with_suffix('.json'), res)
    L = [f'# Điểm quyết định 2 — `{a.cand}` so với `{a.base}`', '',
         f'Cặp đôi trên cùng ảnh, IoU {IOU}, bootstrap {RESAMPLES} lần theo lineage group (cùng mẫu cho hai chính sách). '
         'Tỉ lệ tính bằng %, chỉ số /ảnh để nguyên.', '',
         f"Luật: coverage A+B trên nguồn điện thoại val (DDS2) tăng ≥ 5 điểm và CI > 0 → "
         f"{'**cải thiện rõ**' if res['decision_rule']['clear'] else '**chưa cải thiện rõ**'} "
         f"(tăng {pct(gain)} điểm, CI {[pct(v) for v in gain_ci] if gain_ci else '–'}).", '']
    for name, _, _ in sets:
        L += [f'## {name}', ''] + table(res[name]) + ['']
    L += ['## 21 selfie `test_img` (không nhãn): box hiện theo tầng', '', '| Ảnh | Chặng 1 | Candidate |', '| --- | --- | --- |']
    L += [f"| {s['photo']} | {s['base']['risk']} {s['base']['shown']} | {s['cand']['risk']} {s['cand']['shown']} |" for s in res['selfies_test_img']]
    out.with_suffix('.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'-> {out.with_suffix(".md")}')


if __name__ == '__main__':
    main()
