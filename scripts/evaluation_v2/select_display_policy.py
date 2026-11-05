"""P08a: fit and choose the tiered display policy of the current checkpoint on val, then score it once on
T1, T3, T4 and the test_img selfies (no GPU: raw cache of run_raw_cache.py).

Rule (fixed before looking at T1; plan 6.2 / 7.3, acceptance.yaml):
  - risk groups by display_policy.photo_risk (input check + domain score <= domain_max: the deployed 40.70 for the
    baseline; for a candidate the 95th percentile of its val ACNE-DET/ACNE04 photo domain scores, the rule that
    gave 40.70 in phase 2);
  - calibrators fitted per risk group on val boxes (matched at IoU 0.3, greedy by score; boxes on `ignore` left
    out). Thresholds are chosen on out-of-fold probabilities (5 folds by lineage group) so the choice is not scored
    on the boxes the calibrators were fitted on; the deployed calibrators are refitted on all val boxes;
  - per tier, the smallest threshold on a 0.01 grid at which EVERY source with >= MIN_PHOTOS photos in the group
    reaches its target (ACNE-DET, ACNE04: the T1 targets; DetectDataSkin-2: the T2 targets) with the point value
    and with the 95 % bootstrap lower bound >= target - ci_lower_slack, and the tier shows >= MIN_SUPPORT boxes
    for each such source. Order: tau_C (location precision), then tau_A, then tau_B (on the boxes not shown at A);
  - no threshold reaches the target: tier A or B is switched off (acceptance fallback); for tier C the threshold
    with the best worst-source precision among those still showing >= MIN_SHOWN boxes per photo is kept and
    flagged 'target_not_met' (every photo must get a result, so locations are never switched off);
  - nodule: fewer than NODULE_MIN_GT nodule lesions on typed val photos of a group -> never shown at tier A there;
    after tau_A is chosen, fewer than MIN_SUPPORT nodule boxes shown at tier A on val (out-of-fold) -> nodule is
    unmeasurable there and also never shown at tier A (acceptance.yaml fallback nodule_unmeasurable; added at P11
    from the val counts, thresholds unchanged);
  - --phone-val: the val set also holds one phone-style copy of each val photo (val_phone.py, raw cache set valP,
    source '<source>_phone', same lineage group, scored against the T2 targets); calibrators and thresholds are
    fitted on both, so a tier must hold on the clean photos (their own targets) and on their phone copies;
  - hybrid cache (merge_caches.py): the risk groups named in meta group_classifier read the alt classifier's
    logits (calibrators, thresholds and display), the others the primary one; the risk is the primary one's.
Writes models/candidates/selfie_v1/<tag>_display_policy.json, reports/selfie_v1/risk_coverage/<tag>/*
(raw cache runs/selfie_v1/raw_cache/<tag>).
Usage: venv/Scripts/python.exe scripts/evaluation_v2/select_display_policy.py [--tag baseline]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import digest, write_json  # noqa: E402
from display_policy import GROUP, apply_policy, calibrated, group_features, photo_risk  # noqa: E402
from evaluate_pipeline import case_auc, eval_items, evaluate, match  # noqa: E402
from t3_degrade import VARIANTS, degrade_boxes  # noqa: E402
from val_phone import phone_items  # noqa: E402

CACHE = POLICY = OUT = None  # set from --tag in main()
TAG = 'baseline'
TARGET_SET = dict(acnedet_author='T1', acne04='T1', detectdataskin2='T2')
GRID = [round(v, 2) for v in np.arange(0.05, 1.0, 0.01)]
FOLDS, SEED = 5, 0
MIN_PHOTOS, MIN_SUPPORT, MIN_SHOWN, NODULE_MIN_GT = 10, 30, 1.0, 50
RISK = dict(box_conf=0.30, min_boxes=3, domain_max=None)  # domain_max: see the rule above
CLINICAL = ('acnedet_author', 'acne04')


def load(name):
    return [json.loads(line) for line in (CACHE / f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]


def fit_logistic(X, y):
    from sklearn.linear_model import LogisticRegression
    if len(set(y)) < 2 or sum(y) < 20:
        return None
    m = LogisticRegression(C=1.0, max_iter=1000).fit(np.asarray(X), np.asarray(y))
    return [float(m.intercept_[0])] + [float(v) for v in m.coef_[0]]


def training_rows(items, cache, g, policy):
    """-> per box above box_conf (raw order kept per photo): photo path, lineage group, features and labels."""
    rows = []
    for it in items:
        rec = cache[it['path']]
        boxes = [b for b in rec['boxes'] if b['score'] > RISK['box_conf']]
        res = match(it['boxes'], [dict(bbox=b['bbox'], score=b['score']) for b in boxes], 0.3)
        for b, (kind, j) in zip(boxes, res):
            f = group_features(b, policy, g)
            true = it['boxes'][j]['label'] if kind == 'tp' else None
            rows.append(dict(path=it['path'], group=it['lineage_group_id'], typed=it['typed'], f=f, ignored=kind == 'ignored',
                             yC=int(kind == 'tp'), yA=int(true == f['type']), yB=int(true is not None and GROUP.get(true) == f['group'])))
    return rows


def fit_cals(rows):
    use = [r for r in rows if not r['ignored']]
    typed = [r for r in use if r['typed']]
    return dict(C=fit_logistic([r['f']['C'] for r in use], [r['yC'] for r in use]),
                A=fit_logistic([r['f']['A'] for r in typed], [r['yA'] for r in typed]),
                B=fit_logistic([r['f']['B'] for r in typed], [r['yB'] for r in typed]))


def oof_q(rows):
    """Out-of-fold {A, B, C} per row, folds by lineage group."""
    groups = sorted({r['group'] for r in rows})
    rng = np.random.default_rng(SEED)
    fold = {g: i % FOLDS for i, g in enumerate(rng.permutation(groups))}
    q = [None] * len(rows)
    for k in range(FOLDS):
        cals = fit_cals([r for r in rows if fold[r['group']] != k])
        for i, r in enumerate(rows):
            if fold[r['group']] == k:
                q[i] = {t: calibrated(cals[t], r['f'][t]) for t in ('A', 'B', 'C')}
    return q


def preds_with(items, cache, policy, qmap):
    return {it['path']: dict(path=it['path'], boxes=apply_policy(cache[it['path']], policy, qmap.get(it['path']))['boxes'])
            for it in items}


def source_ok(rep, tier_metric, shown_key, targets, slack, sources, need_support):
    """-> (all sources meet target, worst point value, details)."""
    det, ok, worst = {}, True, 1.0
    for s in sources:
        e = rep['per_source'][s]
        v, ci = e['metrics'][tier_metric], e['ci95'].get(tier_metric)
        tgt = targets['T2' if s.endswith('_phone') else TARGET_SET[s]][tier_metric]
        n = e['metrics'][shown_key] if shown_key else None
        good = v is not None and v >= tgt and (ci is None or ci[0] >= tgt - slack) and (not need_support or (n or 0) >= MIN_SUPPORT)
        det[s] = dict(value=v, ci95=ci, target=tgt, shown=n, ok=good)
        ok &= good
        worst = min(worst, v if v is not None else 0.0)
    return ok, worst, det


def choose(items, cache, policy, risk, qmap, acc, curves):
    """Fill policy['groups'][risk] thresholds by the rule in the module docstring."""
    g = policy['groups'][risk]
    targets = acc['targets']
    slack = targets['T1']['ci_lower_slack']
    counts = Counter(it['source'] for it in items)
    sources = [s for s, n in counts.items() if n >= MIN_PHOTOS]
    typed_sources = [s for s in sources if any(it['typed'] for it in items if it['source'] == s)]
    log = dict(sources=dict(counts), scored_sources=sources, typed_sources=typed_sources)

    def run(resamples):
        return evaluate(items, preds_with(items, cache, policy, qmap), (0.3,), resamples)['by_iou']['0.3']

    def scan(tier, metric, shown_key, srcs, need_support):
        best = None
        for t in GRID:
            g[f'tau_{tier}'] = t
            rep = run(0)
            m = rep['overall']['metrics']
            ok, worst, det = source_ok(rep, metric, shown_key, targets, slack, srcs, need_support)
            curves.append(dict(risk=risk, tier=tier, tau=t, worst=worst, photos=m['photos'], shown=m['shown'],
                               per_source={s: dict(value=d['value'], shown=d['shown']) for s, d in det.items()},
                               coverage_A_or_B_on_found=m['coverage_A_or_B_on_found'], location_recall=m['location_recall']))
            if ok:
                ok, worst, det = source_ok(run(200), metric, shown_key, targets, slack, srcs, need_support)
                if ok:
                    return t, det, 'met'
            if tier == 'C' and m['shown'] / max(m['photos'], 1) >= MIN_SHOWN and (best is None or worst > best[1]):
                best = (t, worst, det)
        return (best[0], best[2], 'target_not_met') if best else (None, None, 'off')

    g['tau_A'] = g['tau_B'] = None
    g['tau_C'], log['C'], log['C_status'] = scan('C', 'location_precision', None, sources, False)
    if typed_sources and g['cal']['A'] is not None:
        t, d, st = scan('A', 'tier_A_precision', 'tier_A_shown', typed_sources, True)
        g['tau_A'], log['A'], log['A_status'] = (t, d, st) if st == 'met' else (None, d, 'off')
    if typed_sources and g['cal']['B'] is not None:
        t, d, st = scan('B', 'tier_B_precision', 'tier_B_shown', typed_sources, True)
        g['tau_B'], log['B'], log['B_status'] = (t, d, st) if st == 'met' else (None, d, 'off')
    return log


def main():
    import argparse
    import yaml
    global CACHE, POLICY, OUT, TAG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--phone-val', action='store_true', help='also fit and choose on the valP phone copies')
    args = ap.parse_args()
    TAG = args.tag
    CACHE = ROOT / 'runs/selfie_v1/raw_cache' / TAG
    POLICY = ROOT / f'models/candidates/selfie_v1/{TAG}_display_policy.json'
    OUT = ROOT / 'reports/selfie_v1/risk_coverage' / TAG
    meta = json.loads((CACHE / 'meta.json').read_text(encoding='utf-8'))
    acc = yaml.safe_load((ROOT / 'configs/selfie_v1/acceptance.yaml').read_text(encoding='utf-8'))
    lineage = {json.loads(line)['path']: json.loads(line) for line in (ROOT / 'datasets/selfie_v1/lineage.jsonl').read_text(encoding='utf-8').splitlines()}
    val_cache = {r['key']: r for r in load('val')}
    val = [it for it in eval_items('val') if lineage[it['path']]['own_locked_split'] != 'train']
    if args.phone_val:
        val_cache.update({r['key']: r for r in load('valP')})
        val += phone_items(val)
    if TAG == 'baseline':
        gate = json.loads((ROOT / 'models/phase2/input_gate.json').read_text(encoding='utf-8'))
        risk_cfg = dict(RISK, domain_max=gate['thresholds']['domain_max'])
    else:
        doms = [photo_risk(val_cache[it['path']], dict(RISK, domain_max=np.inf))[1] for it in val if it['source'] in CLINICAL]
        risk_cfg = dict(RISK, domain_max=float(np.percentile([d for d in doms if d is not None], 95)))
    policy = dict(version=f'selfie_v1-{TAG}-policy-1', detector_path=meta.get('detector', 'models/phase1/best_acne_detector.pt'),
                  classifier_path=meta.get('classifier', 'models/phase2/best_acne_type_robust.pth'),
                  domain_stats=meta.get('domain_stats', 'models/phase2/input_gate_features.npz'), detector_sha16=meta['versions']['detector'],
                  classifier_sha16=meta['versions']['type_model'], temperature=meta['temperature'], classes=meta['classes'],
                  risk=risk_cfg, groups={})
    if 'alt' in meta:
        policy.update(alt_classifier_path=meta['alt']['classifier'], alt_classifier_sha16=meta['alt']['sha16'])
    by_risk = defaultdict(list)
    for it in val:
        by_risk[photo_risk(val_cache[it['path']], risk_cfg)[0]].append(it)
    selection, curves = {}, []
    for risk in ('low', 'high'):
        items = by_risk[risk]
        g = dict(logits_key='logits_alt', temperature=meta['alt']['temperature']) if meta.get('group_classifier', {}).get(risk) == 'alt' else {}
        rows = training_rows(items, val_cache, g, policy)
        nodule_gt = sum(1 for it in items if it['typed'] for b in it['boxes'] if b['label'] == 'nodule')
        policy['groups'][risk] = dict(g, cal=fit_cals(rows), tau_A=None, tau_B=None, tau_C=None,
                                      no_type_A=['nodule'] if nodule_gt < NODULE_MIN_GT else [])
        q = oof_q(rows)
        qmap = defaultdict(list)
        for r, qq in zip(rows, q):
            qmap[r['path']].append(qq)
        selection[risk] = choose(items, val_cache, policy, risk, qmap, acc, curves)
        selection[risk].update(boxes=len(rows), nodule_gt=nodule_gt)
        g = policy['groups'][risk]
        if g['tau_A'] is not None and 'nodule' not in g['no_type_A']:
            rep = evaluate(items, preds_with(items, val_cache, policy, qmap), (0.3,), 0)['by_iou']['0.3']['overall']['metrics']
            n = rep['per_type']['nodule']['shown_A']
            selection[risk]['nodule_shown_A'] = n
            if n < MIN_SUPPORT:
                g['no_type_A'].append('nodule')
                selection[risk]['nodule_fallback'] = 'nodule_unmeasurable'
        print(risk, {k: selection[risk].get(f'{k}_status') for k in 'CAB'},
              {k: policy['groups'][risk][f'tau_{k}'] for k in 'CAB'}, flush=True)
    policy['selection'] = dict(rule=__doc__.split('Rule')[1].split('Writes')[0].strip(), per_risk=selection, phone_val=args.phone_val,
                               val_photos=len(val), val_sha256=digest(ROOT / 'datasets/selfie_v1/lineage.jsonl'),
                               acceptance_sha256=digest(ROOT / 'configs/selfie_v1/acceptance.yaml'))
    write_json(POLICY, policy)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'val_threshold_curves.json', curves)
    score(policy, acc, val, val_cache, lineage)


def run_set(items, cache, policy, resamples=1000, target=None):
    preds, risks = {}, Counter()
    for it in items:
        d = apply_policy(cache[it['path']], policy)
        preds[it['path']] = dict(path=it['path'], status='ok', warnings=d['risk_reasons'], boxes=d['boxes'])
        risks[(it['source'], d['risk'])] += 1
    rep = evaluate(items, preds, (0.3, 0.5), resamples, target=target)
    rep['risk_by_source'] = {f'{s}:{r}': n for (s, r), n in sorted(risks.items())}
    return rep, preds


def score(policy, acc, val, val_cache, lineage):
    res = dict(policy=str(POLICY.relative_to(ROOT)), policy_sha256=digest(POLICY))
    res['val_in_sample'], _ = run_set(val, val_cache, policy)
    t1_cache = {r['key']: r for r in load('T1')}
    t1 = eval_items('T1')
    res['T1'], _ = run_set(t1, t1_cache, policy, target=acc['targets']['T1'])
    t3_cache = {r['key']: r for r in load('T3')}
    res['T3'] = {}
    for v in VARIANTS:
        its = [dict(it, path=f"{it['path']}#{v}", boxes=degrade_boxes(it['boxes'], v, it['width'], it['height'])) for it in t1]
        res['T3'][v], _ = run_set(its, t3_cache, policy, resamples=200)
    soft, label = defaultdict(float), {}
    risks = Counter()
    for r in load('T4'):
        d = apply_policy(r, policy)
        risks[d['risk']] += 1
        label[r['case_id']] = r['acne']
        soft[r['case_id']] = max(soft[r['case_id']], sum(b['q_lesion'] for b in d['boxes']))
    cases = sorted(label)
    res['T4'] = dict(case_auc([soft[c] for c in cases], [label[c] for c in cases]), photos_by_risk=dict(risks),
                     score='max over photos of the summed qC of the shown boxes')
    selfies = [r for r in load('check') if r['key'].startswith('test_img')]
    res['selfies_test_img'] = []
    for r in selfies:
        d = apply_policy(r, policy)
        res['selfies_test_img'].append(dict(photo=r['key'], risk=d['risk'], reasons=d['risk_reasons'],
                                             shown=dict(Counter(b['tier'] for b in d['boxes']))))
    write_json(OUT / 'results.json', res)
    write_summary(policy, res)


def fmt(v, pct=True):
    return '–' if v is None else f'{v * 100:.1f}' if pct else f'{v:.3f}'


def ci(e, k):
    c = e['ci95'].get(k)
    return fmt(e['metrics'][k]) + (f' [{fmt(c[0])}–{fmt(c[1])}]' if c else '')


HEAD = ('| Tập | Ảnh | P vị trí | R vị trí | FP/ảnh | P tầng A (n) | P tầng B (n) | Coverage A+B | Box hiện/ảnh (A/B/C) | ECE |\n'
        '| --- | ---: | --- | --- | ---: | --- | --- | ---: | --- | ---: |')


def row(label, e):
    m = e['metrics']
    n = m['photos'] or 1
    tiers = m['shown_by_tier']
    shown = (m['shown']) / n
    return (f"| {label} | {m['photos']} | {ci(e, 'location_precision')} | {ci(e, 'location_recall')} | {fmt(m['fp_per_photo'], False)} | "
            f"{ci(e, 'tier_A_precision')} ({m['tier_A_shown']}) | {ci(e, 'tier_B_precision')} ({m['tier_B_shown']}) | "
            f"{fmt(m['coverage_A_or_B_on_found'])} | {shown:.1f} (đúng chỗ: {tiers['A']}/{tiers['B']}/{tiers['C']}) | {fmt(m['ece_displayed'], False)} |")


def write_summary(policy, res):
    L = ['# P08a — Hiển thị theo tầng cho checkpoint hiện tại' if TAG == 'baseline' else f'# P08b — Hiển thị theo tầng cho candidate `{TAG}`', '',
         f"Chính sách: `{res['policy']}` (sha256 `{res['policy_sha256'][:16]}…`). Ngưỡng chọn trên val (out-of-fold), "
         'T1/T3/T4 chấm một lần sau khi chốt. IoU 0,3; CI 95% bootstrap theo lineage group.', '', '## Ngưỡng', '']
    for risk, g in policy['groups'].items():
        s = policy['selection']['per_risk'][risk]
        L.append(f"- **Rủi ro {risk}**: τ_C {g['tau_C']} ({s.get('C_status')}), τ_A {g['tau_A']} ({s.get('A_status', 'off')}), "
                 f"τ_B {g['tau_B']} ({s.get('B_status', 'off')}); không hiện loại A cho {g['no_type_A'] or 'không'}; "
                 f"ảnh val: {s['sources']}")
    for name, key in (('T1 (nhãn bác sĩ)', 'T1'), ('val (trong mẫu, chỉ tham khảo)', 'val_in_sample')):
        e = res[key]['by_iou']['0.3']
        L += ['', f'## {name}', '', f"Nhóm rủi ro: {res[key]['risk_by_source']}", '', HEAD, row('tổng', e['overall'])]
        L += [row(s, v) for s, v in sorted(e['per_source'].items())]
        if 'acceptance' in e:
            L += ['', f"Nghiệm thu (acceptance.yaml T1): {e['acceptance']}"]
    t1A = res['T1']['by_iou']['0.3']['overall']['metrics']['tier_A_precision']
    t1B = res['T1']['by_iou']['0.3']['overall']['metrics']['tier_B_precision']
    L += ['', '## T3 (ảnh T1 mô phỏng điện thoại)', '', HEAD]
    for v, r in res['T3'].items():
        L.append(row(v, r['by_iou']['0.3']['overall']))
    L += ['', 'Mức giảm so với T1 sạch (mục tiêu ≤ 5 điểm, chỉ tính khi tầng còn hiển thị): ' + ', '.join(
        f"{v}: A {fmt(None if t1A is None or r['by_iou']['0.3']['overall']['metrics']['tier_A_precision'] is None else t1A - r['by_iou']['0.3']['overall']['metrics']['tier_A_precision'])}"
        f" / B {fmt(None if t1B is None or r['by_iou']['0.3']['overall']['metrics']['tier_B_precision'] is None else t1B - r['by_iou']['0.3']['overall']['metrics']['tier_B_precision'])}"
        for v, r in res['T3'].items())]
    t4 = res['T4']
    L += ['', f"## T4 SCIN đầu/cổ", '', f"AUC {fmt(t4['auc'], False)} CI {t4.get('ci95')} ({t4['positives']} ca mụn / {t4['negatives']} ca không mụn); "
          f"ảnh theo nhóm rủi ro {t4['photos_by_risk']}.", '', '## 21 selfie `test_img` (không có nhãn)', '']
    L += [f"- {s['photo']}: rủi ro {s['risk']} {s['reasons']}, box hiện {s['shown']}" for s in res['selfies_test_img']]
    (OUT / 'summary.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'-> {OUT / "summary.md"}')


if __name__ == '__main__':
    main()
