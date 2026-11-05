"""P11: the one final evaluation of the served bundle (configs/selfie_v1/serving.json) on T1, T2, T3, T4 and T5.

--run (GPU): the real TieredPipeline, batch 1, on every photo of T1, T2 (first and only use), T3 (T1 x
t3_degrade.VARIANTS), T4 (SCIN head/neck, case labels) and T5 (test_img selfies + T5_SCIN SCIN photos with a face
found, seeded); raw records + seconds per photo + errors -> runs/selfie_v1/final/<set>.jsonl (resumable). The policy
sha256 is stored in meta.json; a different served policy refuses to append.
--score (no GPU): acceptance.yaml verdicts per item (pass / fail / inconclusive / not_shown), the pre-registered
fallbacks when tier A or B fails on T2 (never a retune), T5 from the owner's answers (reviews/t5_boxes.jsonl),
service checks -> reports/selfie_v1/final_evaluation/{results.json, summary.md} and release_decision.json.
Usage: venv/Scripts/python.exe -u scripts/evaluation_v2/final_evaluation.py --run
       venv/Scripts/python.exe scripts/evaluation_v2/final_evaluation.py --score
"""
import argparse
import copy
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import digest, write_json  # noqa: E402
from display_policy import apply_policy  # noqa: E402
from evaluate_pipeline import case_auc, eval_items, evaluate, verdict  # noqa: E402
from t3_degrade import VARIANTS, degrade, degrade_boxes  # noqa: E402

RUN = ROOT / 'runs/selfie_v1/final'
OUT = ROOT / 'reports/selfie_v1/final_evaluation'
SERVING = ROOT / 'configs/selfie_v1/serving.json'
ACCEPT = ROOT / 'configs/selfie_v1/acceptance.yaml'
T5_ANSWERS = ROOT / 'datasets/selfie_v1/reviews/t5_boxes.jsonl'
T5_SCIN, SEED = 80, 0
DISPLAY_ITEMS = [f'{t}.{k}' for t in ('T1', 'T2') for k in ('tier_A_precision', 'tier_B_precision', 'location_precision')]
BASELINE_T1_MACRO_F1 = ROOT / 'reports/selfie_v1/baseline/baseline_results.json'


def policy_of_serving():
    cfg = json.loads(SERVING.read_text(encoding='utf-8'))
    return ROOT / cfg['policy']


def t5_items(t4_records):
    """T5: every test_img selfie + T5_SCIN SCIN photos (from the T4 run) in which the face detector found a face."""
    selfies = [dict(key=str(p.relative_to(ROOT)).replace('\\', '/')) for p in sorted((ROOT / 'test_img').glob('*.jpg'))]
    faces = sorted(r['key'] for r in t4_records if r.get('input_check', {}).get('metrics', {}).get('face'))
    pick = np.random.default_rng(SEED).permutation(len(faces))[:T5_SCIN]
    return selfies + [dict(key=faces[i]) for i in sorted(pick)]


def items_of(name, t4_records=None):
    from run_raw_cache import scin_items
    if name in ('T1', 'T2'):
        return [dict(key=it['path']) for it in eval_items(name)]
    if name == 'T3':
        return [dict(key=f"{it['path']}#{v}", path=it['path'], variant=v, boxes=it['boxes']) for it in eval_items('T1') for v in VARIANTS]
    if name == 'T4':
        return scin_items()
    if name == 'T5':
        return t5_items(t4_records)
    raise ValueError(name)


def load_image(item):
    img = cv2.imread(str(ROOT / item.get('path', item['key'])))
    if img is not None and 'variant' in item:
        img, _ = degrade(img, item['boxes'], item['variant'], item['path'])
    return img


def read(name):
    f = RUN / f'{name}.jsonl'
    return [json.loads(line) for line in f.read_text(encoding='utf-8').splitlines()] if f.exists() else []


def run():
    from tiered_pipeline import TieredPipeline, present, raw_analyze
    pipe = TieredPipeline(SERVING)
    RUN.mkdir(parents=True, exist_ok=True)
    meta = dict(policy=str(pipe.policy_path.relative_to(ROOT)), policy_sha256=digest(pipe.policy_path), versions=pipe.versions())
    old = RUN / 'meta.json'
    if old.exists() and json.loads(old.read_text(encoding='utf-8'))['policy_sha256'] != meta['policy_sha256']:
        raise SystemExit(f'{RUN} was made with another served policy; move it away first')
    old.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    pipe.analyze(np.zeros((64, 64, 3), np.uint8))  # warm-up, not timed
    for name in ('T1', 'T2', 'T3', 'T4', 'T5'):
        items = items_of(name, read('T4') if name == 'T5' else None)
        done = {r['key'] for r in read(name)}
        todo = [it for it in items if it['key'] not in done]
        print(f'{name}: {len(items)} photos, {len(todo)} to run', flush=True)
        with (RUN / f'{name}.jsonl').open('a', encoding='utf-8') as out:
            for k, it in enumerate(todo):
                rec = dict(key=it['key'], set=name, **{x: it[x] for x in ('case_id', 'acne', 'shot') if x in it})
                img = load_image(it)
                if img is None:
                    rec.update(error='unreadable')
                else:
                    t = time.perf_counter()
                    try:
                        raw = raw_analyze(pipe, img)
                        present(raw, pipe.policy, pipe.published)
                        rec.update(raw, seconds=round(time.perf_counter() - t, 4))
                    except Exception as e:  # counted as a processing error, never hidden
                        rec.update(error=f'{type(e).__name__}: {e}')
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
                if (k + 1) % 50 == 0:
                    print(f'  {name} {k + 1}/{len(todo)}', flush=True)
    print('done', flush=True)


def scored(items, cache, policy, resamples, target=None):
    preds = {it['path']: dict(path=it['path'], status='ok', boxes=apply_policy(cache[it['path']], policy)['boxes']) for it in items}
    rep = evaluate(items, preds, (0.3, 0.5), resamples, target=target)
    rep['risk'] = dict(Counter(apply_policy(cache[it['path']], policy)['risk'] for it in items))
    return rep


def with_fallback(policy, t2_acceptance, fb):
    """acceptance.yaml fallbacks after a T2 failure -> (policy actually released, list of fallbacks applied)."""
    p, applied = copy.deepcopy(policy), []
    if t2_acceptance.get('tier_A_precision') == 'fail':
        p['groups']['high']['tau_A'] = None
        applied.append(dict(item='tier_A_precision', rule=fb['tier_A_fails_T2']))
    if t2_acceptance.get('tier_B_precision') == 'fail':
        for g in p['groups'].values():
            g['tau_A'] = g['tau_B'] = None
        applied.append(dict(item='tier_B_precision', rule=fb['tier_B_fails_T2']))
    return p, applied


def t5_result(policy, records, target):
    if not T5_ANSWERS.exists():
        return dict(status='inconclusive', reason='owner review not done')
    latest = {}
    for line in T5_ANSWERS.read_text(encoding='utf-8').splitlines():
        a = json.loads(line)
        latest[a['id']] = a['answer']
    task = json.loads((ROOT / 'reports/selfie_v1/review/t5_boxes/items.json').read_text(encoding='utf-8'))
    ids = [it['id'] for it in task['items']]
    answered = [latest[i] for i in ids if i in latest]
    c = Counter(answered)
    share = c['no'] / len(answered) if answered else None
    errors = sum(1 for r in records if 'error' in r)
    status = 'inconclusive' if len(answered) < len(ids) else (
        'pass' if share <= target['max_share_boxes_clearly_off_skin_or_spot'] and errors <= target['processing_errors'] else 'fail')
    return dict(status=status, boxes_in_task=len(ids), answered=len(answered), answers=dict(c), share_clearly_off=share,
                processing_errors=errors, photos=len(records), boxes_shown_total=task.get('boxes_shown_total'),
                sampled=task.get('sampled'))


def score():
    acc = yaml.safe_load(ACCEPT.read_text(encoding='utf-8'))
    meta = json.loads((RUN / 'meta.json').read_text(encoding='utf-8'))
    policy_path = ROOT / meta['policy']
    if digest(policy_path) != meta['policy_sha256']:
        raise SystemExit('served policy changed since the run')
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    R = acc['bootstrap']['resamples']
    cache = {s: {r['key']: r for r in read(s)} for s in ('T1', 'T2', 'T3', 'T4', 'T5')}
    res = dict(utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), policy=meta['policy'], policy_sha256=meta['policy_sha256'],
               versions=meta['versions'], acceptance_sha256=digest(ACCEPT))
    errors = {s: [k for k, r in c.items() if 'error' in r] for s, c in cache.items()}
    t1, t2 = eval_items('T1'), eval_items('T2')
    res['T1'] = scored(t1, cache['T1'], policy, R, acc['targets']['T1'])
    res['T2'] = scored(t2, cache['T2'], policy, R, acc['targets']['T2'])
    t2_acc = res['T2']['by_iou']['0.3']['acceptance']
    released, applied = with_fallback(policy, t2_acc, acc['fallback'])
    res['fallbacks_applied'] = applied
    if applied:  # the released configuration, scored on the same photos (reported, not used to choose anything)
        res['T1_released'] = scored(t1, cache['T1'], released, R, acc['targets']['T1'])
        res['T2_released'] = scored(t2, cache['T2'], released, R, acc['targets']['T2'])
    rel_t1 = res.get('T1_released', res['T1'])['by_iou']['0.3']['overall']
    rel_t2 = res.get('T2_released', res['T2'])['by_iou']['0.3']['overall']
    base = json.loads(BASELINE_T1_MACRO_F1.read_text(encoding='utf-8'))['normal']['T1']['by_iou']['0.3']['overall']['metrics']['macro_f1_box_type']
    f1 = rel_t1['metrics']['macro_f1_box_type']
    res['regression'] = dict(baseline_T1_macro_f1=base, T1_macro_f1=f1, max_drop=acc['targets']['T1']['macro_f1_regression_max_drop'],
                             status='inconclusive' if f1 is None else 'pass' if base - f1 <= acc['targets']['T1']['macro_f1_regression_max_drop'] else 'fail')
    res['T3'] = {}
    t1A, t1B = (rel_t1['metrics'][k] for k in ('tier_A_precision', 'tier_B_precision'))
    for v in VARIANTS:
        its = [dict(it, path=f"{it['path']}#{v}", boxes=degrade_boxes(it['boxes'], v, it['width'], it['height'])) for it in t1]
        rep = scored(its, cache['T3'], released, 200)
        m = rep['by_iou']['0.3']['overall']['metrics']
        drops = {}
        for tier, clean in (('A', t1A), ('B', t1B)):
            d = m[f'tier_{tier}_precision']
            drops[tier] = None if clean is None or d is None else clean - d
        st = 'not_shown' if all(x is None for x in drops.values()) else (
            'pass' if all(x is None or x <= acc['targets']['T3']['max_tier_precision_drop'] for x in drops.values()) else 'fail')
        res['T3'][v] = dict(metrics={k: m[k] for k in ('tier_A_precision', 'tier_A_shown', 'tier_B_precision', 'tier_B_shown',
                                                        'location_precision', 'location_recall', 'coverage_A_or_B_on_found')},
                            ci95=rep['by_iou']['0.3']['overall']['ci95'], drop=drops, status=st, risk=rep['risk'])
    soft, label, risks = defaultdict(float), {}, Counter()
    for r in read('T4'):  # one line per (case, image): an image shared by two cases counts for both
        if 'error' in r:
            continue
        d = apply_policy(r, released)
        risks[d['risk']] += 1
        label[r['case_id']] = r['acne']
        soft[r['case_id']] = max(soft[r['case_id']], sum(b['q_lesion'] for b in d['boxes']))
    cases = sorted(label)
    t4 = case_auc([soft[c] for c in cases], [label[c] for c in cases], resamples=R, min_per_side=acc['targets']['T4']['min_cases_per_side'])
    t4['acceptance'] = t4['status'] if t4['status'] == 'inconclusive' else 'pass' if t4['auc'] >= acc['targets']['T4']['auc_min'] else 'fail'
    res['T4'] = dict(t4, photos_by_risk=dict(risks), score='max over photos of the summed qC of the shown boxes')
    res['T5'] = t5_result(released, list(cache['T5'].values()), acc['targets']['T5'])
    secs = [r['seconds'] for s in ('T1', 'T2', 'T4', 'T5') for r in cache[s].values() if 'seconds' in r]
    decodable = sum(1 for s in cache.values() for r in s.values() if r.get('error') != 'unreadable')
    with_result = sum(1 for s in cache.values() for r in s.values() if 'error' not in r)
    p95 = float(np.percentile(secs, 95))
    res['service'] = dict(photos=sum(len(s) for s in cache.values()), decodable=decodable, with_result=with_result,
                          errors={s: e[:20] for s, e in errors.items() if e},
                          photos_with_result=with_result / decodable if decodable else None,
                          latency_seconds=dict(n=len(secs), p50=float(np.percentile(secs, 50)), p95=p95, max=float(max(secs))),
                          status=dict(photos_with_result='pass' if with_result == decodable else 'fail',
                                      latency_p95='pass' if p95 <= acc['targets']['service']['latency_p95_seconds'] else 'fail'))
    published = dict(clinical_type_precision=rel_t1['metrics']['tier_A_precision'], phone_type_precision=rel_t2['metrics']['tier_A_precision'],
                     source='reports/selfie_v1/final_evaluation/results.json (P11, IoU 0.3, T1 / T2 overall tier A precision of the released policy)')
    res['published'] = published
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'results.json', res)
    write_json(OUT / 'released_policy.json', released)
    decision = dict(utc=res['utc'], policy=meta['policy'], policy_sha256=meta['policy_sha256'],
                    released_policy='reports/selfie_v1/final_evaluation/released_policy.json', released_policy_sha256=digest(OUT / 'released_policy.json'),
                    fallbacks_applied=applied, versions=meta['versions'], published=published,
                    verdicts=dict(T1=res.get('T1_released', res['T1'])['by_iou']['0.3']['acceptance'],
                                  T2=rel_t2_acc(res), T3={v: x['status'] for v, x in res['T3'].items()}, T4=t4['acceptance'],
                                  T5=res['T5']['status'], regression=res['regression']['status'], service=res['service']['status']),
                    rollback=dict(how="configs/selfie_v1/serving.json: set 'policy' to models/candidates/selfie_v1/baseline_display_policy.json "
                                       "and restart api/server.py; or run api/server.py --engine legacy [--demo]",
                                  baseline_policy_sha256=digest(ROOT / 'models/candidates/selfie_v1/baseline_display_policy.json')))
    flat = {f'{k}.{i}': x for k, v in decision['verdicts'].items() for i, x in (v.items() if isinstance(v, dict) else [('', v)])}
    shown_ok = all(flat.get(k) in ('pass', 'not_shown') for k in DISPLAY_ITEMS) and flat['T5.'] == 'pass' and         all(x == 'pass' for k, x in flat.items() if k.startswith('service.'))
    decision.update(decision='release_for_internal_demo' if shown_ok else 'hold',
                    rule='release for the internal localhost demo when every displayed-precision item of T1/T2 passes (or the '
                         'tier is not shown), T5 passes and the service checks pass; every other item is reported as a limitation',
                    unmet=sorted(k.rstrip('.') for k, x in flat.items() if x == 'fail'),
                    inconclusive=sorted(k.rstrip('.') for k, x in flat.items() if x == 'inconclusive'))
    write_json(OUT / 'release_decision.json', decision)
    write_summary(res, decision)
    print(json.dumps(decision['verdicts'], ensure_ascii=False, indent=1))
    print('published', published, 'fallbacks', applied)


def pct(v):
    return '–' if v is None else f'{v * 100:.1f}%'


def num(v, digits, scale=1):
    return '–' if v is None else f'{v * scale:.{digits}f}'


def ci(e, k):
    c = e['ci95'].get(k)
    return pct(e['metrics'][k]) + (f' [{pct(c[0])}–{pct(c[1])}]' if c else '')


def role_table(rep):
    L = ['| IoU | P tầng A (n) | P tầng B (n) | P vị trí | R vị trí | FP/ảnh | Coverage A+B | ECE |', '| --- | --- | --- | --- | --- | ---: | --- | ---: |']
    for iou in ('0.3', '0.5'):
        e = rep['by_iou'][iou]['overall']
        m = e['metrics']
        L.append(f"| {iou} | {ci(e, 'tier_A_precision')} ({m['tier_A_shown']}) | {ci(e, 'tier_B_precision')} ({m['tier_B_shown']}) | "
                 f"{ci(e, 'location_precision')} | {ci(e, 'location_recall')} | {m['fp_per_photo']:.2f} | {ci(e, 'coverage_A_or_B_on_found')} | "
                 f"{num(m['ece_displayed'], 3)} |")
    m = rep['by_iou']['0.3']['overall']['metrics']
    L += ['', 'Tầng A theo loại (IoU 0,3): ' + ', '.join(f"{t} {v['shown_A']} box / {pct(v['precision_A'])} (GT {v['gt']})" for t, v in m['per_type'].items())]
    L += [f"Nhóm rủi ro: {rep['risk']}. Nghiệm thu: {rep['by_iou']['0.3'].get('acceptance')}"]
    return L


def write_summary(res, decision):
    v = decision['verdicts']
    L = ['# P11 — Đánh giá cuối', '', f"Chạy {res['utc']}. Chính sách phục vụ `{res['policy']}` (sha256 `{res['policy_sha256'][:16]}…`), "
         f"phiên bản {res['versions']}. `acceptance.yaml` sha256 `{res['acceptance_sha256'][:16]}…`. Mỗi tập chạy một lần bằng pipeline thật "
         '(GPU, batch 1); ngưỡng không đổi sau khi xem kết quả.', '',
         '## Kết luận theo `acceptance.yaml`', '', '| Mục | Kết quả |', '| --- | --- |',
         f"| T1 (ACNE-DET, nhãn bác sĩ) | {v['T1']} |", f"| T2 (DDS2 proxy, nhãn chưa xác minh) | {v['T2']} |",
         f"| T3 (T1 giả lập điện thoại, giảm ≤ 5 điểm) | {v['T3']} |", f"| T4 (SCIN mức ca, AUC ≥ 0,70) | {v['T4']} |",
         f"| T5 (chủ dự án duyệt, ≤ 10% box rõ ràng sai chỗ) | {v['T5']} |", f"| Regression macro-F1 T1 (giảm ≤ 0,03) | {v['regression']} |",
         f"| Dịch vụ (100% ảnh có kết quả, p95 ≤ 5 s) | {v['service']} |", '',
         f"Dự phòng đã áp dụng: {res['fallbacks_applied'] or 'không'}.", '',
         '## T1', ''] + role_table(res['T1']) + ['', '## T2 (lần dùng đầu tiên và duy nhất)', ''] + role_table(res['T2'])
    for k, name in (('T1_released', 'T1 sau dự phòng'), ('T2_released', 'T2 sau dự phòng')):
        if k in res:
            L += ['', f'## {name}', ''] + role_table(res[k])
    r = res['regression']
    L += ['', f"Regression: macro-F1 box+loại T1 {r['T1_macro_f1']:.3f} so với baseline P05 {r['baseline_T1_macro_f1']:.3f} → {r['status']}.",
          '', '## T3', '', '| Biến thể | P tầng A (n) | P tầng B (n) | Giảm A / B (điểm) | R vị trí | Kết quả |', '| --- | --- | --- | --- | --- | --- |']
    for name, x in res['T3'].items():
        m, d = x['metrics'], x['drop']
        L.append(f"| {name} | {pct(m['tier_A_precision'])} ({m['tier_A_shown']}) | {pct(m['tier_B_precision'])} ({m['tier_B_shown']}) | "
                 f"{num(d['A'], 1, 100)} / {num(d['B'], 1, 100)} | "
                 f"{pct(m['location_recall'])} | {x['status']} |")
    t4 = res['T4']
    L += ['', '## T4', '', f"AUC {t4['auc']:.3f} CI {[round(c, 3) for c in t4.get('ci95', [])]} ({t4['positives']} ca mụn / {t4['negatives']} ca không mụn) → "
          f"{t4['acceptance']}. Ảnh theo nhóm rủi ro: {t4['photos_by_risk']}.", '', '## T5', '', f"{res['T5']}", '', '## Dịch vụ', '',
          f"{res['service']}", '', '## Số được công bố (mục 6.3)', '', f"{res['published']}"]
    (OUT / 'summary.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def rel_t2_acc(res):
    return res.get('T2_released', res['T2'])['by_iou']['0.3']['acceptance']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--score', action='store_true')
    a = ap.parse_args()
    if a.run:
        run()
    if a.score:
        score()


if __name__ == '__main__':
    main()
