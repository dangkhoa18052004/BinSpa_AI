"""P05: baseline (deployed checkpoint) in normal and demo mode, scored with evaluate_pipeline.py.

Both modes are recomputed from the raw cache (run_raw_cache.py), with the rules of AcnePipeline.analyze:
  normal: a failed input check -> 'retake_photo', nothing shown; boxes with detector score > operating conf (0.48);
          photo domain score = median crop distance of those boxes (needs >= 3, else 'unverified'); unverified or
          above domain_max -> 'location_only' (every type withheld); otherwise a type is shown when the calibrated
          top-1 probability (rounded to 4 decimals, as deployed) >= tau 0.87, else location only.
  demo:   never blocks; every box above DEMO_CONF 0.30 is shown with its top-1 type.
Tiers for the evaluator: a shown type = tier A (prob = calibrated top-1), a withheld type = tier C.
The `check` set compares the recomputation with the stored output of the deployed code.
Scored sets: T1 (target T1), val (minus photos the current model was trained on: own_locked_split 'train'),
T3 (each variant vs clean T1), T4 (SCIN case AUC). T2 is not touched.
Writes reports/selfie_v1/baseline/eval/*.json, summary.md, and runs/selfie_v1/preds/baseline/*.jsonl.
Usage: venv/Scripts/python.exe scripts/evaluation_v2/baseline_eval.py [--check-only]
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
from evaluate_pipeline import GROUP, case_auc, eval_items, evaluate  # noqa: E402
from t3_degrade import VARIANTS, degrade_boxes  # noqa: E402

CACHE = ROOT / 'runs/selfie_v1/raw_cache/baseline'
PREDS = ROOT / 'runs/selfie_v1/preds/baseline'
OUT = ROOT / 'reports/selfie_v1/baseline'
MODES = ('normal', 'demo')


def softmax(z, T):
    z = np.asarray(z, float) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def display(rec, mode, meta):
    """Raw cache record -> {status, boxes in evaluator format (+ type_status)}."""
    classes, tau = meta['classes'], meta['tau']
    fails = rec['input_check']['fails']
    if mode == 'normal' and fails:
        return dict(status='retake_photo', boxes=[], warnings=fails)
    conf = meta['normal_conf'] if mode == 'normal' else meta['demo_conf']
    keep = [b for b in rec['boxes'] if b['score'] > conf]
    status = 'ok' if mode == 'normal' else 'demo'
    if mode == 'normal' and keep:
        d = float(np.median([b['crop_domain'] for b in keep])) if len(keep) >= meta['min_lesions_domain'] else None
        if d is None or d > meta['gate']['domain_max']:
            status = 'location_only'
    boxes = []
    for b in keep:
        p = softmax(b['logits'], meta['temperature'])
        k = int(p.argmax())
        prob = round(float(p[k]), 4)
        confident = prob >= tau and status != 'location_only'
        shown = mode == 'demo' or confident
        boxes.append(dict(bbox=b['bbox'], score=b['score'], tier='A' if shown else 'C',
                          type=classes[k] if shown else None, group=GROUP[classes[k]] if shown else None,
                          prob=prob if shown else None, type_status='confident' if confident else 'uncertain'))
    return dict(status=status, boxes=boxes, warnings=fails)


def compare_with_deployed(recs, meta):
    """-> list of differences between the recomputed and the deployed output (empty = identical)."""
    diffs = []
    for r in recs:
        for mode in MODES:
            dep, sim = r['deployed'][mode], display(r, mode, meta)
            dep_status = dep['analysis_status']
            if dep_status != sim['status']:
                diffs.append(f"{r['key']} {mode}: status {dep_status} vs {sim['status']}")
                continue
            dp = dep['predictions']
            if len(dp) != len(sim['boxes']):
                diffs.append(f"{r['key']} {mode}: {len(dp)} vs {len(sim['boxes'])} boxes")
                continue
            for x, y in zip(dp, sim['boxes']):
                if max(abs(u - v) for u, v in zip(x['bbox'], y['bbox'])) > 0.06:
                    diffs.append(f"{r['key']} {mode}: bbox {x['bbox']} vs {y['bbox']}")
                if x.get('type_status', 'confident') != y['type_status'] or \
                        (mode == 'demo' and x['type'] != y['type']) or abs(x['top3'][0]['probability'] - (y['prob'] or x['top3'][0]['probability'])) > 2e-4:
                    diffs.append(f"{r['key']} {mode}: type {x.get('type_status')}/{x['top3'][0]} vs {y['type_status']}/{y['type']} {y['prob']}")
    return diffs


def load_cache(name):
    f = CACHE / f'{name}.jsonl'
    return {json.loads(line)['key']: json.loads(line) for line in f.read_text(encoding='utf-8').splitlines()} if f.exists() else {}


def preds_of(cache, mode, meta):
    out = {}
    for key, rec in cache.items():
        d = display(rec, mode, meta)
        out[key] = dict(path=key, status=d['status'], warnings=d['warnings'],
                        boxes=[{k: v for k, v in b.items() if k != 'type_status'} for b in d['boxes']])
    return out


def save_preds(name, mode, preds):
    PREDS.mkdir(parents=True, exist_ok=True)
    with (PREDS / f'{name}_{mode}.jsonl').open('w', encoding='utf-8') as f:
        for r in preds.values():
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def status_table(items, cache, mode, meta):
    """-> {source: Counter(status and failed checks)}."""
    t = defaultdict(Counter)
    for it in items:
        rec = cache[it['path']]
        d = display(rec, mode, meta)
        t[it['source']][d['status']] += 1
        for f in rec['input_check']['fails']:
            t[it['source']][f'check:{f}'] += 1
    return {k: dict(v) for k, v in t.items()}


def t4(recs, mode, meta):
    """recs: every T4 cache line (one SCIN photo can belong to several cases, so not keyed by path).
    Case score = max over the case's photos of the summed detector score of the shown boxes (soft lesion count);
    secondary: max shown-box count."""
    soft, count, label = defaultdict(float), defaultdict(int), {}
    for rec in recs:
        d = display(rec, mode, meta)
        cid = rec['case_id']
        label[cid] = rec['acne']
        soft[cid] = max(soft[cid], sum(b['score'] for b in d['boxes']))
        count[cid] = max(count[cid], len(d['boxes']))
    cases = sorted(label)
    y = [label[c] for c in cases]
    return dict(soft_count=case_auc([soft[c] for c in cases], y), box_count=case_auc([count[c] for c in cases], y),
                photos=len(recs), photos_with_failed_check=sum(bool(r['input_check']['fails']) for r in recs),
                failed_checks=dict(Counter(f for r in recs for f in r['input_check']['fails'])))


def fmt(v, pct=True):
    return '–' if v is None else f'{v * 100:.1f}' if pct else f'{v:.2f}'


def ci_txt(e, name):
    m, ci = e['metrics'][name], e.get('ci95', {}).get(name)
    return fmt(m) + (f' [{fmt(ci[0])}–{fmt(ci[1])}]' if ci else '')


def split_A(m):
    """Tier A errors split: share of A boxes on no lesion GT, type accuracy of A boxes that hit a lesion."""
    c = m['confusion_A']
    tot = sum(c.values())
    fp = sum(v for k, v in c.items() if k.startswith('none>'))
    ok = sum(v for k, v in c.items() if k.split('>')[0] == k.split('>')[1])
    return (fp / tot if tot else None), (ok / (tot - fp) if tot - fp else None)


def row(label, e):
    m = e['metrics']
    off, acc = split_A(m)
    return (f"| {label} | {m['photos']} | {ci_txt(e, 'location_precision')} | {ci_txt(e, 'location_recall')} | "
            f"{fmt(m['fp_per_photo'], False)} | {ci_txt(e, 'tier_A_precision')} (n {m['tier_A_shown']}) | "
            f"{fmt(off)} | {fmt(acc)} | {fmt(m['coverage_A_on_found'])} | {fmt(m['ece_displayed'], False)} | "
            f"{fmt(m['macro_f1_box_type'], False)} |")


HEAD = ('| Tập | Ảnh | Precision vị trí | Recall vị trí | FP/ảnh | Precision tầng A (số box) | Coverage A | ECE | Macro-F1 |\n'
        '| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: |')


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--resamples', type=int, default=1000)
    a = p.parse_args()
    meta = json.loads((CACHE / 'meta.json').read_text(encoding='utf-8'))
    acc = yaml.safe_load((ROOT / 'configs/selfie_v1/acceptance.yaml').read_text(encoding='utf-8'))
    thr = (acc['matching']['primary_iou'], acc['matching']['secondary_iou'])
    diffs = compare_with_deployed(list(load_cache('check').values()), meta)
    print(f'check: {len(diffs)} differences from the deployed code')
    for d in diffs[:20]:
        print('  ', d)
    if a.check_only:
        return
    if diffs:
        raise SystemExit('recomputation differs from the deployed code; fix before scoring')
    lineage = {json.loads(line)['path']: json.loads(line) for line in (ROOT / 'datasets/selfie_v1/lineage.jsonl').read_text(encoding='utf-8').splitlines()}
    results = dict(versions=meta['versions'], check_differences=0)
    caches = {n: load_cache(n) for n in ('T1', 'val', 'T3', 'T4')}
    items = dict(T1=eval_items('T1'), val=[it for it in eval_items('val') if lineage[it['path']]['own_locked_split'] != 'train'])
    results['val_excluded_seen_in_training'] = len(eval_items('val')) - len(items['val'])
    t3_items = {v: [] for v in VARIANTS}
    for it in items['T1']:
        for v in VARIANTS:
            t3_items[v].append(dict(it, path=f"{it['path']}#{v}", boxes=degrade_boxes(it['boxes'], v, it['width'], it['height'])))
    (OUT / 'eval').mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        res = results.setdefault(mode, {})
        for name in ('T1', 'val'):
            preds = preds_of(caches[name], mode, meta)
            save_preds(name, mode, preds)
            r = evaluate(items[name], preds, thr, a.resamples, target=acc['targets'].get(name))
            r['status_by_source'] = status_table(items[name], caches[name], mode, meta)
            write_json(OUT / 'eval' / f'{name}_{mode}.json', r)
            res[name] = r
        preds = preds_of(caches['T3'], mode, meta)
        save_preds('T3', mode, preds)
        res['T3'] = {}
        for v, its in t3_items.items():
            r = evaluate(its, preds, thr, a.resamples)
            write_json(OUT / 'eval' / f'T3_{v}_{mode}.json', r)
            res['T3'][v] = r
        res['T4'] = t4([json.loads(line) for line in (CACHE / 'T4.jsonl').read_text(encoding='utf-8').splitlines()], mode, meta)
        write_json(OUT / 'eval' / f'T4_{mode}.json', res['T4'])
    secs = [r['seconds'] for c in caches.values() for r in c.values()]
    results['latency_seconds'] = dict(p50=float(np.percentile(secs, 50)), p95=float(np.percentile(secs, 95)), photos=len(secs),
                                      note='input check + detector + crops + classifier + domain, image already decoded')
    write_json(OUT / 'baseline_results.json', results)
    write_summary(results)
    print(f'-> {OUT}')


def write_summary(res):
    L = ['# P05 — Baseline (checkpoint đang triển khai), chế độ thường và demo', '',
         f"Detector `{res['versions']['detector']}`, classifier `{res['versions']['type_model']}`. "
         'Tính lại từ cache thô; khớp 100% với code đang chạy trên bộ `check`. IoU 0,3; CI 95% bootstrap theo lineage group.',
         f"val bỏ {res['val_excluded_seen_in_training']} ảnh mà model hiện tại đã train.", '']
    for mode in MODES:
        r = res[mode]
        L += [f'## Chế độ {mode}', '', HEAD]
        e = r['T1']['by_iou']['0.3']
        L.append(row('T1 (ACNE-DET, nhãn bác sĩ)', e['overall']))
        v = r['val']['by_iou']['0.3']
        L.append(row('val (gộp)', v['overall']))
        for s, ev in sorted(v['per_source'].items()):
            L.append(row(f'val · {s}', ev))
        L += ['', f"Nghiệm thu T1 (acceptance.yaml): {e['acceptance']}", '']
        L += ['Trạng thái ảnh theo nguồn (val): ' + '; '.join(f'{s}: {dict(c)}' for s, c in sorted(r['val']['status_by_source'].items())), '']
        clean = e['overall']['metrics']['tier_A_precision']
        L += ['T3 (ảnh T1 mô phỏng điện thoại):', '', HEAD]
        for vname, rv in r['T3'].items():
            L.append(row(vname, rv['by_iou']['0.3']['overall']))
        drops = {vname: (None if rv['by_iou']['0.3']['overall']['metrics']['tier_A_precision'] is None or clean is None
                         else clean - rv['by_iou']['0.3']['overall']['metrics']['tier_A_precision']) for vname, rv in r['T3'].items()}
        L += ['', 'Mức giảm precision tầng A so với T1 sạch (mục tiêu ≤ 5 điểm): ' +
              ', '.join(f'{k} {fmt(d)}' for k, d in drops.items()), '']
        t = r['T4']
        L += [f"T4 SCIN đầu/cổ: AUC soft-count {fmt(t['soft_count']['auc'], False)} CI {t['soft_count'].get('ci95')} "
              f"({t['soft_count']['positives']} ca mụn / {t['soft_count']['negatives']} ca không mụn, {t['soft_count']['status']}); "
              f"AUC số box {fmt(t['box_count']['auc'], False)}; {t['photos_with_failed_check']}/{t['photos']} ảnh trượt kiểm tra đầu vào.", '']
    lat = res['latency_seconds']
    L += [f"Latency (đã decode ảnh): p50 {lat['p50']:.2f} s, p95 {lat['p95']:.2f} s trên {lat['photos']} ảnh.", '']
    (OUT / 'summary.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    main()
