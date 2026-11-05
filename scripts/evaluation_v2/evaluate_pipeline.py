"""P04: the single selfie_v1 evaluator (plan section 6.1, configs/selfie_v1/acceptance.yaml).

Scores *displayed* lesions: each shown box carries a tier
  A  specific type (one of the five), B  group only (inflammatory / non_inflammatory), C  location only.
Definitions (fixed, used by every later milestone):
  - Matching per photo, one-to-one, shown boxes in descending detector score; each takes the unmatched lesion GT
    (any label except `ignore`) with the highest IoU >= thr. thr 0.3 is primary, 0.5 always reported.
    An unmatched box with IoU >= thr to an `ignore` GT box is `ignored` (counts neither way); any other unmatched
    box is a false positive (a second box on an already matched lesion is a `duplicate` FP).
  - location precision = TP / (shown - ignored); location recall = TP / lesion GT.
  - tier A precision = A boxes whose matched GT has the shown type / A boxes (FPs included, ignored excluded).
  - tier B precision = B boxes whose matched GT is in the shown group / B boxes. Acceptance uses the boxes shown
    at tier B only (what a customer reads as "mụn viêm"); group precision over A and B together is also reported.
  - Type and group metrics use only photos whose GT is fully typed (no `acne_untyped` box, e.g. ACNE04-v2 has none);
    location metrics use every photo.
  - coverage on found = share of matched GT shown at tier A (or A or B).
  - ECE (15 equal bins) of the displayed probability `prob` of A/B boxes: `ece_displayed` over every A/B box
    (a false positive counts as wrong), `ece_on_found` over matched boxes only.
  - macro-F1 box+type: per class, TP = A box of that class on a GT of that class; precision over A boxes,
    recall over typed GT; mean over classes with GT or predictions.
  - Every photo of the role is in the denominator; a photo with warnings or a non-ok status is scored like any other.
    A photo without a prediction record is an error unless --allow-missing (then it counts as zero shown boxes).
  - 95 % CI: bootstrap over lineage groups (1000 resamples), from additive per-photo counts.
Prediction file (JSONL, one record per photo, keyed by `path` as in datasets/selfie_v1/annotations):
  {"path": ..., "status": "ok", "warnings": [...],
   "boxes": [{"bbox": [x1, y1, x2, y2], "score": 0.71, "tier": "A", "type": "papule", "group": "inflammatory",
              "prob": 0.93}, ...]}
  bbox in pixels of the annotated file; tier B needs `group`, tier A needs `type` (group derived); `prob` optional.
Evaluation set for a role: one photo per Roboflow source image (copies of one source share labels and lineage; the
unaugmented export copy test > valid > train is kept, DetectDataSkin-2 before xnbio), ACNE-DET from the author
release, ACNE04 from ACNE04-v2; quarantine and unlabelled photos never enter.
Usage:
  venv/Scripts/python.exe scripts/evaluation_v2/evaluate_pipeline.py --role val --pred preds.jsonl --out report.json
  ... --list-items --role val --out items.jsonl     (the photos a prediction run must cover)
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import digest, write_json  # noqa: E402

DATA = ROOT / 'datasets/selfie_v1'
ACCEPTANCE = ROOT / 'configs/selfie_v1/acceptance.yaml'
CLASSES = ['closed_comedo', 'open_comedo', 'papule', 'pustule', 'nodule']
GROUP = dict(closed_comedo='non_inflammatory', open_comedo='non_inflammatory', papule='inflammatory',
             pustule='inflammatory', nodule='inflammatory')
GROUPS = ['inflammatory', 'non_inflammatory']
TIERS = ('A', 'B', 'C')
BINS = 15
# annotation file used for each lineage source when scoring (ACNE-DET mirror duplicates the author release)
ANNOTATION_OF = dict(acnedet_author='acnedet_author', acne04='acne04v2', detectdataskin2='detectdataskin2', xnbio='xnbio')
DEFAULT_SOURCES = dict(T1=['acnedet_author'], val=['acnedet_author', 'acne04', 'detectdataskin2', 'xnbio'],
                       T2=['detectdataskin2', 'xnbio'], train=['acnedet_author', 'acne04', 'detectdataskin2', 'xnbio'],
                       acne04_test=['acne04'])
EXPORT_RANK = dict(test=0, valid=1, train=2)
SOURCE_RANK = dict(detectdataskin2=0, xnbio=1)


# ---------------------------------------------------------------- evaluation set

def source_key(row):
    """Roboflow copies of one source image share the part of the name before '.rf.'; xnbio and DDS2 share it too."""
    return row['image'].split('.rf.')[0] if row['source'] in SOURCE_RANK else f"{row['source']}:{row['image']}"


def eval_items(role, sources=None, data=DATA):
    """-> GT records of `role` (one per source photo), each with `lineage_group_id`, `eval_source`, `typed`."""
    sources = sources or DEFAULT_SOURCES[role]
    lineage = [json.loads(line) for line in (data / 'lineage.jsonl').read_text(encoding='utf-8').splitlines()]
    keep = {r['path']: r for r in lineage if r['role'] == role and r['source'] in sources
            and not r['quarantined'] and not r['unlabelled']}
    ann = {}
    for src in {ANNOTATION_OF[s] for s in sources}:
        for line in (data / 'annotations' / f'{src}.jsonl').read_text(encoding='utf-8').splitlines():
            a = json.loads(line)
            if a['path'] in keep:
                ann[a['path']] = a
    missing = sorted(set(keep) - set(ann))
    if missing:
        raise SystemExit(f'{len(missing)} {role} photos have no annotation record, e.g. {missing[:3]}')
    chosen = {}
    for path, r in keep.items():
        a = ann[path]
        rank = (EXPORT_RANK.get(a.get('split_export'), 3), SOURCE_RANK.get(r['source'], 0), a['image'])
        k = source_key(r)
        if k not in chosen or rank < chosen[k][0]:
            chosen[k] = (rank, r)
    items = []
    for _, r in sorted(chosen.values(), key=lambda v: v[1]['path']):
        a = ann[r['path']]
        items.append(dict(path=r['path'], source=r['source'], image=r['image'], width=a['width'], height=a['height'],
                          lineage_group_id=r['lineage_group_id'], boxes=a['boxes'],
                          typed=not any(b['label'] == 'acne_untyped' for b in a['boxes'])))
    return items


# ---------------------------------------------------------------- per-photo counts

def iou(a, b):
    """a: (n, 4), b: (m, 4) xyxy -> (n, m)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def check_box(b):
    t = b.get('tier')
    if t not in TIERS:
        raise ValueError(f'tier must be A, B or C: {b}')
    if t == 'A':
        if b.get('type') not in CLASSES:
            raise ValueError(f'tier A box needs a type of {CLASSES}: {b}')
        if b.get('group') not in (None, GROUP[b['type']]):
            raise ValueError(f'group does not match type: {b}')
    if t == 'B' and b.get('group') not in GROUPS:
        raise ValueError(f'tier B box needs a group of {GROUPS}: {b}')
    p = b.get('prob')
    if p is not None and not 0 <= p <= 1:
        raise ValueError(f'prob outside [0, 1]: {b}')


def shown_group(b):
    return GROUP[b['type']] if b['tier'] == 'A' else b.get('group')


def bin_of(p):
    return min(int(p * BINS), BINS - 1)


def match(gt_boxes, shown, thr):
    """-> list (one per shown box, input order) of ('tp', gt index) / ('ignored', None) / ('dup', None) / ('fp', None)."""
    lesion = [i for i, g in enumerate(gt_boxes) if g['label'] != 'ignore']
    ignore = [i for i, g in enumerate(gt_boxes) if g['label'] == 'ignore']
    out = [None] * len(shown)
    if not shown:
        return out
    pb = [b['bbox'] for b in shown]
    m_les = iou(pb, [gt_boxes[i]['bbox'] for i in lesion]) if lesion else np.zeros((len(shown), 0))
    m_ign = iou(pb, [gt_boxes[i]['bbox'] for i in ignore]) if ignore else np.zeros((len(shown), 0))
    taken = set()
    for k in sorted(range(len(shown)), key=lambda k: -shown[k].get('score', 0.0)):  # stable: ties keep input order
        best, best_iou = None, thr
        for j in range(len(lesion)):
            if j not in taken and m_les[k, j] >= best_iou and (best is None or m_les[k, j] > best_iou):
                best, best_iou = j, m_les[k, j]
        if best is not None:
            taken.add(best)
            out[k] = ('tp', lesion[best])
        elif m_ign.shape[1] and m_ign[k].max() >= thr:
            out[k] = ('ignored', None)
        elif m_les.shape[1] and m_les[k].max() >= thr:
            out[k] = ('dup', None)
        else:
            out[k] = ('fp', None)
    return out


def photo_counts(item, shown, thr):
    """Additive counts of one photo at one IoU threshold (summed over photos / lineage groups)."""
    for b in shown:
        check_box(b)
    c = Counter(photos=1, typed_photos=int(item['typed']))
    gt = item['boxes']
    for g in gt:
        if g['label'] != 'ignore':
            c['gt'] += 1
            if item['typed']:
                c[f"gt:{g['label']}"] += 1
    res = match(gt, shown, thr)
    for b, (kind, j) in zip(shown, res):
        c['shown'] += 1
        if kind == 'ignored':
            c['ignored'] += 1
            continue
        tp = kind == 'tp'
        c['tp' if tp else 'fp'] += 1
        c['dup'] += kind == 'dup'
        if tp:
            c[f"found_{b['tier']}"] += 1
        if not item['typed'] or b['tier'] == 'C':
            continue
        true = gt[j]['label'] if tp else None
        group_ok = tp and GROUP[true] == shown_group(b)
        if b['tier'] == 'A':
            type_ok = tp and true == b['type']
            c['a_shown'] += 1
            c['a_correct'] += type_ok
            c[f"a_shown:{b['type']}"] += 1
            c[f"a_correct:{b['type']}"] += type_ok
            c[f"confusion:{true or 'none'}>{b['type']}"] += 1
            ok = type_ok
        else:
            c['b_shown'] += 1
            c['b_correct'] += group_ok
            ok = group_ok
        c['ab_shown'] += 1
        c['ab_group_correct'] += group_ok
        if b.get('prob') is None:
            c['no_prob'] += 1
            continue
        k = bin_of(b['prob'])
        for name, use in (('disp', True), ('found', tp)):
            if use:
                c[f'ece_{name}_n:{k}'] += 1
                c[f'ece_{name}_conf:{k}'] += b['prob']
                c[f'ece_{name}_ok:{k}'] += ok
    return c


# ---------------------------------------------------------------- summary

def ratio(a, b):
    return None if not b else a / b


def ece(c, name):
    n = sum(c.get(f'ece_{name}_n:{k}', 0) for k in range(BINS))
    if not n:
        return None
    return sum(abs(c.get(f'ece_{name}_ok:{k}', 0) - c.get(f'ece_{name}_conf:{k}', 0))
               for k in range(BINS) if c.get(f'ece_{name}_n:{k}', 0)) / n


def macro_f1(c):
    f1 = []
    for t in CLASSES:
        tp, pred, gt = c.get(f'a_correct:{t}', 0), c.get(f'a_shown:{t}', 0), c.get(f'gt:{t}', 0)
        if pred or gt:
            f1.append(2 * tp / (pred + gt))
    return float(np.mean(f1)) if f1 else None


def summarize(c):
    """Counts -> metrics (None where the denominator is 0)."""
    tp = c.get('tp', 0)
    scored = c.get('shown', 0) - c.get('ignored', 0)
    m = dict(
        photos=c.get('photos', 0), typed_photos=c.get('typed_photos', 0), lesion_gt=c.get('gt', 0),
        shown=c.get('shown', 0), ignored=c.get('ignored', 0), tp=tp, fp=c.get('fp', 0), duplicates=c.get('dup', 0),
        location_precision=ratio(tp, scored), location_recall=ratio(tp, c.get('gt', 0)),
        fp_per_photo=ratio(c.get('fp', 0), c.get('photos', 0)),
        tier_A_shown=c.get('a_shown', 0), tier_A_precision=ratio(c.get('a_correct', 0), c.get('a_shown', 0)),
        tier_B_shown=c.get('b_shown', 0), tier_B_precision=ratio(c.get('b_correct', 0), c.get('b_shown', 0)),
        group_precision_A_or_B=ratio(c.get('ab_group_correct', 0), c.get('ab_shown', 0)),
        coverage_A_on_found=ratio(c.get('found_A', 0), tp),
        coverage_A_or_B_on_found=ratio(c.get('found_A', 0) + c.get('found_B', 0), tp),
        shown_by_tier={t: c.get(f'found_{t}', 0) for t in TIERS},
        ece_displayed=ece(c, 'disp'), ece_on_found=ece(c, 'found'), boxes_without_prob=c.get('no_prob', 0),
        macro_f1_box_type=macro_f1(c),
        per_type={t: dict(shown_A=c.get(f'a_shown:{t}', 0), precision_A=ratio(c.get(f'a_correct:{t}', 0), c.get(f'a_shown:{t}', 0)),
                          gt=c.get(f'gt:{t}', 0)) for t in CLASSES})
    m['confusion_A'] = {k.split(':', 1)[1]: v for k, v in sorted(c.items()) if k.startswith('confusion:')}
    m['reliability_displayed'] = [dict(bin=[k / BINS, (k + 1) / BINS], n=c.get(f'ece_disp_n:{k}', 0),
                                       mean_prob=ratio(c.get(f'ece_disp_conf:{k}', 0), c.get(f'ece_disp_n:{k}', 0)),
                                       accuracy=ratio(c.get(f'ece_disp_ok:{k}', 0), c.get(f'ece_disp_n:{k}', 0)))
                                  for k in range(BINS) if c.get(f'ece_disp_n:{k}', 0)]
    return m


CI_METRICS = ['location_precision', 'location_recall', 'fp_per_photo', 'tier_A_precision', 'tier_B_precision',
              'group_precision_A_or_B', 'coverage_A_on_found', 'coverage_A_or_B_on_found', 'ece_displayed',
              'ece_on_found', 'macro_f1_box_type']


def bootstrap(group_counts, resamples=1000, seed=0, ci=0.95):
    """group_counts: {group id: Counter}. Resample groups with replacement -> {metric: [low, high]}."""
    groups = sorted(group_counts)
    if not groups or not resamples:
        return {}
    keys = sorted({k for c in group_counts.values() for k in c})
    M = np.array([[group_counts[g].get(k, 0) for k in keys] for g in groups], float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(resamples, len(groups)))
    W = np.stack([np.bincount(d, minlength=len(groups)) for d in draws])
    vals = defaultdict(list)
    for row in W @ M:
        s = summarize(dict(zip(keys, row)))
        for name in CI_METRICS:
            if s[name] is not None:
                vals[name].append(s[name])
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in vals.items() if len(v) >= resamples / 2}


def verdict(metrics, ci, target):
    """Acceptance items of one role (acceptance.yaml targets.T1 / T2) -> {item: pass / fail / inconclusive / not_shown}.
    not_shown: the policy shows no box at that tier (its precision is undefined; coverage carries the cost)."""
    slack = target.get('ci_lower_slack', 0.0)
    out = {}
    pairs = [('tier_A_precision', 'tier_A_precision'), ('tier_B_precision', 'tier_B_precision'),
             ('location_precision', 'location_precision'), ('location_recall', 'location_recall'),
             ('coverage_A_or_B_on_found', 'coverage_A_or_B_on_found')]
    for item, metric in pairs:
        if item not in target:
            continue
        v, lo = metrics.get(metric), (ci.get(metric) or [None])[0]
        tier = metric[len('tier_'):-len('_precision')] if metric.startswith('tier_') else None
        if v is None and tier and metrics.get(f'tier_{tier}_shown') == 0:
            out[item] = 'not_shown'
        elif v is None:
            out[item] = 'inconclusive'
        else:
            out[item] = 'pass' if v >= target[item] and (lo is None or lo >= target[item] - slack) else 'fail'
    if 'ece_max' in target:
        v = metrics.get('ece_displayed')
        out['ece_max'] = 'inconclusive' if v is None else 'pass' if v <= target['ece_max'] else 'fail'
    return out


def evaluate(items, predictions, thresholds=(0.3, 0.5), resamples=1000, seed=0, allow_missing=False, target=None):
    """items: eval_items(); predictions: {path: record}. -> report dict (overall and per source, per IoU threshold)."""
    missing = [it['path'] for it in items if it['path'] not in predictions]
    if missing and not allow_missing:
        raise ValueError(f'{len(missing)} photos have no prediction, e.g. {missing[:3]}')
    statuses = Counter(predictions[it['path']].get('status', 'ok') if it['path'] in predictions else 'missing' for it in items)
    warned = sum(bool(predictions.get(it['path'], {}).get('warnings')) for it in items)
    report = dict(photos=len(items), groups=len({it['lineage_group_id'] for it in items}), missing=len(missing),
                  statuses=dict(statuses), photos_with_warnings=warned, by_iou={})
    for thr in thresholds:
        total, by_source = Counter(), defaultdict(Counter)
        by_group, by_source_group = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
        for it in items:
            c = photo_counts(it, predictions.get(it['path'], {}).get('boxes', []), thr)
            total.update(c)
            by_source[it['source']].update(c)
            by_group[it['lineage_group_id']].update(c)
            by_source_group[it['source']][it['lineage_group_id']].update(c)
        overall = summarize(total)
        overall_ci = bootstrap(by_group, resamples, seed)
        sources = {s: dict(metrics=summarize(c), ci95=bootstrap(by_source_group[s], resamples, seed)) for s, c in by_source.items()}
        worst = {}
        for name in ('location_precision', 'location_recall', 'tier_A_precision', 'tier_B_precision', 'coverage_A_or_B_on_found'):
            vals = {s: v['metrics'][name] for s, v in sources.items() if v['metrics'][name] is not None}
            if vals:
                s = min(vals, key=vals.get)
                worst[name] = dict(source=s, value=vals[s])
        entry = dict(overall=dict(metrics=overall, ci95=overall_ci), per_source=sources, worst_source=worst)
        if target is not None:
            entry['acceptance'] = verdict(overall, overall_ci, target)
        report['by_iou'][str(thr)] = entry
    return report


# ---------------------------------------------------------------- T4 (case level)

def case_auc(scores, labels, groups=None, resamples=1000, seed=0, min_per_side=20):
    """AUC of a per-case acne score, acne (1) vs non-acne (0) cases; bootstrap CI over `groups` (default: case).
    Fewer than min_per_side cases on a side -> status inconclusive."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    groups = np.arange(len(s)) if groups is None else np.asarray(groups)
    npos, nneg = int(y.sum()), int((1 - y).sum())

    def auc(s, y):
        p, n = s[y == 1], s[y == 0]
        if not len(p) or not len(n):
            return None
        return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))

    out = dict(auc=auc(s, y), positives=npos, negatives=nneg,
               status='inconclusive' if min(npos, nneg) < min_per_side else 'measured')
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(resamples):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])
        v = auc(s[pick], y[pick])
        if v is not None:
            vals.append(v)
    if vals:
        out['ci95'] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


# ---------------------------------------------------------------- CLI

def load_predictions(path):
    preds = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            if r['path'] in preds:
                raise ValueError(f"two prediction records for {r['path']}")
            preds[r['path']] = r
    return preds


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--role', required=True, choices=sorted(DEFAULT_SOURCES))
    p.add_argument('--sources', nargs='+', help='lineage sources (default per role)')
    p.add_argument('--pred', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--list-items', action='store_true', help='write the photos of the role (JSONL) and stop')
    p.add_argument('--resamples', type=int, default=None, help='bootstrap resamples (default from acceptance.yaml)')
    p.add_argument('--allow-missing', action='store_true')
    a = p.parse_args()
    acc = yaml.safe_load(ACCEPTANCE.read_text(encoding='utf-8'))
    items = eval_items(a.role, a.sources)
    if a.list_items:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(''.join(json.dumps(dict(path=it['path'], source=it['source'], lineage_group_id=it['lineage_group_id']),
                                            ensure_ascii=False) + '\n' for it in items), encoding='utf-8')
        print(f'{len(items)} {a.role} photos -> {a.out}')
        return
    if a.pred is None:
        p.error('--pred is required unless --list-items')
    resamples = a.resamples if a.resamples is not None else acc['bootstrap']['resamples']
    report = evaluate(items, load_predictions(a.pred), (acc['matching']['primary_iou'], acc['matching']['secondary_iou']),
                      resamples, allow_missing=a.allow_missing, target=acc['targets'].get(a.role))
    manifest = json.loads((DATA / 'split_manifest.json').read_text(encoding='utf-8'))
    report = dict(role=a.role, sources=a.sources or DEFAULT_SOURCES[a.role],
                  inputs=dict(predictions=str(a.pred), predictions_sha256=digest(a.pred),
                              lineage_sha256=manifest['lineage_sha256'], acceptance_sha256=digest(ACCEPTANCE),
                              acceptance_locked=digest(ACCEPTANCE) == manifest['locked_files_sha256'].get('configs/selfie_v1/acceptance.yaml')),
                  **report)
    write_json(a.out, report)
    o = report['by_iou'][str(acc['matching']['primary_iou'])]['overall']['metrics']
    fmt = lambda v: '-' if v is None else f'{v:.3f}'  # noqa: E731
    print(f"{a.role}: {report['photos']} photos | loc P {fmt(o['location_precision'])} R {fmt(o['location_recall'])} | "
          f"A {fmt(o['tier_A_precision'])} (n {o['tier_A_shown']}) | B {fmt(o['tier_B_precision'])} (n {o['tier_B_shown']}) | "
          f"cov A+B {fmt(o['coverage_A_or_B_on_found'])} | ECE {fmt(o['ece_displayed'])} -> {a.out}")


if __name__ == '__main__':
    main()
