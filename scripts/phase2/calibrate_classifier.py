"""Confidence calibration and abstention ("uncertain") for the acne-type classifier of the two-stage pipeline.

Question: can the classifier probability be shown as a confidence, and can low-confidence types be withheld
(only the lesion location is shown, type = "uncertain") so that the types that ARE shown are right >= 95 %?

Crops are what the deployed pipeline classifies: square context crops around DETECTOR boxes at the detector's
operating confidence, matched class-agnostically to an annotation (IoU >= 0.5) for the true type. Annotated (GT)
crops are reported alongside. Unmatched detector boxes (false detections) have no true type; for them we report
the share withheld.

1. 5-fold CV (method, all 275 ACNE-DET images): per fold the fold's det1 detector (boxes cached by
   cv_evaluate.py; operating conf = class-agnostic max-F1 on the fold's val) and the fold's cfg_a_robust
   classifier (cv_robust.py). Temperature T and threshold tau fitted on the fold's val, applied to its test,
   pooled over the 275 images; bootstrap over the 50 patient groups.
2. Deployed checkpoints (models/phase2/pipeline_metadata.json): T and tau fitted on the master val split
   (40 images), reported once on master test (45 images, already seen by earlier studies).
3. DetectDataSkin-2 (exploratory: other domain, labels unreviewed, already seen): deployed classifier on the
   supplied GT crops with the deployed T and tau. The old cfg_a ECE (frozen probe) is recomputed for reference.

The rule (RULE) is written to reports/phase2_calibration/rule.json before any result and never overwritten.
--promote writes the deployed T / tau into models/phase2/pipeline_metadata.json as the verdict allows.
Usage: venv/Scripts/python.exe -u scripts/phase2/calibrate_classifier.py [--promote]
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

from common2 import ACNE_CLASSES, CROPS, MASTER_SPLIT, MODELS, REPORTS, ROOT, SOURCE, digest, read_json, write_json
from compare_det5 import GRID, ece, greedy, load_detector, load_split
from cv_evaluate import fold_images
from make_crops import crop_box
from predict_acne_type import TypeClassifier, softmax

OUT = REPORTS / 'phase2_calibration'
CACHE = OUT / 'logits'
CV_RUNS = ROOT / 'runs/phase2_cv'
K = 5
NC = len(ACNE_CLASSES)
TARGET = 0.95
TAUS = np.round(np.arange(0.30, 0.991, 0.01), 2)
RULE = dict(
    crops='detector boxes at the operating confidence, matched to an annotation at IoU >= 0.5 (class-agnostic, greedy by score)',
    temperature='single T minimising NLL on the val crops, bounded [0.25, 5]',
    threshold=f'tau = smallest value in 0.30..0.99 (step 0.01) whose val accuracy among crops with calibrated top-1 '
              f'probability >= tau is >= {TARGET}; if none reaches it, the tau with the best val accepted accuracy '
              f'among those keeping >= 50 % of crops (flagged)',
    deploy_temperature='pooled 5-fold CV test ECE after T <= ECE before T',
    deploy_threshold=f'pooled 5-fold CV test accepted accuracy >= {TARGET - 0.02}',
    deployed_fit='if deployed: T and tau refitted on the master val split for the deployed checkpoints',
)


# ---------------------------------------------------------------- data
def with_source(images):
    with MASTER_SPLIT.open(encoding='utf-8-sig') as f:
        sid = {int(r['image_id']): int(r['source_id']) for r in csv.DictReader(f)}
    return [dict(im, source_id=sid[im['id']]) for im in images]


def matched_sorted(boxes, gt):
    """boxes (n, 5) xyxy conf -> sorted by conf desc, true type per box (-1 = no annotation at IoU >= 0.5)."""
    b = boxes[np.argsort(-boxes[:, 4], kind='stable')]
    tp, idx = greedy(b, gt, 0.5)
    true = np.full(len(b), -1)
    true[tp] = gt[idx[tp], 4]
    return b, true


def agnostic_conf(images, boxes):
    """Class-agnostic max-F1 detector confidence on GRID. Greedy matching in score order means a box's match
    does not depend on lower-scored boxes, so one matching at the lowest score serves every threshold."""
    scores, flags, n_gt = [], [], 0
    for im in images:
        b, true = matched_sorted(boxes[im['id']], im['gt'])
        scores.append(b[:, 4]); flags.append(true >= 0); n_gt += len(im['gt'])
    s, t = np.concatenate(scores), np.concatenate(flags)
    f1 = [2 * t[s >= c].sum() / max(t[s >= c].sum() + (s >= c).sum() + n_gt, 1) for c in GRID]
    return float(GRID[int(np.argmax(f1))])


def det_crops(clf, images, boxes, conf):
    z, y, src = [], [], []
    for im in images:
        b = boxes[im['id']]
        b, true = matched_sorted(b[b[:, 4] >= conf], im['gt'])
        if not len(b):
            continue
        image = cv2.imread(str(SOURCE / im['file']))
        h, w = image.shape[:2]
        crops = [image[y0:y0 + s, x0:x0 + s] for x0, y0, s in
                 (crop_box([x1, y1, x2 - x1, y2 - y1], w, h) for x1, y1, x2, y2 in b[:, :4])]
        z.append(clf.logits(crops)); y.append(true); src += [im['source_id']] * len(b)
    return dict(z=np.concatenate(z), y=np.concatenate(y), src=np.array(src))


def gt_crops(clf, manifest, split):
    with manifest.open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == split]
    return dict(z=clf.logits([cv2.imread(str(CROPS / r['crop_path'])) for r in rows]),
                y=np.array([ACNE_CLASSES.index(r['cls']) for r in rows]), src=np.array([int(r['source_id']) for r in rows]))


def cached(name, key, fn):
    """Logits are cached per (name, key); key holds the checkpoint hash and settings that produced them."""
    path = CACHE / f'{name}.npz'
    if path.exists():
        d = np.load(path)
        if str(d['key']) == key:
            return {k: d[k] for k in ('z', 'y', 'src')}
    d = fn()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, key=key, **d)
    return d


# ---------------------------------------------------------------- metrics
def fit_temperature(z, y):
    from scipy.optimize import minimize_scalar
    return float(minimize_scalar(lambda T: nll(softmax(z / T), y), bounds=(0.25, 5), method='bounded').x)


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def choose_tau(p, y):
    conf, ok = p.max(1), p.argmax(1) == y
    for t in TAUS:
        m = conf >= t
        if m.any() and ok[m].mean() >= TARGET:
            return float(t), True
    return float(max((ok[conf >= t].mean(), t) for t in TAUS if (conf >= t).mean() >= 0.5)[1]), False


def calibration(p, y):
    conf, ok = p.max(1), p.argmax(1) == y
    return dict(n=int(len(y)), accuracy=float(ok.mean()), mean_confidence=float(conf.mean()), ece=ece(conf, ok),
                brier=float(np.mean(((p - np.eye(NC)[y]) ** 2).sum(1))), nll=nll(p, y))


def selective(p, y, tau):
    from sklearn.metrics import f1_score
    conf, pred = p.max(1), p.argmax(1)
    ok, m = pred == y, conf >= tau
    mean = lambda v: float(v.mean()) if len(v) else None
    o = np.argsort(-conf, kind='stable')
    per_class = {c: dict(n=int((y == i).sum()), coverage=mean(m[y == i]), accepted_accuracy=mean(ok[m & (y == i)]),
                         accepted_precision=mean(ok[m & (pred == i)]), accepted_predicted=int((m & (pred == i)).sum()))
                 for i, c in enumerate(ACNE_CLASSES)}
    return dict(tau=tau, coverage=mean(m), accepted_accuracy=mean(ok[m]), withheld_accuracy=mean(ok[~m]),
                all_accuracy=mean(ok),
                accepted_macro_f1=float(f1_score(y[m], pred[m], average='macro', labels=list(range(NC)), zero_division=0)),
                all_macro_f1=float(f1_score(y, pred, average='macro', labels=list(range(NC)), zero_division=0)),
                aurc=float(np.mean(1 - np.cumsum(ok[o]) / np.arange(1, len(o) + 1))), per_class=per_class)


def evaluate(z, y, T, tau):
    """Det crops: y = -1 for false detections (only their withheld share is defined)."""
    fp = y < 0
    zm, ym = z[~fp], y[~fp]
    out = dict(before=calibration(softmax(zm), ym), after=calibration(softmax(zm / T), ym),
               selective=selective(softmax(zm / T), ym, tau))
    if fp.any():
        out['false_detections'] = dict(n=int(fp.sum()), share_of_boxes=float(fp.mean()),
                                       withheld=float((softmax(z[fp] / T).max(1) < tau).mean()))
    return out


def reliability(conf, ok, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    i = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    return [(float(conf[i == b].mean()), float(ok[i == b].mean()), int((i == b).sum())) for b in range(bins) if (i == b).any()]


# ---------------------------------------------------------------- plots
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']  # categorical slots 1-3, fixed order
MARKERS = ['o', 's', '^']


def plot(panels, curves, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.4 * (len(panels) + 1), 4.2))
    for ax, (title, series) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], color='#9a9a94', lw=1, ls='--', label='Hiệu chỉnh hoàn hảo')
        for (label, conf, ok), c, mk in zip(series, COLORS, MARKERS):
            r = reliability(conf, ok)
            ax.plot([a for a, _, _ in r], [b for _, b, _ in r], color=c, lw=2, marker=mk, ms=6,
                    label=f'{label} (ECE {ece(conf, ok):.3f})')
        ax.set(title=title, xlabel='Độ tin cậy dự đoán', ylabel='Tỷ lệ đúng thực tế', xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8, loc='upper left', frameon=False)
    ax = axes[-1]
    for (label, cov, acc, point), c, mk in zip(curves, COLORS, MARKERS):
        ax.plot(cov, acc, color=c, lw=2, label=f'{label} (τ: hiển thị {point[0]:.0%}, đúng {point[1]:.1%})')
        ax.plot(*point, color=c, marker=mk, ms=8, ls='none')
    ax.axhline(TARGET, color='#9a9a94', lw=1, ls='--')
    ax.set(title='Đúng loại theo tỷ lệ được hiển thị', xlabel='Tỷ lệ nốt được hiển thị loại (coverage)',
           ylabel='Tỷ lệ đúng loại trong phần hiển thị', xlim=(0, 1))
    ax.legend(fontsize=8, loc='center left', frameon=False)
    for a in axes:
        a.grid(color='#e6e5e0', lw=0.6); a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def risk_curve(p, y):
    conf, ok = p.max(1), p.argmax(1) == y
    o = np.argsort(-conf, kind='stable')
    n = np.arange(1, len(o) + 1)
    return n / len(o), np.cumsum(ok[o]) / n


# ---------------------------------------------------------------- stages
def stage_cv():
    with (ROOT / 'datasets/phase2_cv/folds.csv').open(encoding='utf-8-sig') as f:
        group_of = {int(r['source_id']): r['group'] for r in csv.DictReader(f)}
    per_fold, pooled = [], {kind: dict(z=[], y=[], src=[], T=[], tau=[]) for kind in ('det', 'gt')}
    for k in range(K):
        weights = CV_RUNS / f'fold{k}/efficientnet_b0_cfg_a_robust/best.pth'
        clf = None
        data, row = {}, dict(fold=k)
        for split in ('val', 'test'):
            images = fold_images(k, split)
            raw = json.loads((REPORTS / f'phase2_cv/predictions/fold{k}_pipeline_{split}.json').read_text())
            boxes = {im['id']: np.array(raw[str(im['id'])]).reshape(-1, 8)[:, [0, 1, 2, 3, 7]] for im in images}
            if split == 'val':
                row['detector_conf'] = conf = agnostic_conf(images, boxes)
            key = f'{digest(weights)}|{conf}'

            def load():
                nonlocal clf
                clf = clf or TypeClassifier(weights)
                return clf
            data['det', split] = cached(f'cv_fold{k}_det_{split}', key, lambda: det_crops(load(), images, boxes, conf))
            data['gt', split] = cached(f'cv_fold{k}_gt_{split}', key,
                                       lambda: gt_crops(load(), ROOT / f'datasets/phase2_cv/fold{k}/crop_manifest.csv', split))
        v = data['det', 'val']; m = v['y'] >= 0
        T = fit_temperature(v['z'][m], v['y'][m])
        tau, reached = choose_tau(softmax(v['z'][m] / T), v['y'][m])
        row.update(temperature=T, tau=tau, tau_reached_target_on_val=reached,
                   val_det_crops=int(m.sum()), val_false_detections=int((~m).sum()))
        for kind in ('det', 'gt'):
            t = data[kind, 'test']
            row[f'test_{kind}'] = evaluate(t['z'], t['y'], T, tau)
            for a in ('z', 'y', 'src'):
                pooled[kind][a].append(t[a])
            pooled[kind]['T'].append(np.full(len(t['y']), T)); pooled[kind]['tau'].append(np.full(len(t['y']), tau))
        per_fold.append(row)
        s = row['test_det']
        print(f"fold {k}: det conf {row['detector_conf']:.2f} T {T:.3f} tau {tau:.2f} | test ECE "
              f"{s['before']['ece']:.4f} -> {s['after']['ece']:.4f} | coverage {s['selective']['coverage']:.3f} "
              f"accepted acc {s['selective']['accepted_accuracy']:.4f}", flush=True)
        del clf

    res = {}
    arrays = {}
    for kind, d in pooled.items():
        z, y, src, T, tau = (np.concatenate(d[a]) for a in ('z', 'y', 'src', 'T', 'tau'))
        keep = y >= 0
        pb, pa = softmax(z), softmax(z / T[:, None])
        arrays[kind] = dict(pb=pb[keep], pa=pa[keep], y=y[keep], accept=pa[keep].max(1) >= tau[keep],
                            groups=np.array([group_of[s] for s in src[keep]]))
        r = dict(before=calibration(pb[keep], y[keep]), after=calibration(pa[keep], y[keep]))
        sel = selective(pa[keep], y[keep], 0.0)  # per-crop taus differ by fold -> recompute with the fold masks
        acc_mask, ok = arrays[kind]['accept'], pa[keep].argmax(1) == y[keep]
        sel.update(tau='per fold', coverage=float(acc_mask.mean()), accepted_accuracy=float(ok[acc_mask].mean()),
                   withheld_accuracy=float(ok[~acc_mask].mean()) if (~acc_mask).any() else None)
        from sklearn.metrics import f1_score
        pred = pa[keep].argmax(1)
        sel['accepted_macro_f1'] = float(f1_score(y[keep][acc_mask], pred[acc_mask], average='macro', labels=list(range(NC)), zero_division=0))
        for i, c in enumerate(ACNE_CLASSES):
            yc = y[keep] == i
            sel['per_class'][c].update(coverage=float(acc_mask[yc].mean()),
                                       accepted_accuracy=float(ok[acc_mask & yc].mean()) if (acc_mask & yc).any() else None,
                                       accepted_precision=float(ok[acc_mask & (pred == i)].mean()) if (acc_mask & (pred == i)).any() else None,
                                       accepted_predicted=int((acc_mask & (pred == i)).sum()))
        r['selective'] = sel
        if kind == 'det':
            fp = ~keep
            r['false_detections'] = dict(n=int(fp.sum()), share_of_boxes=float(fp.mean()),
                                         withheld=float((softmax(z[fp] / T[fp, None]).max(1) < tau[fp]).mean()))
        res[kind] = r

    # bootstrap over patient groups (det crops)
    a = arrays['det']
    ug = np.unique(a['groups'])
    idx_of = {g: np.where(a['groups'] == g)[0] for g in ug}
    ok = a['pa'].argmax(1) == a['y']
    rng = np.random.default_rng(42)
    draws = dict(ece_before=[], ece_after=[], ece_diff=[], coverage=[], accepted_accuracy=[])
    for _ in range(1000):
        i = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug))])
        eb, ea = ece(a['pb'][i].max(1), ok[i]), ece(a['pa'][i].max(1), ok[i])
        draws['ece_before'].append(eb); draws['ece_after'].append(ea); draws['ece_diff'].append(ea - eb)
        draws['coverage'].append(a['accept'][i].mean()); draws['accepted_accuracy'].append(ok[i][a['accept'][i]].mean())
    res['bootstrap'] = dict(unit='patient group', groups=len(ug), n_boot=1000, seed=42,
                            **{k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()})
    return dict(per_fold=per_fold, pooled=res), arrays


def stage_deployed():
    pmeta, dmeta = read_json(MODELS / 'pipeline_metadata.json'), read_json(ROOT / 'models/phase1/model_metadata.json')
    clf_w, det_w = ROOT / pmeta['classifier'], ROOT / pmeta['detector']
    if digest(clf_w) != pmeta['classifier_sha256'] or digest(det_w) != dmeta['selected_weights_sha256']:
        sys.exit('deployed checkpoint hash differs from its metadata')
    conf, imgsz = float(dmeta['operating_conf']), int(dmeta['imgsz'])
    key = f'{digest(clf_w)}|{digest(det_w)}|{conf}|{imgsz}'
    state = {}

    def models():
        if not state:
            state['clf'], state['det'] = TypeClassifier(clf_w), load_detector(det_w)
        return state['clf'], state['det']

    def det_split(split):
        clf, det = models()
        images = with_source(load_split(split))
        boxes = {}
        for im in images:
            r = det.predict(str(SOURCE / im['file']), imgsz=imgsz, conf=conf, max_det=500, verbose=False)[0]
            boxes[im['id']] = np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()]).reshape(-1, 5)
        return det_crops(clf, images, boxes, conf)

    data = {}
    for split in ('val', 'test'):
        data['det', split] = cached(f'deployed_det_{split}', key, lambda: det_split(split))
        data['gt', split] = cached(f'deployed_gt_{split}', key, lambda: gt_crops(models()[0], CROPS / 'crop_manifest.csv', split))
    v = data['det', 'val']; m = v['y'] >= 0
    T = fit_temperature(v['z'][m], v['y'][m])
    tau, reached = choose_tau(softmax(v['z'][m] / T), v['y'][m])
    out = dict(classifier=pmeta['classifier'], classifier_sha256=pmeta['classifier_sha256'], detector_conf=conf,
               temperature=T, tau=tau, tau_reached_target_on_val=reached,
               val_det=evaluate(v['z'], v['y'], T, tau),
               test_det=evaluate(data['det', 'test']['z'], data['det', 'test']['y'], T, tau),
               test_gt=evaluate(data['gt', 'test']['z'], data['gt', 'test']['y'], T, tau))
    return out, data, state


def stage_dds2(T, tau, state):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from audit_detectdataskin2 import OUT as DDS, TYPE_MAP, read
    from evaluate_detectdataskin2 import cut
    pmeta = read_json(MODELS / 'pipeline_metadata.json')
    clf_w = ROOT / pmeta['classifier']
    manifest = DDS / 'image_manifest.json'

    def compute():
        clf = state.get('clf') or TypeClassifier(clf_w)
        z, y, src = [], [], []
        for n, row in enumerate(r for r in read(manifest) if not r['exclude_reasons']):
            ann = [a for a in row['annotations'] if a['category'] in TYPE_MAP]
            if not ann:
                continue
            image = cv2.imread(row['path'])
            z.append(clf.logits([cut(image, a['bbox']) for a in ann]))
            y += [ACNE_CLASSES.index(TYPE_MAP[a['category']]) for a in ann]; src += [n] * len(ann)
        return dict(z=np.concatenate(z), y=np.array(y), src=np.array(src))
    d = cached('dds2_gt', f'{digest(clf_w)}|{digest(manifest)}', compute)
    with (DDS / 'frozen_evaluation/classifier_gt_crop_predictions.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    old_p = np.array([[float(r[f'p_{c}']) for c in ACNE_CLASSES] for r in rows])
    old_y = np.array([ACNE_CLASSES.index(r['true']) for r in rows])
    return dict(status='exploratory: other domain, labels unreviewed, results already seen before',
                reference_cfg_a=dict(**calibration(old_p, old_y), selective_at_deployed_tau=selective(old_p, old_y, tau)),
                robust=evaluate(d['z'], d['y'], T, tau)), d, (old_p, old_y)


def promote(result):
    path = MODELS / 'pipeline_metadata.json'
    meta = read_json(path)
    v, dep = result['verdict'], result['deployed']
    meta['type_calibration'] = dict(
        temperature=dep['temperature'] if v['deploy_temperature'] else 1.0,
        accept_threshold=dep['tau'] if v['deploy_threshold'] else None,
        meaning='type probability = softmax(logits / temperature); top-1 probability below accept_threshold -> '
                'type_status "uncertain" (location shown, type withheld)',
        fitted_on='master val split, detector crops matched to annotations', rule=RULE, verdict=v,
        report='reports/phase2_calibration_summary.md', promoted_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, meta)
    print('promoted into', path, meta['type_calibration']['temperature'], meta['type_calibration']['accept_threshold'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--promote', action='store_true', help='write the verdict-approved T / tau into pipeline_metadata.json')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rule_path = OUT / 'rule.json'
    if rule_path.exists():
        if json.loads(rule_path.read_text(encoding='utf-8'))['rule'] != RULE:
            sys.exit('rule.json differs from RULE: the rule is fixed once written')
    else:
        write_json(rule_path, dict(rule=RULE, target_accepted_accuracy=TARGET, locked_utc=datetime.now(timezone.utc).isoformat()))

    cv, cv_arrays = stage_cv()
    dep, dep_data, state = stage_deployed()
    dds2, dds2_data, old = stage_dds2(dep['temperature'], dep['tau'], state)
    pc = cv['pooled']['det']
    verdict = dict(deploy_temperature=bool(pc['after']['ece'] <= pc['before']['ece']),
                   deploy_threshold=bool(pc['selective']['accepted_accuracy'] >= TARGET - 0.02))
    result = dict(rule=RULE, target_accepted_accuracy=TARGET, cv=cv, deployed=dep, detectdataskin2=dds2, verdict=verdict)
    write_json(OUT / 'calibration.json', result)

    ok = lambda p, y: (p.max(1), p.argmax(1) == y)
    a = cv_arrays['det']
    dt, mt = dep_data['det', 'test'], dep_data['det', 'test']['y'] >= 0
    py = dt['y'][mt]
    dz, dy = dds2_data['z'], dds2_data['y']
    T, tau = dep['temperature'], dep['tau']
    panels = [('CV 5 fold, 275 ảnh (crop từ detector)', [('Trước', *ok(a['pb'], a['y'])), ('Sau temperature', *ok(a['pa'], a['y']))]),
              ('DetectDataSkin-2 (thăm dò, crop nhãn)', [('cfg_a cũ', *ok(*old)), ('Robust, trước', *ok(softmax(dz), dy)),
                                                        ('Robust, sau T', *ok(softmax(dz / T), dy))])]
    acc_mask = a['accept']; ok_cv = a['pa'].argmax(1) == a['y']
    curves = [('CV 5 fold', *risk_curve(a['pa'], a['y']), (acc_mask.mean(), ok_cv[acc_mask].mean())),
              ('Model triển khai, test 45 ảnh', *risk_curve(softmax(dt['z'][mt] / T), py),
               (dep['test_det']['selective']['coverage'], dep['test_det']['selective']['accepted_accuracy'])),
              ('DetectDataSkin-2', *risk_curve(softmax(dz / T), dy),
               (dds2['robust']['selective']['coverage'], dds2['robust']['selective']['accepted_accuracy']))]
    plot(panels, curves, OUT / 'calibration.png')

    s = pc['selective']
    print(json.dumps(dict(cv_pooled_det=dict(ece=[pc['before']['ece'], pc['after']['ece']], coverage=s['coverage'],
                                             accepted_accuracy=s['accepted_accuracy'], all_accuracy=pc['after']['accuracy'],
                                             false_detections_withheld=pc['false_detections']['withheld']),
                          bootstrap=cv['pooled']['bootstrap'],
                          deployed=dict(T=T, tau=tau, test_ece=[dep['test_det']['before']['ece'], dep['test_det']['after']['ece']],
                                        test_coverage=dep['test_det']['selective']['coverage'],
                                        test_accepted_accuracy=dep['test_det']['selective']['accepted_accuracy']),
                          dds2=dict(cfg_a_ece=dds2['reference_cfg_a']['ece'], robust_ece=[dds2['robust']['before']['ece'], dds2['robust']['after']['ece']],
                                    robust_accuracy=dds2['robust']['before']['accuracy'],
                                    coverage=dds2['robust']['selective']['coverage'],
                                    accepted_accuracy=dds2['robust']['selective']['accepted_accuracy']),
                          verdict=verdict), indent=1))
    if args.promote:
        promote(result)


if __name__ == '__main__':
    main()
