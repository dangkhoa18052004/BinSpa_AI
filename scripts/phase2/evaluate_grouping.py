"""Two groups instead of five types: non-inflammatory (closed + open comedo) vs inflammatory (papule + pustule + nodule).

No retraining: group probability = sum of the calibrated type probabilities (temperature from
calibrate_classifier.py), on the logits cached by calibrate_classifier.py (reports/phase2_calibration/logits).
Hierarchical display per lesion: type if its probability >= tau (the deployed type threshold); otherwise the group
if the group probability >= tau2; otherwise unknown.
Rule (written to reports/phase2_grouping/rule.json before any result, never overwritten): see RULE.
Evaluated: 5-fold CV detector crops (T, tau, tau2 per fold from that fold's val), deployed checkpoints (master val
fit, master test once) and DetectDataSkin-2 labelled crops (exploratory, other domain).
Usage: venv/Scripts/python.exe -u scripts/phase2/evaluate_grouping.py [--promote]
"""
import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np

from common2 import ACNE_CLASSES, MODELS, REPORTS, read_json, write_json
from compare_det5 import ece
from predict_acne_type import softmax

CAL = REPORTS / 'phase2_calibration'
OUT = REPORTS / 'phase2_grouping'
GROUP_OF = np.array([0, 0, 1, 1, 1])  # closed, open -> 0 non-inflammatory; papule, pustule, nodule -> 1 inflammatory
GROUPS = ['non_inflammatory', 'inflammatory']
TARGET = 0.95
TAUS = np.round(np.arange(0.50, 0.991, 0.01), 2)
RULE = dict(
    group_probability='sum of the temperature-calibrated type probabilities over the group',
    tau2=f'smallest value in 0.50..0.99 whose val group accuracy among crops with group probability >= tau2 is >= {TARGET} '
         '(detector crops matched to annotations; per CV fold, master val for the deployed checkpoints)',
    display='type if type probability >= tau (deployed type threshold); else group if group probability >= tau2; else unknown',
    deploy=f'pooled 5-fold CV test (detector crops): (a) accuracy of lesions shown only as a group >= {TARGET - 0.02} AND '
           '(b) share of lesions with a type or a group >= share with a type + 0.05',
)


def group_probs(p):
    return np.stack([p[:, GROUP_OF == g].sum(1) for g in (0, 1)], 1)


def choose_tau2(pg, yg):
    conf, ok = pg.max(1), pg.argmax(1) == yg
    for t in TAUS:
        m = conf >= t
        if m.any() and ok[m].mean() >= TARGET:
            return float(t)
    return float(TAUS[-1])


def display(p, y, tau, tau2):
    """-> dict of per-lesion masks/flags for the hierarchical display."""
    pg, yg = group_probs(p), GROUP_OF[y]
    typed = p.max(1) >= tau
    grouped = ~typed & (pg.max(1) >= tau2)
    return dict(typed=typed, grouped=grouped, type_ok=p.argmax(1) == y, group_ok=pg.argmax(1) == yg, pg=pg, yg=yg)


def summary(p, y, tau, tau2):
    d = display(p, y, tau, tau2)
    t, g = d['typed'], d['grouped']
    acc = lambda ok, m: float(ok[m].mean()) if m.any() else None
    shown_ok = np.where(t, d['type_ok'], d['group_ok'])
    pg = d['pg']
    bal = np.mean([(pg.argmax(1)[d['yg'] == k] == k).mean() for k in (0, 1) if (d['yg'] == k).any()])
    return dict(n=int(len(y)), tau=tau, tau2=tau2,
                type_accuracy_all=float(d['type_ok'].mean()), group_accuracy_all=float(d['group_ok'].mean()),
                group_balanced_accuracy_all=float(bal), group_ece=ece(pg.max(1), d['group_ok']),
                shown_type=float(t.mean()), shown_group_only=float(g.mean()), shown_any=float((t | g).mean()),
                accuracy_type_shown=acc(d['type_ok'], t), accuracy_group_only_shown=acc(d['group_ok'], g),
                accuracy_any_shown=acc(shown_ok, t | g),
                per_true_group={GROUPS[k]: dict(n=int((d['yg'] == k).sum()), shown_any=acc(t | g, d['yg'] == k),
                                                group_accuracy_all=acc(d['group_ok'], d['yg'] == k)) for k in (0, 1)})


def load(name):
    d = np.load(CAL / 'logits' / f'{name}.npz')
    m = d['y'] >= 0
    return d['z'][m], d['y'][m]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--promote', action='store_true', help='write tau2 into models/phase2/pipeline_metadata.json if the rule passes')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rule_path = OUT / 'rule.json'
    if rule_path.exists():
        if json.loads(rule_path.read_text(encoding='utf-8'))['rule'] != RULE:
            sys.exit('rule.json differs from RULE: the rule is fixed once written')
    else:
        write_json(rule_path, dict(rule=RULE, locked_utc=datetime.now(timezone.utc).isoformat()))

    cal = read_json(CAL / 'calibration.json')
    folds, pooled = [], dict(p=[], y=[], tau=[], tau2=[])
    for row in cal['cv']['per_fold']:
        k, T, tau = row['fold'], row['temperature'], row['tau']
        zv, yv = load(f'cv_fold{k}_det_val')
        tau2 = choose_tau2(group_probs(softmax(zv / T)), GROUP_OF[yv])
        zt, yt = load(f'cv_fold{k}_det_test')
        pt = softmax(zt / T)
        folds.append(dict(fold=k, tau=tau, tau2=tau2, test=summary(pt, yt, tau, tau2)))
        pooled['p'].append(pt); pooled['y'].append(yt)
        pooled['tau'].append(np.full(len(yt), tau)); pooled['tau2'].append(np.full(len(yt), tau2))
    p, y, tau_v, tau2_v = (np.concatenate(pooled[k]) for k in ('p', 'y', 'tau', 'tau2'))
    # pooled hierarchical display with each fold's own thresholds
    pg, yg = group_probs(p), GROUP_OF[y]
    typed = p.max(1) >= tau_v
    grouped = ~typed & (pg.max(1) >= tau2_v)
    type_ok, group_ok = p.argmax(1) == y, pg.argmax(1) == yg
    acc = lambda ok, m: float(ok[m].mean()) if m.any() else None
    cv = dict(n=int(len(y)), type_accuracy_all=float(type_ok.mean()), group_accuracy_all=float(group_ok.mean()),
              group_ece=ece(pg.max(1), group_ok), shown_type=float(typed.mean()), shown_group_only=float(grouped.mean()),
              shown_any=float((typed | grouped).mean()), accuracy_type_shown=acc(type_ok, typed),
              accuracy_group_only_shown=acc(group_ok, grouped),
              accuracy_any_shown=acc(np.where(typed, type_ok, group_ok), typed | grouped),
              per_true_group={GROUPS[k]: dict(n=int((yg == k).sum()), shown_any=acc(typed | grouped, yg == k),
                                              group_accuracy_all=acc(group_ok, yg == k)) for k in (0, 1)},
              per_true_type={c: dict(n=int((y == i).sum()), shown_type=acc(typed, y == i), shown_any=acc(typed | grouped, y == i),
                                     group_accuracy_all=acc(group_ok, y == i)) for i, c in enumerate(ACNE_CLASSES)})

    dep = cal['deployed']
    T, tau = dep['temperature'], dep['tau']
    zv, yv = load('deployed_det_val')
    tau2 = choose_tau2(group_probs(softmax(zv / T)), GROUP_OF[yv])
    zt, yt = load('deployed_det_test')
    deployed = dict(temperature=T, tau=tau, tau2=tau2, test=summary(softmax(zt / T), yt, tau, tau2))
    zd, yd = load('dds2_gt')
    dds2 = summary(softmax(zd / T), yd, tau, tau2)

    # Post-hoc (written after reading the locked result): tau2 fitted only on the val crops NOT shown as a type,
    # i.e. on the crops the group label is actually for. The locked rule fitted it on all val crops, where the easy
    # typed crops inflate the accuracy, so group-only lesions fell short.
    P, Y, TT, T2, fold_t2 = [], [], [], [], []
    for row in cal['cv']['per_fold']:
        k, Tk, tk = row['fold'], row['temperature'], row['tau']
        zv, yv = load(f'cv_fold{k}_det_val'); pv = softmax(zv / Tk); rest = pv.max(1) < tk
        t2 = choose_tau2(group_probs(pv[rest]), GROUP_OF[yv[rest]]); fold_t2.append(t2)
        zk, yk = load(f'cv_fold{k}_det_test'); P.append(softmax(zk / Tk)); Y.append(yk)
        TT.append(np.full(len(yk), tk)); T2.append(np.full(len(yk), t2))
    p2, y2, tt, t2v = (np.concatenate(v) for v in (P, Y, TT, T2))
    pg2 = group_probs(p2); ty2 = p2.max(1) >= tt; gr2 = ~ty2 & (pg2.max(1) >= t2v)
    ok_t2, ok_g2 = p2.argmax(1) == y2, pg2.argmax(1) == GROUP_OF[y2]
    zdv, ydv = load('deployed_det_val'); pdv = softmax(zdv / T); rest = pdv.max(1) < tau
    dep_t2 = choose_tau2(group_probs(pdv[rest]), GROUP_OF[ydv[rest]])
    posthoc = dict(note='tau2 fitted on val crops not shown as a type; post-hoc, CV test already seen', fold_tau2=fold_t2,
                   shown_type=float(ty2.mean()), shown_group_only=float(gr2.mean()), shown_any=float((ty2 | gr2).mean()),
                   accuracy_group_only_shown=acc(ok_g2, gr2), accuracy_any_shown=acc(np.where(ty2, ok_t2, ok_g2), ty2 | gr2),
                   deployed_tau2=dep_t2, deployed_test=summary(softmax(zt / T), yt, tau, dep_t2),
                   detectdataskin2=summary(softmax(zd / T), yd, tau, dep_t2))
    verdict = dict(a=(cv['accuracy_group_only_shown'] or 0) >= TARGET - 0.02, b=cv['shown_any'] >= cv['shown_type'] + 0.05)
    verdict['deploy'] = verdict['a'] and verdict['b']
    res = dict(rule=RULE, cv_per_fold=folds, cv_pooled=cv, deployed=deployed, detectdataskin2=dds2, verdict=verdict, posthoc_residual=posthoc)
    write_json(OUT / 'grouping.json', res)
    f = lambda d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if not isinstance(v, dict)}
    print('CV pooled', f(cv)); print('per true type', json.dumps(cv['per_true_type'], indent=0))
    print('deployed', deployed['tau2'], f(deployed['test'])); print('DDS2', f(dds2)); print('verdict', verdict)
    print('post-hoc residual tau2', f(posthoc), 'deployed test', f(posthoc['deployed_test']), 'DDS2', f(posthoc['detectdataskin2']))
    if args.promote and verdict['deploy']:
        path = MODELS / 'pipeline_metadata.json'
        meta = read_json(path)
        meta['type_calibration'].update(group_threshold=tau2, groups=dict(non_inflammatory=['closed_comedo', 'open_comedo'],
                                                                          inflammatory=['papule', 'pustule', 'nodule']),
                                        group_report='reports/phase2_grouping_summary.md')
        write_json(path, meta)
        print('promoted group_threshold', tau2)


if __name__ == '__main__':
    main()
