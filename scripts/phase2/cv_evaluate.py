"""Evaluate the 5-fold cross-validation: one-stage RT-DETR-L 5-class vs pipeline (RT-DETR-L 1-class ->
EfficientNet-B0), both trained per fold on identical data from the ACNE04-only checkpoint (cv_run.py).

Per fold: operating confidence = max macro-F1 on that fold's validation images (class-aware, IoU 0.5), then the
fold's test images. Every one of the 275 images is a test image exactly once, so the pooled numbers cover the
whole dataset. Reported:
  * per fold and mean +/- SD: mAP50, macro-F1
  * pooled over 275 images: mAP50, mAP@IoU0.3, macro-F1, per-class AP50/P/R/F1 (209 nodules instead of 34)
  * paired bootstrap over the 50 patient groups (1000x): pooled CIs and det5 - pipeline differences
Note: CV estimates the two METHODS. The deployed checkpoints differ (they started from ACNE-DET fine-tuned
weights and used one split); their single-split numbers are in reports/phase2_det5_summary.md.
Usage: venv/Scripts/python.exe -u scripts/phase2/cv_evaluate.py
"""
import csv

import numpy as np

from common2 import ACNE_CLASSES, REPORTS, ROOT, digest, write_csv, write_json
from compare_det5 import System, choose_conf, load_split, per_image_class_stats, run, summarise

CV = ROOT / 'datasets/phase2_cv'
RUNS = ROOT / 'runs/phase2_cv'
OUT = REPORTS / 'phase2_cv'
K = 5
SYSTEMS = ('pipeline', 'det5')


def fold_images(k, split):
    """load_split() of compare_det5 filtered to the fold's split (images are the same ACNE-DET files)."""
    keep = {p.stem for p in (CV / f'fold{k}/det5/images' / split).glob('*.jpg')}
    images = load_split('train') + load_split('val') + load_split('test')
    with (ROOT / 'datasets/acnedet_master_split.csv').open(encoding='utf-8-sig') as f:
        sid = {int(r['image_id']): int(r['source_id']) for r in csv.DictReader(f)}
    out = [dict(im, source_id=sid[im['id']]) for im in images if f'acnedet_{sid[im["id"]]:03d}' in keep]
    assert len(out) == len(keep), (k, split, len(out), len(keep))
    return sorted(out, key=lambda i: i['id'])


def system(k, name):
    if name == 'det5':
        return System('det5_cv', detector=RUNS / f'fold{k}/det5/weights/best.pt')
    return System('pipeline_cv', detector=RUNS / f'fold{k}/det1/weights/best.pt',
                  classifier=RUNS / f'fold{k}/efficientnet_b0_cfg_a/best.pth')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (CV / 'folds.csv').open(encoding='utf-8-sig') as f:
        group_of = {int(r['source_id']): r['group'] for r in csv.DictReader(f)}
    per_fold, images_all, weights = [], [], {}
    pooled_all, pooled30_all, pooled_op, pooled30_op = ({s: [] for s in SYSTEMS} for _ in range(4))
    for k in range(K):
        val, test = fold_images(k, 'val'), fold_images(k, 'test')
        images_all += test
        row = dict(fold=k, val_images=len(val), test_images=len(test))
        for name in SYSTEMS:
            s = system(k, name)
            vp, _ = run(s, val, cache=OUT / f'predictions/fold{k}_{name}_val.json')
            conf = choose_conf(per_image_class_stats(vp, val, 0.5))
            tp, ms = run(s, test, cache=OUT / f'predictions/fold{k}_{name}_test.json')
            st = per_image_class_stats(tp, test, 0.5)
            sm = summarise(st, conf=conf)
            op = [p[p[:, 4] >= conf] for p in tp]  # per-fold operating point applied before pooling
            pooled_all[name] += st
            pooled30_all[name] += per_image_class_stats(tp, test, 0.3)
            pooled_op[name] += per_image_class_stats(op, test, 0.5)
            pooled30_op[name] += per_image_class_stats(op, test, 0.3)
            row.update({f'{name}_conf': conf, f'{name}_map50': sm['map'], f'{name}_macro_f1': sm['macro_f1'],
                        f'{name}_ms_per_image': ms})
            del s
        per_fold.append(row)
        print(row, flush=True)
    for k in range(K):
        weights[k] = {n: digest(RUNS / f'fold{k}' / p) for n, p in (('det5', 'det5/weights/best.pt'), ('det1', 'det1/weights/best.pt'),
                                                                     ('classifier', 'efficientnet_b0_cfg_a/best.pth'))}

    # ---- pooled over all 275 images
    pooled = {}
    for name in SYSTEMS:
        a, o = summarise(pooled_all[name]), summarise(pooled_op[name], conf=0.0)
        pooled[name] = dict(map50=a['map'], map30=summarise(pooled30_all[name])['map'], macro_f1=o['macro_f1'], macro_f1_iou30=summarise(pooled30_op[name], conf=0.0)['macro_f1'],
                            per_class={c: dict(ap50=a['ap'][i], precision=o['p'][i], recall=o['r'][i], f1=o['f1'][i],
                                               tp=o['tp'][i], fp=o['fp'][i], fn=o['fn'][i]) for i, c in enumerate(ACNE_CLASSES)})
        pooled[name].update({f'fold_mean_{m}': float(np.mean([r[f'{name}_{m}'] for r in per_fold])) for m in ('map50', 'macro_f1')})
        pooled[name].update({f'fold_sd_{m}': float(np.std([r[f'{name}_{m}'] for r in per_fold], ddof=1)) for m in ('map50', 'macro_f1')})

    # ---- paired bootstrap over patient groups
    groups = np.array([group_of[im['source_id']] for im in images_all])
    ug = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(42)
    draws = {n: dict(map=[], f1=[]) for n in SYSTEMS}
    for _ in range(1000):
        idx = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug))])
        for n in SYSTEMS:
            draws[n]['map'].append(summarise(pooled_all[n], idx)['map'])
            draws[n]['f1'].append(summarise(pooled_op[n], idx, 0.0)['macro_f1'])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    dm = np.array(draws['det5']['map']) - draws['pipeline']['map']
    df = np.array(draws['det5']['f1']) - draws['pipeline']['f1']
    boot = dict(unit='patient group', groups=len(ug), n_boot=1000, seed=42,
                **{n: dict(map50_ci=ci(d['map']), macro_f1_ci=ci(d['f1'])) for n, d in draws.items()},
                det5_minus_pipeline=dict(map50=float(dm.mean()), map50_ci=ci(dm), p_map50_le0=float((dm <= 0).mean()),
                                         macro_f1=float(df.mean()), macro_f1_ci=ci(df), p_macro_f1_le0=float((df <= 0).mean())))
    write_csv(OUT / 'per_fold.csv', per_fold, list(per_fold[0]))
    write_json(OUT / 'cv_results.json', dict(images=len(images_all), per_fold=per_fold, pooled=pooled, bootstrap=boot,
                                             weights_sha256=weights))
    for n in SYSTEMS:
        print(n, {k: round(v, 4) for k, v in pooled[n].items() if isinstance(v, float)}, boot[n])
    print('det5 - pipeline', boot['det5_minus_pipeline'])


if __name__ == '__main__':
    main()
