"""Does training the phase-2 classifier with phone-photo degradations fix the pipeline's stress-test collapse?

Compares the deployed EfficientNet-B0 cfg_a with EfficientNet-B0 cfg_a_robust (same recipe + Degrade
augmentation, scripts/phase2/train_classifier.py), both behind the same deployed phase-1 detector, and puts
the result next to the one-stage RT-DETR-L 5-class detector from compare_det5.py.
Rule fixed before any result of the robust model was seen, validation only (written to selection_robust.json
before the test split is read): the robust classifier replaces the deployed one if
  (a) its macro-F1 on clean annotated val crops >= deployed - 0.01, and
  (b) the pipeline's end-to-end val mAP50 averaged over the 7 stress conditions is higher than with the
      deployed classifier.
Nothing is installed. The stress conditions are the ones in compare_det5.py; the augmentation ranges cover
them, so the robust model's stress numbers are in-distribution for it (stated in the report).
Usage: venv/Scripts/python.exe -u scripts/phase2/evaluate_robust.py
"""
from datetime import datetime, timezone

import numpy as np
import torch

from common2 import ACNE_CLASSES, ROOT, RUNS, digest, read_json, write_json
from compare_det5 import (OUT, PRED, STRESS, System, bootstrap, choose_conf, load_split, per_image_class_stats, run,
                          selective, agnostic, size_recall, summarise)
from evaluate_classifier import load_model, metrics
from train_classifier import CropSet, load_split as load_crops, predict, transforms

ROBUST = RUNS / 'efficientnet_b0_cfg_a_robust/best.pth'
DEPLOYED = ROOT / 'models/phase2/best_acne_type.pth'


def crop_metrics(path, split):
    model, ckpt = load_model(path)
    cache, y, _ = load_crops(split)
    loader = torch.utils.data.DataLoader(CropSet(cache, y, transforms(False, ckpt['mean'], ckpt['std'])), 64)
    m = metrics(y, predict(model, loader))
    del model; torch.cuda.empty_cache()
    return {k: m[k] for k in ('accuracy', 'balanced_accuracy', 'macro_f1', 'per_class')}


def stress_map(system, images, conf):
    out = {}
    for name, fn in STRESS.items():
        p, _ = run(system, images, transform=fn)
        out[name] = summarise(per_image_class_stats(p, images, 0.5), conf=conf)['map']
        print(' ', system.name, name, round(out[name], 4), flush=True)
    return out


def main():
    if not (RUNS / 'efficientnet_b0_cfg_a_robust/completed.json').exists():
        raise RuntimeError('train efficientnet_b0 cfg_a_robust first')
    val, test = load_split('val'), load_split('test')
    systems = {'pipeline': System('pipeline', DEPLOYED), 'pipeline_robust': System('pipeline_robust', ROBUST)}

    # ---- validation only
    crops_val = {'pipeline': crop_metrics(DEPLOYED, 'val'), 'pipeline_robust': crop_metrics(ROBUST, 'val')}
    vstats, confs, vclean, vstress = {}, {}, {}, {}
    for k, s in systems.items():
        p, _ = run(s, val, cache=PRED / f'{k}_val.json')
        vstats[k] = per_image_class_stats(p, val, 0.5)
        confs[k] = choose_conf(vstats[k])
        vclean[k] = summarise(vstats[k], conf=confs[k])
        vstress[k] = stress_map(s, val, confs[k])
        print('val', k, 'crop macro-F1', round(crops_val[k]['macro_f1'], 4), 'mAP50', round(vclean[k]['map'], 4),
              'stress mean', round(np.mean(list(vstress[k].values())), 4), flush=True)
    a = crops_val['pipeline_robust']['macro_f1'] >= crops_val['pipeline']['macro_f1'] - 0.01
    b = np.mean(list(vstress['pipeline_robust'].values())) > np.mean(list(vstress['pipeline'].values()))
    selection = dict(rule='robust replaces deployed classifier if val crop macro-F1 >= deployed - 0.01 AND mean val '
                          'stress mAP50 (7 conditions) > deployed', condition_a=bool(a), condition_b=bool(b),
                     chosen='pipeline_robust' if a and b else 'pipeline',
                     val={k: dict(crop_macro_f1=crops_val[k]['macro_f1'], conf=confs[k], map50=vclean[k]['map'],
                                  macro_f1=vclean[k]['macro_f1'], stress_map50=vstress[k],
                                  stress_mean=float(np.mean(list(vstress[k].values())))) for k in systems},
                     weights_sha256=dict(deployed=digest(DEPLOYED), robust=digest(ROBUST)),
                     locked_before_test=True, locked_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT / 'selection_robust.json', selection)
    print('selection', selection['chosen'], flush=True)

    # ---- test
    prev = read_json(OUT / 'comparison.json')
    rt_conf = prev['selection']['val']['rtdetr_det5']['conf']
    crops_test = {'pipeline': crop_metrics(DEPLOYED, 'test'), 'pipeline_robust': crop_metrics(ROBUST, 'test')}
    tstats, res = {}, {}
    for k, s in systems.items():
        p, _ = run(s, test, cache=PRED / f'{k}_test.json')
        tstats[k] = per_image_class_stats(p, test, 0.5)
        sm = summarise(tstats[k], conf=confs[k])
        cm, pairs = agnostic(p, test, confs[k])
        res[k] = dict(conf=confs[k], map50=sm['map'], macro_f1=sm['macro_f1'], crop=crops_test[k],
                      per_class_ap50={c: sm["ap"][i] for i, c in enumerate(ACNE_CLASSES)},
                      type_accuracy_on_found=float((pairs[:, 0] == pairs[:, 1]).mean()),
                      recall_by_size=size_recall(tstats[k], confs[k]), selective_type_accuracy=selective(pairs))
    rp, _ = run(System('rtdetr_det5'), test, cache=PRED / 'rtdetr_det5_test.json')
    tstats['rtdetr_det5'] = per_image_class_stats(rp, test, 0.5)
    boot = bootstrap({'pipeline': tstats['pipeline'], 'pipeline_robust': tstats['pipeline_robust'],
                      'rtdetr_det5': tstats['rtdetr_det5']}, {**confs, 'rtdetr_det5': rt_conf})
    stress = {'pipeline': {n: v['pipeline']['map50'] for n, v in prev['stress'].items()},
              'rtdetr_det5': {n: v['rtdetr_det5']['map50'] for n, v in prev['stress'].items()},
              'pipeline_robust': stress_map(systems["pipeline_robust"], test, confs["pipeline_robust"])}
    write_json(OUT / 'comparison_robust.json', dict(selection=selection, test=res, bootstrap=boot, stress_test_map50=stress))
    for k, r in res.items():
        print('test', k, 'mAP50', round(r['map50'], 4), 'macroF1', round(r['macro_f1'], 4), 'crop macro-F1', round(r['crop']['macro_f1'], 4))
    print('bootstrap', boot)
    print('stress', {k: {n: round(x, 3) for n, x in v.items()} for k, v in stress.items()})


if __name__ == '__main__':
    main()
