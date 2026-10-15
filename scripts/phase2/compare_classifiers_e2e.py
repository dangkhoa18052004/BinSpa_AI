"""Two-stage pipeline with each phase-2 classifier behind the deployed phase-1 detector, end to end.

Reuses the cached phase-1 boxes of compare_det5.py (reports/phase2_det5/predictions/pipeline_{val,test}.json:
detector run at conf 0.05, column 7 = detector confidence), re-crops them from the full-resolution images and
classifies them with each candidate; score = detector conf x classifier probability, operating threshold =
max val macro-F1. Candidates: EfficientNet-B0 cfg_a (deployed), ConvNeXt-Tiny cfg_a, EfficientNet-B0
cfg_a_robust. Reports val and test mAP50 / macro-F1, type accuracy on found lesions and a paired bootstrap
vs the deployed classifier. Runs on CPU by default (--device cuda if the GPU is free).
Usage: venv/Scripts/python.exe -u scripts/phase2/compare_classifiers_e2e.py
"""
import argparse
import json

import cv2
import numpy as np

from common2 import ACNE_CLASSES, REPORTS, ROOT, RUNS, SOURCE, digest, write_json
from compare_det5 import PRED, agnostic, bootstrap, choose_conf, load_split, per_image_class_stats, summarise
from make_crops import crop_box
from predict_acne_type import TypeClassifier

CLASSIFIERS = {'efficientnet_b0': ROOT / 'models/phase2/best_acne_type.pth',
               'convnext_tiny': RUNS / 'convnext_tiny_cfg_a/best.pth',
               'efficientnet_b0_robust': RUNS / 'efficientnet_b0_cfg_a_robust/best.pth'}
OUT = REPORTS / 'phase2_det5'


def reclassify(clf, images, cached):
    out = []
    for im in images:
        p = np.array(cached[str(im['id'])]).reshape(-1, 8)
        if not len(p):
            out.append(p); continue
        image = cv2.imread(str(SOURCE / im['file']))
        crops = [image[y0:y0 + s, x0:x0 + s] for x0, y0, s in
                 (crop_box([x1, y1, x2 - x1, y2 - y1], im['w'], im['h']) for x1, y1, x2, y2 in p[:, :4])]
        res = []
        for i in range(0, len(crops), 128):
            res += clf(crops[i:i + 128], k=1)
        cls = np.array([ACNE_CLASSES.index(t[0]['type']) for t in res]); prob = np.array([t[0]['probability'] for t in res])
        out.append(np.column_stack([p[:, :4], p[:, 7] * prob, cls, prob, p[:, 7]]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    val, test = load_split('val'), load_split('test')
    cache = {s: json.loads((PRED / f'pipeline_{s}.json').read_text()) for s in ('val', 'test')}
    res, stats, confs = {}, {}, {}
    for name, path in CLASSIFIERS.items():
        clf = TypeClassifier(path, device=a.device)
        vp, tp = reclassify(clf, val, cache['val']), reclassify(clf, test, cache['test'])
        vs = per_image_class_stats(vp, val, 0.5)
        confs[name] = choose_conf(vs)
        v = summarise(vs, conf=confs[name])
        stats[name] = per_image_class_stats(tp, test, 0.5)
        t = summarise(stats[name], conf=confs[name])
        _, pairs = agnostic(tp, test, confs[name])
        res[name] = dict(weights=str(path.relative_to(ROOT)), weights_sha256=digest(path), conf=confs[name],
                         val_map50=v['map'], val_macro_f1=v['macro_f1'], test_map50=t['map'], test_macro_f1=t['macro_f1'],
                         test_per_class_ap50={c: t['ap'][i] for i, c in enumerate(ACNE_CLASSES)},
                         test_type_accuracy_on_found=float((pairs[:, 0] == pairs[:, 1]).mean()))
        print(name, {k: round(x, 4) for k, x in res[name].items() if isinstance(x, float)}, flush=True)
    boot = bootstrap({'pipeline': stats['efficientnet_b0'], **{k: s for k, s in stats.items() if k != 'efficientnet_b0'}},
                     {'pipeline': confs['efficientnet_b0'], **{k: c for k, c in confs.items() if k != 'efficientnet_b0'}})
    write_json(OUT / 'classifiers_e2e.json', dict(detector=str(ROOT / 'models/phase1/best_acne_detector.pt'), results=res,
                                                  bootstrap_vs_efficientnet_b0=boot))
    print(json.dumps(boot, indent=1))


if __name__ == '__main__':
    main()
