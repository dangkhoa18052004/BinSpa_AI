"""5-fold CV check of the deployed robust classifier (EfficientNet-B0 cfg_a_robust) against cfg_a.

For every fold: trains cfg_a_robust on the fold's crops (cfg_a and the fold's 1-class detector det1 already
exist from cv_run.py), then evaluates the two-stage pipeline det1 -> classifier with each classifier on the
fold's test images: clean (reusing the cached det1 boxes of cv_evaluate.py) and under three phone-photo
degradations (blur, noise, JPEG q25; det1 re-run on the degraded image, crops cut from it). The operating
threshold of each classifier is its max macro-F1 on the fold's validation images (clean).
Rule fixed before any result (written to reports/phase2_cv/robust_rule.json first): the robust classifier stays
deployed if, pooled over the 275 test images,
  (a) clean macro-F1 >= cfg_a - 0.01, and
  (b) mean mAP50 over the three degradations > cfg_a;
otherwise the pipeline goes back to models/phase2/best_acne_type.pth (cfg_a).
Usage: venv/Scripts/python.exe -u scripts/phase2/cv_robust.py
"""
import json
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

from common2 import ACNE_CLASSES, REPORTS, ROOT, SOURCE, write_json
from compare_det5 import STRESS, choose_conf, load_detector, per_image_class_stats, summarise
from cv_evaluate import fold_images
from make_crops import crop_box
from predict_acne_type import TypeClassifier

OUT = REPORTS / 'phase2_cv'
RUNS = ROOT / 'runs/phase2_cv'
K = 5
CLFS = ('efficientnet_b0_cfg_a', 'efficientnet_b0_cfg_a_robust')
DEGRADE = ('blur', 'noise_s8', 'jpeg_q25')
RULE = ('robust stays if pooled clean macro-F1 >= cfg_a - 0.01 AND mean pooled mAP50 over blur, noise_s8, '
        'jpeg_q25 > cfg_a; otherwise revert to models/phase2/best_acne_type.pth')


def classify(clf, image, boxes):
    """boxes (n, >=5): x1 y1 x2 y2 detconf -> (n, 8) rows as in compare_det5."""
    if not len(boxes):
        return np.zeros((0, 8))
    h, w = image.shape[:2]
    crops = [image[y0:y0 + s, x0:x0 + s] for x0, y0, s in
             (crop_box([x1, y1, x2 - x1, y2 - y1], w, h) for x1, y1, x2, y2 in boxes[:, :4])]
    res = []
    for i in range(0, len(crops), 256):
        res += clf(crops[i:i + 256], k=1)
    cls = np.array([ACNE_CLASSES.index(t[0]['type']) for t in res]); p = np.array([t[0]['probability'] for t in res])
    return np.column_stack([boxes[:, :4], boxes[:, 4] * p, cls, p, boxes[:, 4]])


def main():
    write_json(OUT / 'robust_rule.json', dict(rule=RULE, locked_utc=datetime.now(timezone.utc).isoformat()))
    for k in range(K):
        if not (RUNS / f'fold{k}/efficientnet_b0_cfg_a_robust/completed.json').exists():
            rc = subprocess.call([sys.executable, '-u', 'scripts/phase2/train_classifier.py', '--model', 'efficientnet_b0',
                                  '--config', 'cfg_a_robust', '--fold', str(k)], cwd=ROOT)
            if rc:
                sys.exit(f'fold {k} robust training failed')
        print('fold', k, 'robust trained', flush=True)

    pooled = {c: {cond: [] for cond in ('clean',) + DEGRADE} for c in CLFS}
    pooled_all = {c: [] for c in CLFS}
    per_fold = []
    for k in range(K):
        val, test = fold_images(k, 'val'), fold_images(k, 'test')
        cached = {s: json.loads((OUT / f'predictions/fold{k}_pipeline_{s}.json').read_text()) for s in ('val', 'test')}
        boxes = {s: {im['id']: np.array(cached[s][str(im['id'])]).reshape(-1, 8)[:, [0, 1, 2, 3, 7]] for im in imgs}
                 for s, imgs in (('val', val), ('test', test))}
        det = load_detector(RUNS / f'fold{k}/det1/weights/best.pt')
        clfs = {c: TypeClassifier(RUNS / f'fold{k}/{c}/best.pth') for c in CLFS}
        preds = {c: {cond: [] for cond in ('val', 'clean') + DEGRADE} for c in CLFS}
        for split, imgs in (('val', val), ('test', test)):
            for im in imgs:
                image = cv2.imread(str(SOURCE / im['file']))
                for c in CLFS:
                    preds[c]['val' if split == 'val' else 'clean'].append(classify(clfs[c], image, boxes[split][im['id']]))
                if split == 'test':
                    for cond in DEGRADE:
                        bad = STRESS[cond](image)
                        r = det.predict(bad, imgsz=1280, conf=0.05, max_det=500, verbose=False)[0]
                        b = np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()]).reshape(-1, 5)
                        for c in CLFS:
                            preds[c][cond].append(classify(clfs[c], bad, b))
        row = dict(fold=k)
        for c in CLFS:
            conf = choose_conf(per_image_class_stats(preds[c]['val'], val, 0.5))
            row[f'{c}_conf'] = conf
            st = per_image_class_stats(preds[c]['clean'], test, 0.5)
            pooled_all[c] += st
            for cond in ('clean',) + DEGRADE:
                op = [p[p[:, 4] >= conf] for p in preds[c][cond]]
                s_all = st if cond == 'clean' else per_image_class_stats(preds[c][cond], test, 0.5)
                pooled[c][cond].append((s_all, per_image_class_stats(op, test, 0.5)))
                row[f'{c}_{cond}_map50'] = summarise(s_all)['map']
                row[f'{c}_{cond}_macro_f1'] = summarise(per_image_class_stats(op, test, 0.5), conf=0.0)['macro_f1']
        per_fold.append(row)
        print({x: (round(v, 4) if isinstance(v, float) else v) for x, v in row.items()}, flush=True)
        del det, clfs

    res = {}
    for c in CLFS:
        res[c] = {}
        for cond in ('clean',) + DEGRADE:
            s_all = sum((a for a, _ in pooled[c][cond]), []); s_op = sum((o for _, o in pooled[c][cond]), [])
            res[c][cond] = dict(map50=summarise(s_all)['map'], macro_f1=summarise(s_op, conf=0.0)['macro_f1'])
        res[c]['degraded_mean_map50'] = float(np.mean([res[c][d]['map50'] for d in DEGRADE]))
    base, rob = res['efficientnet_b0_cfg_a'], res['efficientnet_b0_cfg_a_robust']
    a = rob['clean']['macro_f1'] >= base['clean']['macro_f1'] - 0.01
    b = rob['degraded_mean_map50'] > base['degraded_mean_map50']
    verdict = dict(rule=RULE, condition_a=bool(a), condition_b=bool(b), keep_robust=bool(a and b))
    write_json(OUT / 'cv_robust.json', dict(per_fold=per_fold, pooled=res, verdict=verdict))
    for c in CLFS:
        print(c, json.dumps({k: (round(v, 4) if isinstance(v, float) else {x: round(y, 4) for x, y in v.items()}) for k, v in res[c].items()}))
    print('verdict', verdict)


if __name__ == '__main__':
    main()
