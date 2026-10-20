"""Fit and evaluate the input check (input_gate.py) of the two-stage pipeline.

Thresholds come ONLY from the ACNE-DET master val split (40 images), degraded on purpose:
  blur, noise, downscale, dark, bright, each at increasing levels. For every degraded photo the deployed pipeline
  runs (detector at its operating conf -> crop -> calibrated classifier) and is scored by type-aware F1
  (box IoU >= 0.5 AND top-1 type right; 2 TP / (predictions + lesions)), pooled per level.
  A level is unacceptable when its F1 < 0.8 x clean F1. Per family the threshold on the matching metric
  (blur -> sharpness, noise -> noise, downscale -> face width, dark -> brightness min, bright -> brightness max and
  clipped share) is the midpoint of that metric's median at the last acceptable and the first unacceptable level.
  Domain threshold: 95th percentile of the clean val photos' domain score; the score itself is fitted on the
  classifier features of the master TRAIN crops.
Evaluated with the thresholds fixed (nothing below is used for fitting):
  * ACNE-DET master test (45 images), clean and at every degradation level
  * ACNE04 test (176 images, phase-1 detector data, binary acne boxes)
  * DetectDataSkin-2 screened (1,275 images, other source; type accuracy of its labelled crops)
  * test_img selfies (no labels; decisions listed for the owner to review)
Rule (written to reports/input_gate/rule.json before any result): the quality check is deployed if
  (a) at most 10 % of the clean ACNE-DET test photos are rejected, and
  (b) pooled over the clean and degraded ACNE-DET test photos, type F1 of accepted photos > rejected photos.
  The domain check is deployed if (a') at most 10 % of clean test photos with >= 3 lesions are out of domain and
  (b') on DetectDataSkin-2, labelled-crop type accuracy of in-domain photos > out-of-domain photos.
Usage: venv/Scripts/python.exe -u scripts/phase2/evaluate_input_gate.py [--promote]
"""
import argparse
import csv
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from common2 import ACNE_CLASSES, CROPS, MODELS, REPORTS, ROOT, SOURCE, digest, read_json, write_json
from compare_det5 import greedy, load_detector, load_split
from input_gate import GATE, GATE_STATS, MIN_LESIONS, DomainScore, FaceTool, judge, measure
from make_crops import crop_box
from predict_acne_type import TypeClassifier, softmax

OUT = REPORTS / 'input_gate'
DROP = 0.8
FAMILIES = {  # mild -> severe
    'blur': [0.0005, 0.001, 0.002, 0.003, 0.004, 0.006],     # Gaussian sigma as a share of image width
    'noise': [4, 8, 12, 16, 24, 32],                          # Gaussian sigma, 8-bit levels
    'downscale': [0.5, 0.35, 0.25, 0.18, 0.125, 0.09],       # resize factor
    'dark': [0.75, 0.6, 0.45, 0.35, 0.25],                    # intensity gain
    'bright': [1.25, 1.5, 1.75, 2.0, 2.5],
}
METRIC = {'blur': ('sharpness', 'min'), 'noise': ('noise', 'max'), 'downscale': ('face_width_px', 'min'),
          'dark': ('brightness', 'min'), 'bright': ('brightness', 'max')}
RULE = dict(
    thresholds='master val only; level unacceptable if pooled type-F1 < 0.8 x clean; threshold = midpoint of the '
               'metric medians at the last acceptable and first unacceptable level (family not failing -> most '
               'severe level median); clipped_max from the bright family the same way; domain = 95th pct clean val',
    deploy_quality='(a) <= 10 % of clean ACNE-DET test photos rejected AND (b) pooled clean+degraded test type-F1 '
                   'accepted > rejected',
    deploy_domain="(a') <= 10 % of clean test photos with >= 3 lesions out of domain AND (b') DetectDataSkin-2 "
                  'labelled-crop accuracy in-domain > out-of-domain',
)


def degrade(img, family, level, seed):
    if family == 'blur':
        return cv2.GaussianBlur(img, (0, 0), level * img.shape[1]), 1.0
    if family == 'noise':
        n = np.random.default_rng(seed).normal(0, level, img.shape)
        return np.clip(img + n, 0, 255).astype(np.uint8), 1.0
    if family == 'downscale':
        return cv2.resize(img, None, fx=level, fy=level, interpolation=cv2.INTER_AREA), level
    return np.clip(img.astype(np.float32) * level, 0, 255).astype(np.uint8), 1.0


class Pipeline:
    def __init__(self):
        pmeta, dmeta = read_json(MODELS / 'pipeline_metadata.json'), read_json(ROOT / 'models/phase1/model_metadata.json')
        cal = pmeta['type_calibration']
        self.clf = TypeClassifier(ROOT / pmeta['classifier'], temperature=cal['temperature'])
        self.det = load_detector(ROOT / pmeta['detector'])
        self.conf, self.imgsz = float(dmeta['operating_conf']), int(dmeta['imgsz'])
        self.face = FaceTool()
        self.domain = DomainScore.load() if GATE_STATS.exists() else None
        self.versions = dict(classifier=pmeta['classifier_sha256'], detector=pmeta['detector_sha256'],
                             temperature=cal['temperature'])

    def __call__(self, img):
        """-> boxes (n, 4), predicted type (n,), calibrated top-1 prob (n,), photo domain score, quality metrics."""
        r = self.det.predict(img, imgsz=self.imgsz, conf=self.conf, max_det=500, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy().reshape(-1, 4)
        order = np.argsort(-r.boxes.conf.cpu().numpy())
        b = b[order]
        h, w = img.shape[:2]
        crops = [img[y0:y0 + s, x0:x0 + s] for x0, y0, s in (crop_box([x1, y1, x2 - x1, y2 - y1], w, h) for x1, y1, x2, y2 in b)]
        f, z = self.clf.embed(crops)
        p = softmax(z / self.clf.temperature)
        return b, p.argmax(1), p.max(1), self.domain.photo(f) if self.domain and len(f) else None, measure(img, self.face)


def score(boxes, types, gt):
    """gt (m, 5) xyxy cls (cls -1 = type unknown) -> counts for type-aware and location F1."""
    det = np.column_stack([boxes, np.zeros(len(boxes))])
    tp, idx = greedy(det, gt, 0.5)
    ok_type = np.array([tp[j] and gt[idx[j], 4] == types[j] for j in range(len(boxes))], bool)
    return dict(n_gt=int(len(gt)), n_pred=int(len(boxes)), tp_loc=int(tp.sum()), tp_type=int(ok_type.sum()))


def f1(rows, key='tp_type'):
    tp = sum(r[key] for r in rows); n = sum(r['n_pred'] + r['n_gt'] for r in rows)
    return 2 * tp / n if n else None


def run_rows(name, items, pipe, families=True):
    """items: (image id, path, gt (m, 5) in original pixels). Cached per set in reports/input_gate/rows_<name>.json."""
    path = OUT / f'rows_{name}.json'
    key = json.dumps(pipe.versions, sort_keys=True)
    if path.exists():
        d = json.loads(path.read_text(encoding='utf-8'))
        if d['key'] == key:
            return d['rows']
    rows = []
    for n, (iid, file, gt) in enumerate(items):
        img = cv2.imread(str(file))
        conds = [('clean', 1.0)] + ([(f, lv) for f, lvs in FAMILIES.items() for lv in lvs] if families else [])
        for fam, lv in conds:
            im, s = (img, 1.0) if fam == 'clean' else degrade(img, fam, lv, seed=n)
            g = gt.copy(); g[:, :4] *= s
            b, t, p, dom, m = pipe(im)
            rows.append(dict(image=str(iid), family=fam, level=lv, metrics=m, domain=dom, n_lesions=int(len(b)),
                             **score(b, t, g)))
        if n % 10 == 0:
            print(name, n + 1, '/', len(items), flush=True)
    write_json(path, dict(key=key, rows=rows))
    return rows


def fit_domain(clf):
    with (CROPS / 'crop_manifest.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'train']
    feats, _ = clf.embed([cv2.imread(str(CROPS / r['crop_path'])) for r in rows])
    y = np.array([ACNE_CLASSES.index(r['cls']) for r in rows])
    ds = DomainScore.fit(feats, y, len(ACNE_CLASSES))
    ds.save()
    return len(rows)


def fit_thresholds(val):
    clean = [r for r in val if r['family'] == 'clean']
    f0 = f1(clean)
    med = lambda rows, k: float(np.median([r['metrics'][k] for r in rows if r['metrics'].get('face')])) if any(r['metrics'].get('face') for r in rows) else None
    curve, thr = {}, {}
    for fam, levels in FAMILIES.items():
        metric, side = METRIC[fam]
        pts = [('clean', f0, med(clean, metric), med(clean, 'clipped'))]
        for lv in levels:
            rows = [r for r in val if r['family'] == fam and r['level'] == lv]
            pts.append((lv, f1(rows), med(rows, metric), med(rows, 'clipped')))
        curve[fam] = [dict(level=lv, type_f1=f, **{metric: v, 'clipped': c}) for lv, f, v, c in pts]
        bad = next((i for i, (_, f, v, _) in enumerate(pts) if i and (f < DROP * f0 or v is None)), None)
        if bad is None:
            thr_v, thr_c, reached = pts[-1][2], pts[-1][3], False
        else:
            prev, cur = pts[bad - 1], pts[bad]
            thr_v = prev[2] if cur[2] is None else (prev[2] + cur[2]) / 2
            thr_c = prev[3] if cur[3] is None else (prev[3] + cur[3]) / 2
            reached = True
        thr[f'{metric}_{side}'] = float(thr_v)
        if fam == 'bright':
            thr['clipped_max'] = float(thr_c)
        curve[fam].append(dict(threshold=thr_v, failure_reached=reached, first_unacceptable=pts[bad][0] if bad else None))
    doms = [r['domain'] for r in clean if r['domain'] is not None]
    thr['domain_max'] = float(np.percentile(doms, 95))
    return thr, dict(clean_type_f1=f0, curves=curve, domain_clean_val=dict(n=len(doms), values=sorted(doms)))


def decide(r, thr):
    q = judge(r['metrics'], thr)
    d = None if r['domain'] is None else r['domain'] > thr['domain_max']
    return q, d


def summarise_acnedet(test, thr):
    out = dict(per_condition=[])
    for fam, lv in [('clean', 1.0)] + [(f, lv) for f, lvs in FAMILIES.items() for lv in lvs]:
        rows = [r for r in test if r['family'] == fam and r['level'] == lv]
        acc = [r for r in rows if not decide(r, thr)[0]]
        out['per_condition'].append(dict(family=fam, level=lv, n=len(rows), rejected=1 - len(acc) / len(rows),
                                         type_f1_all=f1(rows), type_f1_accepted=f1(acc)))
    clean = [r for r in test if r['family'] == 'clean']
    reasons = {}
    for r in test:
        for q in decide(r, thr)[0]:
            reasons[q] = reasons.get(q, 0) + 1
    acc = [r for r in test if not decide(r, thr)[0]]; rej = [r for r in test if decide(r, thr)[0]]
    dom_clean = [decide(r, thr)[1] for r in clean if r['domain'] is not None]
    out.update(clean_rejected=sum(bool(decide(r, thr)[0]) for r in clean) / len(clean),
               clean_rejected_files=[r['image'] for r in clean if decide(r, thr)[0]],
               pooled=dict(accepted=len(acc), rejected=len(rej), type_f1_accepted=f1(acc), type_f1_rejected=f1(rej)),
               reasons=reasons, clean_out_of_domain=float(np.mean(dom_clean)) if dom_clean else None)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--promote', action='store_true', help='write the verdict-approved thresholds to models/phase2/input_gate.json')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rule_path = OUT / 'rule.json'
    if rule_path.exists():
        if json.loads(rule_path.read_text(encoding='utf-8'))['rule'] != RULE:
            sys.exit('rule.json differs from RULE: the rule is fixed once written')
    else:
        write_json(rule_path, dict(rule=RULE, families=FAMILIES, locked_utc=datetime.now(timezone.utc).isoformat()))

    pipe = Pipeline()
    if not GATE_STATS.exists():
        print('domain score fitted on', fit_domain(pipe.clf), 'train crops', flush=True)
        pipe.domain = DomainScore.load()
    pipe.versions['domain_stats'] = digest(GATE_STATS)

    master = lambda split: [(im['id'], SOURCE / im['file'], im['gt']) for im in load_split(split)]
    val = run_rows('acnedet_val', master('val'), pipe)
    thr, fit = fit_thresholds(val)
    print('thresholds', json.dumps(thr), flush=True)
    test = run_rows('acnedet_test', master('test'), pipe)

    a04 = []
    for img in sorted(glob.glob(str(ROOT / 'datasets/phase1_acne_detection/images/test/*.jpg'))):
        h, w = cv2.imread(img).shape[:2]
        lab = ROOT / 'datasets/phase1_acne_detection/labels/test' / (Path(img).stem + '.txt')
        y = np.loadtxt(lab, ndmin=2) if lab.exists() and lab.stat().st_size else np.zeros((0, 5))
        g = np.column_stack([(y[:, 1] - y[:, 3] / 2) * w, (y[:, 2] - y[:, 4] / 2) * h, (y[:, 1] + y[:, 3] / 2) * w,
                             (y[:, 2] + y[:, 4] / 2) * h, -np.ones(len(y))])
        a04.append((img, img, g))
    acne04 = run_rows('acne04_test', a04, pipe, families=False)

    sys.path.insert(0, str(ROOT / 'scripts'))
    from audit_detectdataskin2 import ACNE, OUT as DDS, TYPE_MAP, read
    from evaluate_detectdataskin2 import cut
    dds_items, dds_types = [], {}
    for row in read(DDS / 'image_manifest.json'):
        if row['exclude_reasons']:
            continue
        acne = [a for a in row['annotations'] if a['category'] in ACNE]
        g = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3],
                       ACNE_CLASSES.index(TYPE_MAP[a['category']]) if a['category'] in TYPE_MAP else -1] for a in acne]).reshape(-1, 5)
        dds_items.append((row['uid'], row['path'], g))
        dds_types[row['uid']] = (row['path'], [a for a in row['annotations'] if a['category'] in TYPE_MAP])
    dds2 = run_rows('dds2', dds_items, pipe, families=False)
    crop_path = OUT / 'dds2_crop_accuracy.json'
    if crop_path.exists():
        crop_acc = json.loads(crop_path.read_text(encoding='utf-8'))
    else:
        crop_acc = {}
        for uid, (path, ann) in dds_types.items():
            if ann:
                img = cv2.imread(path)
                z = pipe.clf.logits([cut(img, a['bbox']) for a in ann])
                y = [ACNE_CLASSES.index(TYPE_MAP[a['category']]) for a in ann]
                crop_acc[uid] = [len(y), int((z.argmax(1) == np.array(y)).sum())]
        write_json(crop_path, crop_acc)

    selfies = run_rows('selfies', [(f, f, np.zeros((0, 5))) for f in sorted(glob.glob(str(ROOT / 'test_img/*.jpg')))], pipe, families=False)

    # ---------------- results
    res = dict(rule=RULE, thresholds=thr, fit=fit, acnedet_test=summarise_acnedet(test, thr))
    acc = [r for r in acne04 if not decide(r, thr)[0]]; rej = [r for r in acne04 if decide(r, thr)[0]]
    reasons = {}
    for r in acne04:
        for q in decide(r, thr)[0]:
            reasons[q] = reasons.get(q, 0) + 1
    res['acne04_test'] = dict(n=len(acne04), rejected=len(rej) / len(acne04), reasons=reasons,
                              location_f1_accepted=f1(acc, 'tp_loc'), location_f1_rejected=f1(rej, 'tp_loc'),
                              out_of_domain=float(np.mean([decide(r, thr)[1] for r in acne04 if r['domain'] is not None])))
    reasons = {}
    for r in dds2:
        for q in decide(r, thr)[0]:
            reasons[q] = reasons.get(q, 0) + 1
    def crop_accuracy(rows):
        c = [crop_acc[r['image']] for r in rows if r['image'] in crop_acc]
        return (sum(x[1] for x in c) / sum(x[0] for x in c), sum(x[0] for x in c)) if c else (None, 0)
    qa = [r for r in dds2 if not decide(r, thr)[0]]; qr = [r for r in dds2 if decide(r, thr)[0]]
    di = [r for r in dds2 if decide(r, thr)[1] is False]; do = [r for r in dds2 if decide(r, thr)[1] is True]
    both = [r for r in dds2 if not decide(r, thr)[0] and decide(r, thr)[1] is not True]
    res['dds2'] = dict(n=len(dds2), quality_rejected=len(qr) / len(dds2), reasons=reasons,
                       with_domain_score=len(di) + len(do), out_of_domain=len(do) / max(len(di) + len(do), 1),
                       crop_accuracy=dict(all=crop_accuracy(dds2), quality_accepted=crop_accuracy(qa), quality_rejected=crop_accuracy(qr),
                                          in_domain=crop_accuracy(di), out_of_domain=crop_accuracy(do), passes_both=crop_accuracy(both)),
                       location_f1=dict(all=f1(dds2, 'tp_loc'), quality_accepted=f1(qa, 'tp_loc'), quality_rejected=f1(qr, 'tp_loc')),
                       passes_both=len(both) / len(dds2))
    res['selfies'] = [dict(image=Path(r['image']).name, reasons=decide(r, thr)[0], out_of_domain=decide(r, thr)[1],
                           domain=r['domain'], lesions=r['n_lesions'],
                           **{k: v for k, v in r['metrics'].items() if k != 'face'}) for r in selfies]
    t = res['acnedet_test']
    verdict = dict(quality_a=t['clean_rejected'] <= 0.10,
                   quality_b=(t['pooled']['type_f1_rejected'] or 0) < (t['pooled']['type_f1_accepted'] or 0),
                   domain_a=t['clean_out_of_domain'] is not None and t['clean_out_of_domain'] <= 0.10,
                   domain_b=(res['dds2']['crop_accuracy']['in_domain'][0] or 0) > (res['dds2']['crop_accuracy']['out_of_domain'][0] or 1))
    verdict.update(deploy_quality=verdict['quality_a'] and verdict['quality_b'], deploy_domain=verdict['domain_a'] and verdict['domain_b'])
    res['verdict'] = verdict
    write_json(OUT / 'input_gate_results.json', res)
    print(json.dumps(dict(thresholds=thr, clean_val_f1=fit['clean_type_f1'], acnedet_test={k: v for k, v in t.items() if k != 'per_condition'},
                          acne04=res['acne04_test'], dds2=res['dds2'], verdict=verdict), indent=1, ensure_ascii=False))
    if args.promote:
        gate = dict(thresholds=thr, quality_check=verdict['deploy_quality'], domain_check=verdict['deploy_domain'],
                    domain_stats=str(GATE_STATS.relative_to(ROOT)), domain_stats_sha256=digest(GATE_STATS),
                    classifier_sha256=pipe.versions['classifier'], verdict=verdict, rule=RULE,
                    report='reports/input_gate_summary.md', promoted_utc=datetime.now(timezone.utc).isoformat())
        write_json(GATE, gate)
        print('promoted', GATE)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
