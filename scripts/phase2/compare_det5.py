"""End-to-end comparison: current two-stage pipeline vs two one-stage 5-class detectors on ACNE-DET.

Systems (each turns a full-resolution face image into boxes with an acne type and a score):
  pipeline      deployed phase-1 detector (RT-DETR-L @1280 + ACNE-DET, 1 class, run at conf 0.05) ->
                square context crop -> EfficientNet-B0; score = detector conf x classifier probability
  rtdetr_det5   RT-DETR-L @1280 fine-tuned on 5 classes  (scripts/phase2/train_det5.py)
  yolo26s_det5  YOLO26s  @1280 fine-tuned on 5 classes
  pipeline_deployed (report only) the pipeline exactly as shipped: detector at its operating conf, every
                box kept, class = classifier top-1.
Metrics, ground truth = ACNE-DET boxes of the 5 acne classes, master split:
  * class-aware AP per class and mAP at IoU 0.5 (main) and 0.3 (sensitivity; lesions are tiny)
  * one operating threshold per system = max val macro-F1 @IoU0.5; at it: per-class P/R/F1, macro-F1,
    6x6 confusion matrix (class-agnostic matching, background row/column), type accuracy on found lesions,
    recall by lesion size
  * paired bootstrap over test images (1000x): mAP50 and macro-F1 CIs and differences vs pipeline
  * calibration: ECE of the box score (P(correct box and type)) before/after Platt scaling fitted on val;
    for the classifier also temperature scaling on annotated val crops -> test-crop ECE
  * selective prediction: type accuracy of found lesions when keeping the most confident 100..50 %
  * stress test on test images: darker, brighter, blur, low resolution, JPEG q25, warm white balance, noise
  * label review list: confident disagreements on VAL (never test) for a human to check
Selection rule, fixed before the test split is read and written to selection.json first:
  the candidate with the highest val mAP50 replaces the pipeline only if it beats the pipeline's val mAP50
  by >= 0.01 AND its val macro-F1 is not lower; otherwise the pipeline stays. Nothing is installed.
Usage: venv/Scripts/python.exe -u scripts/phase2/compare_det5.py
"""
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np
import torch

from common2 import ACNE_CLASSES, MASTER_SPLIT, REPORTS, ROOT, SOURCE, digest, load_coco, read_json, write_csv, write_json
from make_crops import crop_box
from predict_acne_type import TypeClassifier

OUT = REPORTS / 'phase2_det5'
PRED = OUT / 'predictions'
NC = len(ACNE_CLASSES)
DET1 = ROOT / 'models/phase1/best_acne_detector.pt'
DET1_CONF = read_json(ROOT / 'models/phase1/model_metadata.json')['operating_conf']
CANDIDATES = {'rtdetr_det5': ROOT / 'runs/phase2_det5/rtdetr_1280_det5/weights/best.pt',
              'yolo26s_det5': ROOT / 'runs/phase2_det5/yolo26s_1280_det5/weights/best.pt'}
SYSTEMS = ['pipeline', *CANDIDATES]
GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
SIZE_BINS = [(0, 24), (24, 48), (48, 96), (96, 1e9)]
STRESS = {
    'darker_0.7': lambda im: np.clip(im * 0.7, 0, 255).astype(np.uint8),
    'brighter_1.3': lambda im: np.clip(im * 1.3, 0, 255).astype(np.uint8),
    'blur': lambda im: cv2.GaussianBlur(im, (0, 0), im.shape[1] * 0.002),
    'low_res_x0.25': lambda im: cv2.resize(cv2.resize(im, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA),
                                           (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR),
    'jpeg_q25': lambda im: cv2.imdecode(cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 25])[1], cv2.IMREAD_COLOR),
    'warm_wb': lambda im: np.clip(im * np.array([0.85, 1.0, 1.12]), 0, 255).astype(np.uint8),
    'noise_s8': lambda im: np.clip(im + np.random.default_rng(0).normal(0, 8, im.shape), 0, 255).astype(np.uint8),
}


# ---------------------------------------------------------------- data and inference
def load_split(split):
    data, cats = load_coco()
    with MASTER_SPLIT.open(encoding='utf-8-sig') as f:
        ids = {int(r['image_id']) for r in csv.DictReader(f) if r['split'] == split}
    gts = defaultdict(list)
    for a in data['annotations']:
        if cats[a['category_id']] in ACNE_CLASSES:
            x, y, w, h = a['bbox']
            gts[a['image_id']].append([x, y, x + w, y + h, ACNE_CLASSES.index(cats[a['category_id']])])
    images = sorted((i for i in data['images'] if i['id'] in ids), key=lambda i: i['id'])
    return [dict(id=i['id'], file=i['file_name'], w=i['width'], h=i['height'],
                 gt=np.array(gts[i['id']], dtype=float).reshape(-1, 5)) for i in images]


def load_detector(weights):
    from ultralytics import RTDETR, YOLO
    ckpt = torch.load(str(weights), map_location='cpu', weights_only=False)
    detr = 'RTDETR' in type(ckpt.get('model') or ckpt.get('ema')).__name__
    del ckpt
    return (RTDETR if detr else YOLO)(str(weights))


class System:

    def __init__(self, name, classifier=ROOT / 'models/phase2/best_acne_type.pth', detector=None):
        self.name = name
        if name.startswith('pipeline'):
            self.det, self.clf = load_detector(detector or DET1), TypeClassifier(classifier)
        else:
            self.det, self.clf = load_detector(detector or CANDIDATES[name]), None

    def predict(self, image):
        """-> (n, 8): x1 y1 x2 y2 score cls aux detconf. aux = classifier prob (pipeline) or score."""
        if self.clf is None:
            b = self.det.predict(image, imgsz=1280, conf=0.01, max_det=500, verbose=False)[0].boxes
            s = b.conf.cpu().numpy()
            return np.column_stack([b.xyxy.cpu().numpy(), s, b.cls.cpu().numpy(), s, s]).reshape(-1, 8)
        h, w = image.shape[:2]
        r = self.det.predict(image, imgsz=1280, conf=0.05, max_det=500, verbose=False)[0]
        boxes, dconf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        crops = [image[y0:y0 + s, x0:x0 + s] for x0, y0, s in
                 (crop_box([x1, y1, x2 - x1, y2 - y1], w, h) for x1, y1, x2, y2 in boxes)]
        out = []
        for i in range(0, len(crops), 256):
            out += self.clf(crops[i:i + 256], k=1)
        if not out:
            return np.zeros((0, 8))
        cls = np.array([ACNE_CLASSES.index(t[0]['type']) for t in out])
        p = np.array([t[0]['probability'] for t in out])
        return np.column_stack([boxes, dconf * p, cls, p, dconf])


def run(system, images, transform=None, cache=None):
    if cache and cache.exists():
        d = json.loads(cache.read_text())
        return [np.array(d[str(im['id'])]).reshape(-1, 8) for im in images], d['_ms_per_image']
    preds, t0 = [], 0.0
    for k, im in enumerate(images):
        image = cv2.imread(str(SOURCE / im['file']))
        if transform:
            image = transform(image)
        torch.cuda.synchronize(); t = time.perf_counter()
        p = system.predict(image)
        torch.cuda.synchronize()
        if k:  # first image = warm-up
            t0 += time.perf_counter() - t
        preds.append(p)
    ms = t0 / max(len(images) - 1, 1) * 1000
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({**{str(im['id']): p.round(4).tolist() for im, p in zip(images, preds)}, '_ms_per_image': ms}))
    return preds, ms


# ---------------------------------------------------------------- matching and metrics
def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def greedy(det, gt, thr):
    """det sorted by score desc; returns (tp flag per det, matched gt index or -1)."""
    o = iou_matrix(det[:, :4], gt[:, :4])
    used, tp, idx = np.zeros(len(gt), bool), np.zeros(len(det), bool), -np.ones(len(det), int)
    for j in range(len(det)):
        if not len(gt):
            break
        row = np.where(used, -1, o[j])
        k = int(row.argmax())
        if row[k] >= thr:
            used[k] = True; tp[j] = True; idx[j] = k
    return tp, idx


def per_image_class_stats(preds, images, thr):
    """For every image and class: scores and TP flags of that class's detections, and the GT count."""
    stats = []
    for p, im in zip(preds, images):
        s = {}
        for c in range(NC):
            d = p[p[:, 5] == c]; d = d[np.argsort(-d[:, 4])]
            g = im['gt'][im['gt'][:, 4] == c]
            tp, idx = greedy(d, g, thr)
            s[c] = dict(score=d[:, 4], tp=tp, n_gt=len(g), gt_size=np.maximum(g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]),
                        hit_gt=idx)
        stats.append(s)
    return stats


def ap(score, tp, n_gt):
    if n_gt == 0:
        return np.nan
    if not len(score):
        return 0.0
    o = np.argsort(-score, kind='stable')
    ctp = np.cumsum(tp[o]); cfp = np.cumsum(~tp[o])
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    r = np.linspace(0, 1, 101)  # COCO 101-point interpolation
    q = np.searchsorted(rec, r, side='left')
    return float(np.mean([prec[i] if i < len(prec) else 0 for i in q]))


def summarise(stats, idx=None, conf=None):
    idx = range(len(stats)) if idx is None else idx
    out = dict(ap={}, p={}, r={}, f1={}, tp={}, fp={}, fn={})
    for c in range(NC):
        sc = np.concatenate([stats[i][c]['score'] for i in idx]); tp = np.concatenate([stats[i][c]['tp'] for i in idx])
        n = int(sum(stats[i][c]['n_gt'] for i in idx))
        out['ap'][c] = ap(sc, tp, n)
        if conf is not None:
            keep = sc >= conf
            t, f = int(tp[keep].sum()), int((~tp[keep]).sum())
            out['tp'][c], out['fp'][c], out['fn'][c] = t, f, n - t
            out['p'][c] = t / max(t + f, 1); out['r'][c] = t / max(n, 1)
            out['f1'][c] = 2 * t / (2 * t + f + n - t) if n + f else np.nan  # class absent and never predicted: undefined
    out['map'] = float(np.nanmean(list(out['ap'].values())))
    if conf is not None:
        out['macro_f1'] = float(np.nanmean(list(out['f1'].values())))
    return out


def choose_conf(stats):
    f = [summarise(stats, conf=c)['macro_f1'] for c in GRID]
    return float(GRID[int(np.argmax(f))])


def agnostic(preds, images, conf, thr=0.5):
    """Class-agnostic matching at the operating point -> confusion matrix and found-lesion pairs."""
    cm = np.zeros((NC + 1, NC + 1), int)  # rows true (last = background), cols predicted (last = missed)
    pairs = []  # (true cls, pred cls, score, aux, gt size)
    for p, im in zip(preds, images):
        d = p[p[:, 4] >= conf]; d = d[np.argsort(-d[:, 4])]
        g = im['gt']
        tp, idx = greedy(d, g, thr)
        for j in range(len(d)):
            if tp[j]:
                k = idx[j]; t = int(g[k, 4]); cm[t, int(d[j, 5])] += 1
                pairs.append((t, int(d[j, 5]), d[j, 4], d[j, 6], max(g[k, 2] - g[k, 0], g[k, 3] - g[k, 1])))
            else:
                cm[NC, int(d[j, 5])] += 1
        for k in set(range(len(g))) - set(idx[tp].tolist()):
            cm[int(g[k, 4]), NC] += 1
    return cm, np.array(pairs).reshape(-1, 5)


def size_recall(stats, conf):
    out = {}
    for lo, hi in SIZE_BINS:
        hit = n = 0
        for s in stats:
            for c in range(NC):
                sz = s[c]['gt_size']; m = (sz >= lo) & (sz < hi)
                n += int(m.sum())
                found = s[c]['hit_gt'][s[c]['tp'] & (s[c]['score'] >= conf)]
                hit += int(m[found].sum()) if len(found) else 0
        out[f'{int(lo)}-{int(hi) if hi < 1e8 else "inf"}px'] = dict(n=n, recall=hit / max(n, 1))
    return out


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if not len(conf):
        return np.nan
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf >= lo) & (conf < lo + 1 / bins) if lo < 1 - 1 / bins else (conf >= lo)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def box_correctness(stats, floor=0.05):
    sc, ok = [], []
    for s in stats:
        for c in range(NC):
            m = s[c]['score'] >= floor
            sc.append(s[c]['score'][m]); ok.append(s[c]['tp'][m])
    return np.concatenate(sc), np.concatenate(ok)


def platt(score_val, ok_val):
    from sklearn.linear_model import LogisticRegression
    z = lambda s: np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6).fit(z(score_val), ok_val.astype(int))
    return lambda s: lr.predict_proba(z(s))[:, 1], dict(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))


def classifier_temperature():
    """Temperature scaling of the deployed classifier on annotated val crops; ECE on annotated test crops."""
    from scipy.optimize import minimize_scalar
    clf = TypeClassifier()
    root = ROOT / 'datasets/phase2_acne_type'
    res = {}
    for split in ('val', 'test'):
        paths = sorted(root.glob(f'{split}/*/*.jpg'))
        y = np.array([ACNE_CLASSES.index(p.parent.name) for p in paths])
        probs = []
        for i in range(0, len(paths), 256):
            crops = [cv2.imread(str(p)) for p in paths[i:i + 256]]
            probs += [[dict((t['type'], t['probability']) for t in o).get(c, 0) for c in ACNE_CLASSES] for o in clf(crops, k=NC)]
        res[split] = (np.log(np.clip(np.array(probs), 1e-7, 1)), y)
    lv, yv = res['val']
    nll = lambda T: -np.mean(torch.log_softmax(torch.tensor(lv / T), 1).numpy()[np.arange(len(yv)), yv])
    T = float(minimize_scalar(nll, bounds=(0.25, 5), method='bounded').x)
    lt, yt = res['test']
    out = dict(temperature=T, n_val=len(yv), n_test=len(yt))
    for name, t in (('before', 1.0), ('after', T)):
        p = torch.softmax(torch.tensor(lt / t), 1).numpy()
        out[f'test_ece_{name}'] = ece(p.max(1), p.argmax(1) == yt)
        out[f'test_mean_conf_{name}'] = float(p.max(1).mean())
    out['test_accuracy'] = float((lt.argmax(1) == yt).mean())
    return out


def selective(pairs):
    if not len(pairs):
        return {}
    o = np.argsort(-pairs[:, 3])
    ok = (pairs[o, 0] == pairs[o, 1])
    return {f'{int(c * 100)}%': float(ok[:max(1, int(round(len(ok) * c)))].mean()) for c in (1, .9, .8, .7, .6, .5)}


def bootstrap(stats_by, confs, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(next(iter(stats_by.values())))
    draws = {k: dict(map=[], f1=[]) for k in stats_by}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, st in stats_by.items():
            s = summarise(st, idx, confs[k])
            draws[k]['map'].append(s['map']); draws[k]['f1'].append(s['macro_f1'])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {k: dict(map50_ci=ci(d['map']), macro_f1_ci=ci(d['f1'])) for k, d in draws.items()}
    for k in stats_by:
        if k != 'pipeline':
            dm = np.array(draws[k]['map']) - draws['pipeline']['map']
            df = np.array(draws[k]['f1']) - draws['pipeline']['f1']
            out[k]['diff_vs_pipeline'] = dict(map50=float(dm.mean()), map50_ci=ci(dm), macro_f1=float(df.mean()), macro_f1_ci=ci(df))
    return out


def plot_cm(cm, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ACNE_CLASSES + ['background']
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(NC + 1):
        for j in range(NC + 1):
            ax.text(j, i, f'{cm[i, j]}\n{norm[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if norm[i, j] > .6 else 'black')
    ax.set_xticks(range(NC + 1), ACNE_CLASSES + ['missed'], rotation=30, ha='right')
    ax.set_yticks(range(NC + 1), labels)
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def label_review(preds, images, conf, name):
    """Confident disagreements on VAL: wrong type with score >= 0.6, or a box with no lesion at score >= 0.7."""
    rows = []
    for p, im in zip(preds, images):
        d = p[p[:, 4] >= max(conf, 0.6)]; d = d[np.argsort(-d[:, 4])]
        tp, idx = greedy(d, im['gt'], 0.3)
        for j in range(len(d)):
            if tp[j] and int(im['gt'][idx[j], 4]) != int(d[j, 5]):
                kind, true = 'type_disagreement', ACNE_CLASSES[int(im['gt'][idx[j], 4])]
            elif not tp[j] and d[j, 4] >= 0.7:
                kind, true = 'no_annotated_lesion', ''
            else:
                continue
            rows.append(dict(system=name, image=im['file'], kind=kind, annotated=true, predicted=ACNE_CLASSES[int(d[j, 5])],
                             score=round(float(d[j, 4]), 3), x1=int(d[j, 0]), y1=int(d[j, 1]), x2=int(d[j, 2]), y2=int(d[j, 3])))
    return sorted(rows, key=lambda r: -r['score'])


def pct(x):
    return f'{x:.3f}' if isinstance(x, float) and not np.isnan(x) else str(x)


# ---------------------------------------------------------------- main
def main():
    for w in CANDIDATES.values():
        if not w.exists():
            raise RuntimeError(f'missing {w}; train it with scripts/phase2/train_det5.py first')
    OUT.mkdir(parents=True, exist_ok=True)
    val, test = load_split('val'), load_split('test')
    systems = {k: System(k) for k in SYSTEMS}
    weights = {'pipeline': dict(detector=digest(DET1), classifier=digest(ROOT / 'models/phase2/best_acne_type.pth')),
               **{k: dict(detector=digest(w)) for k, w in CANDIDATES.items()}}

    # ---- validation: operating points and the locked choice
    vp, vs, vsum, confs = {}, {}, {}, {}
    for k, s in systems.items():
        vp[k], _ = run(s, val, cache=PRED / f'{k}_val.json')
        vs[k] = per_image_class_stats(vp[k], val, 0.5)
        confs[k] = choose_conf(vs[k])
        vsum[k] = summarise(vs[k], conf=confs[k])
        print('val', k, 'conf', confs[k], 'mAP50', round(vsum[k]['map'], 4), 'macroF1', round(vsum[k]['macro_f1'], 4), flush=True)
    best = max(CANDIDATES, key=lambda k: vsum[k]['map'])
    replace = vsum[best]['map'] - vsum['pipeline']['map'] >= 0.01 and vsum[best]['macro_f1'] >= vsum['pipeline']['macro_f1']
    selection = dict(rule='best one-stage candidate by val mAP50 replaces the pipeline only if val mAP50 >= pipeline + 0.01 '
                          'and val macro-F1 >= pipeline; operating conf = max val macro-F1 @IoU0.5',
                     val={k: dict(conf=confs[k], map50=vsum[k]['map'], macro_f1=vsum[k]['macro_f1']) for k in SYSTEMS},
                     best_candidate=best, chosen=best if replace else 'pipeline', weights_sha256=weights,
                     locked_before_test=True, locked_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT / 'selection.json', selection)
    print('selection', selection['chosen'], flush=True)

    # ---- test
    tp_, ts, res, ms = {}, {}, {}, {}
    for k, s in systems.items():
        tp_[k], ms[k] = run(s, test, cache=PRED / f'{k}_test.json')
        ts[k] = per_image_class_stats(tp_[k], test, 0.5)
        s50 = summarise(ts[k], conf=confs[k])
        s30 = summarise(per_image_class_stats(tp_[k], test, 0.3), conf=confs[k])
        cm, pairs = agnostic(tp_[k], test, confs[k])
        plot_cm(cm, f'{k} — test, conf {confs[k]}', OUT / f'confusion_{k}.png')
        res[k] = dict(conf=confs[k], map50=s50['map'], map30=s30['map'], macro_f1=s50['macro_f1'], macro_f1_iou30=s30['macro_f1'],
                      per_class={c: dict(ap50=s50['ap'][i], precision=s50['p'][i], recall=s50['r'][i], f1=s50['f1'][i],
                                         tp=s50['tp'][i], fp=s50['fp'][i], fn=s50['fn'][i]) for i, c in enumerate(ACNE_CLASSES)},
                      found_lesions=int(len(pairs)), type_accuracy_on_found=float((pairs[:, 0] == pairs[:, 1]).mean()) if len(pairs) else None,
                      confusion_matrix=cm.tolist(), recall_by_size=size_recall(ts[k], confs[k]),
                      selective_type_accuracy=selective(pairs), ms_per_image=ms[k])
        print('test', k, {x: round(res[k][x], 4) for x in ('map50', 'map30', 'macro_f1')}, flush=True)

    # pipeline exactly as deployed (detector operating conf, no combined-score threshold)
    dep = [p[p[:, 7] >= DET1_CONF] for p in tp_['pipeline']]
    dep_s = per_image_class_stats(dep, test, 0.5)
    d = summarise(dep_s, conf=0.0)
    cm, pairs = agnostic(dep, test, 0.0)
    res['pipeline_deployed'] = dict(detector_conf=DET1_CONF, macro_f1=d['macro_f1'],
                                    per_class={c: dict(precision=d['p'][i], recall=d['r'][i], f1=d['f1'][i]) for i, c in enumerate(ACNE_CLASSES)},
                                    found_lesions=int(len(pairs)), type_accuracy_on_found=float((pairs[:, 0] == pairs[:, 1]).mean()))

    # ---- bootstrap, calibration
    boot = bootstrap(ts, confs)
    calib = {}
    for k in SYSTEMS:
        sv, ov = box_correctness(vs[k]); st, ot = box_correctness(ts[k])
        f, params = platt(sv, ov)
        calib[k] = dict(score='P(box and type correct @IoU0.5)', platt=params, n_test=int(len(st)),
                        test_ece_before=ece(st, ot), test_ece_after=ece(f(st), ot),
                        test_mean_score=float(st.mean()), test_precision=float(ot.mean()))
    calib['classifier_temperature'] = classifier_temperature()

    # ---- stress test (test images, val operating points)
    stress = {}
    for name, fn in STRESS.items():
        stress[name] = {}
        for k, s in systems.items():
            p, _ = run(s, test, transform=fn)
            st = summarise(per_image_class_stats(p, test, 0.5), conf=confs[k])
            stress[name][k] = dict(map50=st['map'], macro_f1=st['macro_f1'], map50_drop=res[k]['map50'] - st['map'],
                                   macro_f1_drop=res[k]['macro_f1'] - st['macro_f1'])
        print('stress', name, {k: round(v['map50'], 3) for k, v in stress[name].items()}, flush=True)

    # ---- label review on validation
    review = sum((label_review(vp[k], val, confs[k], k) for k in SYSTEMS), [])
    if review:
        write_csv(OUT / 'label_review_val.csv', review, list(review[0]))

    write_json(OUT / 'comparison.json', dict(selection=selection, test=res, bootstrap=boot, calibration=calib, stress=stress,
                                             label_review_rows=len(review), weights_sha256=weights))
    rows = [dict(system=k, conf=res[k]['conf'], test_map50=res[k]['map50'], map50_ci=boot[k]['map50_ci'], test_map30=res[k]['map30'],
                 test_macro_f1=res[k]['macro_f1'], macro_f1_ci=boot[k]['macro_f1_ci'], type_acc_found=res[k]['type_accuracy_on_found'],
                 ms_per_image=res[k]['ms_per_image']) for k in SYSTEMS]
    write_csv(OUT / 'comparison.csv', rows, list(rows[0]))
    print(json.dumps(dict(selection=selection['chosen'], bootstrap=boot), indent=1))


if __name__ == '__main__':
    main()
