"""Framework-independent detector evaluation on the fixed phase-1 splits.

Every detector (Ultralytics YOLO/RT-DETR, torchvision Faster R-CNN) is reduced to
per-image predictions in original pixel coordinates; metrics are then computed with
the same rules for all models:

- mAP50 / mAP50-95: Ultralytics `ap_per_class` (101-point-free continuous AP,
  IoU 0.50:0.05:0.95) with Ultralytics greedy IoU matching. Raw predictions use
  conf>=0.001 and max_det=500 (RT-DETR is capped by its 300 queries).
- Operating point: confidence threshold maximising F1 on VALIDATION (IoU 0.5,
  Ultralytics smoothed max-F1 rule). The same threshold is applied to test, so the
  test set never chooses a threshold. Precision/Recall/F1/TP/FP/FN are reported there.
- Supplementary: Ultralytics-style P/R (max-F1 on the evaluated split itself, same
  as the numbers in reports/metrics/*.csv) and counts at a fixed conf=0.25.
- Latency: batch 1, FP32, CUDA, same images, after warm-up; preprocess/inference/
  postprocess in ms/image, image decoding excluded (as in Ultralytics speed).
"""
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from common import DATASET, REPORTS, ROOT, digest, read_json, verify_dataset, write_csv, write_json

ADD = REPORTS / 'phase1_additional_models'
RUNS_ADD = ROOT / 'runs' / 'phase1_additional'
MODELS_DIR = ROOT / 'models' / 'phase1'
IOUV = np.linspace(0.5, 0.95, 10)
RAW_CONF, MAX_DET, NMS_IOU, IMGSZ = 0.001, 500, 0.7, 640
FIXED_CONF = 0.25
NAMES = {0: 'acne'}


def portable_data_yaml():
    """data.yaml in the dataset points to the old D: location; write a copy that
    references the same split folders here. The dataset itself is never touched."""
    path = ADD / 'data_portable.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (f'path: "{DATASET.as_posix()}"\ntrain: images/train\nval: images/val\ntest: images/test\n'
            'names:\n  0: acne\n')
    if not path.exists() or path.read_text(encoding='utf-8') != text:
        path.write_text(text, encoding='utf-8')
    return path


def load_split(split, dataset=DATASET, manifest=None):
    """Records in manifest order with GT boxes (xyxy, original pixels)."""
    manifest = manifest or verify_dataset(dataset, require_review=True)
    records = []
    for row in manifest['images']:
        if row['split'] != split:
            continue
        w, h = row['width'], row['height']
        boxes = []
        for line in (dataset / 'labels' / split / (Path(row['file_name']).stem + '.txt')).read_text().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        records.append(dict(file_name=row['file_name'], path=dataset / 'images' / split / row['file_name'],
                            width=w, height=h, gt=np.array(boxes, dtype=np.float64).reshape(-1, 4)))
    if len(records) != manifest['counts'][split]:
        raise RuntimeError(f'{split}: expected {manifest["counts"][split]} images, got {len(records)}')
    return records


def box_iou(a, b):
    """IoU matrix (len(a), len(b)) for xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(iou, thresholds=IOUV):
    """Ultralytics BaseValidator.match_predictions (greedy, highest IoU first).
    iou: (labels, detections). Returns bool (detections, len(thresholds))."""
    correct = np.zeros((iou.shape[1], len(thresholds)), dtype=bool)
    for i, threshold in enumerate(thresholds):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def counts_at(records, preds, conf, iou_thr=0.5):
    """TP/FP/FN when only detections with score>=conf are kept (re-matched)."""
    tp = fp = fn = 0
    per_image = []
    for rec, pred in zip(records, preds):
        keep = pred['scores'] >= conf
        boxes = pred['boxes'][keep]
        correct = match(box_iou(rec['gt'], boxes), [iou_thr])[:, 0] if len(boxes) else np.zeros(0, bool)
        t = int(correct.sum())
        per_image.append(dict(file_name=rec['file_name'], gt=len(rec['gt']), pred=int(keep.sum()), tp=t,
                              fp=int(len(boxes) - t), fn=int(len(rec['gt']) - t)))
        tp += t; fp += len(boxes) - t; fn += len(rec['gt']) - t
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(conf=float(conf), iou=iou_thr, tp=int(tp), fp=int(fp), fn=int(fn), precision=p, recall=r, f1=f1), per_image


def compute_metrics(records, preds, out_dir=None, conf_op=None):
    """AP metrics + curves; conf_op=None means 'choose max-F1 threshold on this split'."""
    from ultralytics.utils.metrics import ap_per_class, smooth
    tps, confs, n_gt = [], [], 0
    for rec, pred in zip(records, preds):
        n_gt += len(rec['gt'])
        if len(pred['boxes']):
            tps.append(match(box_iou(rec['gt'], pred['boxes'])) if len(rec['gt']) else
                       np.zeros((len(pred['boxes']), 10), bool))
            confs.append(pred['scores'])
    tp = np.concatenate(tps) if tps else np.zeros((0, 10), bool)
    conf = np.concatenate(confs) if confs else np.zeros(0)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    _, _, p, r, f1, ap, _, _, _, f1_curve, x, _ = ap_per_class(
        tp.astype(float), conf, np.zeros(len(conf)), np.zeros(n_gt), plot=bool(out_dir),
        save_dir=Path(out_dir or '.'), names=NAMES)
    max_f1_conf = float(x[smooth(f1_curve.mean(0), 0.1).argmax()])
    chosen = max_f1_conf if conf_op is None else conf_op
    op, per_image = counts_at(records, preds, chosen)
    fixed, _ = counts_at(records, preds, FIXED_CONF)
    result = dict(images=len(records), gt_boxes=n_gt, predictions=int(len(conf)),
                  mAP50=float(ap[0, 0]), mAP50_95=float(ap[0].mean()),
                  ultra_precision=float(p[0]), ultra_recall=float(r[0]), ultra_f1=float(f1[0]),
                  split_max_f1_conf=max_f1_conf, operating_point=op, fixed_conf_025=fixed)
    if out_dir:
        plot_confusion(op, out_dir)
        write_csv(Path(out_dir) / 'per_image_counts.csv', per_image, list(per_image[0]))
    return result, per_image


def plot_confusion(op, out_dir):
    """1-class detection confusion matrix (Ultralytics layout [pred, true], background last)."""
    from ultralytics.utils.metrics import ConfusionMatrix
    cm = ConfusionMatrix(names=NAMES, task='detect')
    cm.matrix[:] = 0
    cm.matrix[0, 0], cm.matrix[0, 1], cm.matrix[1, 0] = op['tp'], op['fp'], op['fn']
    for normalize in (False, True):
        cm.plot(normalize=normalize, save_dir=out_dir)
    # Record the operating point used so the PNG is interpretable on its own.
    write_json(Path(out_dir) / 'confusion_matrix.json', dict(
        matrix_pred_by_true=cm.matrix.tolist(), conf=op['conf'], match_iou=op['iou'],
        note='rows=predicted [acne, background], cols=true [acne, background]; background/background is undefined for detection'))


# --------------------------------------------------------------------------- predictors

class UltralyticsPredictor:
    def __init__(self, weights, imgsz=IMGSZ, device=0):
        from ultralytics import RTDETR, YOLO
        import torch
        ckpt = torch.load(str(weights), map_location='cpu', weights_only=False)
        is_detr = 'RTDETR' in type(ckpt.get('model') or ckpt.get('ema')).__name__
        del ckpt
        self.model = (RTDETR if is_detr else YOLO)(str(weights))
        self.kind = 'rtdetr' if is_detr else 'yolo'
        self.imgsz, self.device = imgsz, device

    def __call__(self, image_bgr):
        res = self.model.predict(image_bgr, imgsz=self.imgsz, conf=RAW_CONF, iou=NMS_IOU, max_det=MAX_DET,
                                 device=self.device, verbose=False)[0]
        return (res.boxes.xyxy.cpu().numpy().astype(np.float64), res.boxes.conf.cpu().numpy().astype(np.float64),
                dict(res.speed))

    def params(self):
        return sum(p.numel() for p in self.model.model.parameters())


class FasterRCNNPredictor:
    """Splits the torchvision forward into pre/inference/post exactly like Ultralytics:
    preprocess = to-tensor + normalize/resize; inference = backbone + RPN + box head;
    postprocess = box decoding, score filter, NMS, rescale to original."""

    def __init__(self, checkpoint, device='cuda'):
        import torch
        from train_faster_rcnn import build_model
        state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
        self.model = build_model(state['config'], pretrained=False)
        self.model.load_state_dict({k: v.float() for k, v in state['model'].items()})
        self.model.roi_heads.score_thresh = RAW_CONF
        self.model.roi_heads.detections_per_img = MAX_DET
        self.model.to(device).eval()
        self.device, self.torch = device, torch

    def __call__(self, image_bgr):
        torch, m = self.torch, self.model
        with torch.inference_mode():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            # Same pipeline as training (train_faster_rcnn.AcneDataset): INTER_AREA to long side 1024,
            # then torchvision resizes to 640. Feeding the ~3000 px original straight to torchvision's
            # non-antialiased bilinear resize aliases tiny lesions (val mAP50-95 0.117 -> 0.100).
            s = min(1.0, 1024 / max(image_bgr.shape[:2]))
            small = cv2.resize(image_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image_bgr
            x = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).to(self.device).permute(2, 0, 1).float().div_(255)
            images, _ = m.transform([x])
            torch.cuda.synchronize(); t1 = time.perf_counter()
            features = m.backbone(images.tensors)
            proposals, _ = m.rpn(images, features)
            rh = m.roi_heads
            box_features = rh.box_head(rh.box_roi_pool(features, proposals, images.image_sizes))
            logits, regression = rh.box_predictor(box_features)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            boxes, scores, _ = rh.postprocess_detections(logits, regression, proposals, images.image_sizes)
            det = m.transform.postprocess([dict(boxes=boxes[0], scores=scores[0], labels=torch.ones_like(scores[0]))],
                                          images.image_sizes, [tuple(small.shape[:2])])[0]
            b = det['boxes'].cpu().numpy().astype(np.float64) / s  # back to original pixels
            scores = det['scores'].cpu().numpy().astype(np.float64)
            t3 = time.perf_counter()
        return b, scores, dict(preprocess=(t1 - t0) * 1e3, inference=(t2 - t1) * 1e3, postprocess=(t3 - t2) * 1e3)

    def params(self):
        return sum(p.numel() for p in self.model.parameters())


def make_predictor(weights, imgsz=IMGSZ):
    weights = Path(weights)
    return FasterRCNNPredictor(weights) if weights.suffix == '.pth' else UltralyticsPredictor(weights, imgsz)


def collect(predictor, records, warmup=10):
    import torch
    images = [cv2.imread(str(r['path'])) for r in records]
    for image in images[:warmup]:
        predictor(image)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    preds, timings = [], []
    for image in images:
        boxes, scores, speed = predictor(image)
        order = np.argsort(-scores, kind='stable')
        preds.append(dict(boxes=boxes[order].reshape(-1, 4), scores=scores[order]))
        timings.append(speed)
    speed = {k: float(np.mean([t[k] for t in timings])) for k in ('preprocess', 'inference', 'postprocess')}
    speed['total'] = sum(speed.values())
    speed['peak_vram_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
    return preds, speed


def draw_overlays(records, preds, conf, out_dir, names):
    """Green = ground truth, cyan = matched prediction, red = unmatched prediction (IoU 0.5)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec, pred in zip(records, preds):
        if rec['file_name'] not in names:
            continue
        image = cv2.imread(str(rec['path']))
        keep = pred['scores'] >= conf
        boxes = pred['boxes'][keep]
        correct = match(box_iou(rec['gt'], boxes), [0.5])[:, 0] if len(boxes) else np.zeros(0, bool)
        s = min(1.0, 1600 / max(image.shape[:2]))
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for b in rec['gt']:
            cv2.rectangle(image, tuple(int(v * s) for v in b[:2]), tuple(int(v * s) for v in b[2:]), (0, 255, 0), 2)
        for b, ok in zip(boxes, correct):
            cv2.rectangle(image, tuple(int(v * s) for v in b[:2]), tuple(int(v * s) for v in b[2:]),
                          (255, 255, 0) if ok else (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / rec['file_name']), image, [cv2.IMWRITE_JPEG_QUALITY, 92])


def save_predictions(path, records, preds):
    write_json(path, [dict(file_name=r['file_name'], boxes=np.round(p['boxes'], 2).tolist(),
                           scores=np.round(p['scores'], 5).tolist()) for r, p in zip(records, preds)])


def evaluate_model(name, weights, out_root, split_list=('val', 'test'), illustrate=True, imgsz=IMGSZ):
    """Validation first (chooses conf_op), then test with that conf_op. Writes immediately."""
    weights = Path(weights)
    out_root = Path(out_root)
    manifest = verify_dataset(DATASET, require_review=True)
    predictor = make_predictor(weights, imgsz)
    summary = dict(model=name, weights=str(weights), weights_sha256=digest(weights),
                   model_mb=weights.stat().st_size / 1024 ** 2, params_m=predictor.params() / 1e6,
                   manifest_sha256=digest(DATASET / 'manifest.json'), imgsz=imgsz, raw_conf=RAW_CONF,
                   max_det=MAX_DET, nms_iou=NMS_IOU if isinstance(predictor, UltralyticsPredictor) else 'torchvision box_nms_thresh=0.5',
                   latency_protocol='batch=1, FP32, CUDA, mean over all split images after 10 warm-up images; decode excluded')
    conf_op = None
    for split in split_list:
        if split == 'test' and conf_op is None:
            raise RuntimeError('Validation must be evaluated before test to fix the operating threshold')
        records = load_split(split, manifest=manifest)
        preds, speed = collect(predictor, records)
        out = out_root / f'eval_{split}'
        metrics, per_image = compute_metrics(records, preds, out, conf_op)
        if split == 'val':
            conf_op = metrics['split_max_f1_conf']
        metrics.update(speed_ms=speed, split=split)
        save_predictions(out / 'predictions.json', records, preds)
        write_json(out / 'metrics.json', dict(summary, **metrics))
        summary[split] = metrics
        if illustrate and split == 'test':
            sample = set(r['file_name'] for r in random.Random(42).sample(records, 20))
            draw_overlays(records, preds, conf_op, out_root / 'predictions_test', sample)
            # Failure cases: most missed lesions and most unmatched detections at conf_op.
            worst_fn = sorted(per_image, key=lambda r: -r['fn'])[:5]
            worst_fp = sorted(per_image, key=lambda r: -r['fp'])[:5]
            draw_overlays(records, preds, conf_op, out_root / 'failure_cases_test',
                          {r['file_name'] for r in worst_fn + worst_fp})
            write_json(out_root / 'failure_cases_test' / 'failure_cases.json',
                       dict(conf=conf_op, iou=0.5, most_false_negatives=worst_fn, most_false_positives=worst_fp,
                            legend='green=GT, cyan=matched prediction, red=unmatched prediction'))
        write_json(out_root / 'common_eval_summary.json', summary)
    return summary


def flat_row(summary, split='test'):
    m = summary[split]
    op = m['operating_point']
    return dict(model=summary['model'], split=split, precision=op['precision'], recall=op['recall'], f1=op['f1'],
                mAP50=m['mAP50'], mAP50_95=m['mAP50_95'], tp=op['tp'], fp=op['fp'], fn=op['fn'],
                conf_threshold=op['conf'], match_iou=op['iou'],
                preprocess_ms=m['speed_ms']['preprocess'], inference_ms=m['speed_ms']['inference'],
                postprocess_ms=m['speed_ms']['postprocess'], total_ms=m['speed_ms']['total'],
                peak_vram_mb=m['speed_ms']['peak_vram_mb'], model_mb=summary['model_mb'], params_m=summary['params_m'],
                ultra_precision=m['ultra_precision'], ultra_recall=m['ultra_recall'],
                tp_conf025=m['fixed_conf_025']['tp'], fp_conf025=m['fixed_conf_025']['fp'], fn_conf025=m['fixed_conf_025']['fn'],
                weights=summary['weights'], weights_sha256=summary['weights_sha256'])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--splits', default='val,test')
    a = ap.parse_args()
    print(json.dumps(evaluate_model(a.name, a.weights, a.out, tuple(a.splits.split(','))), indent=1, default=str)[:3000])


# --------------------------------------------------------------------------- bootstrap

def image_stats(records, preds, conf_op):
    """Per-image pieces needed to recompute metrics on any resample of images."""
    stats = []
    for rec, pred in zip(records, preds):
        tp = match(box_iou(rec['gt'], pred['boxes'])) if len(rec['gt']) and len(pred['boxes']) else \
            np.zeros((len(pred['boxes']), 10), bool)
        keep = pred['scores'] >= conf_op
        boxes = pred['boxes'][keep]
        t = int(match(box_iou(rec['gt'], boxes), [0.5])[:, 0].sum()) if len(boxes) and len(rec['gt']) else 0
        stats.append(dict(tp=tp, conf=pred['scores'], n_gt=len(rec['gt']), op=(t, len(boxes) - t, len(rec['gt']) - t)))
    return stats


def resample_metrics(stats, idx):
    from ultralytics.utils.metrics import ap_per_class
    tp = np.concatenate([stats[i]['tp'] for i in idx])
    conf = np.concatenate([stats[i]['conf'] for i in idx])
    n_gt = sum(stats[i]['n_gt'] for i in idx)
    ap = ap_per_class(tp.astype(float), conf, np.zeros(len(conf)), np.zeros(n_gt))[5]
    t, f, n = (sum(stats[i]['op'][k] for i in idx) for k in range(3))
    p, r = t / max(t + f, 1), t / max(t + n, 1)
    return dict(mAP50=float(ap[0, 0]), mAP50_95=float(ap[0].mean()), precision=p, recall=r,
                f1=2 * p * r / (p + r) if p + r else 0.0)


def bootstrap(stats_by_model, reference, n_boot=1000, seed=42):
    """95% percentile CIs per model and paired differences vs `reference`, using the SAME
    image resamples for every model (paired bootstrap over the 176 test images)."""
    n = len(next(iter(stats_by_model.values())))
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in stats_by_model}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, stats in stats_by_model.items():
            draws[k].append(resample_metrics(stats, idx))
    out = {}
    for k, rows in draws.items():
        res = {}
        for metric in rows[0]:
            values = np.array([r[metric] for r in rows])
            res[f'{metric}_ci_low'], res[f'{metric}_ci_high'] = np.percentile(values, [2.5, 97.5])
            ref = np.array([r[metric] for r in draws[reference]])
            diff = values - ref
            res[f'{metric}_diff_vs_ref'] = float(diff.mean())
            res[f'{metric}_diff_ci_low'], res[f'{metric}_diff_ci_high'] = np.percentile(diff, [2.5, 97.5])
            res[f'{metric}_p_better_than_ref'] = float((diff > 0).mean())
        out[k] = {key: float(v) for key, v in res.items()}
    return out
