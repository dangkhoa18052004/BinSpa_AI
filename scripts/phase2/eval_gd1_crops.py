"""Phase-1 detector -> phase-2 classifier on ACNE-DET test images (domain-shift check).

For every phase-1 candidate (RT-DETR-L @1280 deployed, YOLO26s @1280 fallback) on the 45 test images:
1. Detector generalisation: recall of the 5 acne classes (IoU >= 0.3 and 0.5, greedy one-to-one by
   score), and the share of detections that hit any annotated lesion (acne or not) at IoU >= 0.3.
   The detectors were trained on ACNE04-v2 (other camera, annotators and box convention), so this is
   a cross-dataset test, not a phase-1 metric.
2. Pipeline crops: each detection matched to an acne lesion is cut with the training crop rule from
   the DETECTED box and classified; macro-F1/accuracy against the matched lesion's label, next to the
   same lesions classified from their annotated boxes (crop-geometry effect only).
Uses the locked phase-2 model; the phase-2 test split was already used once for its own metrics.
Usage: venv/Scripts/python.exe scripts/phase2/eval_gd1_crops.py
"""
import csv
from collections import defaultdict

import cv2
import numpy as np
import torch

from common2 import ACNE_CLASSES, MASTER_SPLIT, REPORTS, ROOT, SOURCE, load_coco, write_json
from make_crops import crop_box
from predict_acne_type import TypeClassifier

DETECTORS = {'rtdetr_1280': (ROOT / 'models/phase1/rtdetr_1280_best.pt', 0.459),
             'yolo26s_1280': (ROOT / 'models/phase1/yolo26s_1280_best.pt', 0.272)}


def iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:, :2]).T
    x2, y2 = np.minimum(a[2:], b[:, 2:]).T
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def greedy(det, gt, thr):
    """det sorted by score; returns {det index: gt index}."""
    used, out = set(), {}
    for j, d in enumerate(det):
        if not len(gt):
            break
        o = iou(d, gt)
        o[list(used)] = -1
        k = int(o.argmax())
        if o[k] >= thr:
            used.add(k); out[j] = k
    return out


def main():
    from sklearn.metrics import f1_score
    from ultralytics import RTDETR, YOLO
    data, cats = load_coco()
    with MASTER_SPLIT.open(encoding='utf-8-sig') as f:
        test_ids = {int(r['image_id']) for r in csv.DictReader(f) if r['split'] == 'test'}
    anns = defaultdict(list)
    for a in data['annotations']:
        anns[a['image_id']].append(a)
    images = [i for i in data['images'] if i['id'] in test_ids]
    clf = TypeClassifier()
    result = {}
    for name, (weights, conf) in DETECTORS.items():
        ckpt = torch.load(str(weights), map_location='cpu', weights_only=False)
        detr = 'RTDETR' in type(ckpt.get('model') or ckpt.get('ema')).__name__
        del ckpt
        det_model = (RTDETR if detr else YOLO)(str(weights))
        n_gt = defaultdict(int); hit = {0.3: defaultdict(int), 0.5: defaultdict(int)}
        n_det = n_det_any = 0
        y_true, y_det_crop, y_gt_crop = [], [], []
        for img in images:
            image = cv2.imread(str(SOURCE / img['file_name']))
            r = det_model.predict(image, imgsz=1280, conf=conf, max_det=500, verbose=False)[0]
            det = r.boxes.xyxy.cpu().numpy()
            det = det[np.argsort(-r.boxes.conf.cpu().numpy())]
            acne = [a for a in anns[img['id']] if cats[a['category_id']] in ACNE_CLASSES]
            gt = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                           for a in acne]).reshape(-1, 4)
            every = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                              for a in anns[img['id']]]).reshape(-1, 4)
            for a in acne:
                n_gt[cats[a['category_id']]] += 1
            n_det += len(det)
            n_det_any += sum(1 for d in det if len(every) and iou(d, every).max() >= 0.3)
            for thr in (0.3, 0.5):
                for j, k in greedy(det, gt, thr).items():
                    hit[thr][cats[acne[k]['category_id']]] += 1
            pairs = greedy(det, gt, 0.3)
            crops_det, crops_gt = [], []
            for j, k in pairs.items():
                x1, y1, x2, y2 = det[j]
                x0, y0, s = crop_box([x1, y1, x2 - x1, y2 - y1], img['width'], img['height'])
                crops_det.append(image[y0:y0 + s, x0:x0 + s])
                x0, y0, s = crop_box(acne[k]['bbox'], img['width'], img['height'])
                crops_gt.append(image[y0:y0 + s, x0:x0 + s])
                y_true.append(ACNE_CLASSES.index(cats[acne[k]['category_id']]))
            y_det_crop += [ACNE_CLASSES.index(t[0]['type']) for t in clf(crops_det)]
            y_gt_crop += [ACNE_CLASSES.index(t[0]['type']) for t in clf(crops_gt)]
        y_true, y_det_crop, y_gt_crop = map(np.array, (y_true, y_det_crop, y_gt_crop))
        labels = list(range(len(ACNE_CLASSES)))
        result[name] = dict(
            images=len(images), detections=n_det, detections_on_any_lesion_iou30=n_det_any / max(n_det, 1),
            recall={f'iou{int(t * 100)}': dict(all=sum(hit[t].values()) / sum(n_gt.values()),
                                              **{c: hit[t][c] / max(n_gt[c], 1) for c in ACNE_CLASSES}) for t in hit},
            gt_lesions=dict(n_gt),
            classifier_on_matched=dict(n=len(y_true),
                                       detector_crop=dict(accuracy=float((y_det_crop == y_true).mean()),
                                                          macro_f1=float(f1_score(y_true, y_det_crop, labels=labels, average='macro', zero_division=0))),
                                       annotated_crop=dict(accuracy=float((y_gt_crop == y_true).mean()),
                                                           macro_f1=float(f1_score(y_true, y_gt_crop, labels=labels, average='macro', zero_division=0))),
                                       matched_per_class={c: int((y_true == k).sum()) for k, c in enumerate(ACNE_CLASSES)}))
        print(name, result[name], flush=True)
    write_json(REPORTS / 'phase2_evaluation/gd1_pipeline_on_acnedet.json', result)


if __name__ == '__main__':
    main()
