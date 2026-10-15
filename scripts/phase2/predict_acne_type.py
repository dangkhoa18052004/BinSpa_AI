"""Acne lesion type: top-3 classes with probabilities.

Two modes:
  --crop  one or more crop images (already cut around a lesion), classified by the pipeline classifier
          (models/phase2/pipeline_metadata.json; fallback models/phase2/best_acne_type.pth)
  --face  a face photo. Default engine 'pipeline' (the project's two-stage design: find acne, then tell the
          type): phase-1 detector (models/phase1/best_acne_detector.pt) -> square context crop (2x lesion side,
          min 32 px) -> classifier top-3. With type_calibration in pipeline_metadata.json the type probabilities
          are temperature-calibrated and each lesion gets type_status 'confident' or 'uncertain' (type withheld,
          location shown; scripts/phase2/calibrate_classifier.py). Engine 'det5': the one-stage RT-DETR-L 5-class detector
          (models/phase2/best_acne_det5.pt) with a calibrated confidence = P(box and type correct); about 7-10x
          faster, but less accurate than the two-stage pipeline in 5-fold CV (reports/phase2_cv_summary.md).
          --conf overrides the detector confidence of either engine.
Output: JSON with boxes (face mode), types, model versions and the disclaimer.
Usage:
  venv/Scripts/python.exe scripts/phase2/predict_acne_type.py --face test_img/test6.jpg --out pred.json
  venv/Scripts/python.exe scripts/phase2/predict_acne_type.py --face test_img/test6.jpg --conf 0.30 --out pred030.json
  venv/Scripts/python.exe scripts/phase2/predict_acne_type.py --face test_img/test6.jpg --engine det5
  venv/Scripts/python.exe scripts/phase2/predict_acne_type.py --crop a.jpg b.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from common2 import ACNE_CLASSES, MODELS, ROOT, digest, read_json
from make_crops import crop_box

DISCLAIMER = 'Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da.'


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class TypeClassifier:
    def __init__(self, path=MODELS / 'best_acne_type.pth', device='cuda', temperature=1.0):
        import timm
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model = timm.create_model(ckpt['timm_name'], pretrained=False, num_classes=len(ckpt['classes']))
        self.model.load_state_dict(ckpt['model'])
        self.model.to(device).eval()
        self.classes, self.size, self.device = ckpt['classes'], ckpt['input'], device
        self.mean = torch.tensor(ckpt['mean']).view(1, 3, 1, 1)
        self.std = torch.tensor(ckpt['std']).view(1, 3, 1, 1)
        self.version = digest(path)[:16]
        self.temperature = temperature  # probabilities = softmax(logits / T); 1.0 = uncalibrated

    def logits(self, crops_bgr, batch=64):
        """Every forward pass gets exactly `batch` inputs (zero-padded): each new batch size costs cuDNN
        seconds of kernel selection on this GPU, and every face has a different lesion count."""
        out = []
        for i in range(0, len(crops_bgr), batch):
            x = np.stack([cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1] for c in crops_bgr[i:i + batch]])
            n = len(x)
            x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255
            x = torch.nn.functional.interpolate(x, size=self.size, mode='bilinear', antialias=True, align_corners=False)
            x = ((x - self.mean) / self.std)
            x = torch.cat([x, x.new_zeros((batch - n, *x.shape[1:]))]).to(self.device)
            with torch.inference_mode():
                out.append(self.model(x)[:n].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), np.float32)

    def __call__(self, crops_bgr, k=3):
        if not crops_bgr:
            return []
        prob = softmax(self.logits(crops_bgr) / self.temperature)
        out = []
        for p in prob:
            top = np.argsort(-p)[:k]
            out.append([dict(type=self.classes[i], probability=round(float(p[i]), 4)) for i in top])
        return out


def detect(image_path, conf=None, weights=ROOT / 'models/phase1/best_acne_detector.pt',
           meta_path=ROOT / 'models/phase1/model_metadata.json'):
    """-> boxes xyxy, scores, class ids, weights version, threshold used."""
    from ultralytics import RTDETR, YOLO
    meta = read_json(meta_path)
    ckpt = torch.load(str(weights), map_location='cpu', weights_only=False)
    detr = 'RTDETR' in type(ckpt.get('model') or ckpt.get('ema')).__name__
    del ckpt
    model = (RTDETR if detr else YOLO)(str(weights))
    threshold = meta['operating_conf'] if conf is None else conf
    r = model.predict(str(image_path), imgsz=meta['imgsz'], conf=threshold, max_det=500, verbose=False)[0]
    return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
            digest(weights)[:16], threshold)


def pipeline_classifier():
    """Classifier of the two-stage pipeline: models/phase2/pipeline_metadata.json (promote_classifier.py),
    else the original models/phase2/best_acne_type.pth."""
    meta = MODELS / 'pipeline_metadata.json'
    return ROOT / read_json(meta)['classifier'] if meta.exists() else MODELS / 'best_acne_type.pth'


def pipeline_type_model():
    """-> (classifier with the calibrated temperature, accept threshold or None, calibration info for the output).
    models/phase2/pipeline_metadata.json 'type_calibration' (calibrate_classifier.py --promote); absent = T 1, no status."""
    meta = MODELS / 'pipeline_metadata.json'
    cal = (read_json(meta).get('type_calibration') if meta.exists() else None) or {}
    T, tau = cal.get('temperature', 1.0), cal.get('accept_threshold')
    info = dict(temperature=round(T, 4), accept_threshold=tau) if cal else None
    return TypeClassifier(pipeline_classifier(), temperature=T), tau, info


def with_status(top3, tau):
    """'confident': show the type; 'uncertain': calibrated top-1 probability below the threshold, show only the location."""
    return dict(top3=top3) if tau is None else dict(top3=top3, type_status='confident' if top3[0]['probability'] >= tau else 'uncertain')


def calibrate(score, platt):
    z = np.log(np.clip(score, 1e-4, 1 - 1e-4) / (1 - np.clip(score, 1e-4, 1 - 1e-4)))
    return 1 / (1 + np.exp(-(platt['a'] * z + platt['b'])))


def face_det5(path, conf):
    meta = read_json(MODELS / 'det5_metadata.json')
    boxes, score, cls, version, threshold = detect(path, conf, MODELS / 'best_acne_det5.pt', MODELS / 'det5_metadata.json')
    calibrated = calibrate(score, meta['calibration'])
    return dict(image=str(path), engine='det5', lesion_count=len(boxes), model_version=version,
                confidence_threshold=threshold,
                predictions=[dict(bbox=[round(float(v), 1) for v in b], type=meta['classes'][c], score=round(float(s), 4),
                                  calibrated_confidence=round(float(p), 4))
                             for b, s, c, p in zip(boxes, score, cls, calibrated)])


def face_pipeline(path, conf):
    clf, tau, cal = pipeline_type_model()
    image = cv2.imread(str(path))
    boxes, score, _, det_version, threshold = detect(path, conf)
    crops = []
    for (x1, y1, x2, y2) in boxes:
        x0, y0, side = crop_box([x1, y1, x2 - x1, y2 - y1], image.shape[1], image.shape[0])
        crops.append(image[y0:y0 + side, x0:x0 + side])
    preds = [dict(bbox=[round(float(v), 1) for v in b], detection_confidence=round(float(c), 4), **with_status(t, tau))
             for b, c, t in zip(boxes, score, clf(crops))]
    out = dict(image=str(path), engine='pipeline', lesion_count=len(boxes), detector_version=det_version,
               detector_confidence_threshold=threshold, type_model_version=clf.version)
    if cal:
        out.update(type_calibration=cal, uncertain_type_count=sum(p['type_status'] == 'uncertain' for p in preds) if tau else None)
    return dict(out, predictions=preds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--crop', nargs='+', type=Path)
    g.add_argument('--face', type=Path)
    p.add_argument('--engine', choices=['pipeline', 'det5'], default='pipeline', help='--face only (default pipeline)')
    p.add_argument('--conf', type=float, help='Detector confidence for --face only; default: deployed metadata')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.conf is not None:
        if a.crop:
            p.error('--conf applies only to --face; --crop already classifies every supplied crop')
        if not 0 <= a.conf <= 1:
            p.error('--conf must be a finite number between 0 and 1')
    if a.crop:
        clf, tau, cal = pipeline_type_model()
        crops = [cv2.imread(str(c)) for c in a.crop]
        result = dict(predictions=[dict(crop=str(c), **with_status(t, tau)) for c, t in zip(a.crop, clf(crops))],
                      type_model_version=clf.version)
        if cal:
            result['type_calibration'] = cal
    else:
        result = (face_det5 if a.engine == 'det5' else face_pipeline)(a.face, a.conf)
    result.update(disclaimer=DISCLAIMER)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
