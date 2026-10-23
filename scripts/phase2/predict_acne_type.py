"""Acne lesion type: top-3 classes with probabilities.

Two modes:
  --crop  one or more crop images (already cut around a lesion), classified by the pipeline classifier
          (models/phase2/pipeline_metadata.json; fallback models/phase2/best_acne_type.pth)
  --face  a face photo. Default engine 'pipeline' (the project's two-stage design: find acne, then tell the
          type): phase-1 detector (models/phase1/best_acne_detector.pt) -> square context crop (2x lesion side,
          min 32 px) -> classifier top-3. With type_calibration in pipeline_metadata.json the type probabilities
          are temperature-calibrated and each lesion gets type_status 'confident' or 'uncertain' (type withheld,
          location shown; scripts/phase2/calibrate_classifier.py). With models/phase2/input_gate.json the photo
          is checked first (face, size, sharpness, noise, light; scripts/phase2/input_gate.py): a failed check returns
          analysis_status 'retake_photo' with reasons, a photo unlike the training photos (or with only 1-2 lesions,
          too few to verify) 'location_only'.
          --no-check skips it. Engine 'det5': the one-stage RT-DETR-L 5-class detector
          (models/phase2/best_acne_det5.pt) with a calibrated confidence = P(box and type correct); about 7-10x
          faster, but less accurate than the two-stage pipeline in 5-fold CV (reports/phase2_cv_summary.md).
          --conf overrides the detector confidence of either engine.
Output: JSON with boxes (face mode), types, model versions and the disclaimer.
Usage:
  venv/Scripts/python.exe scripts/phase2/predict_acne_type.py --face test_img/test6.jpg --out pred.json
  venv/Scripts/python.exe scripts/phase2/predict_acne_type.py --face test_img/test6.jpg --conf 0.30 --out pred030.json
  venv/Scripts/python.exe scripts/phase2/predict_acne_type.py --face test_img/test6.jpg --demo
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
TYPE_VI = dict(closed_comedo='Mụn đầu trắng', open_comedo='Mụn đầu đen', papule='Mụn sẩn', pustule='Mụn mủ', nodule='Mụn nốt')
# Demo mode (no labelled customer photos yet): the input and domain checks are still measured and reported, but
# never withhold the analysis; the detector runs at DEMO_CONF (finds about 2x more lesions on phone photos than
# the operating threshold). Types on photos unlike the training photos were 21-28 % correct on DetectDataSkin-2.
DEMO_CONF = 0.30
DEMO_MESSAGE = ('Chế độ thử nghiệm: model được huấn luyện trên ảnh da liễu chụp bằng máy chuyên dụng, '
                'loại mụn trên ảnh điện thoại có độ tin cậy thấp.')


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

    def embed(self, crops_bgr, batch=64):
        """-> (pooled penultimate features, logits). Every forward pass gets exactly `batch` inputs (zero-padded):
        each new batch size costs cuDNN seconds of kernel selection on this GPU, and every face has a different
        lesion count."""
        feats, logits = [], []
        for i in range(0, len(crops_bgr), batch):
            x = np.stack([cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1] for c in crops_bgr[i:i + batch]])
            n = len(x)
            x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255
            x = torch.nn.functional.interpolate(x, size=self.size, mode='bilinear', antialias=True, align_corners=False)
            x = ((x - self.mean) / self.std)
            x = torch.cat([x, x.new_zeros((batch - n, *x.shape[1:]))]).to(self.device)
            with torch.inference_mode():
                f = self.model.forward_head(self.model.forward_features(x), pre_logits=True)
                feats.append(f[:n].float().cpu().numpy())
                logits.append(self.model.get_classifier()(f)[:n].float().cpu().numpy())
        if not feats:
            return np.zeros((0, self.model.num_features), np.float32), np.zeros((0, len(self.classes)), np.float32)
        return np.concatenate(feats), np.concatenate(logits)

    def logits(self, crops_bgr, batch=64):
        return self.embed(crops_bgr, batch)[1]

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
    meta = read_json(meta_path)
    model = load_yolo_or_detr(weights)
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


def topk(prob, classes, k=3):
    return [[dict(type=classes[i], probability=round(float(p[i]), 4)) for i in np.argsort(-p)[:k]] for p in prob]


def load_yolo_or_detr(weights):
    from ultralytics import RTDETR, YOLO
    ckpt = torch.load(str(weights), map_location='cpu', weights_only=False)
    detr = 'RTDETR' in type(ckpt.get('model') or ckpt.get('ema')).__name__
    del ckpt
    return (RTDETR if detr else YOLO)(str(weights))


class AcnePipeline:
    """The deployed two-stage pipeline with every model loaded once (CLI --face and api/server.py).
    check: apply the input check of models/phase2/input_gate.json (evaluate_input_gate.py).
    analysis_status: 'ok'; 'retake_photo' (quality check failed: no lesions are reported, reasons say why);
    'location_only' (photo unlike the training photos, or only 1-2 lesions to verify it: every type 'uncertain').
    demo=True: status 'demo' instead of 'retake_photo' / 'location_only'; the checks are still reported, every
    lesion keeps its own calibrated type_status and gets 'type' / 'type_vi' (top-1), detector at DEMO_CONF."""

    def __init__(self, conf=None, check=True, weights=ROOT / 'models/phase1/best_acne_detector.pt',
                 meta_path=ROOT / 'models/phase1/model_metadata.json', demo=False):
        from input_gate import DomainScore, FaceTool, load_gate
        meta = read_json(meta_path)
        self.detector, self.det_version = load_yolo_or_detr(weights), digest(weights)[:16]
        self.demo = demo
        self.imgsz = meta['imgsz']
        self.conf = conf if conf is not None else DEMO_CONF if demo else meta['operating_conf']
        self.clf, self.tau, self.cal = pipeline_type_model()
        self.gate = load_gate() if check else None
        self.face = FaceTool() if self.gate and self.gate['quality_check'] else None
        self.domain = DomainScore.load(ROOT / self.gate['domain_stats']) if self.gate and self.gate['domain_check'] else None

    def versions(self):
        return dict(detector=self.det_version, type_model=self.clf.version, detector_confidence_threshold=self.conf,
                    type_calibration=self.cal, input_check=bool(self.face), domain_check=bool(self.domain), demo=self.demo)

    def analyze(self, image, name=None):
        """image: BGR uint8 array."""
        from input_gate import MESSAGES, judge, measure
        out = dict(image=name, engine='pipeline', analysis_status='ok', image_size=[int(image.shape[1]), int(image.shape[0])])
        if self.face:
            metrics = measure(image, self.face)
            fails = judge(metrics, self.gate['thresholds'])
            out['input_check'] = dict(passed=not fails, reasons=[dict(code=f, message=MESSAGES[f]) for f in fails],
                                      metrics={k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
            if fails and not self.demo:
                return dict(out, analysis_status='retake_photo', lesion_count=None, predictions=[])
        r = self.detector.predict(image, imgsz=self.imgsz, conf=self.conf, max_det=500, verbose=False)[0]
        boxes, score = r.boxes.xyxy.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy()
        crops = []
        for (x1, y1, x2, y2) in boxes:
            x0, y0, side = crop_box([x1, y1, x2 - x1, y2 - y1], image.shape[1], image.shape[0])
            crops.append(image[y0:y0 + side, x0:x0 + side])
        feats, logits = self.clf.embed(crops)
        tops = topk(softmax(logits / self.clf.temperature), self.clf.classes)
        preds = [dict(bbox=[round(float(v), 1) for v in b], detection_confidence=round(float(c), 4), **with_status(t, self.tau))
                 for b, c, t in zip(boxes, score, tops)]
        if self.domain and len(feats):
            limit = self.gate['thresholds']['domain_max']
            d = self.domain.photo(feats)
            out['domain_check'] = dict(score=None if d is None else round(d, 3), threshold=round(limit, 3),
                                       in_domain=None if d is None else d <= limit)
            if d is None:  # 1-2 lesions: too few to verify the photo -> treated as unverified, types withheld
                out['domain_check'].update(status='unverified', crop_scores=[round(float(v), 3) for v in self.domain.crops(feats)])
                out.update(analysis_status='location_only', message=MESSAGES['domain_unverified'])
            elif d > limit:
                out.update(analysis_status='location_only', message=MESSAGES['out_of_domain'])
            if out['analysis_status'] == 'location_only' and not self.demo:
                for p in preds:
                    p['type_status'] = 'uncertain'
        if self.demo:
            out.update(analysis_status='demo', message=DEMO_MESSAGE)
            for p in preds:
                p.update(type=p['top3'][0]['type'], type_vi=TYPE_VI[p['top3'][0]['type']])
            out['type_counts'] = {t: sum(p['type'] == t for p in preds) for t in self.clf.classes}
        out.update(lesion_count=len(boxes), detector_version=self.det_version, detector_confidence_threshold=self.conf,
                   type_model_version=self.clf.version)
        if self.cal:
            out.update(type_calibration=self.cal,
                       uncertain_type_count=sum(p.get('type_status') == 'uncertain' for p in preds) if self.tau else None)
        return dict(out, predictions=preds)


def face_pipeline(path, conf, check=True, demo=False):
    return AcnePipeline(conf, check, demo=demo).analyze(cv2.imread(str(path)), str(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--crop', nargs='+', type=Path)
    g.add_argument('--face', type=Path)
    p.add_argument('--engine', choices=['pipeline', 'det5'], default='pipeline', help='--face only (default pipeline)')
    p.add_argument('--conf', type=float, help='Detector confidence for --face only; default: deployed metadata')
    p.add_argument('--no-check', action='store_true', help='--face pipeline: skip the input check (analyse anyway)')
    p.add_argument('--demo', action='store_true', help='--face pipeline: demo mode (checks reported, never withheld)')
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
        result = face_det5(a.face, a.conf) if a.engine == 'det5' else face_pipeline(a.face, a.conf, check=not a.no_check, demo=a.demo)
    result.update(disclaimer=DISCLAIMER)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
