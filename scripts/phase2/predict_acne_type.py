"""Acne lesion type: top-3 classes with probabilities.

Two modes:
  --crop  one or more crop images (already cut around a lesion)
  --face  a face photo: runs the phase-1 detector (models/phase1/best_acne_detector.pt, imgsz/conf from
          its metadata), cuts each box with the same square context crop used for training (2x the
          lesion side, min 32 px) and classifies every crop.
Output: JSON with boxes (face mode), top-3 types, model versions and the disclaimer.
Usage:
  venv/Scripts/python.exe scripts/phase2/predict_acne_type.py --face test_img/test6.jpg --out pred.json
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


class TypeClassifier:
    def __init__(self, path=MODELS / 'best_acne_type.pth', device='cuda'):
        import timm
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model = timm.create_model(ckpt['timm_name'], pretrained=False, num_classes=len(ckpt['classes']))
        self.model.load_state_dict(ckpt['model'])
        self.model.to(device).eval()
        self.classes, self.size, self.device = ckpt['classes'], ckpt['input'], device
        self.mean = torch.tensor(ckpt['mean']).view(1, 3, 1, 1)
        self.std = torch.tensor(ckpt['std']).view(1, 3, 1, 1)
        self.version = digest(path)[:16]

    def __call__(self, crops_bgr, k=3):
        if not crops_bgr:
            return []
        x = np.stack([cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1] for c in crops_bgr])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255
        x = torch.nn.functional.interpolate(x, size=self.size, mode='bilinear', antialias=True, align_corners=False)
        x = ((x - self.mean) / self.std).to(self.device)
        with torch.inference_mode():
            prob = torch.softmax(self.model(x).float(), 1).cpu().numpy()
        out = []
        for p in prob:
            top = np.argsort(-p)[:k]
            out.append([dict(type=self.classes[i], probability=round(float(p[i]), 4)) for i in top])
        return out


def detect(image_path):
    sys.path.insert(0, str(ROOT / 'scripts/phase1'))
    from ultralytics import RTDETR, YOLO
    weights = ROOT / 'models/phase1/best_acne_detector.pt'
    meta = read_json(ROOT / 'models/phase1/model_metadata.json')
    ckpt = torch.load(str(weights), map_location='cpu', weights_only=False)
    detr = 'RTDETR' in type(ckpt.get('model') or ckpt.get('ema')).__name__
    del ckpt
    model = (RTDETR if detr else YOLO)(str(weights))
    r = model.predict(str(image_path), imgsz=meta['imgsz'], conf=meta['operating_conf'], max_det=500, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), digest(weights)[:16]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--crop', nargs='+', type=Path)
    g.add_argument('--face', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    clf = TypeClassifier()
    if a.crop:
        crops = [cv2.imread(str(c)) for c in a.crop]
        result = dict(predictions=[dict(crop=str(c), top3=t) for c, t in zip(a.crop, clf(crops))])
    else:
        image = cv2.imread(str(a.face))
        boxes, conf, det_version = detect(a.face)
        crops, regions = [], []
        for (x1, y1, x2, y2) in boxes:
            x0, y0, side = crop_box([x1, y1, x2 - x1, y2 - y1], image.shape[1], image.shape[0])
            crops.append(image[y0:y0 + side, x0:x0 + side]); regions.append((x0, y0, side))
        types = clf(crops)
        result = dict(image=str(a.face), lesion_count=len(boxes), detector_version=det_version,
                      predictions=[dict(bbox=[round(float(v), 1) for v in b], detection_confidence=round(float(c), 4), top3=t)
                                   for b, c, t in zip(boxes, conf, types)])
    result.update(type_model_version=clf.version, disclaimer=DISCLAIMER)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out:
        a.out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
