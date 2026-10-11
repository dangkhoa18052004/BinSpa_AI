"""Scar-group detection on a face photo: bbox + scar_group + confidence.

Uses models/phase3/best_scar_detector.pt with the imgsz and validation-chosen confidence stored in
models/phase3/model_metadata.json (override with --conf). Groups are the two ACNE-DET classes:
atrophic_scar (sẹo lõm) and hypertrophic_scar (sẹo phì đại/lồi); no finer subtype is inferred.
Usage: venv/Scripts/python.exe scripts/phase3/predict_scar.py --source test_img/test6.jpg [--out scar.json]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import digest, read_json  # noqa: E402

WEIGHTS = ROOT / 'models/phase3/best_scar_detector.pt'
DISPLAY = {'atrophic_scar': 'dấu hiệu nhóm sẹo lõm', 'hypertrophic_scar': 'dấu hiệu nhóm sẹo phì đại/lồi'}


def main():
    from ultralytics import YOLO
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--conf', type=float)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    meta = read_json(ROOT / 'models/phase3/model_metadata.json')
    conf = a.conf if a.conf is not None else meta['operating_conf']
    r = YOLO(str(WEIGHTS)).predict(str(a.source), imgsz=meta['imgsz'], conf=conf, max_det=500, verbose=False)[0]
    preds = [dict(bbox=[round(float(v), 1) for v in b], scar_group=r.names[int(c)], label=DISPLAY[r.names[int(c)]],
                  confidence=round(float(s), 4))
             for b, c, s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist(), r.boxes.conf.cpu().tolist())]
    counts = Counter(x['scar_group'] for x in preds)
    result = dict(image=str(a.source), atrophic_count=counts['atrophic_scar'], hypertrophic_count=counts['hypertrophic_scar'],
                  dominant_group=counts.most_common(1)[0][0] if counts else None, predictions=preds,
                  model_version=digest(WEIGHTS)[:16], conf=conf,
                  disclaimer='Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da.')
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out:
        a.out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
