"""One-image or directory inference with boxes, confidence, version and disclaimer.

imgsz and conf default to the values stored in models/phase1/model_metadata.json for the
deployed detector (the threshold chosen on validation); pass them explicitly for other weights.
"""
import argparse
from pathlib import Path
from common import ROOT, digest, read_json, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--weights',type=Path,default=ROOT/'models/phase1/best_acne_detector.pt')
    p.add_argument('--output',type=Path,default=ROOT/'reports/phase1_external_predictions')
    p.add_argument('--imgsz',type=int,help='default: imgsz in model_metadata.json')
    p.add_argument('--device',default='0')
    p.add_argument('--conf',type=float,help='default: operating_conf in model_metadata.json')
    a=p.parse_args()
    if not a.weights.is_file():
        raise FileNotFoundError('No selected model yet. Finish benchmark first.')
    meta=ROOT/'models/phase1/model_metadata.json'
    if a.imgsz is None or a.conf is None:
        m=read_json(meta)
        if m.get('selected_weights_sha256')!=digest(a.weights):
            raise ValueError('--imgsz and --conf are required for weights other than the deployed detector')
        a.imgsz=a.imgsz or m['imgsz'];a.conf=a.conf if a.conf is not None else m['operating_conf']
    import torch
    from ultralytics import RTDETR,YOLO
    ckpt=torch.load(str(a.weights),map_location='cpu',weights_only=False)
    detr='RTDETR' in type(ckpt.get('model') or ckpt.get('ema')).__name__;del ckpt
    model=(RTDETR if detr else YOLO)(str(a.weights));version=digest(a.weights)[:16]
    a.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for r in model.predict(a.source,imgsz=a.imgsz,conf=a.conf,device=a.device,max_det=500,stream=True,verbose=False):
        predictions=[dict(bbox=box,confidence=score,class_name='acne') for box,score in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.conf.cpu().tolist())]
        r.save(filename=str(a.output/Path(r.path).name))
        results.append(dict(file_name=Path(r.path).name,image_size=list(r.orig_shape),lesion_count=len(predictions),predictions=predictions,model_version=version,disclaimer='Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da.'))
    write_json(a.output/'predictions.json',results)
    print('Saved',len(results),'results to',a.output)


if __name__=='__main__':
    main()
