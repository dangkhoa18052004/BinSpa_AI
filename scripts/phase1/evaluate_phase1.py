"""Evaluate frozen models; test metrics never decide the selected model."""
import argparse
import csv
import random
import time
from pathlib import Path
from common import ROOT, DATASET, REPORTS, digest, read_json, verify_dataset, write_csv, write_json


def evaluate(weights, dataset=DATASET, split='val', imgsz=640, batch=8, device='0', output=None):
    from ultralytics import YOLO
    weights,dataset=Path(weights),Path(dataset)
    manifest=verify_dataset(dataset, require_review=True)
    output=Path(output or REPORTS/'phase1_evaluation'/weights.parent.parent.name/split)
    model=YOLO(str(weights))
    metrics=model.val(data=str(dataset/'data.yaml'),split=split,imgsz=imgsz,batch=batch,device=device,workers=0,conf=.001,iou=.7,max_det=500,plots=True,project=str(output.parent),name=output.name,exist_ok=True,verbose=False)
    stats=dict(model=weights.parent.parent.name,split=split,precision=float(metrics.box.mp),recall=float(metrics.box.mr),mAP50=float(metrics.box.map50),mAP50_95=float(metrics.box.map),inference_ms=float(metrics.speed['inference']),preprocess_ms=float(metrics.speed['preprocess']),postprocess_ms=float(metrics.speed['postprocess']),model_mb=weights.stat().st_size/1024**2,weights=str(weights.resolve()),weights_sha256=digest(weights),manifest_sha256=digest(dataset/'manifest.json'),images=manifest['counts'][split],imgsz=imgsz,batch=batch,device=str(device))
    stats.update(max_det=500,conf=.001,iou=.7,workers=0)
    write_json(output/'metrics.json',stats)
    write_csv(output/'metrics.csv',[stats],list(stats))
    return stats


def greedy_matches(predictions,truth,iou_threshold=.5):
    """Predictions in descending confidence order; one GT matched at most once."""
    unmatched=set(range(len(truth)))
    matched=[]; false_positive=[]
    def iou(a,b):
        iw=max(0,min(a[2],b[2])-max(a[0],b[0]))
        ih=max(0,min(a[3],b[3])-max(a[1],b[1]))
        inter=iw*ih
        denom=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/denom if denom>0 else 0
    for index,box in enumerate(predictions):
        best=max(unmatched,key=lambda j:iou(box,truth[j]),default=None)
        if best is not None and iou(box,truth[best])>=iou_threshold:
            unmatched.remove(best);matched.append((index,best))
        else:
            false_positive.append(index)
    return matched,false_positive,sorted(unmatched)


def error_analysis(weights, dataset=DATASET, imgsz=640, device='0', conf=.25, count=20):
    """Run all test images, save 20 overlays, export unmatched detections for region review."""
    from ultralytics import YOLO
    from PIL import Image,ImageDraw
    dataset=Path(dataset)
    manifest=verify_dataset(dataset,require_review=True)
    output=REPORTS/'phase1_predictions'
    output.mkdir(parents=True,exist_ok=True)
    records=[r for r in manifest['images'] if r['split']=='test']
    examples=set(r['file_name'] for r in random.Random(42).sample(records,min(count,len(records))))
    model=YOLO(str(weights)); totals=dict(true_positive=0,false_positive=0,false_negative=0)
    reviews=[];image_stats=[]
    start=time.perf_counter()
    for row in records:
        path=dataset/'images/test'/row['file_name']
        result=model.predict(str(path),imgsz=imgsz,device=device,conf=conf,iou=.7,max_det=500,verbose=False)[0]
        predictions=result.boxes.xyxy.cpu().tolist();scores=result.boxes.conf.cpu().tolist()
        order=sorted(range(len(scores)),key=lambda j:scores[j],reverse=True)
        predictions=[predictions[j] for j in order];scores=[scores[j] for j in order]
        truth=[]
        for line in (dataset/'labels/test'/(path.stem+'.txt')).read_text().splitlines():
            _,cx,cy,bw,bh=map(float,line.split())
            truth.append([(cx-bw/2)*row['width'],(cy-bh/2)*row['height'],(cx+bw/2)*row['width'],(cy+bh/2)*row['height']])
        matched,fp,fn=greedy_matches(predictions,truth)
        totals['true_positive']+=len(matched);totals['false_positive']+=len(fp);totals['false_negative']+=len(fn)
        image_stats.append(dict(file_name=path.name,tp=len(matched),fp=len(fp),fn=len(fn)))
        for j in fp:
            reviews.append(dict(file_name=path.name,detection_index=j,confidence=scores[j],x1=predictions[j][0],y1=predictions[j][1],x2=predictions[j][2],y2=predictions[j][3],region='',notes=''))
        if path.name in examples:
            with Image.open(path) as source:
                image=source.convert('RGB')
            image.thumbnail((1600,1600));sx=image.width/row['width'];sy=image.height/row['height']
            draw=ImageDraw.Draw(image)
            for b in truth:
                draw.rectangle((b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy),outline='lime',width=2)
            for j,b in enumerate(predictions):
                draw.rectangle((b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy),outline='red' if j in fp else 'cyan',width=2)
            image.save(output/path.name,quality=92)
    write_csv(output/'image_error_counts.csv',image_stats,['file_name','tp','fp','fn'])
    reviewfile=output/'false_positive_region_review.csv'
    # Preserve manual review work on rerun.
    write_csv(output/('false_positive_region_review.generated.csv' if reviewfile.exists() else reviewfile.name),reviews,['file_name','detection_index','confidence','x1','y1','x2','y2','region','notes'])
    write_json(output/'error_summary.json',dict(**totals,conf=conf,match_iou=.5,test_images=len(records),illustrations=len(examples),elapsed_seconds=time.perf_counter()-start,region_review_status='pending_manual_review',allowed_regions=['eyes','lips','mole','hair_beard','other'],note='Unmatched to dataset labels does not prove a lesion is absent; manually inspect annotation omissions. Green=GT, cyan=matched prediction, red=unmatched prediction.'))
    return totals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=ROOT/'models/phase1/best_acne_detector.pt')
    p.add_argument('--dataset',type=Path,default=DATASET)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',default='0')
    p.add_argument('--error-analysis',action='store_true')
    a=p.parse_args()
    print(evaluate(a.weights,a.dataset,a.split,a.imgsz,a.batch,a.device))
    if a.error_analysis:
        print(error_analysis(a.weights,a.dataset,a.imgsz,a.device))


if __name__=='__main__':
    main()
