"""Frozen deployed-model probe on the screened DetectDataSkin-2 export.

No training or confidence tuning. No scar-subtype scores, because source labels
do not identify those subtypes. The dataset has no verified patient IDs or expert
label review, so these are exploratory cross-source metrics against provided labels.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
from common import digest
from evaluate_detector_common import box_iou, compute_metrics, make_predictor
from make_crops import crop_box
from predict_acne_type import TypeClassifier
from audit_detectdataskin2 import ACNE, TYPE_MAP, RAW, OUT, DERIVED, read, write_csv, write_json

CLASSES = ['closed_comedo', 'open_comedo', 'papule', 'pustule', 'nodule']
EVAL = OUT / 'frozen_evaluation'


def probability(classifier, crops):
    import torch
    x = np.stack([cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1] for c in crops])
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255
    x = torch.nn.functional.interpolate(x, size=classifier.size, mode='bilinear', antialias=True, align_corners=False)
    x = ((x - classifier.mean) / classifier.std).to(classifier.device)
    with torch.inference_mode():
        return torch.softmax(classifier.model(x).float(), 1).cpu().numpy()


def cut(image, box):
    x, y, w, h = box
    H, W = image.shape[:2]
    x0, y0, side = crop_box([x, y, w, h], W, H)
    return image[y0:y0+side, x0:x0+side]


def paired_matches(gt, boxes, scores, threshold):
    # COCO-like score-ordered one-to-one matching for fixed-threshold counts.
    overlaps = box_iou(gt, boxes)
    used = np.zeros(len(gt), bool)
    matches = {}
    for j in np.argsort(-scores, kind='stable'):
        column = overlaps[:, j].copy()
        column[used] = -1
        if len(column) and column.max() >= threshold:
            k = int(column.argmax())
            matches[int(j)] = k
            used[k] = True
    return matches


def classifier_metrics(table):
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    y = np.array([CLASSES.index(r['true']) for r in table])
    p = np.array([[r[f'p_{c}'] for c in CLASSES] for r in table])
    pred = p.argmax(1)
    cm = np.bincount(y*5+pred, minlength=25).reshape(5,5)
    pp, rr, ff, support = precision_recall_fscore_support(y, pred, labels=list(range(5)), zero_division=0)
    result = dict(crops=len(y), accuracy=float((pred==y).mean()), macro_precision=float(pp.mean()),
                  macro_recall=float(rr.mean()), macro_f1=float(ff.mean()), confusion_matrix=cm.tolist(),
                  per_class={c:dict(precision=float(pp[k]), recall=float(rr[k]), f1=float(ff[k]), support=int(support[k])) for k,c in enumerate(CLASSES)})
    if len(np.unique(y)) == 5:
        result['roc_auc_ovr_macro'] = float(roc_auc_score(y, p, multi_class='ovr', average='macro'))
    return result


def prf(tp, fp, fn):
    p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.0)


def main():
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; frozen probe expects the project GPU environment')
    EVAL.mkdir(parents=True, exist_ok=True)
    manifest = read(OUT/'image_manifest.json')
    retained = [r for r in manifest if not r['exclude_reasons']]
    if not retained:
        raise RuntimeError('No eligible images after screening')
    detector_meta = read(ROOT/'models/phase1/model_metadata.json')
    classifier_meta = read(ROOT/'models/phase2/model_metadata.json')
    weights = ROOT/'models/phase1/best_acne_detector.pt'
    clf_weights = ROOT/'models/phase2/best_acne_type.pth'
    if digest(weights) != detector_meta['selected_weights_sha256'] or digest(clf_weights) != classifier_meta['weights_sha256']:
        raise RuntimeError('Deployed checkpoint/metadata hash mismatch')
    protocol=dict(locked_utc=datetime.now(timezone.utc).isoformat(), status='exploratory_external_probe',
                  screened_annotations_sha256=digest(DERIVED/'annotations.coco.json'),
                  image_manifest_sha256=digest(OUT/'image_manifest.json'), images=len(retained),
                  detector_sha256=digest(weights),classifier_sha256=digest(clf_weights),
                  detector_imgsz=detector_meta['imgsz'], detector_confidence=float(detector_meta['operating_conf']),
                  raw_confidence=.001, nms_iou=.7, max_det=500, classifier_crop_context=2, classifier_min_crop_side=32,
                  classifier_input=224, classifier_mode='FP32, inference_mode, no augmentation, argmax of five classes',
                  binary_acne_source_classes=sorted(ACNE), type_class_mapping=TYPE_MAP,
                  pipeline_iou_thresholds=[.3,.5], pipeline_matching='class-aware score-ordered greedy, one-to-one within each class',
                  no_training=True, no_threshold_tuning=True, no_patient_independence_claim=True,
                  scar_subtypes_evaluated=False, scar_reason='No atrophic_scar or hypertrophic_scar labels in source',
                  warnings=['Source labels unreviewed by dermatologist', 'Labels assumed complete for FP counts',
                            'Nodule/Nodules mapping provisional; Cysts excluded from five-type evaluation',
                            'Archive 640x640 export, original camera resolution unavailable',
                            'No confidence calibration or formal equivalence claims'])
    write_json(EVAL/'protocol_locked_before_inference.json',protocol)
    print('LOCKED PROTOCOL',json.dumps(protocol),flush=True)
    detector=make_predictor(weights,detector_meta['imgsz'])
    classifier=TypeClassifier(clf_weights)
    if classifier.classes != CLASSES:
        raise RuntimeError('Classifier class order differs from protocol')
    records,predictions=[],[]
    gt_results, detection_results=[],[]
    pending_crops,pending_rows=[],[]
    pipeline={t:{c:dict(tp=0,fp=0,fn=0) for c in CLASSES} for t in [.3,.5]}
    crop_counts_by_image=[]

    def flush():
        if not pending_crops:
            return
        probs=probability(classifier,pending_crops)
        for row,p in zip(pending_rows,probs):
            row.update(pred=CLASSES[int(p.argmax())],**{f'p_{c}':float(p[k]) for k,c in enumerate(CLASSES)})
            gt_results.append(row)
        pending_crops.clear();pending_rows.clear()

    for idx,row in enumerate(retained):
        image=cv2.imread(row['path'])
        if image is None or image.shape[:2]!=(row['height'],row['width']):
            raise RuntimeError(f'Screened image changed: {row["file_name"]}')
        if digest(row['path']) != row['sha256']:
            raise RuntimeError(f'Screened image bytes changed: {row["file_name"]}')
        acne=[a for a in row['annotations'] if a['category'] in ACNE]
        supported=[a for a in row['annotations'] if a['category'] in TYPE_MAP]
        xyxy=lambda aa:np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in aa],dtype=float).reshape(-1,4)
        gt=xyxy(acne);gt_type=xyxy(supported)
        records.append(dict(file_name=row['file_name'],path=Path(row['path']),width=row['width'],height=row['height'],gt=gt))
        types=np.array([CLASSES.index(TYPE_MAP[a['category']]) for a in supported],dtype=int)
        boxes,scores,_=detector(image)
        order=np.argsort(-scores,kind='stable');boxes,scores=boxes[order],scores[order]
        predictions.append(dict(boxes=boxes,scores=scores))
        keep=scores>=protocol['detector_confidence']
        kept_boxes,kept_scores=boxes[keep],scores[keep]
        pred_type=[]
        crops=[cut(image,[b[0],b[1],b[2]-b[0],b[3]-b[1]]) for b in kept_boxes]
        for start in range(0,len(crops),64):
            probs=probability(classifier,crops[start:start+64])
            for j,p in enumerate(probs,start):
                pred_type.append(int(p.argmax()))
                detection_results.append(dict(uid=row['uid'],file_name=row['file_name'],detection_index=j,
                    x1=float(kept_boxes[j,0]),y1=float(kept_boxes[j,1]),x2=float(kept_boxes[j,2]),y2=float(kept_boxes[j,3]),
                    detector_confidence=float(kept_scores[j]),pred=CLASSES[int(p.argmax())],**{f'p_{c}':float(p[k]) for k,c in enumerate(CLASSES)}))
        pred_type=np.array(pred_type,dtype=int)
        for threshold in pipeline:
            for k,c in enumerate(CLASSES):
                selected=pred_type==k
                matches=paired_matches(gt_type[types==k],kept_boxes[selected],kept_scores[selected],threshold)
                tp=len(matches)
                pipeline[threshold][c]['tp']+=tp
                pipeline[threshold][c]['fp']+=int(selected.sum())-tp
                pipeline[threshold][c]['fn']+=int((types==k).sum())-tp
        for a in supported:
            pending_crops.append(cut(image,a['bbox']))
            pending_rows.append(dict(uid=row['uid'],file_name=row['file_name'],source_annotation_id=a['id'],
                                     source_category=a['category'],true=TYPE_MAP[a['category']]))
            if len(pending_crops)==64:
                flush()
        crop_counts_by_image.append(dict(uid=row['uid'],file_name=row['file_name'],acne_labels=len(acne),
                                        supported_type_labels=len(supported),detected_boxes=int(keep.sum())))
        if (idx+1)%50==0:
            print(f'Frozen inference {idx+1}/{len(retained)}, classified GT crops {len(gt_results)}',flush=True)
    flush()
    offsets=np.cumsum([0]+[len(p['scores']) for p in predictions])
    np.savez_compressed(EVAL/'detector_raw_predictions.npz',boxes=np.concatenate([p['boxes'] for p in predictions]),
                        scores=np.concatenate([p['scores'] for p in predictions]),offsets=offsets,
                        file_names=np.array([r['file_name'] for r in records]))
    write_csv(EVAL/'classifier_gt_crop_predictions.csv',gt_results)
    write_csv(EVAL/'pipeline_predictions.csv',detection_results)
    write_csv(EVAL/'per_image_counts.csv',crop_counts_by_image)
    metrics,counts=compute_metrics(records,predictions,conf_op=protocol['detector_confidence'])
    binary={k:metrics[k] for k in ['images','gt_boxes','predictions','mAP50','mAP50_95','operating_point']}
    pipe={}
    for threshold,cc in pipeline.items():
        per={c:prf(**v) for c,v in cc.items()}
        totals={k:sum(v[k] for v in cc.values()) for k in ['tp','fp','fn']}
        pipe[f'iou{threshold}']=dict(per_class=per,micro=prf(**totals),
                                   macro={k:float(np.mean([v[k] for v in per.values()])) for k in ['precision','recall','f1']})
    result=dict(protocol=protocol,phase1_binary_detection=binary,phase2_gt_crop_classifier=classifier_metrics(gt_results),
                phase1_plus_phase2_pipeline=pipe,images_without_acne_annotations=sum(len(r['gt'])==0 for r in records),
                unverified_negative_label_warning='No-acne annotations do not establish dermatologist-confirmed clear skin',
                note='Scores are exploratory against supplied labels; not a final independent test or proof of clinical performance')
    write_json(EVAL/'metrics.json',result)
    write_csv(EVAL/'binary_per_image_counts.csv',counts)
    print('FROZEN METRICS',json.dumps({k:v for k,v in result.items() if k!='protocol'}),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
