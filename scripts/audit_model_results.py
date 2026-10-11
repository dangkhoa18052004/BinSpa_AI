"""Audit frozen results without modifying training/evaluation artifacts.

Default: replay saved predictions, check splits/hashes, recompute classification
metrics and cluster bootstrap. --inference additionally checks checkpoint outputs.
All new artifacts go to reports/model_audit/. No model selection or training.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
OUT = ROOT / 'reports/model_audit'
CLASSES = ['closed_comedo', 'open_comedo', 'papule', 'pustule', 'nodule']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit_results.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def progress(label):
    print(label, flush=True)


def source_path(value):
    # Preserve the original dataset; resolve old D:/BinSpa_AI artifact paths here.
    s = str(value).replace('\\', '/')
    marker = '/BinSpa_AI/'
    if marker in s:
        return ROOT / s.split(marker, 1)[1]
    return Path(value)


def prf(tp, fp, fn):
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)


def datasets():
    from common import verify_dataset
    manifest = verify_dataset(require_review=True)
    master = rows(ROOT / 'datasets/acnedet_master_split.csv')
    by_id = {r['source_id']: r for r in master}
    groups = defaultdict(set)
    hashes = defaultdict(list)
    for r in master:
        groups[r['group']].add(r['split'])
        p = ROOT / 'data_raw/acnedet_roboflow/acnedet.v1i.coco/train' / r['file_name']
        hashes[sha(p)].append((r['source_id'], r['split']))
    source_duplicates = [v for v in hashes.values() if len(v) > 1]
    crops = rows(ROOT / 'datasets/phase2_acne_type/crop_manifest.csv')
    crop_errors = [r['crop_path'] for r in crops if r['split'] != by_id[r['source_id']]['split']]
    scar_counts = {s: Counter() for s in ['train', 'val', 'test']}
    derived_errors = []
    coco = read(ROOT / 'data_raw/acnedet_roboflow/acnedet.v1i.coco/train/_annotations.coco.json')
    cats = {c['id']: c['name'] for c in coco['categories']}
    anns = defaultdict(Counter)
    for a in coco['annotations']:
        anns[a['image_id']][cats[a['category_id']]] += 1
    for r in master:
        s, stem = r['split'], f'acnedet_{int(r["source_id"]):03d}'
        expected_acne = sum(anns[int(r['image_id'])][c] for c in CLASSES)
        for ds, expected in [('phase3_scar_detection', None), ('phase1_combined_acnedet', expected_acne)]:
            lab = ROOT / 'datasets' / ds / 'labels' / s / (stem + '.txt')
            image = ROOT / 'datasets' / ds / 'images' / s / (stem + '.jpg')
            if not lab.is_file() or not image.is_file():
                derived_errors.append(f'missing {ds}/{s}/{stem}')
                continue
            if sha(image) not in hashes or (r['source_id'], s) not in hashes[sha(image)]:
                derived_errors.append(f'image source mismatch {ds}/{s}/{stem}')
            lines = lab.read_text().splitlines()
            if expected is not None and len(lines) != expected:
                derived_errors.append(f'box count mismatch {ds}/{s}/{stem}')
            counts = Counter()
            for line in lines:
                cls, cx, cy, w, h = map(float, line.split())
                if not (np.isfinite([cls, cx, cy, w, h]).all() and 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    derived_errors.append(f'invalid bbox {ds}/{s}/{stem}')
                counts[int(cls)] += 1
            if expected is None:
                scar_counts[s].update(counts)
                for k, c in enumerate(['atrophic_scar', 'hypertrophic_scar']):
                    if counts[k] != anns[int(r['image_id'])][c]:
                        derived_errors.append(f'scar count mismatch {s}/{stem}/{c}')
    return dict(phase1_integrity=True, phase1_split=manifest['counts'],
                phase1_verified_negative_images=sum(r['lesions'] == 0 for r in manifest['images']),
                acnedet_images=len(master), master_groups=len(groups),
                split_images=dict(Counter(r['split'] for r in master)),
                split_groups={s: len({r['group'] for r in master if r['split'] == s}) for s in ['train', 'val', 'test']},
                cross_split_groups=[g for g, v in groups.items() if len(v) > 1],
                exact_duplicate_source_images=source_duplicates, crop_split_errors=crop_errors,
                crop_counts=dict(Counter(r['split'] for r in crops)),
                scar_label_counts={s: dict(v) for s, v in scar_counts.items()}, derived_errors=derived_errors)


def phase1():
    from evaluate_detector_common import compute_metrics, load_split
    records = {s: load_split(s) for s in ['val', 'test']}
    paths = sorted((ROOT / 'reports/phase1_additional_models/common_eval').glob('*/common_eval_summary.json'))
    paths += sorted((ROOT / 'reports/phase1_additional_models').glob('*_1280/common_eval_summary.json'))
    paths += sorted((ROOT / 'reports/phase1_combined').glob('*/common_eval_summary.json'))
    result = []
    for path in paths:
        summ = read(path)
        item = dict(model=summ['model'], summary=str(path.relative_to(ROOT)),
                    checkpoint_hash_matches=sha(source_path(summ['weights'])) == summ['weights_sha256'])
        for s in ['val', 'test']:
            saved = read(path.parent / f'eval_{s}/predictions.json')
            if [p['file_name'] for p in saved] != [r['file_name'] for r in records[s]]:
                raise RuntimeError(f'Prediction/manifest order mismatch: {path} {s}')
            pred = [dict(boxes=np.array(p['boxes']).reshape(-1, 4), scores=np.array(p['scores'])) for p in saved]
            conf = summ['val']['operating_point']['conf']
            m, _ = compute_metrics(records[s], pred, conf_op=conf)
            orig = summ[s]
            item[s] = dict(recomputed_mAP50=m['mAP50'], recomputed_mAP50_95=m['mAP50_95'],
                           ap50_delta=m['mAP50'] - orig['mAP50'], ap_delta=m['mAP50_95'] - orig['mAP50_95'],
                           recorded=orig['operating_point'], recomputed=m['operating_point'],
                           count_deltas={k: m['operating_point'][k] - orig['operating_point'][k] for k in ['tp', 'fp', 'fn']},
                           val_conf_applied_to_test=conf == orig['operating_point']['conf'])
        result.append(item)
        progress(f'replayed phase1 {summ["model"]}: AP delta {item["test"]["ap_delta"]:+.6f}, counts {item["test"]["count_deltas"]}')
    return result


def cm_metrics(cm):
    tp = np.diag(cm)
    p = np.divide(tp, cm.sum(0), out=np.zeros(5), where=cm.sum(0) != 0)
    r = np.divide(tp, cm.sum(1), out=np.zeros(5), where=cm.sum(1) != 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros(5), where=p + r != 0)
    return dict(accuracy=float(tp.sum() / cm.sum()), macro_f1=float(f.mean()),
                macro_precision=float(p.mean()), macro_recall=float(r.mean()), confusion_matrix=cm.tolist())


def phase2():
    from sklearn.metrics import roc_auc_score
    original = read(ROOT / 'reports/phase2_evaluation/test_metrics.json')
    master = {r['source_id']: r for r in rows(ROOT / 'datasets/acnedet_master_split.csv')}
    test_crops = {r['crop_path']: r for r in rows(ROOT / 'datasets/phase2_acne_type/crop_manifest.csv') if r['split'] == 'test'}
    tables, result = {}, {}
    for arch in ['convnext_tiny', 'efficientnet_b0']:
        table = rows(ROOT / f'reports/phase2_evaluation/test_predictions_{arch}.csv')
        y = np.array([CLASSES.index(r['true']) for r in table])
        p = np.array([[float(r[f'p_{c}']) for c in CLASSES] for r in table])
        pred = p.argmax(1)
        cm = np.bincount(y * 5 + pred, minlength=25).reshape(5, 5)
        m = cm_metrics(cm)
        m['roc_auc_ovr_macro'] = float(roc_auc_score(y, p, multi_class='ovr', average='macro'))
        result[arch] = dict(recomputed=m, deltas={k: m[k] - original['test'][arch][k] for k in m if k != 'confusion_matrix'},
                            checkpoint_hash_matches=sha(ROOT / f'runs/phase2/{arch}_cfg_a/best.pth') == original['test'][arch]['weights_sha256'],
                            confusion_matrix_matches=m['confusion_matrix'] == original['test'][arch]['confusion_matrix'],
                            predictions_match_argmax=all(r['pred'] == CLASSES[int(k)] for r, k in zip(table, pred)),
                            crop_truth_errors=[r['crop_path'] for r in table if r['crop_path'] not in test_crops or
                                               (r['true'], r['source_id']) != (test_crops[r['crop_path']]['cls'], test_crops[r['crop_path']]['source_id'])],
                            non_test_sources=sum(master[r['source_id']]['split'] != 'test' for r in table))
        tables[arch] = (table, y, pred)
    if [r['crop_path'] for r in tables['convnext_tiny'][0]] != [r['crop_path'] for r in tables['efficientnet_b0'][0]]:
        raise RuntimeError('Phase2 paired predictions order mismatch')
    boots = {}
    for unit in ['source_image', 'master_group']:
        group = np.array([r['source_id'] if unit == 'source_image' else master[r['source_id']]['group'] for r in tables['convnext_tiny'][0]])
        if unit == 'source_image':
            group = group.astype(int)  # Match the original bootstrap's numeric source ordering.
        ids = np.unique(group)
        cms = {}
        for arch, (_, y, pred) in tables.items():
            cms[arch] = np.array([np.bincount((y * 5 + pred)[group == g], minlength=25).reshape(5, 5) for g in ids])
        rng = np.random.default_rng(42)
        draws = {arch: [] for arch in tables}
        for _ in range(1000):
            idx = rng.integers(0, len(ids), len(ids))
            for arch in tables:
                draws[arch].append(cm_metrics(cms[arch][idx].sum(0))['macro_f1'])
        diff = np.array(draws['convnext_tiny']) - np.array(draws['efficientnet_b0'])
        boots[unit] = dict(units=len(ids), n_boot=1000, seed=42,
                          macro_f1_ci={a: np.percentile(v, [2.5, 97.5]).tolist() for a, v in draws.items()},
                          paired_difference_convnext_minus_efficientnet_ci=np.percentile(diff, [2.5, 97.5]).tolist())
    return dict(models=result, bootstrap=boots, recorded_bootstrap=original['bootstrap'])


def phase3():
    original = read(ROOT / 'reports/phase3_evaluation/test_metrics.json')
    result = {}
    for arch, m in original['test'].items():
        per = {c: dict(**v, **prf(v['tp'], v['fp'], v['fn'])) for c, v in m['counts_at_val_f1_conf'].items()}
        totals = {k: sum(v[k] for v in per.values()) for k in ['tp', 'fp', 'fn']}
        result[arch] = dict(conf_op=m['conf_op'], test_max_f1_conf=m['f1_max_conf'],
                            reported_precision=m['precision'], reported_recall=m['recall'],
                            at_val_conf_per_class=per, at_val_conf_micro=dict(**totals, **prf(**totals)),
                            at_val_conf_macro={k: float(np.mean([v[k] for v in per.values()])) for k in ['precision', 'recall', 'f1']})
    return result


def selection_checks():
    vr = rows(ROOT / 'reports/phase3_evaluation/validation_runs.csv')
    reps = {}
    for arch in ['yolo26s', 'yolo11s']:
        subset = [r for r in vr if r['arch'] == arch]
        reps[arch] = max(subset, key=lambda r: float(r['val_mAP50_95']))
    best = max(float(r['val_mAP50_95']) for r in reps.values())
    tied = [r for r in reps.values() if best - float(r['val_mAP50_95']) <= .002]
    chosen = max(tied, key=lambda r: (float(r['val_hs_recall']), -float(r['val_infer_ms'])))
    meta = read(ROOT / 'models/phase1/model_metadata.json')
    combined = read(ROOT / 'reports/phase1_combined/comparison.json')
    new = combined['candidates'][combined['selection']['chosen']]
    comparison = {k: dict(metadata=meta['test'][f'test_{k}'], current_artifact=new[f'acne04_test_{k}']) for k in ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1']}
    return dict(phase3_documented_rule_chosen=chosen['arch'],
                phase3_actual_chosen=read(ROOT / 'reports/phase3_evaluation/selection.json')['chosen']['arch'],
                bucket_rounding_counterexample=dict(a=.09899, b=.09901, difference=.00002, tolerance=.002,
                                                     buckets=[round(.09899/.002), round(.09901/.002)]),
                phase1_metadata_vs_current_checkpoint_metrics=comparison)


def provenance():
    deployed = {}
    for phase, filename, key in [('phase1', 'best_acne_detector.pt', 'selected_weights_sha256'),
                                 ('phase2', 'best_acne_type.pth', 'weights_sha256'),
                                 ('phase3', 'best_scar_detector.pt', 'weights_sha256')]:
        meta = read(ROOT / 'models' / phase / 'model_metadata.json')
        actual = sha(ROOT / 'models' / phase / filename)
        deployed[phase] = dict(sha256=actual, matches_metadata=actual == meta[key])
    return dict(deployed_checkpoints=deployed, dataset_files_sha256={
        p: sha(ROOT / p) for p in ['datasets/acnedet_master_split.csv',
                                  'datasets/phase2_acne_type/crop_manifest.csv',
                                  'data_raw/acnedet_roboflow/acnedet.v1i.coco/train/_annotations.coco.json']})


def inference_checks():
    import cv2
    import torch
    from ultralytics import YOLO
    from evaluate_detector_common import load_split, make_predictor, box_iou
    from train_classifier import CropSet, predict, transforms
    from evaluate_classifier import load_model
    if not torch.cuda.is_available():
        return dict(status='skipped: CUDA unavailable')
    result = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    manifest = rows(ROOT / 'datasets/phase2_acne_type/crop_manifest.csv')
    test = [r for r in manifest if r['split'] == 'test']
    cache = ROOT / 'datasets/phase2_acne_type/cache_test_256.npy'
    x = np.load(cache, mmap_mode='r')
    cache_errors = []
    for i in np.linspace(0, len(test)-1, 20).astype(int):
        fresh = cv2.resize(cv2.imread(str(ROOT / 'datasets/phase2_acne_type' / test[i]['crop_path'])), (256,256), interpolation=cv2.INTER_AREA)[:,:,::-1]
        if not np.array_equal(fresh, x[i]): cache_errors.append(int(i))
    result['phase2_cache_sample_errors'] = cache_errors
    result['phase2'] = {}
    y = np.array([CLASSES.index(r['cls']) for r in test])
    for arch in ['efficientnet_b0', 'convnext_tiny']:
        model, ckpt = load_model(ROOT / f'runs/phase2/{arch}_cfg_a/best.pth')
        loader = torch.utils.data.DataLoader(CropSet(cache, y, transforms(False, ckpt['mean'], ckpt['std'])), batch_size=64)
        prob = predict(model, loader)
        saved = rows(ROOT / f'reports/phase2_evaluation/test_predictions_{arch}.csv')
        if [r['crop_path'] for r in saved] != [r['crop_path'] for r in test]:
            raise RuntimeError(f'Classifier cache/manifest/prediction order mismatch: {arch}')
        prior = np.array([[float(r[f'p_{c}']) for c in CLASSES] for r in saved])
        result['phase2'][arch] = dict(crops=len(y), prediction_changes=int((prob.argmax(1)!=prior.argmax(1)).sum()),
                                     max_probability_delta=float(np.max(np.abs(prob-prior))))
        progress(f'inference phase2 {arch}: {result["phase2"][arch]}')
        del model
        torch.cuda.empty_cache()
    # Full scar test replay at the deployed validation confidence, greedy by score.
    original = read(ROOT / 'reports/phase3_evaluation/test_metrics.json')
    result['phase3'] = {}
    for arch, prior in original['test'].items():
        sel = original['selection']['representatives'][arch]
        model = YOLO(str(source_path(sel['weights'])))
        totals = {c: dict(tp=0, fp=0, fn=0) for c in ['atrophic_scar','hypertrophic_scar']}
        for path in sorted((ROOT/'datasets/phase3_scar_detection/images/test').glob('*.jpg')):
            r = model.predict(str(path), imgsz=sel['imgsz'], conf=sel['val_f1_conf'], iou=.7, max_det=500, device=0, verbose=False)[0]
            H, W = r.orig_shape
            labels = np.array([list(map(float,l.split())) for l in (ROOT/'datasets/phase3_scar_detection/labels/test'/f'{path.stem}.txt').read_text().splitlines()]).reshape(-1,5)
            for k, c in enumerate(totals):
                lab = labels[labels[:,0]==k,1:]
                gt = np.column_stack(((lab[:,0]-lab[:,2]/2)*W,(lab[:,1]-lab[:,3]/2)*H,(lab[:,0]+lab[:,2]/2)*W,(lab[:,1]+lab[:,3]/2)*H))
                keep = r.boxes.cls.cpu().numpy()==k
                d = r.boxes.xyxy.cpu().numpy()[keep][np.argsort(-r.boxes.conf.cpu().numpy()[keep])]
                iou = box_iou(gt,d)
                used = np.zeros(len(gt),bool)
                tp=0
                for j in range(len(d)):
                    v=iou[:,j].copy();v[used]=-1
                    if len(v) and v.max()>=.5:
                        used[v.argmax()]=True;tp+=1
                totals[c]['tp']+=tp;totals[c]['fp']+=len(d)-tp;totals[c]['fn']+=len(gt)-tp
        result['phase3'][arch] = dict(images=45, counts=totals,
                                     matches_recorded=totals==prior['counts_at_val_f1_conf'])
        progress(f'inference phase3 {arch}: {result["phase3"][arch]}')
        del model
        torch.cuda.empty_cache()
    # Original full-resolution test images, current selected phase1 checkpoint.
    meta = read(ROOT/'models/phase1/model_metadata.json')
    saved = read(ROOT/'reports/phase1_combined/rtdetr_ft/eval_test/predictions.json')
    records = load_split('test')
    predictor = make_predictor(ROOT/'models/phase1/best_acne_detector.pt',meta['imgsz'])
    sampled=[]
    for i in np.linspace(0,len(records)-1,10).astype(int):
        boxes,scores,_=predictor(cv2.imread(str(records[i]['path'])))
        order=np.argsort(-scores,kind='stable');boxes,scores=boxes[order],scores[order]
        previous=saved[i]
        same=boxes.shape==np.array(previous['boxes']).shape
        sampled.append(dict(file_name=records[i]['file_name'],same_number_of_raw_boxes=same,
                            max_box_delta_px=float(np.max(np.abs(boxes-np.array(previous['boxes'])))) if same else None,
                            max_score_delta=float(np.max(np.abs(scores-np.array(previous['scores'])))) if same else None))
    result['phase1_sample']=sampled
    progress('inference phase1: 10 held-out images replayed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference',action='store_true')
    a=p.parse_args()
    data=dict(created_utc=datetime.now(timezone.utc).isoformat(), scope='frozen artifacts; no training or model selection')
    for name, fn in [('datasets',datasets),('provenance',provenance),('phase1_replay',phase1),('phase2_replay',phase2),('phase3_operating_point',phase3),('selection_checks',selection_checks)]:
        progress(f'Audit {name}')
        data[name]=fn();save(data)
    if a.inference:
        progress('Audit checkpoint inference')
        data['inference']=inference_checks();save(data)
    progress(f'Audit complete: {OUT / "audit_results.json"}')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
