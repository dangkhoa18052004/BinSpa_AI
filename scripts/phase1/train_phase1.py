"""Smoke and benchmark, with explicit OOM fallback and validation-only model selection."""
import argparse
import gc
import importlib.metadata
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from common import ROOT, DATASET, REPORTS, digest, read_json, verify_dataset, write_csv, write_json
from evaluate_phase1 import evaluate, error_analysis

RUNS=ROOT/'runs/phase1'
STATUS=REPORTS/'phase1_training_status.json'
MODELS=('yolo26n','yolo26s','yolo11s')


def status(**data):
    write_json(STATUS,dict(updated_utc=datetime.now(timezone.utc).isoformat(),**data))


def check_environment(device):
    import torch
    if device!='cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; install CUDA torch. CPU training must be explicitly requested.')
        torch.zeros((8,8),device='cuda') @ torch.ones((8,8),device='cuda')
        torch.cuda.synchronize()
    env=dict(python=__import__('sys').version,packages={p:importlib.metadata.version(p) for p in ('torch','torchvision','ultralytics')},cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,torch_cuda=torch.version.cuda)
    write_json(REPORTS/'phase1_environment.json',env)
    print(env,flush=True)


def train_one(name, dataset, epochs, imgsz, batch, device, seed, patience, stage, workers, cache=False, recheck_amp=False):
    import torch
    from ultralytics import YOLO
    weights_dir=ROOT/'models/phase1/pretrained'
    weights_dir.mkdir(parents=True,exist_ok=True)
    # Use a local destination so downloads never escape the workspace.
    source=weights_dir/(name+'.pt')
    attempts=[]
    for candidate in dict.fromkeys([batch, min(batch,4), min(batch,2), 1]):
        run_name=f'{stage}_{name}_e{epochs}_s{seed}_b{candidate}_w{workers}'
        if stage=='benchmark':
            run_name += '_cram' if cache=='ram' else '_cfalse'
        run=RUNS/run_name
        if (run/'completed.json').exists():
            saved=read_json(run/'completed.json')
            # Ultralytics can call on_fit_epoch_end once more after final validation.
            if saved['actual_epochs']>saved['epochs']:
                saved['actual_epochs']=saved['epochs']
                write_json(run/'completed.json',saved)
            if saved['manifest_sha256']!=digest(dataset/'manifest.json') or saved['imgsz']!=imgsz or saved['epochs']!=epochs or saved['patience']!=patience:
                raise RuntimeError('Completed run config differs; choose a new run configuration')
            if digest(run/'weights/best.pt')!=saved['best_sha256']:
                raise RuntimeError('Completed checkpoint changed')
            return run/'weights/best.pt',saved['batch']
        if (run/'oom.json').exists():
            attempts.append(read_json(run/'oom.json'))
            continue
        if run.exists():
            # Interrupted runs are resumed by the explicit --resume option, not silently reset.
            raise RuntimeError(f'Incomplete run exists: {run}. Use --resume {run/"weights/last.pt"} or inspect it before retrying.')
        model=None
        import ultralytics.engine.trainer as trainer_module
        original_amp_check=trainer_module.check_amp
        try:
            status(state='training',stage=stage,model=name,epochs=epochs,batch=candidate,run=str(run),attempts=attempts)
            write_json(REPORTS/'phase1_run_contracts'/(run_name+'.json'),dict(manifest_sha256=digest(dataset/'manifest.json'),epochs=epochs,imgsz=imgsz,batch=candidate,patience=patience,seed=seed,stage=stage,cache=cache))
            model=YOLO(str(source))
            certificate=REPORTS/'phase1_amp_certificate.json'
            if stage=='benchmark' and not recheck_amp and certificate.is_file():
                certified=read_json(certificate)
                current=read_json(REPORTS/'phase1_environment.json')
                if certified.get('environment')==current and certified.get('device')==str(device) and certified.get('amp_enabled') is True:
                    trainer_module.check_amp=lambda _: True
                    print('Reusing AMP validation from completed 10-epoch smoke on the same GPU/library versions',flush=True)
                    write_json(REPORTS/'phase1_amp_check_policy.json',dict(policy='reuse_verified_smoke_amp',certificate=str(certificate),reason='Repeated FP16 convolution in Ultralytics AMP self-test stalled; smoke AMP check passed and 10 epochs completed with finite losses. Monitor finite validation metrics during benchmark.'))
            def on_epoch_end(trainer):
                if not all(math.isfinite(float(v)) for v in trainer.metrics.values()):
                    raise RuntimeError('Non-finite validation metrics/losses; aborting benchmark')
                status(state='training',stage=stage,model=name,epoch=min(trainer.epoch+1,epochs),epochs=epochs,batch=int(trainer.batch_size),run=str(trainer.save_dir),metrics={k:float(v) for k,v in trainer.metrics.items()})
            model.add_callback('on_fit_epoch_end',on_epoch_end)
            model.train(data=str(dataset/'data.yaml'),epochs=epochs,imgsz=imgsz,batch=candidate,device=device,seed=seed,deterministic=True,patience=patience,workers=workers,amp=True,cache=cache,optimizer='AdamW',lr0=.001,lrf=.01,weight_decay=.0005,close_mosaic=min(10,epochs),project=str(RUNS),name=run_name,exist_ok=False,plots=True,max_det=500)
            actual=Path(model.trainer.save_dir)
            best=actual/'weights/best.pt'
            if not best.is_file():
                raise RuntimeError('Training produced no best checkpoint')
            actual_batch=int(model.trainer.batch_size)
            effective_cache=model.trainer.train_loader.dataset.cache
            write_json(actual/'completed.json',dict(model=name,epochs=epochs,actual_epochs=min(model.trainer.epoch+1,epochs),imgsz=imgsz,batch=actual_batch,requested_batch=candidate,workers=workers,cache_requested=cache,cache_effective=effective_cache,amp_enabled=bool(model.trainer.amp),patience=patience,seed=seed,manifest_sha256=digest(dataset/'manifest.json'),best_sha256=digest(best)))
            return best,actual_batch
        except Exception as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            attempts.append(dict(batch=candidate,error=str(exc)))
            if run.exists():
                write_json(run/'oom.json',attempts[-1])
            print(f'OOM at batch {candidate}; trying smaller batch',flush=True)
        finally:
            trainer_module.check_amp=original_amp_check
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError('OOM even at batch 1; inspect available VRAM before changing resolution')


def benchmark(args):
    manifest=verify_dataset(args.dataset,require_review=True)
    check_environment(args.device)
    if args.smoke:
        best,batch=train_one('yolo26n',args.dataset,10,args.imgsz,args.batch,args.device,args.seed,10,'smoke',args.workers)
        metrics=evaluate(best,args.dataset,'val',args.imgsz,batch,args.device)
        write_json(REPORTS/'phase1_smoke.json',metrics)
        completed=read_json(best.parent.parent/'completed.json')
        if completed.get('amp_enabled') is True and completed['actual_epochs']==10:
            write_json(REPORTS/'phase1_amp_certificate.json',dict(environment=read_json(REPORTS/'phase1_environment.json'),device=str(args.device),amp_enabled=True,smoke_weights_sha256=digest(best),manifest_sha256=digest(args.dataset/'manifest.json'),validation_metrics=metrics))
        status(state='smoke_complete',metrics=metrics)
        return
    if not (REPORTS/'phase1_smoke.json').exists():
        raise RuntimeError('Run --smoke first')
    smoke=read_json(REPORTS/'phase1_smoke.json')
    if smoke['manifest_sha256']!=digest(args.dataset/'manifest.json'):
        raise RuntimeError('Smoke used a different dataset')
    import psutil
    # Resized uint8 BGR images, not full-resolution original images. Keep a 4 GiB reserve.
    cache_bytes=sum(int(r['width']*args.imgsz/max(r['width'],r['height'])+1)*int(r['height']*args.imgsz/max(r['width'],r['height'])+1)*3 for r in manifest['images'] if r['split']!='test')
    available=psutil.virtual_memory().available
    cache='ram' if args.cache=='ram' or (args.cache=='auto' and available>2*cache_bytes+4*1024**3) else False
    write_json(REPORTS/'phase1_cache_policy.json',dict(requested=args.cache,chosen=cache,estimated_resized_bytes=cache_bytes,available_bytes=available,reserve_bytes=4*1024**3,workers=args.workers))
    print('Benchmark cache policy:',cache,'estimated GiB:',cache_bytes/1024**3,flush=True)
    rows=[]
    for name in MODELS:
        best,batch=train_one(name,args.dataset,args.epochs,args.imgsz,args.batch,args.device,args.seed,args.patience,'benchmark',args.workers,cache,args.recheck_amp)
        result=evaluate(best,args.dataset,'val',args.imgsz,batch,args.device)
        result['architecture']=name
        rows.append(result)
        write_csv(REPORTS/'metrics/phase1_validation_benchmark.csv',rows,list(rows[0]))
    rows.sort(key=lambda r:(-r['mAP50_95'],-r['recall'],r['inference_ms']))
    chosen=rows[0]
    target=ROOT/'models/phase1/best_acne_detector.pt'
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(chosen['weights'],target)
    shutil.copy2(args.dataset/'data.yaml',target.parent/'data.yaml')
    source_run=Path(chosen['weights']).parent.parent
    shutil.copy2(source_run/'args.yaml',target.parent/'train_config.yaml')
    metadata=dict(status='selected_on_validation; pending_camera_and_region_review',architecture=chosen['architecture'],selection_rule='validation mAP50-95 descending, recall descending, inference_ms ascending',validation_ranking=rows,selected_weights_sha256=digest(target),dataset_manifest_sha256=digest(args.dataset/'manifest.json'),dataset_version=manifest['dataset_version'],seed=args.seed,split_counts=manifest['counts'],imgsz=args.imgsz,epochs=args.epochs,created_utc=datetime.now(timezone.utc).isoformat(),disclaimer='Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da.')
    # Lock selection before any test evaluation. Test cannot alter the winner.
    write_json(target.parent/'model_metadata.json',metadata)
    write_csv(REPORTS/'metrics/phase1_validation_ranking.csv',rows,list(rows[0]))
    test_rows=[]
    for row in rows:
        status(state='evaluating_test',model=row['architecture'],selected_model=chosen['architecture'])
        metric=evaluate(row['weights'],args.dataset,'test',args.imgsz,row['batch'],args.device)
        metric['architecture']=row['architecture'];test_rows.append(metric)
        write_csv(REPORTS/'metrics/phase1_test_benchmark.csv',test_rows,list(test_rows[0]))
    totals=error_analysis(target,args.dataset,args.imgsz,args.device)
    status(state='benchmark_complete',selected_model=chosen['architecture'],test_metrics=test_rows,error_counts=totals,remaining=['Manual false-positive anatomical region review','External camera images evaluation'])
    write_summary(manifest,rows,test_rows,totals,target,args.dataset)


def write_summary(manifest, validation, test, errors, target, dataset):
    lines=['# Báo cáo Giai đoạn 1','', 'Đã hoàn thành benchmark; nghiệm thu thực tế còn chờ ảnh camera ngoài dataset và kiểm tra false positive theo vùng.','',f'Dataset: {manifest["dataset_version"]}; split seed={manifest["seed"]}; {manifest["counts"]}.',f'Manifest SHA256: `{digest(dataset/"manifest.json")}`.','', 'Ảnh trùng SHA256 được loại trước split; giữ filename đầu tiên theo thứ tự từ điển. Nhãn invalid được ghi log; cách ly toàn bộ ảnh có nhãn invalid để tránh nhãn thiếu. Không có ID bệnh nhân, nên chưa xác nhận được độc lập theo người.','', '| Model | Val mAP50–95 | Test precision | Test recall | Test mAP50 | Test mAP50–95 | Inference ms |','|---|---:|---:|---:|---:|---:|---:|']
    for v in validation:
        t=next(r for r in test if r['architecture']==v['architecture'])
        lines.append(f'| {v["architecture"]} | {v["mAP50_95"]:.4f} | {t["precision"]:.4f} | {t["recall"]:.4f} | {t["mAP50"]:.4f} | {t["mAP50_95"]:.4f} | {t["inference_ms"]:.2f} |')
    lines+=['',f'Model chọn bằng validation: `{target}`.',f'Đếm lỗi tại conf=0.25, IoU=0.5: {errors}.','', '20 ảnh prediction: `reports/phase1_predictions`. Màu xanh lá: nhãn; cyan: prediction khớp; đỏ: prediction chưa khớp nhãn.','', 'Cần kiểm tra thủ công `false_positive_region_review.csv`, nhất là mắt/môi/nốt ruồi/tóc/râu. Annotation bỏ sót có thể tạo false positive biểu kiến. Chưa có ảnh camera ngoài dataset được cung cấp, chưa thể kết luận độ tin cậy trong môi trường spa.','', 'Không tự động chuyển sang GĐ2. Kết quả chỉ mang tính tham khảo chăm sóc da.']
    (REPORTS/'phase1_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DATASET)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',default='0')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--cache',choices=['auto','none','ram'],default='auto',help='Benchmark only: auto RAM cache when resized images fit with a 4 GiB reserve')
    p.add_argument('--recheck-amp',action='store_true',help='Repeat built-in AMP calibration rather than reuse the matching successful smoke certificate')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    try:
        if args.resume:
            from ultralytics import YOLO
            verify_dataset(args.dataset,require_review=True)
            check_environment(args.device)
            run=args.resume.parent.parent
            contract=read_json(REPORTS/'phase1_run_contracts'/(run.name+'.json'))
            if contract['manifest_sha256']!=digest(args.dataset/'manifest.json'):
                raise RuntimeError('Dataset changed since interrupted run; cannot resume')
            with (run/'args.yaml').open(encoding='utf-8') as f:
                import yaml
                saved=yaml.safe_load(f)
            # Interrupted run config is authoritative; never change split/epochs on resume.
            if Path(saved['data']).resolve()!=(args.dataset/'data.yaml').resolve():
                raise RuntimeError('Resume checkpoint belongs to a different dataset')
            model=YOLO(str(args.resume));model.train(resume=True)
            best=Path(model.trainer.save_dir)/'weights/best.pt'
            write_json(run/'completed.json',dict(model=Path(saved['model']).stem,epochs=saved['epochs'],actual_epochs=min(model.trainer.epoch+1,saved['epochs']),imgsz=saved['imgsz'],batch=int(model.trainer.batch_size),patience=saved['patience'],seed=saved['seed'],manifest_sha256=digest(args.dataset/'manifest.json'),best_sha256=digest(best)))
            status(state='resume_complete',run=str(run))
        else:
            benchmark(args)
    except Exception as exc:
        status(state='failed',error=str(exc),type=type(exc).__name__)
        raise


if __name__=='__main__':
    import multiprocessing
    from filelock import FileLock
    multiprocessing.freeze_support()
    REPORTS.mkdir(parents=True,exist_ok=True)
    with FileLock(str(REPORTS/'phase1_train.lock'),timeout=0):
        main()
