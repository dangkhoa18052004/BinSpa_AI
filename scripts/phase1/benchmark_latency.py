"""Dedicated latency benchmark for the representative model of each architecture.

The latency measured inside benchmark_additional_models.py (3 rounds over all test images)
was dominated by laptop clock-state changes (YOLO26n measured slower than YOLO26s). This
script controls for that:
1. End-to-end (deployment view): original photo (BGR array, decode excluded) -> final boxes,
   same 30 test images for every model, 5 rounds interleaved model-by-model, after warm-up;
   per-component pre/inference/post as in the common evaluator; median over all calls.
2. GPU model-only: forward pass timed with CUDA events, 10 interleaved rounds x 20 calls.
   YOLO/RT-DETR: fused model on a 1x3x640x640 tensor. Faster R-CNN: model(...) on one
   real image (long side 640 after its transform), incl. its internal NMS.
3. Static cost: parameters and GFLOPs at 640 (not affected by clock state).
All FP32, batch 1, CUDA. Writes reports/metrics/phase1_latency_benchmark.csv and adds the
columns to reports/metrics/phase1_all_models_comparison*.csv.
Usage: venv/Scripts/python.exe -u scripts/phase1/benchmark_latency.py
"""
import csv
import random
import subprocess
from datetime import datetime, timezone

import cv2
import numpy as np
import torch

from common import REPORTS, write_csv, write_json
from evaluate_detector_common import ADD, FasterRCNNPredictor, load_split, make_predictor

E2E_IMAGES, E2E_ROUNDS, GPU_ROUNDS, GPU_CALLS = 30, 5, 10, 20


def rows(name):
    with (REPORTS / 'metrics' / name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def gpu_model(predictor):
    """Return (callable forward on a fixed input, params, GFLOPs)."""
    if isinstance(predictor, FasterRCNNPredictor):
        model = predictor.model
        rec = load_split('test')[0]
        image = cv2.imread(str(rec['path']))
        s = 1024 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        x = [torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cuda().permute(2, 0, 1).float().div(255)]
        from torch.utils.flop_counter import FlopCounterMode
        with torch.inference_mode(), FlopCounterMode(display=False) as counter:
            model(x)
        gflops = counter.get_total_flops() / 1e9  # FLOPs (=2xMACs), same convention as Ultralytics get_flops
        return (lambda: model(x)), sum(p.numel() for p in model.parameters()), gflops
    net = predictor.model.model.fuse().eval().cuda().float()
    x = torch.zeros(1, 3, 640, 640, device='cuda')
    from ultralytics.utils.torch_utils import get_flops
    return (lambda: net(x)), sum(p.numel() for p in net.parameters()), get_flops(net, 640)


def gpu_clock():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout.strip()
        return out
    except Exception as exc:  # informational only
        return f'unavailable: {exc}'


def main():
    reps = rows('phase1_all_models_comparison.csv')
    models = {r['model']: dict(label=r['label'], weights=r['weights']) for r in reps}
    records = load_split('test')
    sample = random.Random(42).sample(records, E2E_IMAGES)
    images = [cv2.imread(str(r['path'])) for r in sample]
    predictors = {k: make_predictor(v['weights']) for k, v in models.items()}
    clocks = [dict(stage='start', clock=gpu_clock())]

    # 1. End-to-end, interleaved rounds
    for p in predictors.values():
        for image in images[:10]:
            p(image)
    e2e = {k: [] for k in predictors}
    for r in range(E2E_ROUNDS):
        for k, p in predictors.items():
            for image in images:
                e2e[k].append(p(image)[2])
        clocks.append(dict(stage=f'e2e round {r + 1}', clock=gpu_clock()))
        print(f'e2e round {r + 1}: ' + ', '.join(f'{k}={np.median([sum(t.values()) for t in e2e[k][-E2E_IMAGES:]]):.1f}ms' for k in e2e), flush=True)

    # 2. GPU model-only with CUDA events, interleaved rounds
    forwards, static = {}, {}
    for k, p in predictors.items():
        fn, params, gflops = gpu_model(p)
        forwards[k], static[k] = fn, dict(params_m=params / 1e6, gflops=gflops)
    gpu = {k: [] for k in forwards}
    with torch.inference_mode():
        for fn in forwards.values():
            for _ in range(30):
                fn()
        torch.cuda.synchronize()
        for r in range(GPU_ROUNDS):
            for k, fn in forwards.items():
                for _ in range(GPU_CALLS):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); fn(); end.record()
                    torch.cuda.synchronize()
                    gpu[k].append(start.elapsed_time(end))
        clocks.append(dict(stage='gpu done', clock=gpu_clock()))

    out = []
    for k, v in models.items():
        totals = np.array([sum(t.values()) for t in e2e[k]])
        g = np.array(gpu[k])
        out.append(dict(model=k, label=v['label'],
                        e2e_total_ms_median=float(np.median(totals)), e2e_total_ms_p10=float(np.percentile(totals, 10)),
                        e2e_total_ms_p90=float(np.percentile(totals, 90)),
                        e2e_preprocess_ms=float(np.median([t['preprocess'] for t in e2e[k]])),
                        e2e_inference_ms=float(np.median([t['inference'] for t in e2e[k]])),
                        e2e_postprocess_ms=float(np.median([t['postprocess'] for t in e2e[k]])),
                        gpu_forward_ms_median=float(np.median(g)), gpu_forward_ms_p10=float(np.percentile(g, 10)),
                        gpu_forward_ms_p90=float(np.percentile(g, 90)), **static[k],
                        e2e_calls=len(totals), gpu_calls=len(g)))
        print(out[-1], flush=True)
    write_csv(REPORTS / 'metrics/phase1_latency_benchmark.csv', out, list(out[0]))
    write_json(ADD / 'latency_benchmark_meta.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), gpu=torch.cuda.get_device_name(0), precision='FP32', batch=1,
        e2e=dict(images=E2E_IMAGES, rounds=E2E_ROUNDS, image_selection='random.Random(42).sample(test, 30)',
                 note='original photo array -> boxes in original pixels; JPEG decode excluded'),
        gpu_forward=dict(rounds=GPU_ROUNDS, calls_per_round=GPU_CALLS, timer='CUDA events',
                         input='1x3x640x640 zeros (YOLO/RT-DETR fused); one real test image for Faster R-CNN'),
        gpu_clock_samples=clocks))

    # merge into the comparison tables (all other columns unchanged)
    by_model = {r['model']: r for r in out}
    for name in ('phase1_all_models_comparison.csv', 'phase1_all_models_comparison_all_runs.csv', 'phase1_additional_models.csv'):
        table = rows(name)
        extra = [c for c in out[0] if c not in ('model', 'label')]
        for r in table:
            r.update({c: by_model[r['model']][c] for c in extra} if r['model'] in by_model else {})
        write_csv(REPORTS / 'metrics' / name, table, list(dict.fromkeys([*table[0], *extra])))


if __name__ == '__main__':
    main()
