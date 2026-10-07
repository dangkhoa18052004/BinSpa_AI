"""Print training progress of the additional models (YOLOv5s, Faster R-CNN, RT-DETR).

Usage (cmd or PowerShell): venv\\Scripts\\python.exe scripts\\phase1\\check_additional_progress.py
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ADD = Path(__file__).resolve().parents[2] / 'reports' / 'phase1_additional_models'

for model in ('yolov5s', 'faster_rcnn', 'rtdetr', 'yolo11s_full100', 'faster_rcnn_full50', 'rtdetr_full100'):
    status = ADD / model / 'status.json'
    if not status.exists():
        print(f'{model:<12} chua bat dau')
        continue
    s = json.loads(status.read_text(encoding='utf-8'))
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(s['updated_utc'])).total_seconds() / 60
    line = f"{model:<12} {s.get('state', ''):<15} {s.get('run', s.get('config', ''))}"
    if 'epoch' in s:
        line += f"  epoch {s['epoch']}/{s['epochs']}"
    metrics = s.get('metrics') or s.get('last') or {}
    m5095 = metrics.get('metrics/mAP50-95(B)', metrics.get('val_mAP50_95'))
    if m5095 is not None:
        line += f'  val mAP50-95={m5095:.4f}'
    print(line + f'  (cap nhat {age:.0f} phut truoc)')
    if s.get('state') == 'complete':
        t = s['test']
        print(f"{'':<12} TEST: P={t['precision']:.4f} R={t['recall']:.4f} F1={t['f1']:.4f} "
              f"mAP50={t['mAP50']:.4f} mAP50-95={t['mAP50_95']:.4f}")
    screening = ADD / model / 'screening_val.csv'
    if screening.exists():
        with screening.open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                print(f"{'':<12} screening {r['config']}: val mAP50-95={float(r['mAP50_95']):.4f} "
                      f"recall={float(r['recall']):.4f} best_epoch={r['best_epoch']}")

# ---- Chi tiet run dang chay: batch hien tai, metric theo epoch, anh minh hoa ----
import re

ROOT = ADD.parents[1]
logs = sorted((ADD / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
if logs:
    text = logs[-1].read_bytes()[-20000:].decode('utf-8', 'ignore').replace('\r', '\n')
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    # Ultralytics: "epoch  GPU_mem  loss1 loss2 loss3  instances  size: pct ... i/N"
    bars = re.findall(r'^\s*(\d+/\d+)\s+(\S+G)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+):\s*(\d+)%.*?(\d+/\d+)', text, re.M)
    print(f'\nLog: {logs[-1].name}')
    if bars:
        e, gpu, l1, l2, l3, inst, size, pct, step = bars[-1]
        print(f'  Epoch {e} | batch {step} ({pct}%) | GPU {gpu} | loss box/giou={l1} cls={l2} dfl/l1={l3} | {inst} lesion trong batch')
    else:  # Faster R-CNN prints one dict per epoch
        rows = re.findall(r"^\{'epoch'.*\}$", text, re.M)
        print('  ' + (rows[-1][:300] if rows else 'chua co dong tien do'))

runs = sorted([*(ROOT / 'runs' / 'phase1_additional').glob('*/results.csv'), *(ROOT / 'runs' / 'phase1').glob('*_p100/results.csv')],
              key=lambda p: p.stat().st_mtime)
if runs:
    run = runs[-1].parent
    with runs[-1].open(encoding='utf-8-sig') as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    print(f'\nRun moi nhat: {run}')
    print('  epoch | val P  | val R  | mAP50  | mAP50-95')
    for r in rows[-5:]:
        g = lambda *keys: next((float(r[k]) for k in keys if k in r and r[k] != ''), float('nan'))
        print(f"  {int(float(r['epoch'])):>5} | {g('metrics/precision(B)', 'val_precision'):.4f} | {g('metrics/recall(B)', 'val_recall'):.4f} | "
              f"{g('metrics/mAP50(B)', 'val_mAP50'):.4f} | {g('metrics/mAP50-95(B)', 'val_mAP50_95'):.4f}")
    images = sorted(p.name for p in run.glob('*.jpg')) + sorted(p.name for p in run.glob('*.png'))
    if images:
        print('  Anh xem duoc (mo bang Explorer):', ', '.join(images))
