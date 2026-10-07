"""Train YOLOv5s (Ultralytics `yolov5su.pt`, COCO-pretrained) on the fixed phase-1 split.

`yolov5su` = YOLOv5s backbone/neck with the anchor-free Ultralytics head; the original
anchor-based YOLOv5s needs the separate ultralytics/yolov5 repository and cannot be
loaded by the `ultralytics` package used for the other models.

cfg_a reproduces the protocol of YOLO26n/26s/11s exactly (AdamW 1e-3, batch 8, ...).
Usage: venv/Scripts/python.exe -u scripts/phase1/train_yolov5s.py
"""
from train_ultralytics_common import run_model

BASE = dict(batch=8, patience=20, cache='ram', max_det=500, close_mosaic=10, weight_decay=0.0005, lrf=0.01)
CONFIGS = {
    'cfg_a': dict(BASE, optimizer='AdamW', lr0=0.001),      # identical to the existing 3-model protocol
    'cfg_b': dict(BASE, optimizer='SGD', lr0=0.01, momentum=0.937),  # Ultralytics default for YOLOv5u
    'cfg_c': dict(BASE, optimizer='AdamW', lr0=0.0005),
}

if __name__ == '__main__':
    from ultralytics import YOLO
    run_model('yolov5s', 'yolov5su.pt', YOLO, CONFIGS, screen_epochs=30, final_epochs=100, deploy_name='yolov5s_best.pt')
