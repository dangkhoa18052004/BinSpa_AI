"""Train RT-DETR-L (Ultralytics `rtdetr-l.pt`, COCO-pretrained) on the fixed phase-1 split.

RT-DETR-L (HGNetv2) is the smallest RT-DETR with official pretrained weights in the
`ultralytics` package; R18 weights are not distributed there. Hyperparameters follow
the RT-DETR recipe (AdamW 1e-4, weight decay 1e-4, no bias warm-up LR) rather than
the YOLO recipe. 300 object queries cap detections per image (one image in the dataset
has 434 lesions; 25 have more than 100).
Usage: venv/Scripts/python.exe -u scripts/phase1/train_rtdetr.py [--full-schedule]
"""
from train_ultralytics_common import run_model

BASE = dict(batch=8, patience=20, cache='ram', max_det=300, close_mosaic=10, optimizer='AdamW',
            weight_decay=0.0001, lrf=0.01, warmup_bias_lr=0.0)
CONFIGS = {
    'cfg_a': dict(BASE, lr0=0.0001),              # RT-DETR paper learning rate
    'cfg_b': dict(BASE, lr0=0.0002),
    'cfg_c': dict(BASE, lr0=0.0001, mosaic=0.0),  # paper-style augmentation without mosaic
}

if __name__ == '__main__':
    from ultralytics import RTDETR
    import sys
    # --full-schedule: retrain the validation-chosen config for all 100 epochs (patience=100) as rtdetr_full100.
    run_model('rtdetr', 'rtdetr-l.pt', RTDETR, CONFIGS, screen_epochs=30, final_epochs=100, deploy_name='rtdetr_best.pt',
              full_schedule='--full-schedule' in sys.argv)
