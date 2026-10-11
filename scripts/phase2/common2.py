"""Shared paths and helpers for phase 2 (acne lesion type) on the ACNE-DET Roboflow mirror."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import digest, read_json, write_csv, write_json  # noqa: E402,F401

SOURCE = ROOT / 'data_raw/acnedet_roboflow/acnedet.v1i.coco/train'
ANNOTATIONS = SOURCE / '_annotations.coco.json'
REPORTS = ROOT / 'reports'
DATA_REPORT = REPORTS / 'phase2_data'
MASTER_SPLIT = ROOT / 'datasets/acnedet_master_split.csv'  # frozen source-image split for acne models
CROPS = ROOT / 'datasets/phase2_acne_type'
MODELS = ROOT / 'models/phase2'
RUNS = ROOT / 'runs/phase2'
ACNE_CLASSES = ['closed_comedo', 'open_comedo', 'papule', 'pustule', 'nodule']  # names exactly as in ACNE-DET
# Auxiliary counts used ONLY to reproduce the historical split. No scar task is
# active; keep these columns so retiring a task does not silently change test data.
SPLIT_AUX_CLASSES = ['atrophic_scar', 'hypertrophic_scar']


def source_id(image):
    """Original ACNE-DET number ('257.jpg' -> 257) kept by Roboflow in extra.name."""
    return int(re.match(r'\d+', image['extra']['name']).group())


def load_coco():
    data = json.loads(ANNOTATIONS.read_text(encoding='utf-8'))
    cats = {c['id']: c['name'] for c in data['categories']}
    return data, cats
