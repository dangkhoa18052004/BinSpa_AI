"""P01 (selfie_v1): source registry. Read-only over data_raw/, raw/ and acne04v2/.

For every source: where it is, archive/annotation SHA256, stated licence and what that allows here, label trust,
image and source-image counts, raw label counts, and its role in the plan. Checks configs/selfie_v1/label_mapping.yaml
against the data: every raw label in use must be mapped, every `drop` label must be unused, every mapped label
must exist. Exits non-zero if a check fails.
Writes datasets/selfie_v1/source_registry.json (local, gitignored).
Usage: venv/Scripts/python.exe scripts/data_v2/build_registry.py
"""
import ast
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import digest, write_json  # noqa: E402

RAW = ROOT / 'data_raw'
MAPPING = ROOT / 'configs/selfie_v1/label_mapping.yaml'
OUT = ROOT / 'datasets/selfie_v1/source_registry.json'
INTERNAL = 'internal localhost demo and research only; not for commercial use or a public service'

SOURCES = {
    'acnedet_author': dict(
        path='data_raw/acnedet_author', archive='data_raw/labelme.zip', format='labelme',
        url='https://github.com/INno-Vation/Decoupled-Sequential-Detection-Head (LabelMe release, Baidu link)',
        version='LabelMe release dated 2024-01-31', license_stated='none stated by the authors', usage=INTERNAL,
        role='train + T1 (in-domain test, dermatologist labels); replaces the mirror as the label source',
        overlaps='275/276 images are the ACNE-DET Roboflow mirror; ID 40 only here'),
    'acnedet_mirror': dict(
        path='data_raw/acnedet_roboflow/acnedet.v1i.coco', archive='data_raw/acnedet_roboflow/acnedet.v1i.coco.zip',
        format='coco', url='https://universe.roboflow.com/detect-4fwbi/acnedet', version='v1 (2025-05-13)',
        license_stated='MIT (uploader; the authors state none)', usage=INTERNAL,
        role='provenance of the historical master split only; not counted as a second source',
        overlaps='same images as acnedet_author (pHash 0, identical per-class counts)'),
    'acne04v2': dict(
        path='raw/ACNE04_images', annotations='acne04v2/Acne04-v2_annotations.json', format='acne04v2',
        url='https://github.com/xpwu95/LDL (images); Acne04-v2 labels from the AcneAI paper (MICCAI 2024)',
        version='Acne04-v2', license_stated='ACNE04: academic use, contact the authors otherwise', usage=INTERNAL,
        role='detector train (acne_untyped); locked phase-1 val/test kept for regression',
        overlaps='kbm0q is mostly ACNE04 resized; 131 ACNE04 images are inside DetectDataSkin-2'),
    'kbm0q': dict(
        path='data_raw/acne-kbm0q_roboflow/extracted', archive='data_raw/acne-kbm0q_roboflow/acne-kbm0q.v21.coco.zip',
        format='coco', url='https://universe.roboflow.com/kritsakorn/acne-kbm0q', version='v21 (2022-01-31)',
        license_stated='undefined (README); API reports Private', usage=INTERNAL,
        role='type labels for ACNE04 images (mapped back to full resolution in P02); never train on images matching locked ACNE04 val/test',
        overlaps='756-1021 images match ACNE04 by pHash, 212 match DetectDataSkin-2; 279 match locked ACNE04 val/test'),
    'xnbio': dict(
        path='data_raw/acne-xnbio_roboflow/extracted', archive='data_raw/acne-xnbio_roboflow/acne-xnbio.v19.coco.zip',
        format='coco', url='https://universe.roboflow.com/rungroj/acne-xnbio', version='v19 (2024-11-05)',
        license_stated='CC BY 4.0 (uploader); contains watermarked DermNet/Alamy images', usage=INTERNAL,
        role='merged into DetectDataSkin-2 lineage groups (train/val/T2)',
        overlaps='every source name is in DetectDataSkin-2; 220 byte-identical images'),
    'detectdataskin2': dict(
        path='data_raw/detectdataskin2', archive='data_raw/acnedet_roboflow/DetectDataSkin-2.v1i.coco.zip', format='coco',
        url='https://app.roboflow.com/datasets/detectdataskin-2-y4fto/1', version='v1 (2026-10-11 export)',
        license_stated='CC BY 4.0 (export)', usage=INTERNAL,
        role='train / val / T2 proxy-test by lineage group; evaluated once before (exploratory), so T2 is not unseen',
        overlaps='131 ACNE04 images; contains all of xnbio; 411 source images in several export splits'),
    'scin': dict(
        path='data_raw/scin', annotations='data_raw/scin/scin_labels.csv', format='scin',
        url='https://github.com/google-research-datasets/scin', version='1.0.0',
        license_stated='see the SCIN repository terms (not re-verified here)', usage=INTERNAL,
        role='T4 (case-level acne vs non-acne on real phone photos) and T5 owner review; no lesion labels',
        overlaps='none checked beyond SCIN itself'),
}


def coco_labels(root):
    files, used, cats, images, sources = [], Counter(), set(), 0, set()
    for ann in sorted(Path(root).glob('*/_annotations.coco.json')):
        d = json.loads(ann.read_text(encoding='utf-8'))
        names = {c['id']: c['name'] for c in d['categories']}
        cats |= set(names.values())
        used.update(names[a['category_id']] for a in d['annotations'])
        images += len(d['images'])
        sources |= {i['file_name'].split('.rf.')[0] for i in d['images']}
        files.append((str(ann.relative_to(ROOT)), digest(ann)))
    return dict(annotation_files=dict(files), images=images, source_images=len(sources),
                categories=sorted(cats), label_counts=dict(used.most_common()))


def labelme_labels(root):
    used, n = Counter(), 0
    for p in sorted((Path(root) / 'labelme').glob('*.json')):
        used.update(s['label'] for s in json.loads(p.read_text(encoding='utf-8'))['shapes'])
        n += 1
    images = len(list((Path(root) / 'images').glob('*.jpg')))
    return dict(images=images, source_images=images, label_files=n, categories=sorted(used), label_counts=dict(used.most_common()))


def acne04_labels(spec):
    d = json.loads((ROOT / spec['annotations']).read_text(encoding='utf-8'))
    files = len(list((ROOT / spec['path']).glob('*.jpg')))
    return dict(annotation_sha256=digest(ROOT / spec['annotations']), images=files, labelled_images=len(d['images']),
                source_images=len(d['images']), categories=['acne'], label_counts={'acne': len(d['annotations'])})


def scin_counts(spec):
    base = ROOT / spec['path']
    cases = {r['case_id']: r for r in csv.DictReader(open(base / 'scin_cases.csv', encoding='utf-8'))}
    weights = Counter()
    for r in csv.DictReader(open(base / 'scin_labels.csv', encoding='utf-8')):
        w = ast.literal_eval(r['weighted_skin_condition_label'] or '{}')
        a = sum(v for k, v in w.items() if 'acne' in k.lower())
        weights['acne_case' if a >= 0.5 else 'non_acne_case' if a == 0 else 'uncertain_case'] += 1
    return dict(annotation_sha256={'scin_cases.csv': digest(base / 'scin_cases.csv'), 'scin_labels.csv': digest(base / 'scin_labels.csv')},
                images=len(list((base / 'images').iterdir())), cases=len(cases), case_classes=dict(weights),
                categories=[], label_counts={})


def check_mapping(name, info, mapping):
    m = mapping['sources'][name].get('map', {})
    allowed = set(mapping['canonical_classes']) | {'acne_untyped', 'ignore', 'drop'}
    problems = [f'{name}: raw label "{lab}" in use but not mapped' for lab in info['label_counts'] if lab not in m]
    problems += [f'{name}: "{lab}" mapped to drop but has {info["label_counts"][lab]} annotations'
                 for lab, to in m.items() if to == 'drop' and info['label_counts'].get(lab)]
    problems += [f'{name}: "{lab}" -> unknown canonical "{to}"' for lab, to in m.items() if to not in allowed]
    canonical = Counter()
    for lab, n in info['label_counts'].items():
        canonical[m.get(lab, 'UNMAPPED')] += n
    return problems, dict(canonical.most_common())


def main():
    mapping = yaml.safe_load(MAPPING.read_text(encoding='utf-8'))
    registry, problems = {}, []
    for name, spec in SOURCES.items():
        if spec['format'] == 'coco':
            info = coco_labels(ROOT / spec['path'])
        elif spec['format'] == 'labelme':
            info = labelme_labels(ROOT / spec['path'])
        elif spec['format'] == 'acne04v2':
            info = acne04_labels(spec)
        else:
            info = scin_counts(spec)
        if spec.get('archive'):
            info['archive_sha256'] = digest(ROOT / spec['archive'])
        p, canonical = check_mapping(name, info, mapping)
        problems += p
        src_map = mapping['sources'][name]
        registry[name] = dict(spec, label_trust=src_map['label_trust'], completeness=src_map['completeness'],
                              geometry=src_map['geometry'], **info, canonical_counts=canonical)
        print(f"{name}: images {info['images']}, source images {info.get('source_images', '-')}, canonical {canonical}", flush=True)
    write_json(OUT, dict(version=mapping['version'], mapping_sha256=digest(MAPPING),
                         created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                         person_id='no source has a verified person ID; SCIN case_id is a contribution, not a verified person',
                         sources=registry, mapping_problems=problems))
    if problems:
        print('MAPPING PROBLEMS:\n  ' + '\n  '.join(problems))
        sys.exit(1)
    print('mapping check: ok')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
