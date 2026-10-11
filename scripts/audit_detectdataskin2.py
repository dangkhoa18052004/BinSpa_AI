"""Extract and screen DetectDataSkin-2 without modifying existing model datasets.

Archive paths and existing files are checked before extraction. Source COCO and
images remain untouched. Outputs contain screening reasons, duplicate evidence,
and a derived COCO subset, not a patient-independent clinical test certificate.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import stat
import sys
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data_raw/detectdataskin2'
OUT = ROOT / 'reports/detectdataskin2_audit'
DERIVED = ROOT / 'datasets/detectdataskin2_screened'
DEFAULT_ZIP = ROOT / 'data_raw/acnedet_roboflow/DetectDataSkin-2.v1i.coco.zip'
ACNE = {'Blackhead', 'Whitehead', 'Papule', 'Pustule', 'Nodule', 'Nodules', 'Cysts'}
TYPE_MAP = {'Blackhead': 'open_comedo', 'Whitehead': 'closed_comedo', 'Papule': 'papule',
            'Pustule': 'pustule', 'Nodule': 'nodule', 'Nodules': 'nodule'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        fields = list(dict.fromkeys(k for row in records for k in row))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def extract(archive):
    destination = RAW.resolve()
    if not destination.is_relative_to(ROOT.resolve()):
        raise RuntimeError('Extraction destination outside workspace')
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        targets = set()
        for info in entries:
            if '\\' in info.filename or ':' in info.filename:
                raise RuntimeError(f'Unsafe archive entry: {info.filename}')
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(info.external_attr >> 16):
                raise RuntimeError(f'Unsafe archive entry: {info.filename}')
            key = str(target).lower()
            if key in targets:
                raise RuntimeError(f'Duplicate archive path: {info.filename}')
            targets.add(key)
        extracted = reused = 0
        for info in entries:
            target = destination / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != info.file_size or zlib.crc32(target.read_bytes()) != info.CRC:
                    raise RuntimeError(f'Existing file differs from archive; refusing overwrite: {target}')
                reused += 1
            else:
                with z.open(info) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
        result = dict(archive=str(archive.relative_to(ROOT)), archive_sha256=sha(archive),
                      archive_bytes=archive.stat().st_size, entries=len(entries),
                      uncompressed_bytes=sum(i.file_size for i in entries), extracted=extracted, reused=reused)
    write_json(OUT / 'extraction.json', result)
    print('Archive extracted', result, flush=True)
    return result


def phash(gray):
    x = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(x)[:8, :8].ravel()
    bits = d > np.median(d[1:])
    return int.from_bytes(np.packbits(bits).tobytes(), 'big')


def fingerprint(path):
    with Image.open(path) as im:
        width, height = im.size
    flag = cv2.IMREAD_REDUCED_GRAYSCALE_4 if min(width, height) >= 512 else cv2.IMREAD_GRAYSCALE
    gray = cv2.imread(str(path), flag)
    if gray is None:
        raise RuntimeError('Image decode failed')
    return dict(width=width, height=height, sha256=sha(path), phash=f'{phash(gray):016x}',
                phash_flip=f'{phash(cv2.flip(gray, 1)):016x}')


def canonical(name):
    s = Path(name).name.split('.rf.', 1)[0]
    return re.sub(r'(_jpg|_jpeg|_png|\.jpg|\.jpeg|\.png)$', '', s, flags=re.I).lower()


def load_records():
    records, issues, class_counts = [], [], Counter()
    categories = None
    metadata = {}
    for split in ['train', 'valid', 'test']:
        p = RAW / split / '_annotations.coco.json'
        coco = read(p)
        cats = {c['id']: c['name'] for c in coco['categories']}
        if categories is not None and categories != cats:
            raise RuntimeError('Different category definitions across archive splits')
        categories = cats
        metadata[split] = {k: coco[k] for k in ['info', 'licenses']}
        if len({i['id'] for i in coco['images']}) != len(coco['images']):
            raise RuntimeError(f'Duplicate image IDs in {split}')
        image_ids = {i['id'] for i in coco['images']}
        by_image = defaultdict(list)
        for a in coco['annotations']:
            if a['image_id'] not in image_ids or a['category_id'] not in cats:
                raise RuntimeError(f'Orphan annotation: {split}/{a["id"]}')
            by_image[a['image_id']].append(a)
        for im in coco['images']:
            path = (RAW / split / im['file_name']).resolve()
            if not path.is_relative_to((RAW / split).resolve()):
                raise RuntimeError('Unsafe COCO image path')
            row = dict(uid=f'{split}:{im["id"]}', source_split=split,
                       file_name=f'{split}/{im["file_name"]}', original_key=canonical(im['file_name']),
                       image_id=im['id'], path=str(path), issues=[], annotations=[], exclude_reasons=[], duplicate_of='')
            try:
                row.update(fingerprint(path))
                if (row['width'], row['height']) != (im['width'], im['height']):
                    row['issues'].append('image_dimensions_disagree_with_coco')
            except Exception as exc:
                row['issues'].append(f'image_error:{type(exc).__name__}:{exc}')
                row.update(width=im.get('width'), height=im.get('height'), sha256='', phash='', phash_flip='')
            seen, coordinate_classes = set(), defaultdict(set)
            for a in by_image[im['id']]:
                cls = cats[a['category_id']]
                class_counts[cls] += 1
                b = np.asarray(a['bbox'], dtype=float)
                if b.shape != (4,) or not np.isfinite(b).all() or min(b[2:]) <= 0:
                    row['issues'].append(f'invalid_bbox:{a["id"]}')
                    continue
                x, y, w, h = b
                if x < -.001 or y < -.001 or x + w > row['width'] + .001 or y + h > row['height'] + .001:
                    row['issues'].append(f'out_of_bounds_bbox:{a["id"]}')
                    continue
                coords = tuple(np.round(b, 4))
                key = (cls, coords)
                if key in seen:
                    row['issues'].append(f'duplicate_annotation:{a["id"]}')
                    continue
                seen.add(key)
                coordinate_classes[coords].add(cls)
                if cls not in ACNE | {'Bekas Jerawat', 'Pori-Pori'}:
                    row['issues'].append(f'undefined_leaf_category:{a["id"]}:{cls}')
                row['annotations'].append(dict(id=a['id'], category_id=a['category_id'], category=cls,
                                               bbox=b.tolist(), iscrowd=a.get('iscrowd', 0)))
            if any(len(c) > 1 for c in coordinate_classes.values()):
                row['issues'].append('same_box_conflicting_classes')
            if row['issues']:
                row['exclude_reasons'].append('annotation_or_image_requires_review')
                issues.extend(dict(uid=row['uid'], file_name=row['file_name'], issue=v) for v in row['issues'])
            row['acne_boxes'] = sum(a['category'] in ACNE for a in row['annotations'])
            row['supported_type_boxes'] = sum(a['category'] in TYPE_MAP for a in row['annotations'])
            records.append(row)
        print(f'Indexed {split}: {len(coco["images"])} images', flush=True)
    return records, issues, class_counts, metadata, categories


def reference_records():
    records = []
    manifest = read(ROOT / 'datasets/phase1_acne_detection/manifest.json')
    phase1_split = {r['file_name']: r['split'] for r in manifest['images']}
    with (ROOT / 'datasets/acnedet_master_split.csv').open(encoding='utf-8-sig') as f:
        master = list(csv.DictReader(f))
    paths = [(p, 'ACNE04', phase1_split.get(p.name, 'unused_or_quarantined'))
             for p in sorted((ROOT / 'raw/ACNE04_images').glob('*')) if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    paths += [(ROOT / 'data_raw/acnedet_roboflow/acnedet.v1i.coco/train' / r['file_name'], 'ACNEDET', r['split']) for r in master]
    for i, (p, dataset, split) in enumerate(paths):
        records.append(dict(path=str(p), dataset=dataset, split=split, original_key=canonical(p.name), **fingerprint(p)))
        if (i + 1) % 300 == 0:
            print(f'Fingerprint existing sources {i+1}/{len(paths)}', flush=True)
    write_csv(OUT / 'reference_fingerprints.csv', records)
    return records


def nearest(query, refs):
    q = np.array([int(r['phash'], 16) if r['phash'] else 0 for r in query], dtype=np.uint64)
    f = np.array([int(r['phash_flip'], 16) if r['phash_flip'] else 0 for r in query], dtype=np.uint64)
    v = np.array([int(r['phash'], 16) for r in refs], dtype=np.uint64)
    lut = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    found = []
    for start in range(0, len(q), 128):
        a = np.bitwise_xor(q[start:start+128, None], v[None])
        b = np.bitwise_xor(f[start:start+128, None], v[None])
        da = lut[a.view(np.uint8).reshape(len(a), len(v), 8)].sum(2)
        db = lut[b.view(np.uint8).reshape(len(b), len(v), 8)].sum(2)
        d = np.minimum(da, db)
        for i, dd in enumerate(d):
            k = int(dd.argmin())
            found.append((k, int(dd[k]), bool(db[i, k] < da[i, k])))
    return found


def pixel_correlation(a, b, flip=False):
    def thumbnail(path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    aa = thumbnail(a)
    if flip:
        aa = cv2.flip(aa, 1)
    bb = thumbnail(b)
    return float(np.corrcoef(aa.ravel(), bb.ravel())[0, 1])


def screen(records, refs):
    evidence = []
    sha_refs = {r['sha256']: r for r in refs}
    key_refs = defaultdict(list)
    for r in refs:
        key_refs[r['original_key']].append(r)
    for row, (k, distance, flip) in zip(records, nearest(records, refs)):
        ref = refs[k]
        methods = []
        correlation = None
        if row['sha256'] in sha_refs:
            ref = sha_refs[row['sha256']]
            methods.append('exact_sha256_existing_source')
        named = key_refs.get(row['original_key'], [])
        if re.search(r'levle[0-3]_\d+', row['original_key']):
            methods.append('ACNE04_source_name')
            if named:
                ref = named[0]
        if distance <= 6 and row['phash']:
            correlation = pixel_correlation(row['path'], refs[k]['path'], flip)
            if correlation >= .98:
                methods.append('phash_le6_and_pixel_correlation_ge098')
                ref = refs[k]
        if methods:
            row['exclude_reasons'].append('overlap_existing_source')
        if methods or distance <= 10:
            evidence.append(dict(uid=row['uid'], file_name=row['file_name'], reference=str(Path(ref['path']).relative_to(ROOT)),
                                 reference_dataset=ref['dataset'], reference_split=ref['split'], methods=methods,
                                 phash_distance_to_nearest=distance, nearest_was_flipped=flip, pixel_correlation=correlation,
                                 status='excluded' if methods else 'possible_match_requires_review'))
            if not methods:
                row['exclude_reasons'].append('possible_overlap_requires_review')
    # Keep a single source image, even when the Roboflow splits contain copies.
    sha_seen, key_seen = {}, {}
    internal = []
    for r in sorted(records, key=lambda r: (bool(r['exclude_reasons']), r['file_name'])):
        duplicate = sha_seen.get(r['sha256']) or key_seen.get(r['original_key'])
        if duplicate:
            r['exclude_reasons'].append('internal_duplicate_sha_or_source_name')
            r['duplicate_of'] = duplicate['uid']
            internal.append(dict(uid=r['uid'], file_name=r['file_name'], duplicate_of=duplicate['uid'],
                                 other_file_name=duplicate['file_name'], cross_archive_split=r['source_split'] != duplicate['source_split'],
                                 identical_bytes=r['sha256'] == duplicate['sha256'], method='sha_or_source_name'))
        else:
            if r['sha256']:
                sha_seen[r['sha256']] = r
            key_seen[r['original_key']] = r
    # Perceptual copies with different names: only flag/exclude; never assert patient identity.
    near_internal = []
    remaining = [r for r in records if not r['exclude_reasons']]
    if remaining:
        reference = [dict(r, phash=r['phash']) for r in remaining]
        values = np.array([int(r['phash'], 16) for r in remaining], dtype=np.uint64)
        lut = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
        for i, r in enumerate(remaining):
            if r['exclude_reasons']:
                continue
            dd = np.bitwise_xor(values[i+1:], values[i])
            ds = lut[dd.view(np.uint8).reshape(-1, 8)].sum(1)
            for j in np.where(ds <= 6)[0] + i + 1:
                other = remaining[int(j)]
                if other['exclude_reasons']:
                    continue
                corr = pixel_correlation(r['path'], other['path'])
                if corr >= .98:
                    other['exclude_reasons'].append('internal_perceptual_duplicate')
                    other['duplicate_of'] = r['uid']
                    near_internal.append(dict(uid=other['uid'], file_name=other['file_name'], duplicate_of=r['uid'],
                                              other_file_name=r['file_name'], phash_distance=int(ds[int(j)-i-1]), pixel_correlation=corr,
                                              cross_archive_split=other['source_split'] != r['source_split'], method='perceptual'))
    return evidence, internal, near_internal


def export_subset(records, categories, metadata, archive_sha256):
    keep = [r for r in records if not r['exclude_reasons']]
    images, annotations = [], []
    for idx, r in enumerate(keep):
        images.append(dict(id=idx, file_name=r['file_name'], width=r['width'], height=r['height'],
                           source_uid=r['uid'], archive_split=r['source_split'], patient_id=None))
        for a in r['annotations']:
            annotations.append(dict(id=len(annotations), image_id=idx, category_id=a['category_id'],
                                    bbox=a['bbox'], area=a['bbox'][2]*a['bbox'][3], iscrowd=a['iscrowd'], source_annotation_id=a['id']))
    derived = dict(info=dict(description='Screened external exploratory subset; no patient IDs or expert label review',
                            source_images_root=str(RAW), parent_archive_sha256=archive_sha256),
                   licenses=metadata['train']['licenses'], images=images, annotations=annotations,
                   categories=[dict(id=k, name=v) for k, v in categories.items()])
    write_json(DERIVED / 'annotations.coco.json', derived)
    write_json(DERIVED / 'protocol.json', dict(status='screened_unreviewed_external_probe', train_allowed=False,
               purpose='Frozen checkpoint exploratory evaluation; not final patient-independent test',
               exclusions=['annotation/image issues', 'known source overlap', 'possible near overlap', 'duplicate source images'],
               patient_ids_available=False, label_mapping_assumption=TYPE_MAP, cysts_excluded_from_type_classification=True,
               scar_subtypes_not_available=True, tuning_on_this_dataset=False))
    return keep


def contact_sheets(records, evidence):
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'human_review_queue.csv', [dict(
        uid=r['uid'], file_name=r['file_name'], source_split=r['source_split'],
        source_categories=sorted({a['category'] for a in r['annotations']}),
        total_boxes=len(r['annotations']), acne_boxes=r['acne_boxes'],
        patient_group_verified='', annotation_review_status='unreviewed',
        reviewer='', review_notes='') for r in records if not r['exclude_reasons']])
    selected, seen = [], set()
    for c in sorted(ACNE | {'Bekas Jerawat', 'Pori-Pori'}):
        candidates = [r for r in records if not r['exclude_reasons'] and any(a['category'] == c for a in r['annotations'])]
        for r in candidates[:2]:
            if r['uid'] not in seen:
                selected.append(r); seen.add(r['uid'])
    rows = []
    for start in range(0, len(selected), 12):
        canvas = np.full((3*310, 4*300, 3), 255, np.uint8)
        for pos, r in enumerate(selected[start:start+12]):
            image = cv2.imread(r['path'])
            for a in r['annotations']:
                x, y, w, h = a['bbox']
                color = (0, 170, 0) if a['category'] in ACNE else (190, 0, 190)
                cv2.rectangle(image, (round(x), round(y)), (round(x+w), round(y+h)), color, max(1, round(max(image.shape[:2])/600)))
            scale = min(300/image.shape[1], 270/image.shape[0])
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            top, left = (pos//4)*310, (pos%4)*300
            canvas[top:top+small.shape[0], left:left+small.shape[1]] = small
            cv2.putText(canvas, f'{start+pos+1}: {r["uid"]}', (left+3, top+282), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,0), 1)
            classes = ','.join(sorted({a['category'] for a in r['annotations']}))
            cv2.putText(canvas, classes[:42], (left+3, top+301), cv2.FONT_HERSHEY_SIMPLEX, .35, (0,0,0), 1)
            rows.append(dict(index=start+pos+1, uid=r['uid'], file_name=r['file_name'], categories=classes))
        cv2.imwrite(str(OUT/f'review_sheet_{start//12+1}.jpg'), canvas)
    write_csv(OUT / 'review_sheet_index.csv', rows)
    # Compare transformed copies and originals side by side.
    matches = [e for e in evidence if e['status']=='excluded'][:8]
    by_uid = {r['uid']:r for r in records}
    canvas = np.full((4*240, 4*240, 3), 255, np.uint8)
    for i, e in enumerate(matches):
        for offset, path in enumerate([Path(by_uid[e['uid']]['path']), ROOT/e['reference']]):
            im = cv2.imread(str(path)); scale=min(240/im.shape[1],205/im.shape[0])
            small=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            row,col=(i//2)*240,((i%2)*2+offset)*240
            canvas[row:row+small.shape[0],col:col+small.shape[1]]=small
            cv2.putText(canvas, ('new '+e['uid']) if offset==0 else e['reference_dataset']+'/'+e['reference_split'],
                        (col+2,row+225),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
    if matches:
        cv2.imwrite(str(OUT/'known_overlap_examples.jpg'),canvas)
    variants = [r for r in records if r['duplicate_of'] and
                r['sha256'] != by_uid[r['duplicate_of']]['sha256']][:6]
    canvas = np.full((3*270, 4*240, 3), 255, np.uint8)
    for i, r in enumerate(variants):
        for j, item in enumerate([r, by_uid[r['duplicate_of']]]):
            top, left = (i//2)*270, ((i%2)*2+j)*240
            canvas[top:top+240, left:left+240] = cv2.resize(cv2.imread(item['path']), (240,240))
            cv2.putText(canvas, item['uid'], (left+3, top+257), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1)
    if variants:
        cv2.imwrite(str(OUT/'internal_duplicate_examples.jpg'), canvas)


def finish_reports():
    """Resume report rendering from saved screening evidence without recomputing it."""
    records = read(OUT / 'image_manifest.json')
    summary = read(OUT / 'summary.json')
    with (OUT / 'known_source_overlap.csv').open(encoding='utf-8-sig', newline='') as f:
        evidence = list(csv.DictReader(f))
    by_uid = {r['uid']: r for r in records}
    duplicates = []
    for r in records:
        if not r['duplicate_of']:
            continue
        original = by_uid[r['duplicate_of']]
        entry = dict(uid=r['uid'], file_name=r['file_name'], duplicate_of=original['uid'],
                     other_file_name=original['file_name'],
                     cross_archive_split=r['source_split'] != original['source_split'])
        if 'internal_perceptual_duplicate' in r['exclude_reasons']:
            entry.update(method='perceptual', pixel_correlation=max(pixel_correlation(Path(r['path']), Path(original['path'])),
                         pixel_correlation(Path(r['path']), Path(original['path']), flip=True)),
                         phash_distance=min((int(r['phash'],16)^int(original['phash'],16)).bit_count(),
                                            (int(r['phash_flip'],16)^int(original['phash'],16)).bit_count()))
        else:
            entry.update(method='sha_or_source_name', identical_bytes=r['sha256'] == original['sha256'])
        duplicates.append(entry)
    assert len(duplicates) == summary['internal_duplicates'] + summary['internal_perceptual_duplicates']
    assert sum(not r['exclude_reasons'] for r in records) == summary['remaining_images']
    write_csv(OUT / 'internal_duplicates.csv', duplicates)
    contact_sheets(records, evidence)
    print('Reports completed from saved screening manifest', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path,default=DEFAULT_ZIP)
    parser.add_argument('--finish-reports', action='store_true')
    args=parser.parse_args()
    if args.finish_reports:
        finish_reports()
        return
    OUT.mkdir(parents=True,exist_ok=True)
    extraction=extract(args.zip.resolve())
    records,issues,counts,metadata,categories=load_records()
    print('Source classes',dict(counts),flush=True)
    refs=reference_records()
    evidence,duplicates,near=screen(records,refs)
    kept=export_subset(records,categories,metadata,extraction['archive_sha256'])
    def by_class(rr):
        return dict(Counter(a['category'] for r in rr for a in r['annotations']))
    summary=dict(extraction=extraction,source_images=len(records),source_annotations=sum(counts.values()),
                 archive_splits=dict(Counter(r['source_split'] for r in records)),source_class_counts=dict(counts),
                 dimensions_top=[dict(width=w,height=h,images=n) for (w,h),n in Counter((r['width'],r['height']) for r in records).most_common(12)],
                 source_metadata=metadata,source_issue_count=len(issues),images_with_issues=sum(bool(r['issues']) for r in records),
                 exclusion_reason_counts=dict(Counter(reason for r in records for reason in set(r['exclude_reasons']))),
                 known_source_overlap_images=sum('overlap_existing_source' in r['exclude_reasons'] for r in records),
                 known_source_overlap_by_reference_split=dict(Counter(e['reference_dataset']+'/'+e['reference_split'] for e in evidence if e['status']=='excluded')),
                 internal_duplicates=len(duplicates),internal_cross_split_duplicates=sum(r['cross_archive_split'] for r in duplicates),
                 internal_perceptual_duplicates=len(near),internal_perceptual_cross_split_duplicates=sum(r['cross_archive_split'] for r in near),
                 remaining_images=len(kept),remaining_archive_splits=dict(Counter(r['source_split'] for r in kept)),remaining_class_counts=by_class(kept),
                 remaining_acne_images=sum(r['acne_boxes']>0 for r in kept),remaining_acne_boxes=sum(r['acne_boxes'] for r in kept),
                 remaining_supported_type_boxes=sum(r['supported_type_boxes'] for r in kept),
                 remaining_images_without_acne_labels=sum(r['acne_boxes']==0 for r in kept),
                 patient_ids_verified=False,expert_annotation_review_complete=False,
                 warning='Screening removes documented image overlap, not all possible patient overlap or missing labels; not a final independent clinical test')
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'image_manifest.json',records)
    write_csv(OUT/'image_manifest.csv',[{k:r[k] for k in ['uid','source_split','file_name','original_key','width','height','sha256','phash','phash_flip','acne_boxes','supported_type_boxes','issues','exclude_reasons','duplicate_of']} for r in records])
    write_csv(OUT/'annotation_issues.csv',issues)
    write_csv(OUT/'known_source_overlap.csv',evidence)
    write_csv(OUT/'internal_duplicates.csv',duplicates+near)
    contact_sheets(records,evidence)
    print('SCREENING SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in ['source_metadata','extraction']},ensure_ascii=True),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
