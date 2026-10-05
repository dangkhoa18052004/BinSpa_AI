"""Audit v2 and export unsplit labels; quarantine fully invalid images, dedupe by SHA256."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from common import RAW, ANNOTATIONS, AUDIT, digest, load_annotations, write_csv, write_json, yolo_box


def convert(images_dir=RAW, annotations=ANNOTATIONS, output=AUDIT):
    images_dir, annotations, output = map(Path, (images_dir, annotations, output))
    data = load_annotations(annotations)
    by_id = defaultdict(list)
    for ann in data['annotations']:
        by_id[ann['image_id']].append(ann)
    missing = [im for im in data['images'] if not (images_dir/im['file_name']).is_file()]
    write_csv(output/'missing_files.csv', missing, ['id','file_name','width','height'])
    if missing:
        raise RuntimeError(f'Missing {len(missing)} files; refusing conversion/split')
    records, invalid, clipped, excluded, duplicates = [], [], [], [], []
    groups = defaultdict(list)
    box_pixels = []
    for im in sorted(data['images'], key=lambda i:i['file_name']):
        name, width, height = im['file_name'], im['width'], im['height']
        with Image.open(images_dir/name) as image:
            if image.size != (width, height):
                raise ValueError(f'Dimension mismatch: {name}')
            image.verify()
        labels = []
        invalid_for_image = 0
        for ann in by_id[im['id']]:
            box, reason, was_clipped = yolo_box(ann, width, height)
            if box is None:
                invalid_for_image += 1
                invalid.append(dict(file_name=name, annotation_id=ann['id'], coordinates=str(ann.get('coordinates')), radius=ann.get('radius'), reason=reason))
                continue
            labels.append('0 '+' '.join(f'{v:.9f}' for v in box))
            box_pixels.append((box[2]*width, box[3]*height))
            if was_clipped:
                clipped.append(dict(file_name=name, annotation_id=ann['id']))
        if invalid_for_image:
            # Partially corrupt source labels are incomplete supervision too.
            excluded.append(dict(file_name=name, reason=f'quarantine: {invalid_for_image} invalid annotations; {len(labels)} valid annotations withheld'))
            continue
        if not labels:
            excluded.append(dict(file_name=name, reason='no_valid_lesions; not a verified negative image'))
            continue
        record = dict(file_name=name, image_id=im['id'], width=width, height=height, sha256=digest(images_dir/name), lesions=len(labels), labels=labels)
        groups[record['sha256']].append(record)
    for sha, group in sorted(groups.items()):
        group.sort(key=lambda r:r['file_name'])
        canonical = group[0]
        records.append(canonical)
        for row in group[1:]:
            duplicates.append(dict(file_name=row['file_name'], kept_file=canonical['file_name'], sha256=sha, kept_lesions=canonical['lesions'], excluded_lesions=row['lesions'], label_sets_equal=set(row['labels'])==set(canonical['labels'])))
    # Never overwrite a reviewed dataset here; converted labels are rebuildable audit output.
    label_dir = output/'converted_labels'
    label_dir.mkdir(parents=True, exist_ok=True)
    for row in records:
        (label_dir/(Path(row['file_name']).stem+'.txt')).write_text('\n'.join(row['labels'])+'\n', encoding='utf-8')
        del row['labels']
    write_csv(output/'invalid_annotations.csv', invalid, ['file_name','annotation_id','coordinates','radius','reason'])
    write_csv(output/'clipped_annotations.csv', clipped, ['file_name','annotation_id'])
    write_csv(output/'excluded_images.csv', excluded, ['file_name','reason'])
    write_csv(output/'duplicate_images.csv', duplicates, ['file_name','kept_file','sha256','kept_lesions','excluded_lesions','label_sets_equal'])
    summary = dict(source_images=len(data['images']), source_annotations=len(data['annotations']), mapped_images=len(data['images']), missing_images=0, invalid_annotations=len(invalid), clipped_valid_annotations=len(clipped), excluded_images=len(excluded), duplicate_extras=len(duplicates), duplicate_groups=sum(len(g)>1 for g in groups.values()), retained_images=len(records), retained_lesions=sum(r['lesions'] for r in records), duplicate_policy='Keep lexicographically first eligible filename per SHA256; never merge conflicting annotations', invalid_policy='Log invalid boxes; quarantine entire images containing invalid annotations or no valid lesions, to avoid incomplete supervision', source_annotations_sha256=digest(annotations), box_size_pixels_before_dedupe=dict(min_width=min(v[0] for v in box_pixels), max_width=max(v[0] for v in box_pixels), min_height=min(v[1] for v in box_pixels), max_height=max(v[1] for v in box_pixels)))
    write_json(output/'conversion.json', dict(summary=summary, images=sorted(records,key=lambda r:r['file_name']), images_dir=str(images_dir.resolve())))
    print(summary, flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,default=RAW)
    p.add_argument('--annotations',type=Path,default=ANNOTATIONS)
    p.add_argument('--output',type=Path,default=AUDIT)
    a=p.parse_args(); convert(a.images,a.annotations,a.output)


if __name__=='__main__':
    main()
