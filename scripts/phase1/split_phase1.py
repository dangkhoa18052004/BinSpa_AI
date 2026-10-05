"""Seeded 70/15/15 image split after dedupe; refuse replacement of an existing dataset."""
import argparse
import json
import random
import shutil
from pathlib import Path
from collections import Counter
from common import AUDIT, DATASET, digest, read_json, verify_dataset, write_csv, write_json


def split(conversion=AUDIT/'conversion.json', output=DATASET, seed=42):
    conversion, output = Path(conversion), Path(output)
    data=read_json(conversion)
    if (output/'manifest.json').exists():
        existing=verify_dataset(output)
        if existing['conversion_sha256']!=digest(conversion) or existing['seed']!=seed:
            raise RuntimeError('Existing dataset differs; use a new --output path')
        print('Existing immutable dataset verified; reused')
        return existing
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Output not empty; use a new --output path')
    rows=list(data['images'])
    if len(rows)<7:
        raise RuntimeError('Too few eligible images for three splits')
    random.Random(seed).shuffle(rows)
    ntrain=int(len(rows)*.70); nval=int(len(rows)*.15)
    allrows=[]
    for subset,records in [('train',rows[:ntrain]),('val',rows[ntrain:ntrain+nval]),('test',rows[ntrain+nval:])]:
        image_dir=output/'images'/subset; label_dir=output/'labels'/subset
        image_dir.mkdir(parents=True,exist_ok=True); label_dir.mkdir(parents=True,exist_ok=True)
        for row in records:
            row=dict(row,split=subset)
            source=Path(data['images_dir'])/row['file_name']
            if digest(source)!=row['sha256']:
                raise RuntimeError(f'Source changed since conversion: {source}')
            labelname=Path(row['file_name']).stem+'.txt'
            shutil.copy2(source,image_dir/row['file_name'])
            shutil.copy2(conversion.parent/'converted_labels'/labelname,label_dir/labelname)
            row['label_sha256']=digest(label_dir/labelname)
            allrows.append(row)
    yaml='path: '+json.dumps(output.resolve().as_posix())+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: acne\n'
    (output/'data.yaml').write_text(yaml,encoding='utf-8')
    manifest=dict(dataset_version='acne04v2-clean-v1',seed=seed,ratios=[.70,.15,.15],conversion_sha256=digest(conversion),source=data['summary'],counts=dict(Counter(r['split'] for r in allrows)),images=sorted(allrows,key=lambda r:(r['split'],r['file_name'])))
    write_json(output/'manifest.json',manifest)
    write_csv(conversion.parent/'split_manifest.csv',manifest['images'],['file_name','image_id','split','width','height','sha256','label_sha256','lesions'])
    verify_dataset(output)
    print('Dataset verified:',manifest['counts'],'lesions:',sum(r['lesions'] for r in allrows))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conversion',type=Path,default=AUDIT/'conversion.json')
    p.add_argument('--output',type=Path,default=DATASET)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();split(a.conversion,a.output,a.seed)


if __name__=='__main__':
    main()
