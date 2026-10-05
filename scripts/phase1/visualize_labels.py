import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw
from common import DATASET, REPORTS, digest, verify_dataset, write_json


def visualize(dataset=DATASET, output=REPORTS/'phase1_label_check', count=30, seed=42):
    dataset,output=Path(dataset),Path(output)
    manifest=verify_dataset(dataset)
    samples={}
    for subset in ['train','val','test']:
        records=[r for r in manifest['images'] if r['split']==subset]
        records=random.Random(seed).sample(records,min(count,len(records)))
        folder=output/subset;folder.mkdir(parents=True,exist_ok=True)
        sheet=Image.new('RGB',(1500,330*((len(records)+4)//5)), 'white')
        for index,row in enumerate(records):
            name=row['file_name']
            with Image.open(dataset/'images'/subset/name) as source:
                image=source.convert('RGB')
            width,height=image.size
            image.thumbnail((1600,1600))
            draw=ImageDraw.Draw(image)
            for line in (dataset/'labels'/subset/(Path(name).stem+'.txt')).read_text().splitlines():
                _,cx,cy,bw,bh=map(float,line.split())
                draw.rectangle(((cx-bw/2)*image.width,(cy-bh/2)*image.height,(cx+bw/2)*image.width,(cy+bh/2)*image.height),outline='#00ff00',width=2)
            image.save(folder/name,quality=90)
            image.thumbnail((290,295))
            x,y=(index%5)*300,(index//5)*330
            sheet.paste(image,(x+(300-image.width)//2,y))
            ImageDraw.Draw(sheet).text((x+5,y+297),f'{name}\n{row["lesions"]} lesions',fill='black')
        sheet.save(output/f'{subset}_contact_sheet.jpg',quality=92)
        samples[subset]=[r['file_name'] for r in records]
    write_json(output/'samples.json',dict(manifest_sha256=digest(dataset/'manifest.json'),seed=seed,samples=samples))
    print('Saved label previews:',output)


def main():
    p=argparse.ArgumentParser(description='Render labels, then record an actual visual review')
    p.add_argument('--dataset',type=Path,default=DATASET)
    p.add_argument('--output',type=Path,default=REPORTS/'phase1_label_check')
    p.add_argument('--count',type=int,default=30)
    p.add_argument('--reviewed-by')
    p.add_argument('--notes',default='')
    a=p.parse_args()
    if a.reviewed_by:
        from common import read_json
        manifest=verify_dataset(a.dataset)
        samples=read_json(a.output/'samples.json')
        if samples['manifest_sha256']!=digest(a.dataset/'manifest.json'):
            raise RuntimeError('Previews are stale')
        for subset in ['train','val','test']:
            if len(samples['samples'][subset])<min(30,manifest['counts'][subset]):
                raise RuntimeError('Need 30 previews per split')
            for name in samples['samples'][subset]:
                if not (a.output/subset/name).is_file():
                    raise RuntimeError('Missing preview')
        if not a.notes.strip():
            raise RuntimeError('Provide actual review findings with --notes')
        write_json(a.dataset/'label_review.json',dict(status='reviewed',reviewed_by=a.reviewed_by,notes=a.notes,manifest_sha256=samples['manifest_sha256'],preview_directory=str(a.output.resolve())))
        print('Recorded review for this dataset version')
    else:
        visualize(a.dataset,a.output,a.count)


if __name__=='__main__':
    main()
