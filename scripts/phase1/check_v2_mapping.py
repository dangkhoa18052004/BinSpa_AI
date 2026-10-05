import argparse
from pathlib import Path
from PIL import Image
from common import RAW, ANNOTATIONS, AUDIT, load_annotations, write_csv


def main():
    parser = argparse.ArgumentParser(description='Check every v2 filename and image dimensions')
    parser.add_argument('--images', type=Path, default=RAW)
    parser.add_argument('--annotations', type=Path, default=ANNOTATIONS)
    args = parser.parse_args()
    missing, errors = [], []
    data = load_annotations(args.annotations)
    for im in data['images']:
        path = args.images/im['file_name']
        if not path.is_file():
            missing.append(im)
            continue
        try:
            with Image.open(path) as img:
                if img.size != (im['width'], im['height']):
                    raise ValueError(f'Size {img.size} != metadata')
                img.verify()
        except Exception as exc:
            errors.append(dict(file_name=im['file_name'], error=str(exc)))
    write_csv(AUDIT/'missing_files.csv', missing, ['id','file_name','width','height'])
    write_csv(AUDIT/'image_errors.csv', errors, ['file_name','error'])
    print(f'Found={len(data["images"])-len(missing)}, Missing={len(missing)}, image_errors={len(errors)}')
    if missing or errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
