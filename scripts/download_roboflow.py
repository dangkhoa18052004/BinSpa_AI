"""Download a Roboflow Universe dataset export (COCO by default) into data_raw/.

The API key is read from the ROBOFLOW_API_KEY environment variable so it never
lands in the repo or shell history. Only the zip is downloaded; it is not
extracted. Zip members are listed and checked for unsafe paths so extraction
can be done separately after review.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = 'https://api.roboflow.com'


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f'HTTP {e.code}: {e.read().decode("utf-8", "replace")[:500]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', default='kritsakorn')
    ap.add_argument('--project', default='acne-kbm0q')
    ap.add_argument('--version', type=int, help='default: latest version')
    ap.add_argument('--format', default='coco')
    ap.add_argument('--out', type=Path, help='default: data_raw/<project>_roboflow')
    args = ap.parse_args()

    key = os.environ.get('ROBOFLOW_API_KEY')
    if not key:
        sys.exit('Set ROBOFLOW_API_KEY first (Roboflow > Settings > API Keys).')
    q = urllib.parse.urlencode({'api_key': key})

    info = get_json(f'{API}/{args.workspace}/{args.project}?{q}')
    proj = info.get('project', {})
    versions = info.get('versions', [])
    print(f"Project: {proj.get('name')} | images: {proj.get('images')} | "
          f"classes: {list(proj.get('classes', {}))} | license: {proj.get('license')}")
    if not versions:
        sys.exit('No versions published for this project.')
    print('Versions:', [v['id'].rsplit('/', 1)[-1] for v in versions])
    ver = args.version or max(int(v['id'].rsplit('/', 1)[-1]) for v in versions)

    export = get_json(f'{API}/{args.workspace}/{args.project}/{ver}/{args.format}?{q}')
    link = export.get('export', {}).get('link')
    if not link:
        sys.exit(f'No export link returned: {json.dumps(export)[:500]}')

    out = args.out or ROOT / f'data_raw/{args.project}_roboflow'
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f'{args.project}.v{ver}.{args.format}.zip'
    if zip_path.exists():
        sys.exit(f'{zip_path} already exists; not overwriting.')
    print(f'Downloading v{ver} ({args.format}) -> {zip_path}')
    urllib.request.urlretrieve(link, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        unsafe = [n for n in names if n.startswith(('/', '\\')) or '..' in Path(n).parts]
        imgs = [n for n in names if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f'{zip_path.stat().st_size / 1e6:.1f} MB, {len(names)} files, {len(imgs)} images')
    if unsafe:
        print('WARNING: unsafe paths in zip, do not extract blindly:', unsafe[:10])
    (out / 'download_info.json').write_text(json.dumps({
        'workspace': args.workspace, 'project': args.project, 'version': ver,
        'format': args.format, 'project_info': proj, 'zip': zip_path.name,
        'files': len(names), 'images': len(imgs), 'unsafe_paths': unsafe,
    }, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
