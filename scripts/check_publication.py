"""Check the explicit public file set, or a Git commit tree, before publishing.

Stdlib only. Reports file names/rule names, never matching secret values.
This checks the current tree, not removal of artifacts from Git history.
"""
import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {'.gitignore','.gitattributes','README.md','requirements-phase1.txt','requirements-phase1-lock.txt',
              'extract_acne04_images.py','zip_for_kaggle.py'}
REPORT_FILES = {'reports/publication/README.md','reports/publication/metrics.json',
                'reports/publication/confusion_matrices.json','reports/publication/artifact_manifest.json'}
LEGACY_FIGURES = {'1_precision_recall_map.png','2_map50_95.png','3_inference_time.png','4_model_size.png',
    '5_precision_vs_recall.png','6_tp_fp_fn.png','dataset_stats.png','forest_vs_yolo26s.png',
    'recall_by_density.png','yolo26s_test_F1_curve.png','yolo26s_test_PR_curve.png'}
SECRET_PATTERNS = {
    'github token': r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'cloud access key': r'AKIA[A-Z0-9]{16}',
    'model hub token': r'\bhf_[A-Za-z0-9]{25,}',
    'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'credential URL': r'https?://[^\s/@:]+:[^\s/@]+@',
    'personal machine path': (r'(?i)(?:[A-Z]:[\\/](?:Users|Project|BinSpa_AI)[\\/]|/'
                              r'Users/[^/\s]+/|/' r'home/[^/\s]+/)'),
    'assigned secret literal': r'''(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*["'][A-Za-z0-9_+/-]{20,}["']''',
}


def allowed(path):
    p=PurePosixPath(path)
    if path in ROOT_FILES or path in REPORT_FILES or path=='docs/PRIVACY.md':
        return True
    if p.parts[0]=='scripts' and p.suffix in {'.py','.ps1','.md'}:
        return not any(part in {'__pycache__','Ultralytics'} or part.startswith('.') for part in p.parts)
    if p.parent==PurePosixPath('docs/figures'):
        return p.name in LEGACY_FIGURES
    if path.startswith('docs/figures/research/'):
        if p.suffix=='.png':
            return True
        return path=='docs/figures/research/confusion_matrices/all_models_confusion_matrices.pdf'
    return False


def publication_paths():
    candidates=[ROOT / p for p in ROOT_FILES | REPORT_FILES | {'docs/PRIVACY.md'}]
    candidates+=list((ROOT/'scripts').rglob('*'))+list((ROOT/'docs/figures').rglob('*'))
    return sorted({p.relative_to(ROOT).as_posix() for p in candidates if p.is_file() and allowed(p.relative_to(ROOT).as_posix())})


def check(paths, get_bytes):
    errors=[]
    for path in paths:
        if not allowed(path):
            errors.append((path,'outside publication allowlist'))
            continue
        content=get_bytes(path)
        if len(content)>20_000_000:
            errors.append((path,'unexpected large artifact'))
        suffix=PurePosixPath(path).suffix
        if suffix in {'.png','.pdf'}:
            if suffix=='.png' and not content.startswith(b'\x89PNG\r\n\x1a\n'):
                errors.append((path,'invalid PNG signature'))
            if suffix=='.pdf' and not content.startswith(b'%PDF-'):
                errors.append((path,'invalid PDF signature'))
            continue
        text=content.decode('utf-8-sig')
        for name,pattern in SECRET_PATTERNS.items():
            if re.search(pattern,text):
                errors.append((path,name))
        if suffix=='.py':
            ast.parse(text,filename=path)
        if suffix=='.json':
            json.loads(text)
        if suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)|(?:src|href)="([^"]+)"',text):
                link=next(x for x in target if x)
                if link.startswith(('https://','http://','#','mailto:')):
                    continue
                link=link.split('#')[0]
                resolved=(ROOT / PurePosixPath(path).parent / link).resolve()
                if not resolved.is_relative_to(ROOT):
                    errors.append((path,'link escapes workspace'))
                    continue
                rel=resolved.relative_to(ROOT).as_posix()
                if rel not in paths:
                    errors.append((path,'unpublished local link: '+rel))
    manifest=json.loads(get_bytes('reports/publication/artifact_manifest.json'))
    for row in manifest['artifacts']:
        if row['path'] not in paths:
            errors.append((row['path'],'manifest references unpublished file'))
        elif hashlib.sha256(get_bytes(row['path'])).hexdigest()!=row['sha256']:
            errors.append((row['path'],'artifact hash differs'))
    report=dict(passed=not errors,files=len(paths),bytes=sum(len(get_bytes(p)) for p in paths),
        checks=['publication path allowlist','credential patterns','personal paths','Python syntax',
                'Markdown local links','aggregate artifact hashes','binary signatures'],
        scope='Current publication tree only; prior commits and other branches are not sanitized',errors=errors)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree',help='Inspect this Git tree/commit rather than local publication files')
    parser.add_argument('--manifest',help='Save the selected local file list and check receipt to this JSON')
    args=parser.parse_args()
    if args.tree:
        paths=subprocess.check_output(['git','ls-tree','-r','--name-only',args.tree],cwd=ROOT).decode('utf-8').splitlines()
        # A single Git batch process avoids spawning one process per file.
        request=''.join(f'{args.tree}:{p}\n' for p in paths).encode('utf-8')
        process=subprocess.run(['git','cat-file','--batch'],input=request,stdout=subprocess.PIPE,check=True,cwd=ROOT)
        blobs={}
        offset=0
        for path in paths:
            end=process.stdout.index(b'\n',offset)
            header=process.stdout[offset:end].split()
            size=int(header[2])
            offset=end+1
            blobs[path]=process.stdout[offset:offset+size]
            offset+=size+1
        get_bytes=blobs.__getitem__
    else:
        paths=publication_paths()
        get_bytes=lambda p:(ROOT/p).read_bytes()
    receipt=check(paths,get_bytes)
    if args.manifest:
        output=Path(args.manifest)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(dict(files=paths,verification=receipt),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    raise SystemExit(0 if receipt['passed'] else 1)


if __name__=='__main__':
    main()
