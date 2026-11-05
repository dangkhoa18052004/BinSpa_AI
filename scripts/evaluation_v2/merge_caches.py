"""Hybrid candidate (P08b): one detector, two classifiers, combined from two raw caches of the same detector.

The primary cache gives every field (boxes, input check, logits, crop distances -> photo risk); the alt cache
must hold the same boxes (checked) and only adds its logits to each box as `logits_alt`. meta.json records the alt
classifier and which risk groups read it (`group_classifier`), which select_display_policy.py writes into the
policy. No GPU.
Usage: venv/Scripts/python.exe scripts/evaluation_v2/merge_caches.py --tag hyb_d1 --primary cand_d1c1 --alt cand_d1c0 --alt-groups low
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHES = ROOT / 'runs/selfie_v1/raw_cache'
SETS = ('check', 'val', 'T1', 'T3', 'T4', 'valP')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--primary', required=True)
    ap.add_argument('--alt', required=True)
    ap.add_argument('--alt-groups', nargs='+', choices=['low', 'high'], required=True)
    a = ap.parse_args()
    pm, am = (json.loads((CACHES / t / 'meta.json').read_text(encoding='utf-8')) for t in (a.primary, a.alt))
    if pm['versions']['detector'] != am['versions']['detector'] or pm['classes'] != am['classes']:
        raise SystemExit('the two caches come from different detectors or class lists')
    out = CACHES / a.tag
    out.mkdir(parents=True, exist_ok=True)
    for name in [n for n in SETS if (CACHES / a.primary / f'{n}.jsonl').exists()]:
        alt = {r['key']: r for r in map(json.loads, (CACHES / a.alt / f'{name}.jsonl').read_text(encoding='utf-8').splitlines())}
        lines = []
        for r in map(json.loads, (CACHES / a.primary / f'{name}.jsonl').read_text(encoding='utf-8').splitlines()):
            o = alt[r['key']]
            if [(b['bbox'], b['score']) for b in r['boxes']] != [(b['bbox'], b['score']) for b in o['boxes']]:
                raise SystemExit(f"{name} {r['key']}: boxes differ between {a.primary} and {a.alt}")
            for b, ob in zip(r['boxes'], o['boxes']):
                b['logits_alt'] = ob['logits']
            lines.append(json.dumps(r, ensure_ascii=False))
        (out / f'{name}.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(name, len(lines), flush=True)
    meta = dict(pm, merged_from=dict(primary=a.primary, alt=a.alt),
                alt=dict(classifier=am.get('classifier', 'models/phase2/best_acne_type_robust.pth'),
                         sha16=am['versions']['type_model'], temperature=am['temperature']),
                group_classifier={g: 'alt' for g in a.alt_groups})
    (out / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
