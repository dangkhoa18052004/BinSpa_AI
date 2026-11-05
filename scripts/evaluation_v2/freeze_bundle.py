"""P11: freeze a display policy and every file it names into one folder under models/ (the served, rollback-able bundle).

Copies the detector, classifier and domain statistics a policy points to (often under runs/) into
models/candidates/selfie_v1/<tag>/, keeps files already under models/ where they are, and writes
<tag>/display_policy.json: the same policy with the new paths (thresholds, calibrators, hashes unchanged), plus
bundle.json with the SHA256 of every file. Refuses when a copied file's hash differs from the policy's.
Usage: venv/Scripts/python.exe scripts/evaluation_v2/freeze_bundle.py --tag hyb_d1
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import digest, write_json  # noqa: E402

FILES = dict(detector_path=('detector.pt', 'detector_sha16'), classifier_path=('classifier.pth', 'classifier_sha16'),
             alt_classifier_path=('alt_classifier.pth', 'alt_classifier_sha16'), domain_stats=('domain_stats.npz', None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    src = ROOT / f'models/candidates/selfie_v1/{a.tag}_display_policy.json'
    policy = json.loads(src.read_text(encoding='utf-8'))
    out = ROOT / 'models/candidates/selfie_v1' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, (name, sha_key) in FILES.items():
        if key not in policy:
            continue
        path = ROOT / policy[key].replace('\\', '/')
        sha = digest(path)
        if sha_key and sha[:16] != policy[sha_key]:
            raise SystemExit(f'{path}: sha {sha[:16]} != policy {policy[sha_key]}')
        if not path.resolve().is_relative_to((ROOT / 'models').resolve()):
            shutil.copy2(path, out / name)
            path = out / name
            if digest(path) != sha:
                raise SystemExit(f'copy of {name} differs')
        policy[key] = path.relative_to(ROOT).as_posix()
        files[key] = dict(path=policy[key], sha256=sha)
    target = out / 'display_policy.json'
    write_json(target, policy)
    write_json(out / 'bundle.json', dict(tag=a.tag, policy=target.relative_to(ROOT).as_posix(), policy_sha256=digest(target),
                                         source_policy=src.relative_to(ROOT).as_posix(), source_policy_sha256=digest(src), files=files))
    print(f'-> {target}')


if __name__ == '__main__':
    main()
