"""After P11: the uncertain type guesses ("AI đoán (chưa chắc)", tier B / C boxes): which types to show, how often right.

Owner decision 04/11/2026: show the classifier's top-1 type on boxes below tier A, marked as an uncertain guess.
This layer is outside the acceptance targets (tier A stays the only "shown type"). A guess is right when the box
matches a lesion and the guessed type is its type (IoU 0.3); guesses on non-lesions count as wrong.
Rule, on val only (runs/selfie_v1/raw_cache/<tag>/val.jsonl, same thresholds as the released policy): a type is
guessed in a risk group only with >= MIN_GUESSES guesses there and val precision >= MIN_PRECISION. The kept types go
to configs/selfie_v1/serving.json "guess_types" (applied by tiered_pipeline.present); T1 and T2 (P11 runs) are then
scored once with that filter. Writes reports/selfie_v1/final_evaluation/guess_layer.json.
Usage: venv/Scripts/python.exe scripts/evaluation_v2/measure_guesses.py --tag hyb_d1
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import digest, write_json  # noqa: E402
from display_policy import CLASSES, apply_policy  # noqa: E402
from evaluate_pipeline import eval_items, evaluate  # noqa: E402

RUN = ROOT / 'runs/selfie_v1/final'
POLICY = ROOT / 'reports/selfie_v1/final_evaluation/released_policy.json'
SERVING = ROOT / 'configs/selfie_v1/serving.json'
MIN_GUESSES, MIN_PRECISION = 30, 0.40


def load(path):
    return {r['key']: r for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())}


def guess_preds(items, cache, policy, allowed=None):
    preds, risk = {}, {}
    for it in items:
        d = apply_policy(cache[it['path']], policy)
        risk[it['path']] = d['risk']
        preds[it['path']] = dict(path=it['path'], boxes=[dict(b, tier='A', type=b['guess_type'], group=None, prob=b['guess_prob'])
                                                         for b in d['boxes'] if b.get('guess_type')
                                                         and (allowed is None or b['guess_type'] in allowed[d['risk']])])
    return preds, risk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='hyb_d1')
    a = ap.parse_args()
    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    lineage = {r['path']: r for r in map(json.loads, (ROOT / 'datasets/selfie_v1/lineage.jsonl').read_text(encoding='utf-8').splitlines())}
    val = [it for it in eval_items('val') if lineage[it['path']]['own_locked_split'] != 'train']
    preds, risk = guess_preds(val, load(ROOT / 'runs/selfie_v1/raw_cache' / a.tag / 'val.jsonl'), policy)
    allowed, val_stats = {}, {}
    for g in ('low', 'high'):
        its = [it for it in val if risk[it['path']] == g]
        m = evaluate(its, {k: preds[k] for k in (it['path'] for it in its)}, (0.3,), 0)['by_iou']['0.3']['overall']['metrics']
        val_stats[g] = {t: dict(shown=v['shown_A'], precision=v['precision_A']) for t, v in m['per_type'].items()}
        allowed[g] = [t for t in CLASSES if val_stats[g][t]['shown'] >= MIN_GUESSES and val_stats[g][t]['precision'] >= MIN_PRECISION]
    out = dict(policy_sha256=digest(POLICY), iou=0.3, rule=__doc__.split('\n\n')[1].strip(), min_guesses=MIN_GUESSES,
               min_precision=MIN_PRECISION, val=val_stats, guess_types=allowed)
    print('guess types', allowed)
    for name in ('T1', 'T2'):
        items = eval_items(name)
        p, _ = guess_preds(items, load(RUN / f'{name}.jsonl'), policy, allowed)
        rep = evaluate(items, p, (0.3,), 1000)['by_iou']['0.3']['overall']
        m = rep['metrics']
        out[name] = dict(guesses=m['tier_A_shown'], precision=m['tier_A_precision'], ci95=rep['ci95'].get('tier_A_precision'),
                         on_a_lesion=m['location_precision'],
                         per_type={t: dict(shown=v['shown_A'], precision=v['precision_A']) for t, v in m['per_type'].items()})
        print(name, m['tier_A_shown'], round(m['tier_A_precision'], 3), [round(c, 3) for c in out[name]['ci95']])
    write_json(ROOT / 'reports/selfie_v1/final_evaluation/guess_layer.json', out)
    cfg = json.loads(SERVING.read_text(encoding='utf-8'))
    cfg['guess_types'] = dict(allowed, policy_sha256=cfg['published'].get('policy_sha256'),
                              source='reports/selfie_v1/final_evaluation/guess_layer.json (val rule)')
    cfg['published'].update(guess_precision_clinical=out['T1']['precision'], guess_precision_phone=out['T2']['precision'])
    SERVING.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
