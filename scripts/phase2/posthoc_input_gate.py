"""Post-hoc analysis of the input check (written AFTER reading the test results of evaluate_input_gate.py).

The locked rule failed condition (a): 9 of 45 clean ACNE-DET test photos (20 %) were rejected, all by the noise
check alone; on clean val photos the noise metric ranges 0.031-0.071 while the fitted noise_max is 0.052, so
skin texture and sensor noise are not separable by this metric. Variants tested here (same other thresholds):
  without_noise  no noise check: lets even sigma 32 noise through (type F1 0.01)
  loose_noise    noise_max = midpoint of the clean val maximum and the sigma-16 val median (val data only);
                 the variant deployed in models/phase2/input_gate.json
Because the test split is already seen, the variants' false-rejection rates are checked on photos never used for any threshold: the 190 ACNE-DET master TRAIN photos and the 176 ACNE04 test photos
(clean, clinical). Quality metrics do not depend on the models, so having trained on these photos does not bias
a false-rejection rate. The test-split numbers of the variant are reported as post-hoc.
Usage: venv/Scripts/python.exe -u scripts/phase2/posthoc_input_gate.py
"""
import json
import sys

import cv2
import numpy as np

from common2 import REPORTS, SOURCE, write_json
from compare_det5 import load_split
from evaluate_input_gate import FAMILIES, decide, f1, fit_thresholds
from input_gate import FaceTool, judge, measure

OUT = REPORTS / 'input_gate'


def main():
    rows = lambda n: json.loads((OUT / f'rows_{n}.json').read_text(encoding='utf-8'))['rows']
    thr, _ = fit_thresholds(rows('acnedet_val'))
    val = rows('acnedet_val')
    clean_val_max = max(r['metrics']['noise'] for r in val if r['family'] == 'clean')
    s16 = float(np.median([r['metrics']['noise'] for r in val if r['family'] == 'noise' and r['level'] == 16]))
    loose = dict(thr, noise_max=(clean_val_max + s16) / 2)
    path = OUT / 'train_metrics.json'
    if path.exists():
        train = json.loads(path.read_text(encoding='utf-8'))
    else:
        face = FaceTool()
        train = [dict(image=im['id'], metrics=measure(cv2.imread(str(SOURCE / im['file'])), face)) for im in load_split('train')]
        write_json(path, train)
    out = {}
    for name, t in (('locked', thr), ('without_noise', dict(thr, noise_max=None)), ('loose_noise', loose)):
        rej_train = [r for r in train if judge(r['metrics'], t)]
        reasons = {}
        for r in rej_train:
            for q in judge(r['metrics'], t):
                reasons[q] = reasons.get(q, 0) + 1
        a04 = rows('acne04_test') if (OUT / 'rows_acne04_test.json').exists() else []
        test = rows('acnedet_test')
        clean = [r for r in test if r['family'] == 'clean']
        acc = [r for r in test if not decide(r, t)[0]]; rej = [r for r in test if decide(r, t)[0]]
        per = []
        for fam, lv in [(f, lv) for f, lvs in FAMILIES.items() for lv in lvs]:
            c = [r for r in test if r['family'] == fam and r['level'] == lv]
            per.append(dict(family=fam, level=lv, rejected=sum(bool(decide(r, t)[0]) for r in c) / len(c), type_f1_all=f1(c),
                            type_f1_accepted=f1([r for r in c if not decide(r, t)[0]])))
        out[name] = dict(noise_max=t['noise_max'], train_photos=len(train), train_rejected=len(rej_train) / len(train), train_reasons=reasons,
                         acne04_rejected=sum(bool(decide(r, t)[0]) for r in a04) / len(a04) if a04 else None,
                         test_clean_rejected_posthoc=sum(bool(decide(r, t)[0]) for r in clean) / len(clean),
                         test_pooled_posthoc=dict(accepted=len(acc), rejected=len(rej), type_f1_accepted=f1(acc), type_f1_rejected=f1(rej)),
                         test_per_condition_posthoc=per)
    write_json(OUT / 'posthoc.json', dict(note=__doc__.split('Usage')[0].strip(), **out))
    for n, d in out.items():
        print(n, {k: v for k, v in d.items() if k != 'test_per_condition_posthoc'})


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
