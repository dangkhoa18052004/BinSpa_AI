"""Run every training of the 5-fold cross-validation, resumably, then the evaluation.

Per fold k (datasets/phase2_cv, built by cv_folds.py), each step in its own process so GPU memory is freed:
  1. EfficientNet-B0 cfg_a on the fold's crops      -> runs/phase2_cv/fold{k}/efficientnet_b0_cfg_a
  2. RT-DETR-L 5-class (one stage)                  -> runs/phase2_cv/fold{k}/det5
  3. RT-DETR-L 1-class (the pipeline's detector)    -> runs/phase2_cv/fold{k}/det1
Completed steps are skipped. An incomplete run directory stops the script (inspect and delete it).
Usage: venv/Scripts/python.exe -u scripts/phase2/cv_run.py [--folds 0 1 2 3 4] [--no-eval]
"""
import argparse
import subprocess
import sys
from datetime import datetime

from common2 import ROOT

PY = sys.executable
STEPS = [('efficientnet_b0_cfg_a', ['scripts/phase2/train_classifier.py', '--model', 'efficientnet_b0', '--config', 'cfg_a']),
         ('det5', ['scripts/phase2/train_det5.py', '--model', 'rtdetr', '--task', 'det5', '--resume']),
         ('det1', ['scripts/phase2/train_det5.py', '--model', 'rtdetr', '--task', 'det1', '--resume'])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--no-eval', action='store_true')
    a = p.parse_args()
    for k in a.folds:
        for name, cmd in STEPS:
            run = ROOT / f'runs/phase2_cv/fold{k}/{name}'
            if (run / 'completed.json').exists():
                continue
            log = ROOT / f'runs/phase2_cv/fold{k}/{name}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            print(datetime.now().strftime('%H:%M'), f'fold {k} {name} start', flush=True)
            with log.open('w', encoding='utf-8') as f:
                rc = subprocess.call([PY, '-u', *cmd, '--fold', str(k)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            if rc or not (run / 'completed.json').exists():
                print(f'fold {k} {name} FAILED rc={rc}; see {log}', flush=True)
                sys.exit(1)
            print(datetime.now().strftime('%H:%M'), f'fold {k} {name} done', flush=True)
    if not a.no_eval:
        sys.exit(subprocess.call([PY, '-u', 'scripts/phase2/cv_evaluate.py'], cwd=ROOT))


if __name__ == '__main__':
    main()
