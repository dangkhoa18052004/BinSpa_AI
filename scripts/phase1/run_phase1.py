"""Run smoke then full benchmark sequentially; previews must have been reviewed."""
import subprocess
import sys
import os
import argparse
from datetime import datetime, timezone
from common import ROOT, REPORTS, verify_dataset, write_json


def main():
    from filelock import FileLock
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-training',action='store_true',help='Wait for a currently running standalone smoke/train job before continuing')
    args=parser.parse_args()
    REPORTS.mkdir(parents=True,exist_ok=True)
    with FileLock(str(REPORTS/'phase1_pipeline.lock'),timeout=0):
        verify_dataset(require_review=True)
        write_json(REPORTS/'phase1_process.json',dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),command='run_phase1.py: smoke then 3-model benchmark'))
        if args.wait_for_training:
            print('Waiting for active training to finish before smoke/benchmark continuation...',flush=True)
            with FileLock(str(REPORTS/'phase1_train.lock'),timeout=7200):
                pass
        train=ROOT/'scripts/phase1/train_phase1.py'
        subprocess.run([sys.executable,'-u',str(train),'--smoke'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-u',str(train)],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
