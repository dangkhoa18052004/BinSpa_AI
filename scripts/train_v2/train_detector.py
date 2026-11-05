"""P06: train one detector experiment from configs/selfie_v1/<name>.yaml (Ultralytics RT-DETR).

Writes runs/selfie_v1/<name>/ (Ultralytics run), runs/selfie_v1/<name>_status.json after every epoch and
runs/selfie_v1/<name>/completed.json (config, data hash, start/best sha256, environment). An interrupted run resumes
from its last.pt when started again; a completed run is never retrained.
Usage: venv/Scripts/python.exe -u scripts/train_v2/train_detector.py --config configs/selfie_v1/det_d1.yaml
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from common import REPORTS, digest, read_json, write_json  # noqa: E402
from train_ultralytics_common import best_epoch, environment  # noqa: E402

RUNS = ROOT / 'runs/selfie_v1'


def main():
    import torch
    import ultralytics.engine.trainer as trainer_module
    from ultralytics import RTDETR
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    cfg_path = ROOT / a.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    name, run = cfg['name'], RUNS / cfg['name']
    if (run / 'completed.json').exists():
        print('Already completed:', run)
        return
    cert = REPORTS / 'phase1_amp_certificate.json'
    if cert.is_file() and read_json(cert).get('environment') == environment() and read_json(cert).get('amp_enabled') is True:
        trainer_module.check_amp = lambda _: True  # same AMP policy as the phase-1 runs
    data = ROOT / cfg['data']
    build = read_json(data.parent / 'build.json')
    last = run / 'weights/last.pt'
    status = RUNS / f'{name}_status.json'
    cb = lambda t: write_json(status, dict(updated_utc=datetime.now(timezone.utc).isoformat(), epoch=t.epoch + 1,  # noqa: E731
                                            epochs=cfg['train']['epochs'], metrics={k: float(v) for k, v in t.metrics.items()}))
    if last.exists():
        model = RTDETR(str(last))
        model.add_callback('on_fit_epoch_end', cb)
        model.train(resume=True)
    else:
        if run.exists():
            raise RuntimeError(f'{run} exists without last.pt; inspect it, it is never reset silently')
        model = RTDETR(str(ROOT / cfg['start']))
        model.add_callback('on_fit_epoch_end', cb)
        model.train(data=str(data), device='0', project=str(RUNS), name=name, exist_ok=False, plots=True, **cfg['train'])
    best = run / 'weights/best.pt'
    write_json(run / 'completed.json', dict(config=cfg, config_sha256=digest(cfg_path), start_sha256=digest(ROOT / cfg['start']),
                                            data_build=build, best_epoch=best_epoch(run), actual_epochs=model.trainer.epoch + 1,
                                            best_sha256=digest(best), peak_train_vram_gb=torch.cuda.max_memory_reserved() / 1024 ** 3,
                                            environment=environment(), completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
