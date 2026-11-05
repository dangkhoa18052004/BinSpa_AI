"""P00 (selfie_v1): trace the deployed baseline exactly as the CLI and the API run it.

Records the SHA256 of every artifact the pipeline loads and checks it against the hashes stored in the
metadata, the runtime settings of normal and demo mode (detector confidence, imgsz, temperature, tau, input
gate thresholds), and which metadata fields describe older checkpoints. Then runs every photo in test_img/
through both entry points in both modes:
  cli  cv2.imread -> AcnePipeline.analyze   (scripts/phase2/predict_acne_type.py --face)
  api  bytes -> api/server.py decode (PIL, EXIF transpose) -> AcnePipeline.analyze via the Flask test client
and compares them (status, lesion count, matched boxes, top-1 type, type_status) plus warm latency.
Writes reports/selfie_v1/baseline_manifest.json and reports/selfie_v1/baseline/ (local only, gitignored).
Usage: venv/Scripts/python.exe scripts/evaluation_v2/baseline_trace.py
"""
import io
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
sys.path.insert(0, str(ROOT / 'api'))
from common2 import digest, read_json, write_json  # noqa: E402
from predict_acne_type import DEMO_CONF, AcnePipeline  # noqa: E402
from server import create_app  # noqa: E402

OUT = ROOT / 'reports/selfie_v1'
PHOTOS = ROOT / 'test_img'
ARTIFACTS = [
    'models/phase1/best_acne_detector.pt', 'models/phase1/model_metadata.json',
    'models/phase2/best_acne_type_robust.pth', 'models/phase2/best_acne_type.pth',
    'models/phase2/pipeline_metadata.json', 'models/phase2/model_metadata.json',
    'models/phase2/input_gate.json', 'models/phase2/input_gate_features.npz',
    'models/phase1/face_tools/face_landmarker.task',
    'scripts/phase2/predict_acne_type.py', 'scripts/phase2/input_gate.py', 'scripts/phase2/make_crops.py',
    'api/server.py', 'api/demo.html',
]


def artifacts():
    files = {a: digest(ROOT / a) for a in ARTIFACTS}
    det_meta = read_json(ROOT / 'models/phase1/model_metadata.json')
    pipe_meta = read_json(ROOT / 'models/phase2/pipeline_metadata.json')
    clf_meta = read_json(ROOT / 'models/phase2/model_metadata.json')
    det, clf = files['models/phase1/best_acne_detector.pt'], files['models/phase2/best_acne_type_robust.pth']
    checks = {
        'detector == model_metadata.selected_weights_sha256': det == det_meta['selected_weights_sha256'],
        'detector == pipeline_metadata.detector_sha256': det == pipe_meta['detector_sha256'],
        'pipeline_metadata.classifier is best_acne_type_robust.pth': pipe_meta['classifier'] == 'models/phase2/best_acne_type_robust.pth',
        'classifier == pipeline_metadata.classifier_sha256': clf == pipe_meta['classifier_sha256'],
        'phase2/model_metadata.json describes the deployed classifier': clf_meta['weights_sha256'] == clf,
    }
    return files, checks, det_meta, pipe_meta


def match(a, b, iou_min=0.5):
    """Greedy one-to-one match of two prediction lists by bbox IoU -> [(i, j, iou)]."""
    if not a or not b:
        return []
    A = np.array([p['bbox'] for p in a], float)
    B = np.array([p['bbox'] for p in b], float)
    x1 = np.maximum(A[:, None, 0], B[None, :, 0]); y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2]); y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])  # noqa: E731
    iou = inter / (area(A)[:, None] + area(B)[None, :] - inter + 1e-9)
    pairs, used = [], set()
    for i, j in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
        if iou[i, j] < iou_min:
            break
        if i in {p[0] for p in pairs} or j in used:
            continue
        pairs.append((int(i), int(j), float(iou[i, j])))
        used.add(j)
    return pairs


def compare(cli, api):
    pairs = match(cli['predictions'], api['predictions'])
    top1 = lambda p: p['top3'][0]['type']  # noqa: E731
    return dict(
        status_equal=cli['analysis_status'] == api['analysis_status'],
        lesions_cli=cli['lesion_count'], lesions_api=api['lesion_count'], matched=len(pairs),
        max_bbox_diff_px=max((float(np.abs(np.subtract(cli['predictions'][i]['bbox'], api['predictions'][j]['bbox'])).max())
                              for i, j, _ in pairs), default=0.0),
        top1_agree=sum(top1(cli['predictions'][i]) == top1(api['predictions'][j]) for i, j, _ in pairs),
        status_agree=sum(cli['predictions'][i].get('type_status') == api['predictions'][j].get('type_status') for i, j, _ in pairs),
    )


def exif_orientation(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.getexif().get(0x0112, 1), list(im.size)


def run_mode(demo, photos):
    pipe = AcnePipeline(demo=demo)
    client = create_app(pipe).test_client()
    pipe.analyze(cv2.imread(str(photos[0])), 'warmup')  # cuDNN kernel selection, first-call overheads
    rows, cli_ms, api_ms = [], [], []
    for p in photos:
        img = cv2.imread(str(p))
        t = time.perf_counter()
        cli = pipe.analyze(img, p.name)
        cli_ms.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        resp = client.post('/analyze', data={'image': (io.BytesIO(p.read_bytes()), p.name)}, content_type='multipart/form-data')
        api_ms.append((time.perf_counter() - t) * 1000)
        api = resp.get_json()
        orient, size = exif_orientation(p)
        rows.append(dict(photo=p.name, file_size=size, exif_orientation=orient, cli_image_size=cli['image_size'],
                         api_image_size=api.get('image_size'), cli=cli, api=api, consistency=compare(cli, api)))
    pct = lambda v, q: round(float(np.percentile(v, q)), 1)  # noqa: E731
    latency = dict(cli_analyze_ms_p50=pct(cli_ms, 50), cli_analyze_ms_p95=pct(cli_ms, 95),
                   api_request_ms_p50=pct(api_ms, 50), api_request_ms_p95=pct(api_ms, 95), photos=len(photos))
    return pipe.versions(), rows, latency


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    files, checks, det_meta, pipe_meta = artifacts()
    photos = sorted(PHOTOS.glob('*.jpg'))
    modes = {}
    for name, demo in (('normal', False), ('demo', True)):
        versions, rows, latency = run_mode(demo, photos)
        write_json(OUT / f'baseline/predictions_{name}.json', rows)
        cons = [r['consistency'] for r in rows]
        statuses = {}
        for r in rows:
            statuses[r['cli']['analysis_status']] = statuses.get(r['cli']['analysis_status'], 0) + 1
        modes[name] = dict(
            runtime=versions, latency=latency, cli_status_counts=statuses,
            lesions_total=sum(r['cli']['lesion_count'] or 0 for r in rows),
            type_shown_total=sum(1 for r in rows for p in r['cli']['predictions']
                                 if (name == 'demo' and 'type' in p) or p.get('type_status') == 'confident'),
            cli_vs_api=dict(photos=len(cons), status_equal=sum(c['status_equal'] for c in cons),
                            lesions_cli=sum(c['lesions_cli'] or 0 for c in cons), lesions_api=sum(c['lesions_api'] or 0 for c in cons),
                            matched=sum(c['matched'] for c in cons), top1_agree=sum(c['top1_agree'] for c in cons),
                            status_agree=sum(c['status_agree'] for c in cons),
                            max_bbox_diff_px=max(c['max_bbox_diff_px'] for c in cons)))
        print(name, json.dumps(modes[name], ensure_ascii=False, default=str), flush=True)
    gate = read_json(ROOT / 'models/phase2/input_gate.json')
    manifest = dict(
        milestone='P00', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        artifacts_sha256=files, metadata_checks=checks,
        runtime=dict(
            detector=dict(weights='models/phase1/best_acne_detector.pt', architecture=det_meta['architecture'],
                          imgsz=det_meta['imgsz'], max_det=500, conf_normal=det_meta['operating_conf'], conf_demo=DEMO_CONF),
            crop='square, side = 2 x max(w, h) of the detector box, min 32 px, shifted inside the image (make_crops.crop_box)',
            classifier=dict(weights=pipe_meta['classifier'], architecture=pipe_meta['classifier_architecture'],
                            config=pipe_meta['classifier_config'], temperature=pipe_meta['type_calibration']['temperature'],
                            accept_threshold=pipe_meta['type_calibration']['accept_threshold']),
            input_gate=dict(thresholds=gate['thresholds'], quality_check=gate['quality_check'], domain_check=gate['domain_check'],
                            min_lesions_for_domain=3),
            decode=dict(cli='cv2.imread (applies EXIF orientation)', api='PIL ImageOps.exif_transpose -> RGB -> BGR'),
        ),
        not_used_at_runtime={
            'pipeline_metadata.operating_score_threshold': pipe_meta.get('operating_score_threshold'),
            'pipeline_metadata.end_to_end.*.conf': 'evaluation thresholds on conf x type probability; AcnePipeline uses model_metadata.operating_conf',
            'models/phase2/model_metadata.json': 'describes best_acne_type.pth (crop accuracy 86.60 %), not the deployed robust classifier',
        },
        modes=modes,
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        photos=dict(folder='test_img', count=len(photos), labelled=False),
    )
    write_json(OUT / 'baseline_manifest.json', manifest)
    print(json.dumps(dict(metadata_checks=checks), ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
