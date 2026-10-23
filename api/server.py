"""BinSpa AI local service: acne analysis over HTTP for the Spa website.

Models load once at start-up (scripts/phase2/predict_acne_type.py AcnePipeline: input check -> detector ->
calibrated type classifier -> domain check). Photos are decoded in memory and never written to disk here;
storing them is the website's decision (with the customer's consent).

  GET  /          test page (api/demo.html): take or pick a photo, see boxes and types
  GET  /health    -> {"status": "ok", "models": {...}}
  POST /analyze   multipart/form-data, field "image" (JPEG/PNG/WebP, <= MAX_MB) -> analysis JSON
                  (analysis_status ok | retake_photo | location_only | demo, image_size, predictions[].bbox in
                  pixels of the EXIF-corrected photo, type top3, type_status, disclaimer)

--demo: no labelled customer photos yet, so the checks never withhold the analysis (AcnePipeline demo=True):
every lesion gets a type, flagged low-confidence; the input and domain checks are still reported.
Binds to 127.0.0.1 unless --host is given (--host 0.0.0.0 to test from a phone on the same Wi-Fi).
If BINSPA_AI_KEY is set, /health and /analyze need it in the X-API-Key header.
One GPU, one request at a time (a lock serialises inference).
Usage: venv/Scripts/python.exe api/server.py [--port 8008] [--demo] [--host 0.0.0.0]
"""
import argparse
import hmac
import io
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
from predict_acne_type import DISCLAIMER, AcnePipeline  # noqa: E402

MAX_MB = 15
MAX_PIXELS = 40_000_000
FORMATS = {'JPEG', 'PNG', 'WEBP', 'MPO'}  # MPO: multi-picture JPEG from some phones


def decode(data):
    """bytes -> BGR array, EXIF orientation applied (phone photos are often stored rotated)."""
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    img = Image.open(io.BytesIO(data))
    if img.format not in FORMATS:
        raise ValueError(f'unsupported image format {img.format}')
    img = ImageOps.exif_transpose(img).convert('RGB')
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def create_app(pipeline=None, demo=False):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_MB * 1024 * 1024
    app.json.ensure_ascii = False
    key = os.environ.get('BINSPA_AI_KEY')
    state = dict(pipeline=pipeline, lock=threading.Lock())

    def pipe():
        if state['pipeline'] is None:
            state['pipeline'] = AcnePipeline(demo=demo)
        return state['pipeline']

    @app.before_request
    def check_key():
        if request.path == '/':  # static page; it sends the key itself
            return None
        if key and not hmac.compare_digest(request.headers.get('X-API-Key', ''), key):
            return jsonify(error='unauthorized'), 401

    @app.get('/')
    def page():
        return send_file(Path(__file__).with_name('demo.html'))

    @app.get('/health')
    def health():
        return jsonify(status='ok', models=pipe().versions())

    @app.post('/analyze')
    def analyze():
        f = request.files.get('image')
        if f is None:
            return jsonify(error='missing file field "image"'), 400
        try:
            image = decode(f.read())
        except Exception as e:  # PIL raises many types for bad input
            return jsonify(error=f'cannot read image: {e}'), 400
        t = time.perf_counter()
        with state['lock']:
            result = pipe().analyze(image)
        result.update(processing_ms=round((time.perf_counter() - t) * 1000), disclaimer=DISCLAIMER)
        result.pop('image', None)
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f'image larger than {MAX_MB} MB'), 413

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=8008)
    p.add_argument('--host', default='127.0.0.1', help='0.0.0.0 = reachable from other devices on the network')
    p.add_argument('--demo', action='store_true', help='demo mode: always classify, checks only reported')
    a = p.parse_args()
    app = create_app(AcnePipeline(demo=a.demo))  # load models before accepting requests
    print(f'BinSpa AI service on http://{a.host}:{a.port}' + (' (demo mode)' if a.demo else ''), flush=True)
    app.run(host=a.host, port=a.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
