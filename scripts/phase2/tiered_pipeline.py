"""P09/P10: the tiered display pipeline served by api/server.py (plan sections 2, 6.2, 6.3).

Photo -> input check (reported as warnings, never blocks) -> detector at the policy's box_conf -> crop ->
classifier logits + Mahalanobis crop distance -> display_policy.apply_policy: each box is shown at tier A (type),
B (inflammatory / non-inflammatory), C (location only) or hidden. The checkpoints, domain statistics, calibrators
and thresholds all come from the one policy JSON named in configs/selfie_v1/serving.json, and the checkpoint hashes
must match the policy, so a stage-2 candidate is switched in (or rolled back) by editing that file only.
A hybrid policy also names an alt classifier (alt_classifier_path / alt_classifier_sha16): its logits are added to
every box as `logits_alt` and read by the risk groups whose `logits_key` says so.
Every decodable photo gets a result. 0-2 boxes are processed like any other count: the photo only falls in the
high-risk calibration group, where fewer boxes reach tiers A/B.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
from common2 import digest, read_json  # noqa: E402
from display_policy import GROUP, apply_policy, calibrated, group_features, photo_risk  # noqa: E402

SERVING = ROOT / 'configs/selfie_v1/serving.json'
DEFAULTS = dict(detector_path='models/phase1/best_acne_detector.pt', classifier_path='models/phase2/best_acne_type_robust.pth',
                domain_stats='models/phase2/input_gate_features.npz')
TYPE_VI = dict(closed_comedo='Mụn đầu trắng', open_comedo='Mụn đầu đen', papule='Mụn sẩn', pustule='Mụn mủ', nodule='Mụn nốt')
GROUP_VI = dict(inflammatory='Mụn viêm', non_inflammatory='Mụn không viêm')
RISK_WARNINGS = dict(
    out_of_domain='Ảnh khác kiểu ảnh dùng để huấn luyện AI: AI chỉ ghi loại mụn khi đủ chắc, nhiều nốt chỉ được khoanh vị trí.',
    domain_unverified='AI tìm thấy ít nốt nên chưa so được ảnh với ảnh huấn luyện: AI thận trọng hơn khi ghi loại mụn.')
UNVERIFIED = 'Chưa kiểm định trên selfie có nhãn chuyên gia. Kết quả chỉ để tham khảo chăm sóc da.'


def raw_analyze(pipe, image):
    """Input check, detector boxes above pipe.conf, classifier logits and crop domain distances, nothing withheld.
    pipe: needs face, gate, detector, imgsz, conf, clf, domain (AcnePipeline or TieredPipeline); clf_alt optional."""
    from input_gate import judge, measure
    from make_crops import crop_box
    metrics = measure(image, pipe.face)
    fails = judge(metrics, pipe.gate['thresholds'])
    r = pipe.detector.predict(image, imgsz=pipe.imgsz, conf=pipe.conf, max_det=500, verbose=False)[0]
    boxes, score = r.boxes.xyxy.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy()
    crops = []
    for (x1, y1, x2, y2) in boxes:
        x0, y0, side = crop_box([x1, y1, x2 - x1, y2 - y1], image.shape[1], image.shape[0])
        crops.append(image[y0:y0 + side, x0:x0 + side])
    feats, logits = pipe.clf.embed(crops)
    dist = pipe.domain.crops(feats) if len(feats) else np.zeros(0)
    rec = dict(size=[int(image.shape[1]), int(image.shape[0])], input_check=dict(metrics=metrics, fails=fails),
               boxes=[dict(bbox=[float(v) for v in b], score=float(s), logits=[float(v) for v in lg], crop_domain=float(d))
                      for b, s, lg, d in zip(boxes, score, logits, dist)])
    if getattr(pipe, 'clf_alt', None) is not None:
        for b, lg in zip(rec['boxes'], pipe.clf_alt.embed(crops)[1]):
            b['logits_alt'] = [float(v) for v in lg]
    return rec


def statement(published):
    """Plan 6.3: the only accuracy sentence the page may show, from measured numbers in serving.json."""
    x, y = published.get('clinical_type_precision'), published.get('phone_type_precision')
    if x is None:
        return 'Độ chính xác đang được đánh giá trên dữ liệu kiểm tra có nhãn. ' + UNVERIFIED
    s = f'Trên dữ liệu kiểm tra có nhãn, loại mụn được hiển thị đúng khoảng {round(x * 100)}% (bộ ảnh lâm sàng, nhãn bác sĩ)'
    s += f' và {round(y * 100)}% (bộ ảnh thường, nhãn chưa xác minh). ' if y is not None else '. '
    gx, gy = published.get('guess_precision_clinical'), published.get('guess_precision_phone')
    if gx is not None and gy is not None:
        s += (f'Loại ghi "AI đoán (chưa chắc)" chỉ đúng khoảng {round(gx * 100)}% và {round(gy * 100)}% trên hai bộ ảnh đó, '
              'cần chuyên viên xác nhận. ')
    return s + UNVERIFIED


def present(rec, policy, published, technical=False, guess_types=None):
    """Raw record -> API result (the evaluator scores apply_policy(rec, policy)['boxes'], the same boxes).
    guess_types {risk: [type]}: tier B / C boxes whose top-1 type is listed get guess_type / guess_type_vi /
    guess_probability, an uncertain guess shown as such (measure_guesses.py); None or {} = no guesses."""
    from input_gate import MESSAGES
    d = apply_policy(rec, policy)
    preds = []
    for b in d['boxes']:
        p = dict(bbox=[round(v, 1) for v in b['bbox']], tier=b['tier'], detection_confidence=round(b['score'], 4),
                 lesion_probability=b['q_lesion'])
        if b['tier'] in ('A', 'B'):
            p.update(group=b['group'], group_vi=GROUP_VI[b['group']], probability=b['prob'])
        if b['tier'] == 'A':
            p.update(type=b['type'], type_vi=TYPE_VI[b['type']])
        elif b.get('guess_type') in (guess_types or {}).get(d['risk'], []):
            p.update(guess_type=b['guess_type'], guess_type_vi=TYPE_VI[b['guess_type']], guess_probability=b['guess_prob'])
        preds.append(p)
    fails = rec['input_check']['fails']
    warnings = [dict(code=f, message=MESSAGES[f]) for f in fails]
    warnings += [dict(code=r, message=RISK_WARNINGS[r]) for r in d['risk_reasons'] if r in RISK_WARNINGS]
    out = dict(engine='tiered', analysis_status='ok', image_size=rec['size'], risk=d['risk'], warnings=warnings,
               lesion_count=len(preds), counts=dict(type=dict(Counter(p['type'] for p in preds if p['tier'] == 'A')),
                                                     group=dict(Counter(p['group'] for p in preds if p['tier'] == 'B')),
                                                     location_only=sum(p['tier'] == 'C' for p in preds),
                                                     guess=dict(Counter(p['guess_type'] for p in preds if 'guess_type' in p))),
               predictions=preds, statement=statement(published), policy_version=policy['version'])
    if technical:  # every detector box with its top-1 type: staff view only, never used for acceptance
        g = policy['groups'][d['risk']]
        tech = []
        for b in rec['boxes']:
            if b['score'] <= policy['risk']['box_conf']:
                continue
            f = group_features(b, policy, g)
            tech.append(dict(bbox=[round(v, 1) for v in b['bbox']], detection_confidence=round(b['score'], 4), type=f['type'],
                             type_vi=TYPE_VI[f['type']], type_softmax=round(f['p_type'], 4), group=GROUP[f['type']],
                             q={t: round(calibrated(g['cal'][t], f[t]), 4) for t in ('A', 'B', 'C')},
                             crop_domain=round(b['crop_domain'], 2)))
        out['technical'] = dict(note='Mọi box detector với loại top-1, kể cả box không đủ tin cậy để hiển thị. '
                                     'Chỉ để nhân viên kỹ thuật xem, không dùng để đánh giá.',
                                domain_score=None if d['domain_score'] is None else round(d['domain_score'], 2),
                                risk_reasons=d['risk_reasons'], boxes=tech)
    return out


class TieredPipeline:
    """Models of one display policy, loaded once (api/server.py)."""

    def __init__(self, serving=SERVING):
        from input_gate import DomainScore, FaceTool, load_gate
        from predict_acne_type import TypeClassifier, load_yolo_or_detr
        cfg = read_json(serving)
        self.policy_path = ROOT / cfg['policy']
        self.policy = read_json(self.policy_path)
        self.published = cfg.get('published', {})
        self.guess_types = cfg.get('guess_types', {})
        sha = digest(self.policy_path)
        if self.published.get('policy_sha256') not in (None, sha):
            self.published = {}  # numbers measured on another policy (e.g. after a rollback): show 'being evaluated'
        if self.guess_types.get('policy_sha256') != sha:
            self.guess_types = {}  # guesses were measured for another policy: none shown
        paths = {k: ROOT / self.policy.get(k, v) for k, v in DEFAULTS.items()}
        det_sha, clf_sha = digest(paths['detector_path'])[:16], digest(paths['classifier_path'])[:16]
        if (det_sha, clf_sha) != (self.policy['detector_sha16'], self.policy['classifier_sha16']):
            raise RuntimeError(f'{self.policy_path.name} was fitted for detector {self.policy["detector_sha16"]} / classifier '
                               f'{self.policy["classifier_sha16"]}, found {det_sha} / {clf_sha}: bundle out of sync')
        self.detector = load_yolo_or_detr(paths['detector_path'])
        self.clf = TypeClassifier(paths['classifier_path'], temperature=self.policy['temperature'])
        self.clf_alt = None
        if 'alt_classifier_path' in self.policy:
            alt = ROOT / self.policy['alt_classifier_path']
            if digest(alt)[:16] != self.policy['alt_classifier_sha16']:
                raise RuntimeError(f'{self.policy_path.name}: alt classifier {alt} does not match: bundle out of sync')
            self.clf_alt = TypeClassifier(alt)
        self.domain = DomainScore.load(paths['domain_stats'])
        self.gate = load_gate()
        self.face = FaceTool()
        self.imgsz = read_json(ROOT / 'models/phase1/model_metadata.json')['imgsz']
        self.conf = self.policy['risk']['box_conf']
        self.det_version, self.clf_version = det_sha, clf_sha

    def versions(self):
        return dict(engine='tiered', policy=str(self.policy_path.relative_to(ROOT)), policy_sha256=digest(self.policy_path)[:16],
                    policy_version=self.policy['version'], detector=self.det_version, type_model=self.clf_version,
                    alt_type_model=self.policy.get('alt_classifier_sha16'))

    def analyze(self, image, technical=False):
        """image: BGR uint8 array."""
        out = present(raw_analyze(self, image), self.policy, self.published, technical, self.guess_types)
        return dict(out, detector_version=self.det_version, type_model_version=self.clf_version)
