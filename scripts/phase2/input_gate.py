"""Input check for the two-stage pipeline: is this photo fit for acne analysis?

Two parts:
  quality(bgr)  before analysis. MediaPipe face landmarks (models/phase1/face_tools) -> face width in pixels; on
                the skin of the face (oval minus eyes, brows, lips), resampled to a face width of NORM_WIDTH px and
                contrast-normalised: sharpness (variance of the Laplacian), noise (Immerkaer estimate), brightness
                (mean L*) and clipped share. Thresholds come from models/phase2/input_gate.json, fitted on degraded
                validation images by scripts/phase2/evaluate_input_gate.py.
  domain(feats) after classification. Class-conditional Mahalanobis distance (shared Ledoit-Wolf covariance) of
                the classifier features of the detected lesion crops to the training crops; the photo's score is
                the median over its lesions (needs >= MIN_LESIONS). High = unlike the training photos.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'models/phase1/face_tools'
GATE = ROOT / 'models/phase2/input_gate.json'
GATE_STATS = ROOT / 'models/phase2/input_gate_features.npz'
WORK = 1280         # landmark detection resolution (longest side)
NORM_WIDTH = 1024   # face width the skin metrics are measured at
MIN_LESIONS = 3
# MediaPipe face mesh contours (same as scripts/phase1/label_error_regions.py)
EYES = [[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]]
BROWS = [[70, 63, 105, 66, 107, 55, 65, 52, 53, 46], [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
        150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
MESSAGES = dict(no_face='Không thấy rõ khuôn mặt: chụp chính diện, cả khuôn mặt trong khung hình.',
                face_too_small='Khuôn mặt quá nhỏ hoặc ảnh độ phân giải thấp: chụp gần hơn bằng camera sau.',
                blurry='Ảnh bị mờ: giữ máy chắc tay, lấy nét vào da mặt.',
                noisy='Ảnh bị nhiễu hạt: chụp ở nơi đủ sáng.',
                too_dark='Ảnh quá tối: chụp ở nơi đủ sáng, ánh sáng chiếu đều lên mặt.',
                too_bright='Ảnh quá sáng hoặc bị cháy sáng: tránh đèn flash và nắng trực tiếp.',
                out_of_domain='Ảnh khác nhiều so với ảnh dùng để huấn luyện: loại mụn chưa đủ tin cậy, chỉ hiển thị vị trí.')


class FaceTool:
    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision
        self.landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(TOOLS / 'face_landmarker.task')), num_faces=1,
            min_face_detection_confidence=.3))

    def landmarks(self, bgr):
        """-> (468+, 2) landmark pixels in the original image, or None."""
        import mediapipe as mp
        scale = min(1.0, WORK / max(bgr.shape[:2]))
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        rgb = np.ascontiguousarray(small[:, :, ::-1])
        faces = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).face_landmarks
        if not faces:
            return None
        h, w = small.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in faces[0]]) / scale


def skin_mask(shape, pts):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(pts[OVAL]).astype(np.int32)], 1)
    for part, grow in [(e, .35) for e in EYES] + [(b, .15) for b in BROWS] + [(LIPS, .15)]:
        p = pts[part]
        hole = np.zeros(shape, np.uint8)
        cv2.fillPoly(hole, [np.round(cv2.convexHull(p.astype(np.float32))[:, 0]).astype(np.int32)], 1)
        k = 2 * int(grow * np.ptp(p[:, 0])) + 1
        mask &= 1 - cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.erode(mask, np.ones((5, 5), np.uint8))


def measure(bgr, face_tool):
    """Quality metrics of one photo (no thresholds)."""
    pts = face_tool.landmarks(bgr)
    if pts is None:
        return dict(face=False)
    o = pts[OVAL]
    x0, y0 = np.maximum(o.min(0), 0).astype(int)
    x1, y1 = np.minimum(o.max(0), [bgr.shape[1] - 1, bgr.shape[0] - 1]).astype(int)
    width = float(np.ptp(o[:, 0]))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return dict(face=False)
    s = NORM_WIDTH / max(x1 - x0, 1)
    crop = cv2.resize(bgr[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    mask = skin_mask(crop.shape[:2], (pts - [x0, y0]) * s).astype(bool)
    if mask.sum() < 1000:
        return dict(face=False)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (gray - gray[mask].mean()) / max(gray[mask].std(), 1.0)   # contrast-normalised skin
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    imm = cv2.filter2D(g, cv2.CV_32F, np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32))
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return dict(face=True, face_width_px=width,
                sharpness=float(lap[inner].var()),
                noise=float(np.sqrt(np.pi / 2) * np.abs(imm[inner]).mean() / 6),
                brightness=float(lab[..., 0][mask].mean()),
                clipped=float((crop[mask].max(1) >= 250).mean()))


def judge(m, thr):
    """Metrics + thresholds -> list of failed checks (empty = fit for analysis)."""
    if not m.get('face'):
        return ['no_face']
    fails = []
    if m['face_width_px'] < thr['face_width_px_min']:
        fails.append('face_too_small')
    if m['sharpness'] < thr['sharpness_min']:
        fails.append('blurry')
    if thr.get('noise_max') is not None and m['noise'] > thr['noise_max']:  # None = check off (see input_gate.json)
        fails.append('noisy')
    if m['brightness'] < thr['brightness_min']:
        fails.append('too_dark')
    if m['brightness'] > thr['brightness_max'] or m['clipped'] > thr['clipped_max']:
        fails.append('too_bright')
    return fails


class DomainScore:
    """Class-conditional Mahalanobis (shared covariance) to the training crop features."""
    def __init__(self, means, precision):
        self.means, self.precision = means, precision

    @classmethod
    def fit(cls, feats, y, n_classes):
        from sklearn.covariance import LedoitWolf
        means = np.stack([feats[y == c].mean(0) for c in range(n_classes)])
        cov = LedoitWolf().fit(feats - means[y])
        return cls(means, cov.precision_.astype(np.float32))

    @classmethod
    def load(cls, path=GATE_STATS):
        d = np.load(path)
        return cls(d['means'], d['precision'])

    def save(self, path=GATE_STATS):
        np.savez(path, means=self.means, precision=self.precision)

    def crops(self, feats):
        d = feats[:, None, :] - self.means[None]
        return np.sqrt(np.einsum('ncf,fg,ncg->nc', d, self.precision, d).min(1))

    def photo(self, feats):
        return float(np.median(self.crops(feats))) if len(feats) >= MIN_LESIONS else None


def load_gate(path=GATE):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
