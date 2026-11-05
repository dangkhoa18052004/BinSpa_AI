"""P09/P10: EXIF decoding, the tiered API result, the served bundle check and the HTTP layer (no GPU).
Usage: venv/Scripts/python.exe -m unittest discover -s tests -v
"""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'api'))
sys.path.insert(0, str(ROOT / 'scripts/phase2'))
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
from display_policy import apply_policy  # noqa: E402
from server import create_app, decode  # noqa: E402
from tiered_pipeline import TieredPipeline, present, statement  # noqa: E402

CLASSES = ['closed_comedo', 'open_comedo', 'papule', 'pustule', 'nodule']
POLICY = dict(version='test', temperature=1.0, classes=CLASSES, risk=dict(box_conf=0.30, min_boxes=3, domain_max=40.0),
              groups=dict(low=dict(tau_C=0.5, tau_A=0.8, tau_B=0.7, no_type_A=['nodule'], cal=dict(A=[0, 1, 0], B=[0, 1, 0], C=[0, 1])),
                          high=dict(tau_C=0.5, tau_A=None, tau_B=0.7, no_type_A=[], cal=dict(A=None, B=[0, 1, 0], C=[0, 1]))))


def box(score, logits, x=0):
    return dict(bbox=[x, 0, x + 5, 5], score=score, logits=logits, crop_domain=30.0)


def jpeg(arr, orientation=None):
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    img.save(buf, 'JPEG', quality=95, exif=exif.tobytes())
    return buf.getvalue()


class TestExif(unittest.TestCase):
    def test_orientation_6_is_rotated_clockwise_and_boxes_follow(self):
        stored = np.zeros((20, 40, 3), np.uint8)
        stored[0:6, 0:10] = (255, 0, 0)  # red block at the stored top-left
        bgr = decode(jpeg(stored, 6))
        self.assertEqual(bgr.shape[:2], (40, 20))  # portrait after rotation
        self.assertGreater(bgr[0:10, 14:20, 2].mean(), 200)  # 90 deg clockwise: top-left -> top-right (R channel of BGR)
        self.assertLess(bgr[30:40, 0:6, 2].mean(), 50)

    def test_no_orientation_keeps_pixels(self):
        stored = np.zeros((20, 40, 3), np.uint8)
        stored[0:6, 0:10] = (255, 0, 0)
        bgr = decode(jpeg(stored))
        self.assertEqual(bgr.shape[:2], (20, 40))
        self.assertGreater(bgr[0:6, 0:10, 2].mean(), 200)


class TestPresent(unittest.TestCase):
    def rec(self, fails=(), n=3):
        boxes = [box(0.9, [0, 0, 5, 0, 0], x=10 * i) for i in range(n)] + [box(0.2, [5, 0, 0, 0, 0], x=100)]
        return dict(size=[200, 100], input_check=dict(metrics={}, fails=list(fails)), boxes=boxes)

    def test_same_boxes_as_the_evaluated_policy(self):
        for rec in (self.rec(), self.rec(fails=['no_face']), self.rec(n=2)):
            out = present(rec, POLICY, {})
            ref = apply_policy(rec, POLICY)['boxes']
            self.assertEqual([(p['bbox'], p['tier'], p.get('type'), p.get('group')) for p in out['predictions']],
                             [([round(v, 1) for v in b['bbox']], b['tier'], b['type'], b['group']) for b in ref])
            self.assertEqual(out['analysis_status'], 'ok')  # never blocked

    def test_tiers_counts_and_warnings(self):
        low = present(self.rec(), POLICY, {})
        self.assertEqual(low['risk'], 'low')
        self.assertEqual([p['tier'] for p in low['predictions']], ['A'] * 3)
        self.assertEqual(low['counts'], dict(type=dict(papule=3), group={}, location_only=0, guess={}))
        self.assertEqual(low['predictions'][0]['type_vi'], 'Mụn sẩn')
        high = present(self.rec(fails=['no_face']), POLICY, {})
        self.assertEqual(high['risk'], 'high')
        self.assertEqual({p['tier'] for p in high['predictions']}, {'B'})  # tier A off in the high-risk group
        self.assertEqual([w['code'] for w in high['warnings']], ['no_face'])
        few = present(self.rec(n=2), POLICY, {})
        self.assertEqual([w['code'] for w in few['warnings']], ['domain_unverified'])
        self.assertEqual(len(few['predictions']), 2)  # 1-2 lesions are still shown

    def test_technical_lists_every_box_above_box_conf(self):
        out = present(self.rec(), POLICY, {}, technical=True)
        self.assertEqual(len(out['technical']['boxes']), 3)  # the 0.2 box is below box_conf
        self.assertNotIn('technical', present(self.rec(), POLICY, {}))

    def test_hybrid_group_reads_alt_logits(self):
        rec = self.rec()
        for b in rec['boxes']:
            b['logits_alt'] = [0, 0, 0, 5, 0]  # alt classifier says pustule
        hyb = dict(POLICY, groups=dict(POLICY['groups'], low=dict(POLICY['groups']['low'], logits_key='logits_alt', temperature=1.0)))
        self.assertEqual({p['type'] for p in present(rec, hyb, {})['predictions']}, {'pustule'})
        self.assertEqual({p['type'] for p in present(rec, POLICY, {})['predictions']}, {'papule'})
        high = present(self.rec(fails=['no_face']), hyb, {})  # the high group still reads the primary logits
        self.assertEqual(high['predictions'], present(self.rec(fails=['no_face']), POLICY, {})['predictions'])
        self.assertEqual({b['type'] for b in present(rec, hyb, {}, technical=True)['technical']['boxes']}, {'pustule'})

    def test_uncertain_guesses_only_below_tier_A_and_for_listed_types(self):
        rec = self.rec(fails=['no_face'])  # high risk: tau_A off, so papule boxes fall to tier B
        rec['boxes'].append(box(0.9, [0, 0, 0, 0, 5], x=150))  # nodule top-1
        none = present(rec, POLICY, {})
        self.assertFalse(any('guess_type' in p for p in none['predictions']))  # no guess list -> no guesses
        out = present(rec, POLICY, {}, guess_types=dict(high=['papule']))
        self.assertEqual([p.get('guess_type') for p in out['predictions']], ['papule'] * 3 + [None])  # nodule not listed
        no_nodule = dict(POLICY, groups=dict(POLICY['groups'], high=dict(POLICY['groups']['high'], no_type_A=['nodule'])))
        listed = present(rec, no_nodule, {}, guess_types=dict(high=['papule', 'nodule']))
        self.assertIsNone(listed['predictions'][3].get('guess_type'))  # a type in no_type_A is never guessed
        self.assertEqual(out['predictions'][0]['guess_type_vi'], 'Mụn sẩn')
        self.assertEqual(out['counts']['guess'], dict(papule=3))
        low = present(self.rec(), POLICY, {}, guess_types=dict(low=['papule']))
        self.assertFalse(any('guess_type' in p for p in low['predictions']))  # tier A boxes keep their type, no guess

    def test_statement(self):
        self.assertIn('đang được đánh giá', statement({}))
        s = statement(dict(clinical_type_precision=0.912, phone_type_precision=0.804))
        self.assertIn('91%', s)
        self.assertIn('80%', s)
        self.assertIn('Chưa kiểm định trên selfie có nhãn chuyên gia', s)
        g = statement(dict(clinical_type_precision=0.919, phone_type_precision=0.835, guess_precision_clinical=0.612,
                           guess_precision_phone=0.526))
        self.assertIn('"AI đoán (chưa chắc)" chỉ đúng khoảng 61% và 53%', g)


class TestBundle(unittest.TestCase):
    def test_policy_for_other_checkpoints_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            pol = Path(d) / 'p.json'
            pol.write_text(json.dumps(dict(POLICY, detector_sha16='0' * 16, classifier_sha16='0' * 16)), encoding='utf-8')
            srv = Path(d) / 's.json'
            srv.write_text(json.dumps(dict(policy=str(pol))), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'out of sync'):
                TieredPipeline(srv)


    def test_published_numbers_of_another_policy_are_not_shown(self):
        with tempfile.TemporaryDirectory() as d:
            srv = Path(d) / 's.json'
            pub = dict(clinical_type_precision=0.9, phone_type_precision=0.8, policy_sha256='0' * 64)
            srv.write_text(json.dumps(dict(policy='models/candidates/selfie_v1/baseline_display_policy.json', published=pub)), encoding='utf-8')
            try:
                pipe = TieredPipeline(srv)
            except (RuntimeError, OSError) as e:  # checkpoints missing on this machine
                self.skipTest(str(e))
            self.assertEqual(pipe.published, {})


class FakePipeline:
    def analyze(self, image, technical=False):
        return dict(engine='tiered', image_size=[int(image.shape[1]), int(image.shape[0])], technical=technical, predictions=[])

    def versions(self):
        return dict(engine='fake')


class TestHttp(unittest.TestCase):
    def setUp(self):
        self.client = create_app(FakePipeline()).test_client()

    def post(self, data, query=''):
        return self.client.post('/analyze' + query, data=dict(image=(io.BytesIO(data), 'a.jpg')), content_type='multipart/form-data')

    def test_analyze(self):
        r = self.post(jpeg(np.zeros((20, 40, 3), np.uint8), 6))
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertEqual(j['image_size'], [20, 40])  # EXIF-corrected size
        self.assertFalse(j['technical'])
        self.assertIn('disclaimer', j)
        self.assertTrue(self.post(jpeg(np.zeros((8, 8, 3), np.uint8)), '?technical=1').get_json()['technical'])

    def test_bad_input(self):
        self.assertEqual(self.post(b'not an image').status_code, 400)
        self.assertEqual(self.client.post('/analyze', data={}).status_code, 400)
        self.assertEqual(self.client.get('/health').get_json()['models'], dict(engine='fake'))


if __name__ == '__main__':
    unittest.main()
