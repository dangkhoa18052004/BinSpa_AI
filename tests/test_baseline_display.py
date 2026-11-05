"""P05: rules of the recomputed normal / demo display and the T3 box geometry.
(The recomputation is also compared with the deployed code on real photos by baseline_eval.py --check-only.)
Usage: venv/Scripts/python.exe -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
from baseline_eval import display  # noqa: E402
from t3_degrade import degrade, degrade_boxes  # noqa: E402

META = dict(classes=['closed_comedo', 'open_comedo', 'papule', 'pustule', 'nodule'], tau=0.87, temperature=1.0,
            normal_conf=0.48, demo_conf=0.30, min_lesions_domain=3, gate=dict(domain_max=40.0))
SURE = [0, 0, 10, 0, 0]      # papule, top-1 ~1.0
UNSURE = [0, 0, 1, 0.8, 0]   # papule, top-1 e/(e+e^0.8+3) = 0.3422


def raw(scores, logits, domain=10.0, fails=()):
    return dict(input_check=dict(fails=list(fails)),
                boxes=[dict(bbox=[i, 0, i + 5, 5], score=s, logits=lg, crop_domain=domain) for i, (s, lg) in enumerate(zip(scores, logits))])


class TestDisplay(unittest.TestCase):
    def test_failed_check_blocks_normal_not_demo(self):
        r = raw([0.9] * 3, [SURE] * 3, fails=['face_too_small'])
        self.assertEqual(display(r, 'normal', META), dict(status='retake_photo', boxes=[], warnings=['face_too_small']))
        d = display(r, 'demo', META)
        self.assertEqual((d['status'], len(d['boxes'])), ('demo', 3))

    def test_normal_confidence_threshold_is_strict(self):
        d = display(raw([0.48, 0.49, 0.9, 0.9], [SURE] * 4), 'normal', META)
        self.assertEqual(len(d['boxes']), 3)

    def test_one_or_two_lesions_or_far_domain_withhold_types(self):
        for r in (raw([0.9, 0.9], [SURE] * 2), raw([0.9] * 3, [SURE] * 3, domain=50.0)):
            d = display(r, 'normal', META)
            self.assertEqual(d['status'], 'location_only')
            self.assertEqual({b['tier'] for b in d['boxes']}, {'C'})

    def test_tau_and_demo_types(self):
        r = raw([0.9, 0.9, 0.9, 0.35], [SURE, SURE, UNSURE, SURE])
        d = display(r, 'normal', META)
        self.assertEqual([b['tier'] for b in d['boxes']], ['A', 'A', 'C'])
        self.assertEqual(d['boxes'][0]['type'], 'papule')
        d = display(r, 'demo', META)
        self.assertEqual([b['tier'] for b in d['boxes']], ['A'] * 4)
        self.assertEqual(d['boxes'][2]['type_status'], 'uncertain')
        self.assertAlmostEqual(d['boxes'][2]['prob'], 0.3422, places=4)


class TestT3Geometry(unittest.TestCase):
    def test_downscale_and_rotation_move_boxes_with_the_image(self):
        boxes = [dict(bbox=[100, 200, 140, 240], label='papule')]
        b = degrade_boxes(boxes, 'downscale_1080', 2160, 2880)[0]['bbox']
        self.assertEqual(b, [50.0, 100.0, 70.0, 120.0])
        # a bright square on black: after rotation its centre must stay inside the moved box
        img = np.zeros((400, 300, 3), np.uint8)
        img[200:240, 100:140] = 255
        out, nb = degrade(img, boxes, 'rotate_5', 'x')
        ys, xs = np.nonzero(out[..., 0] > 128)
        x1, y1, x2, y2 = nb[0]['bbox']
        self.assertTrue(x1 <= xs.mean() <= x2 and y1 <= ys.mean() <= y2)
        self.assertLess(x2 - x1, 40 * 1.2)

    def test_photometric_variants_keep_boxes_and_are_seeded(self):
        img = np.full((50, 60, 3), 128, np.uint8)
        boxes = [dict(bbox=[1, 2, 3, 4], label='papule')]
        a, ba = degrade(img, boxes, 'noise_s8', 'k')
        b, _ = degrade(img, boxes, 'noise_s8', 'k')
        self.assertTrue((a == b).all())
        self.assertEqual(ba[0]['bbox'], [1, 2, 3, 4])

    def test_box_leaving_the_image_is_dropped(self):
        self.assertEqual(degrade_boxes([dict(bbox=[0, 0, 4, 4], label='papule')], 'rotate_5', 100, 100), [])


if __name__ == '__main__':
    unittest.main()
