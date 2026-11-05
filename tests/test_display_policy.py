"""P08a: tiered display rules (scripts/evaluation_v2/display_policy.py).
Usage: venv/Scripts/python.exe -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
from display_policy import apply_policy, decide, photo_risk  # noqa: E402

RISK = dict(box_conf=0.30, min_boxes=3, domain_max=40.0)
G = dict(tau_C=0.5, tau_A=0.8, tau_B=0.7, no_type_A=['nodule'])


def f(t='papule'):
    return dict(type=t, group='inflammatory')


class TestDecide(unittest.TestCase):
    def test_tiers(self):
        self.assertIsNone(decide(f(), dict(A=0.9, B=0.9, C=0.4), G))           # below tau_C: hidden
        self.assertEqual(decide(f(), dict(A=0.85, B=0.9, C=0.9), G), 'A')
        self.assertEqual(decide(f(), dict(A=0.6, B=0.75, C=0.9), G), 'B')
        self.assertEqual(decide(f(), dict(A=0.6, B=0.6, C=0.9), G), 'C')
        self.assertEqual(decide(f('nodule'), dict(A=0.99, B=0.9, C=0.9), G), 'B')  # nodule never at tier A

    def test_disabled_tiers(self):
        g = dict(tau_C=0.5, tau_A=None, tau_B=None)
        self.assertEqual(decide(f(), dict(A=1.0, B=1.0, C=0.9), g), 'C')


class TestRisk(unittest.TestCase):
    def rec(self, d, n=3, fails=()):
        return dict(input_check=dict(fails=list(fails)), boxes=[dict(score=0.5, crop_domain=d)] * n)

    def test_risk_groups(self):
        self.assertEqual(photo_risk(self.rec(35), RISK)[0], 'low')
        self.assertEqual(photo_risk(self.rec(45), RISK)[2], ['out_of_domain'])
        self.assertEqual(photo_risk(self.rec(35, n=2), RISK)[2], ['domain_unverified'])
        self.assertEqual(photo_risk(self.rec(35, fails=['blurry']), RISK)[0], 'high')


class TestApply(unittest.TestCase):
    def test_every_photo_gets_a_result_and_probabilities_are_shown(self):
        policy = dict(risk=RISK, temperature=1.0, classes=['closed_comedo', 'open_comedo', 'papule', 'pustule', 'nodule'],
                      groups=dict(low=dict(G, cal=dict(A=[0, 1, 0], B=[0, 1, 0], C=[0, 1])),
                                  high=dict(tau_C=0.9, tau_A=None, tau_B=None, cal=dict(A=None, B=None, C=[0, 1]))))
        box = dict(bbox=[0, 0, 5, 5], score=0.9, logits=[0, 0, 5, 0, 0], crop_domain=30)
        low = apply_policy(dict(input_check=dict(fails=[]), boxes=[box] * 3), policy)
        self.assertEqual(low['risk'], 'low')
        self.assertEqual([b['tier'] for b in low['boxes']], ['A'] * 3)
        self.assertAlmostEqual(low['boxes'][0]['prob'], 0.9, places=3)   # sigmoid(logit(0.9))
        high = apply_policy(dict(input_check=dict(fails=['no_face']), boxes=[box] * 3), policy)
        self.assertEqual((high['risk'], high['boxes']), ('high', []))  # nothing above tau_C, still a result


if __name__ == '__main__':
    unittest.main()
