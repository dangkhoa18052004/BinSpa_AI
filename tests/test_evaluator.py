"""Fixture tests for the selfie_v1 evaluator (P04, plan section 9).

Each fixture has hand-counted TP / FP / FN per tier: wrong type, wrong group, extra box, missed lesion, duplicate
prediction, `ignore` region, photo without lesions, untyped GT, IoU 0.3 vs 0.5. The data part (skipped when
datasets/selfie_v1 is absent) checks the evaluation sets built from the locked split.
Usage: venv/Scripts/python.exe -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
from evaluate_pipeline import (case_auc, eval_items, evaluate, match, photo_counts, summarize,  # noqa: E402
                               verdict)

DATA = ROOT / 'datasets/selfie_v1'


def gt(x, label):
    return dict(bbox=[x, 0, x + 10, 10], label=label)


def box(bbox, tier, score, type=None, group=None, prob=None):
    return dict(bbox=bbox, tier=tier, score=score, type=type, group=group, prob=prob)


# typed photo: papule, closed comedo, a mole (ignore), a missed pustule
TYPED = dict(path='a.jpg', source='acnedet_author', lineage_group_id='g1', typed=True,
             boxes=[gt(0, 'papule'), gt(20, 'closed_comedo'), gt(40, 'ignore'), gt(60, 'pustule')])
TYPED_SHOWN = [
    box([0, 0, 10, 10], 'A', 0.9, 'papule', prob=0.9),             # TP, right type
    box([20, 0, 30, 10], 'A', 0.8, 'papule', prob=0.8),            # TP, wrong type and wrong group
    box([40, 0, 50, 10], 'C', 0.7),                                # on the mole: ignored
    box([100, 100, 110, 110], 'B', 0.6, group='inflammatory', prob=0.6),  # no GT: FP
    box([1, 0, 11, 10], 'A', 0.5, 'papule', prob=0.5),             # second box on the papule: duplicate FP
]
# ACNE04-v2 style photo: one lesion without type
UNTYPED = dict(path='b.jpg', source='acne04', lineage_group_id='g2', typed=False, boxes=[gt(0, 'acne_untyped')])
UNTYPED_SHOWN = [box([0, 0, 10, 10], 'A', 0.9, 'papule', prob=0.9), box([50, 50, 60, 60], 'A', 0.8, 'pustule', prob=0.8)]
# clear skin, nothing shown
EMPTY = dict(path='c.jpg', source='acnedet_author', lineage_group_id='g3', typed=True, boxes=[])


class TestMatch(unittest.TestCase):
    def test_typed_fixture_outcomes(self):
        kinds = [k for k, _ in match(TYPED['boxes'], TYPED_SHOWN, 0.3)]
        self.assertEqual(kinds, ['tp', 'tp', 'ignored', 'fp', 'dup'])

    def test_higher_score_takes_the_lesion_first(self):
        shown = [box([0, 0, 10, 10], 'C', 0.5), box([0, 0, 10, 6], 'C', 0.9)]  # IoU 1.0 (low score), 0.6 (high score)
        self.assertEqual(match([gt(0, 'papule')], shown, 0.3), [('dup', None), ('tp', 0)])

    def test_box_takes_the_best_overlapping_lesion(self):
        # IoU 0.43 with the first lesion, 1.0 with the second
        res = match([gt(0, 'papule'), gt(4, 'pustule')], [box([4, 0, 14, 10], 'C', 0.9)], 0.2)
        self.assertEqual(res, [('tp', 1)])

    def test_lesion_match_wins_over_ignore(self):
        g = [gt(0, 'ignore'), gt(0, 'papule')]
        self.assertEqual(match(g, [box([0, 0, 10, 10], 'C', 0.9)], 0.3), [('tp', 1)])

    def test_iou_threshold(self):
        shown = [box([0, 0, 10, 4], 'C', 0.9)]  # IoU 0.4
        self.assertEqual(match([gt(0, 'papule')], shown, 0.3)[0][0], 'tp')
        self.assertEqual(match([gt(0, 'papule')], shown, 0.5)[0][0], 'fp')


class TestCounts(unittest.TestCase):
    def test_typed_fixture_metrics(self):
        m = summarize(photo_counts(TYPED, TYPED_SHOWN, 0.3))
        self.assertEqual((m['shown'], m['ignored'], m['tp'], m['fp'], m['duplicates'], m['lesion_gt']), (5, 1, 2, 2, 1, 3))
        self.assertAlmostEqual(m['location_precision'], 2 / 4)
        self.assertAlmostEqual(m['location_recall'], 2 / 3)
        self.assertEqual(m['tier_A_shown'], 3)
        self.assertAlmostEqual(m['tier_A_precision'], 1 / 3)
        self.assertEqual(m['tier_B_shown'], 1)
        self.assertAlmostEqual(m['tier_B_precision'], 0.0)
        self.assertAlmostEqual(m['group_precision_A_or_B'], 1 / 4)
        self.assertAlmostEqual(m['coverage_A_on_found'], 1.0)
        self.assertAlmostEqual(m['ece_displayed'], (0.1 + 0.8 + 0.6 + 0.5) / 4)
        self.assertAlmostEqual(m['ece_on_found'], (0.1 + 0.8) / 2)
        # papule F1 2*1/(3+1); closed comedo and pustule 0
        self.assertAlmostEqual(m['macro_f1_box_type'], 0.5 / 3)
        self.assertEqual(m['confusion_A'], {'papule>papule': 1, 'closed_comedo>papule': 1, 'none>papule': 1})
        self.assertEqual(m['per_type']['papule'], dict(shown_A=3, precision_A=1 / 3, gt=1))

    def test_untyped_photo_scores_location_only(self):
        m = summarize(photo_counts(UNTYPED, UNTYPED_SHOWN, 0.3))
        self.assertEqual((m['tp'], m['fp'], m['lesion_gt']), (1, 1, 1))
        self.assertEqual(m['tier_A_shown'], 0)
        self.assertIsNone(m['tier_A_precision'])
        self.assertIsNone(m['ece_displayed'])
        self.assertEqual(m['coverage_A_on_found'], 1.0)

    def test_bad_boxes_are_rejected(self):
        with self.assertRaises(ValueError):
            photo_counts(TYPED, [box([0, 0, 10, 10], 'A', 0.9)], 0.3)            # tier A without type
        with self.assertRaises(ValueError):
            photo_counts(TYPED, [box([0, 0, 10, 10], 'A', 0.9, 'papule', 'non_inflammatory')], 0.3)
        with self.assertRaises(ValueError):
            photo_counts(TYPED, [box([0, 0, 10, 10], 'B', 0.9)], 0.3)            # tier B without group
        with self.assertRaises(ValueError):
            photo_counts(TYPED, [box([0, 0, 10, 10], 'X', 0.9)], 0.3)


class TestEvaluate(unittest.TestCase):
    def preds(self):
        return {'a.jpg': dict(path='a.jpg', status='ok', warnings=['dark'], boxes=TYPED_SHOWN),
                'b.jpg': dict(path='b.jpg', status='ok', boxes=UNTYPED_SHOWN),
                'c.jpg': dict(path='c.jpg', status='retake_photo', boxes=[])}

    def test_every_photo_in_denominator_and_sources_split(self):
        r = evaluate([TYPED, UNTYPED, EMPTY], self.preds(), (0.3,), resamples=50)
        self.assertEqual((r['photos'], r['groups'], r['photos_with_warnings']), (3, 3, 1))
        self.assertEqual(r['statuses'], {'ok': 2, 'retake_photo': 1})
        o = r['by_iou']['0.3']['overall']['metrics']
        self.assertEqual((o['photos'], o['tp'], o['fp'], o['lesion_gt']), (3, 3, 3, 4))
        self.assertAlmostEqual(o['fp_per_photo'], 1.0)
        self.assertAlmostEqual(o['tier_A_precision'], 1 / 3)        # untyped photo adds nothing to tier A
        self.assertEqual(set(r['by_iou']['0.3']['per_source']), {'acnedet_author', 'acne04'})
        self.assertEqual(r['by_iou']['0.3']['per_source']['acnedet_author']['metrics']['photos'], 2)
        self.assertEqual(r['by_iou']['0.3']['worst_source']['location_precision']['source'], 'acnedet_author')

    def test_missing_prediction(self):
        p = self.preds()
        del p['c.jpg']
        with self.assertRaises(ValueError):
            evaluate([TYPED, UNTYPED, EMPTY], p, (0.3,), resamples=0)
        r = evaluate([TYPED, UNTYPED, EMPTY], p, (0.3,), resamples=0, allow_missing=True)
        self.assertEqual((r['missing'], r['by_iou']['0.3']['overall']['metrics']['photos']), (1, 3))

    def test_bootstrap_ci_brackets_point_and_is_seeded(self):
        items = [dict(TYPED, path=f'{i}.jpg', lineage_group_id=f'g{i // 2}') for i in range(20)]
        preds = {f'{i}.jpg': dict(path=f'{i}.jpg', boxes=TYPED_SHOWN if i % 3 else TYPED_SHOWN[:1]) for i in range(20)}
        r1 = evaluate(items, preds, (0.3,), resamples=200)
        r2 = evaluate(items, preds, (0.3,), resamples=200)
        o = r1['by_iou']['0.3']['overall']
        lo, hi = o['ci95']['tier_A_precision']
        self.assertLessEqual(lo, o['metrics']['tier_A_precision'])
        self.assertGreaterEqual(hi, o['metrics']['tier_A_precision'])
        self.assertLess(lo, hi)
        self.assertEqual(r1['by_iou']['0.3']['overall']['ci95'], r2['by_iou']['0.3']['overall']['ci95'])

    def test_iou_05_is_reported(self):
        r = evaluate([TYPED], {'a.jpg': dict(path='a.jpg', boxes=TYPED_SHOWN)}, (0.3, 0.5), resamples=0)
        self.assertEqual(set(r['by_iou']), {'0.3', '0.5'})


class TestVerdict(unittest.TestCase):
    TARGET = dict(tier_A_precision=0.9, location_recall=0.5, ece_max=0.05, ci_lower_slack=0.05)

    def test_pass_fail_inconclusive(self):
        m = dict(tier_A_precision=0.92, location_recall=0.4, ece_displayed=None)
        v = verdict(m, dict(tier_A_precision=[0.86, 0.97]), self.TARGET)
        self.assertEqual(v, dict(tier_A_precision='pass', location_recall='fail', ece_max='inconclusive'))

    def test_tier_not_shown(self):
        v = verdict(dict(tier_B_precision=None, tier_B_shown=0), {}, dict(tier_B_precision=0.92))
        self.assertEqual(v['tier_B_precision'], 'not_shown')

    def test_wide_ci_fails(self):
        v = verdict(dict(tier_A_precision=0.92), dict(tier_A_precision=[0.80, 0.99]), dict(tier_A_precision=0.9, ci_lower_slack=0.05))
        self.assertEqual(v['tier_A_precision'], 'fail')


class TestCaseAuc(unittest.TestCase):
    def test_perfect_and_inconclusive(self):
        r = case_auc([0.9] * 25 + [0.1] * 25, [1] * 25 + [0] * 25, resamples=100)
        self.assertEqual((r['auc'], r['status']), (1.0, 'measured'))
        r = case_auc([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0], resamples=0)
        self.assertEqual((r['auc'], r['status']), (0.875, 'inconclusive'))


@unittest.skipUnless((DATA / 'lineage.jsonl').exists(), 'selfie_v1 split not built')
class TestEvalSetsOnDisk(unittest.TestCase):
    def test_T1_is_the_45_acnedet_author_photos(self):
        items = eval_items('T1')
        self.assertEqual(len(items), 45)
        self.assertTrue(all(it['source'] == 'acnedet_author' and it['typed'] for it in items))

    def test_T2_one_photo_per_roboflow_source(self):
        items = eval_items('T2')
        self.assertEqual(len(items), 284)
        keys = [it['image'].split('.rf.')[0] for it in items]
        self.assertEqual(len(keys), len(set(keys)))

    def test_val_has_no_quarantine_or_unlabelled_photo_and_no_other_role(self):
        rows = {json.loads(line)['path']: json.loads(line) for line in (DATA / 'lineage.jsonl').read_text(encoding='utf-8').splitlines()}
        for it in eval_items('val'):
            r = rows[it['path']]
            self.assertEqual(r['role'], 'val')
            self.assertFalse(r['quarantined'] or r['unlabelled'])


if __name__ == '__main__':
    unittest.main()
