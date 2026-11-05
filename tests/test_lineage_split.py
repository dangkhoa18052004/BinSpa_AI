"""Leakage tests for the selfie_v1 split (P03).

Unit part: union-find and role assignment on a small synthetic fixture.
Data part (skipped when datasets/selfie_v1 is absent): the locked split on disk has no group or Roboflow source
name in two roles, T1 is exactly the ACNE-DET master test, every locked ACNE04 test photo stays protected, and the
locked config files still match their hashes.
Usage: venv/Scripts/python.exe -m unittest discover -s tests -v
"""
import csv
import json
import re
import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/data_v2'))
sys.path.insert(0, str(ROOT / 'scripts/phase1'))
from build_lineage_split import UnionFind, assign_roles  # noqa: E402
from common import digest  # noqa: E402

DATA = ROOT / 'datasets/selfie_v1'


class TestUnionFind(unittest.TestCase):
    def test_copies_flip_resize_mirror_end_in_one_group(self):
        # 0 original, 1 flipped copy (same key), 2 resized mirror (same sha as 0), 3 unrelated
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 0)
        self.assertEqual(len({uf.find(i) for i in (0, 1, 2)}), 1)
        self.assertNotEqual(uf.find(3), uf.find(0))


class TestAssignRoles(unittest.TestCase):
    def test_most_protective_locked_role_wins(self):
        groups = {'g1': [0, 1], 'g2': [2]}
        role, reason = assign_roles(groups, {0: 'train', 1: 'T1', 2: 'val'}, {'g1': 'other', 'g2': 'other'}, {'g1': 2, 'g2': 1})
        self.assertEqual(role['g1'], 'T1')
        self.assertTrue(reason['g1'].startswith('locked_conflict'))
        self.assertEqual(role['g2'], 'val')

    def test_free_groups_split_by_photo_share_and_seed(self):
        groups = {f'g{i}': [i] for i in range(200)}
        pools = {g: 'roboflow' for g in groups}
        sizes = {g: 1 for g in groups}
        r1, _ = assign_roles(groups, {}, pools, sizes, seed=1)
        r2, _ = assign_roles(groups, {}, pools, sizes, seed=1)
        self.assertEqual(r1, r2)
        share = {k: sum(v == k for v in r1.values()) / 200 for k in ('T2', 'val', 'train')}
        self.assertAlmostEqual(share['T2'], 0.20, delta=0.02)
        self.assertAlmostEqual(share['val'], 0.15, delta=0.02)

    def test_other_pool_never_gets_T2(self):
        groups = {f'g{i}': [i] for i in range(50)}
        role, _ = assign_roles(groups, {}, {g: 'other' for g in groups}, {g: 1 for g in groups})
        self.assertNotIn('T2', role.values())


@unittest.skipUnless((DATA / 'lineage.jsonl').exists(), 'selfie_v1 split not built')
class TestLockedSplitOnDisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in (DATA / 'lineage.jsonl').read_text(encoding='utf-8').splitlines()]
        cls.manifest = json.loads((DATA / 'split_manifest.json').read_text(encoding='utf-8'))

    def test_group_has_one_role(self):
        roles = defaultdict(set)
        for r in self.rows:
            if r['role'] != 'quarantine':
                roles[r['lineage_group_id']].add(r['role'])
        self.assertEqual([g for g, v in roles.items() if len(v) > 1], [])

    def test_roboflow_copies_share_a_role(self):
        roles = defaultdict(set)
        for r in self.rows:
            if r['source'] in ('detectdataskin2', 'xnbio', 'kbm0q') and r['role'] != 'quarantine':
                roles[(r['source'], r['image'].split('.rf.')[0])].add(r['role'])
        self.assertEqual([k for k, v in roles.items() if len(v) > 1], [])

    def test_T1_is_exactly_the_acnedet_master_test(self):
        with (ROOT / 'datasets/acnedet_master_split.csv').open(encoding='utf-8-sig') as f:
            master = list(csv.DictReader(f))
        test_ids = {int(m['source_id']) for m in master if m['split'] == 'test'}
        t1 = {int(r['image'].split('.')[0]) for r in self.rows if r['source'] == 'acnedet_author' and r['role'] == 'T1'}
        self.assertEqual(t1, test_ids)

    def test_locked_acne04_test_photos_stay_protected(self):
        bad = [r['image'] for r in self.rows if r['source'] == 'acne04' and r['own_locked_split'] == 'acne04_test'
               and r['role'] != 'acne04_test']
        self.assertEqual(bad, [])

    def test_no_T2_photo_from_clinical_families(self):
        t2 = [r for r in self.rows if r['role'] == 'T2' and r['source'].startswith('acnedet')]
        self.assertEqual(t2, [])

    def test_locked_files_unchanged(self):
        for path, sha in self.manifest['locked_files_sha256'].items():
            self.assertEqual(digest(ROOT / path), sha, f'{path} changed after the split was locked')
        self.assertEqual(digest(DATA / 'lineage.jsonl'), self.manifest['lineage_sha256'])

    def test_acne04_named_roboflow_copies_follow_acne04(self):
        role_of = {r['image']: r['role'] for r in self.rows if r['source'] == 'acne04'}
        bad = []
        for r in self.rows:
            m = re.match(r'(levle\d+_\d+)_jpe?g', r['image']) if r['source'] != 'acne04' else None
            if m and m.group(1) + '.jpg' in role_of and r['role'] not in (role_of[m.group(1) + '.jpg'], 'quarantine'):
                bad.append(r['image'])
        self.assertEqual(bad, [])


if __name__ == '__main__':
    unittest.main()
