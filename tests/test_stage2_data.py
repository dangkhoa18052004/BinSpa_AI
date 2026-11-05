"""P06/P07: no leakage into the stage-2 training sets, and box geometry of the detector dataset builder.
Usage: venv/Scripts/python.exe -m unittest discover -s tests -v
"""
import csv
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/data_v2'))
sys.path.insert(0, str(ROOT / 'scripts/evaluation_v2'))
from build_det_dataset import phone_degrade, write  # noqa: E402
from val_phone import phone_image, phone_items  # noqa: E402

DATA = ROOT / 'datasets/selfie_v1'
LINEAGE = [json.loads(x) for x in (DATA / 'lineage.jsonl').read_text(encoding='utf-8').splitlines()] if (DATA / 'lineage.jsonl').exists() else []
HELD_OUT = ('T1', 'T2', 'acne04_test')


def groups_of(role):
    return {r['lineage_group_id'] for r in LINEAGE if r['role'] == role}


class TestDetectorGeometry(unittest.TestCase):
    def test_labels_follow_resize_and_degrade_scale(self):
        img = np.zeros((2000, 1000, 3), np.uint8)
        boxes = [dict(bbox=[100, 200, 300, 400], label='papule'), dict(bbox=[0, 0, 50, 50], label='ignore')]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for sub in ('images/train', 'labels/train'):
                (out / sub).mkdir(parents=True)
            self.assertEqual(write(out, 'train', 'a', img, boxes, 1.0), 1)  # ignore box left out
            line = (out / 'labels/train/a.txt').read_text().split()
            np.testing.assert_allclose([float(v) for v in line[1:]], [200 / 1000, 300 / 2000, 200 / 1000, 200 / 2000], atol=1e-5)
            deg, s = phone_degrade(img, 'k')
            self.assertEqual(deg.shape[:2], (round(2000 * s), round(1000 * s)))
            write(out, 'train', 'b', deg, boxes, s)  # same normalised box after the downscale
            line_b = (out / 'labels/train/b.txt').read_text().split()
            np.testing.assert_allclose([float(v) for v in line_b[1:]], [float(v) for v in line[1:]], atol=2e-3)
        self.assertEqual(phone_degrade(img, 'k')[1], s)  # seeded by key

    def test_val_phone_boxes_follow_the_image(self):
        for w, h in ((1000, 2000), (1333, 777)):
            it = dict(path=f'x/{w}.jpg', source='acne04', width=w, height=h, boxes=[dict(bbox=[100, 200, 300, 400], label='papule')])
            p = phone_items([it])[0]
            img = phone_image(np.zeros((h, w, 3), np.uint8), it['path'])
            self.assertEqual(img.shape[:2], (p['height'], p['width']))
            np.testing.assert_allclose(p['boxes'][0]['bbox'], [100 * p['width'] / w, 200 * p['height'] / h, 300 * p['width'] / w, 400 * p['height'] / h])
            self.assertEqual((p['path'], p['source']), (f'x/{w}.jpg#phone', 'acne04_phone'))


@unittest.skipUnless((DATA / 'crops_v1/crop_manifest.csv').exists(), 'crops_v1 not built')
class TestClassifierLeakage(unittest.TestCase):
    def test_crop_splits_follow_lineage_roles(self):
        role = {r['path']: r['role'] for r in LINEAGE}
        with (DATA / 'crops_v1/crop_manifest.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            self.assertEqual(role[r['source_image']], r['split'], r['source_image'])
        train = {r['lineage_group_id'] for r in rows if r['split'] == 'train'}
        self.assertFalse(train & {r['lineage_group_id'] for r in rows if r['split'] == 'val'})
        for h in HELD_OUT:
            self.assertFalse(train & groups_of(h), h)


@unittest.skipUnless((DATA / 'det_v1/build.json').exists(), 'det_v1 not built')
class TestDetectorLeakage(unittest.TestCase):
    def test_train_images_come_from_train_role_only(self):
        by_name = {}
        for r in LINEAGE:
            by_name.setdefault(f"{r['source']}_{Path(r['path']).stem}", set()).add(r['role'])
        for split, allowed in (('train', {'train'}), ('val', {'val'})):
            for p in (DATA / 'det_v1/images' / split).iterdir():
                name = re.sub(r'_(phone|rep\d)$', '', p.stem)
                self.assertIn(name, by_name, p.name)
                self.assertTrue(by_name[name] <= allowed, (p.name, by_name[name]))


if __name__ == '__main__':
    unittest.main()
