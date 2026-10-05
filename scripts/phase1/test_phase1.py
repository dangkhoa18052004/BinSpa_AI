"""Regression tests for observed corrupt annotations and duplicate leakage."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from common import ROOT, yolo_box, write_json, verify_dataset
from convert_v2_to_yolo import convert
from split_phase1 import split
from evaluate_phase1 import greedy_matches


class Phase1Tests(unittest.TestCase):
    def test_corrupt_source_coordinates_are_rejected(self):
        box,reason,_=yolo_box(dict(coordinates=[98018,304639],radius=1076),3112,3456)
        self.assertIsNone(box)
        self.assertEqual(reason,'outside_image_or_empty')

    def test_clip_recomputes_center(self):
        box,reason,clipped=yolo_box(dict(coordinates=[2,5],radius=4),10,10)
        self.assertTrue(clipped)
        self.assertEqual(box,(.3,.5,.6,.8))

    def test_bad_radius_and_nan_are_rejected(self):
        for radius in [0,-1,float('nan')]:
            self.assertIsNone(yolo_box(dict(coordinates=[5,5],radius=radius),10,10)[0])

    def test_one_truth_cannot_match_multiple_predictions(self):
        matched,fp,fn=greedy_matches([[0,0,10,10],[0,0,10,10]],[[0,0,10,10]])
        self.assertEqual((len(matched),fp,fn),(1,[1],[]))

    def test_dedupe_quarantine_split_and_mutation_detection(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp);raw=root/'raw';raw.mkdir()
            images=[];anns=[]
            for i in range(10):
                name=f'{i}.png'
                Image.new('RGB',(20,20),(i*20,0,0)).save(raw/name)
                images.append(dict(id=i,file_name=name,width=20,height=20))
                anns.append(dict(id=i,image_id=i,coordinates=[10,10],radius=2 if i!=9 else 0))
            # A partly corrupted image must be quarantined, even if a valid box remains.
            anns.append(dict(id=10,image_id=9,coordinates=[10,10],radius=2))
            (raw/'1.png').write_bytes((raw/'0.png').read_bytes())
            source=root/'annotations.json';write_json(source,dict(images=images,annotations=anns))
            summary=convert(raw,source,root/'audit')
            self.assertEqual((summary['retained_images'],summary['duplicate_extras'],summary['excluded_images']),(8,1,1))
            dataset=root/'dataset'
            manifest=split(root/'audit/conversion.json',dataset)
            self.assertEqual(len({r['sha256'] for r in manifest['images']}),8)
            verify_dataset(dataset)
            split(root/'audit/conversion.json',dataset)
            r=manifest['images'][0]
            label=dataset/'labels'/r['split']/(Path(r['file_name']).stem+'.txt')
            label.write_text('0 0.5 0.5 1 1\n')
            with self.assertRaisesRegex(RuntimeError,'Dataset modified'):
                verify_dataset(dataset)

    def test_missing_image_blocks_before_split(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp);source=root/'annotations.json'
            write_json(source,dict(images=[dict(id=0,file_name='missing.jpg',width=20,height=20)],annotations=[]))
            with self.assertRaisesRegex(RuntimeError,'Missing'):
                convert(root,source,root/'audit')
            self.assertTrue((root/'audit/missing_files.csv').is_file())


if __name__=='__main__':
    unittest.main()
