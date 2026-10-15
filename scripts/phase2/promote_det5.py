"""Install the one-stage RT-DETR-L 5-class detector as the face-photo acne-type model.

Decision recorded in reports/phase2_det5_summary.md: RT-DETR-L 5-class beat the detector -> classifier
pipeline on the locked validation rule and on test (mAP50 +0.054, bootstrap CI excludes 0) and was the most
robust under the stress test. Copies the checkpoint to models/phase2/best_acne_det5.pt and writes
models/phase2/det5_metadata.json with the operating confidence (max val macro-F1) and the Platt calibration
(fitted on val) used to report P(box and type correct). The crop classifier (best_acne_type.pth) and the
phase-1 detector are left untouched: --crop mode, lesion counting and the DetectDataSkin-2 audit still use them.
Usage: venv/Scripts/python.exe scripts/phase2/promote_det5.py
"""
import shutil
from datetime import datetime, timezone

from common2 import ACNE_CLASSES, MODELS, REPORTS, ROOT, digest, read_json, write_json

SOURCE = ROOT / 'runs/phase2_det5/rtdetr_1280_det5/weights/best.pt'
TARGET = MODELS / 'best_acne_det5.pt'


def main():
    comp = read_json(REPORTS / 'phase2_det5/comparison.json')
    sel = comp['selection']
    if sel['chosen'] != 'rtdetr_det5':
        raise RuntimeError(f'locked selection is {sel["chosen"]}, not rtdetr_det5')
    if digest(SOURCE) != sel['weights_sha256']['rtdetr_det5']['detector']:
        raise RuntimeError('checkpoint changed since the comparison; re-run compare_det5.py')
    shutil.copy2(SOURCE, TARGET)
    test, boot = comp['test']['rtdetr_det5'], comp['bootstrap']['rtdetr_det5']
    write_json(MODELS / 'det5_metadata.json', dict(
        status='final', task='acne lesion detection + type, one stage', architecture='rtdetr-l', imgsz=1280,
        classes=ACNE_CLASSES, weights_sha256=digest(TARGET), source_run=str(SOURCE.relative_to(ROOT)),
        start_weights='runs/phase1_combined/rtdetr_1280_ft_acnedet/weights/best.pt',
        operating_conf=sel['val']['rtdetr_det5']['conf'],
        calibration=dict(method='Platt on logit(score), fitted on ACNE-DET val', **comp['calibration']['rtdetr_det5']['platt'],
                         meaning='calibrated = P(box matches a lesion at IoU>=0.5 and the type is right)',
                         test_ece_before=comp['calibration']['rtdetr_det5']['test_ece_before'],
                         test_ece_after=comp['calibration']['rtdetr_det5']['test_ece_after']),
        validation=sel['val']['rtdetr_det5'],
        test=dict(map50=test['map50'], map50_ci=boot['map50_ci'], macro_f1=test['macro_f1'], macro_f1_ci=boot['macro_f1_ci'],
                  diff_vs_pipeline=boot['diff_vs_pipeline'], per_class=test['per_class'], ms_per_image=test['ms_per_image']),
        replaces_for_face_mode='phase-1 detector + models/phase2/best_acne_type.pth (still available: --engine pipeline)',
        dataset='ACNE-DET Roboflow mirror v1, master split datasets/acnedet_master_split.csv',
        report='reports/phase2_det5_summary.md', promoted_utc=datetime.now(timezone.utc).isoformat(),
        disclaimer='Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da.'))
    print('installed', TARGET, digest(TARGET)[:16])


if __name__ == '__main__':
    main()
