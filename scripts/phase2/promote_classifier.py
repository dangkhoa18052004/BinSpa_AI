"""Install the phase-2 classifier used by the two-stage face pipeline (find acne -> tell the type).

Deployment decision of the project owner: keep the two-stage structure and use the classifier with the highest
score. Evidence (reports/phase2_det5/classifiers_e2e.json, deployed phase-1 detector in front, 45 test images):
EfficientNet-B0 cfg_a_robust has the highest test mAP50, macro-F1 and type accuracy on found lesions and is the
most robust under the stress test. Caveats recorded in the metadata: it was NOT the validation choice (lower
clean-validation scores), so picking it by test makes its test numbers optimistic, and the gains over the
previous classifier are not statistically significant (bootstrap CIs touch 0).
Copies to models/phase2/best_acne_type_robust.pth and writes models/phase2/pipeline_metadata.json, read by
predict_acne_type.py. models/phase2/best_acne_type.pth is left untouched (audit scripts pin its hash).
Usage: venv/Scripts/python.exe scripts/phase2/promote_classifier.py
"""
import shutil
from datetime import datetime, timezone

from common2 import MODELS, REPORTS, ROOT, RUNS, digest, read_json, write_json

NAME = 'efficientnet_b0_robust'
SOURCE = RUNS / 'efficientnet_b0_cfg_a_robust/best.pth'
TARGET = MODELS / 'best_acne_type_robust.pth'


def main():
    e2e = read_json(REPORTS / 'phase2_det5/classifiers_e2e.json')
    r = e2e['results'][NAME]
    if digest(SOURCE) != r['weights_sha256']:
        raise RuntimeError('checkpoint changed since classifiers_e2e.json; re-run compare_classifiers_e2e.py')
    best = {m: max(e2e['results'], key=lambda k: e2e['results'][k][m]) for m in ('test_map50', 'test_macro_f1', 'test_type_accuracy_on_found')}
    if set(best.values()) != {NAME}:
        raise RuntimeError(f'{NAME} is not the highest on every test score: {best}')
    shutil.copy2(SOURCE, TARGET)
    write_json(MODELS / 'pipeline_metadata.json', dict(
        status='final', structure='phase-1 detector -> crop -> phase-2 classifier',
        detector='models/phase1/best_acne_detector.pt', detector_sha256=digest(ROOT / 'models/phase1/best_acne_detector.pt'),
        classifier=str(TARGET.relative_to(ROOT)).replace('\\', '/'), classifier_sha256=digest(TARGET),
        classifier_architecture='efficientnet_b0', classifier_config='cfg_a_robust', operating_score_threshold=r['conf'],
        end_to_end=e2e['results'], bootstrap_vs_previous=e2e['bootstrap_vs_efficientnet_b0'][NAME]['diff_vs_pipeline'],
        decision='owner: two-stage structure, highest score; highest test mAP50, macro-F1 and type accuracy of the three',
        caveats=['chosen on test, not on validation (validation preferred the previous classifier): test numbers are optimistic',
                 'differences vs previous classifier not significant (bootstrap CIs touch 0)',
                 'stress-test gains are in-distribution for this model (its augmentation covers the stress levels)'],
        previous_classifier='models/phase2/best_acne_type.pth', report='reports/phase2_det5_summary.md',
        promoted_utc=datetime.now(timezone.utc).isoformat(),
        disclaimer='Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da.'))
    print('installed', TARGET, digest(TARGET)[:16])


if __name__ == '__main__':
    main()
