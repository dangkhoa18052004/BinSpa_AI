"""Verify saved frozen predictions and render the exploratory evaluation report.

Calibration diagnostics describe supplied GT crops/labels; they neither train
temperature scaling nor validate confidence clinically. Fixed ten equal-width bins.
"""
import csv
import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/detectdataskin2_audit'
EVAL = OUT / 'frozen_evaluation'
CLASSES = ['closed_comedo', 'open_comedo', 'papule', 'pustule', 'nodule']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    metrics = read(EVAL / 'metrics.json')
    protocol = read(EVAL / 'protocol_locked_before_inference.json')
    summary = read(OUT / 'summary.json')
    assert protocol == metrics['protocol']
    assert sha(ROOT / 'models/phase1/best_acne_detector.pt') == protocol['detector_sha256']
    assert sha(ROOT / 'models/phase2/best_acne_type.pth') == protocol['classifier_sha256']
    assert sha(OUT / 'image_manifest.json') == protocol['image_manifest_sha256']
    assert sha(ROOT / 'datasets/detectdataskin2_screened/annotations.coco.json') == protocol['screened_annotations_sha256']
    gt = rows(EVAL / 'classifier_gt_crop_predictions.csv')
    binary = rows(EVAL / 'binary_per_image_counts.csv')
    assert len(gt) == summary['remaining_supported_type_boxes'] == 18846
    assert len(binary) == protocol['images'] == summary['remaining_images'] == 1275
    p = np.array([[float(r['p_' + c]) for c in CLASSES] for r in gt])
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert np.allclose(p.sum(1), 1, atol=1e-6)
    y = np.array([CLASSES.index(r['true']) for r in gt])
    pred = p.argmax(1)
    assert all(CLASSES[k] == r['pred'] for k, r in zip(pred, gt))
    cm = np.bincount(y * 5 + pred, minlength=25).reshape(5, 5)
    assert cm.tolist() == metrics['phase2_gt_crop_classifier']['confusion_matrix']
    assert abs(np.trace(cm) / cm.sum() - metrics['phase2_gt_crop_classifier']['accuracy']) < 1e-12
    op = metrics['phase1_binary_detection']['operating_point']
    for key in ['tp', 'fp', 'fn']:
        assert sum(int(r[key]) for r in binary) == op[key]
    assert sum(int(r['gt']) for r in binary) == summary['remaining_acne_boxes'] == 18974
    assert abs(op['precision'] - op['tp'] / (op['tp'] + op['fp'])) < 1e-12
    assert abs(op['recall'] - op['tp'] / (op['tp'] + op['fn'])) < 1e-12
    with np.load(EVAL / 'detector_raw_predictions.npz') as archive:
        raw = {key: archive[key] for key in archive.files}
    retained = [r for r in read(OUT / 'image_manifest.json') if not r['exclude_reasons']]
    assert list(raw['file_names']) == [r['file_name'] for r in retained]
    assert len(raw['offsets']) == 1276 and raw['offsets'][0] == 0
    assert raw['offsets'][-1] == len(raw['scores']) == len(raw['boxes'])
    assert np.isfinite(raw['boxes']).all() and np.isfinite(raw['scores']).all()
    assert len(raw['scores']) == metrics['phase1_binary_detection']['predictions']
    for idx, r in enumerate(binary):
        scores = raw['scores'][raw['offsets'][idx]:raw['offsets'][idx+1]]
        assert int((scores >= protocol['detector_confidence']).sum()) == int(r['pred'])
    confidence = p.max(1)
    correct = pred == y
    bins = []
    for i in range(10):
        low, high = i / 10, (i + 1) / 10
        selected = (confidence >= low) & ((confidence < high) if i < 9 else (confidence <= high))
        n = int(selected.sum())
        bins.append(dict(low=low, high=high, crops=n,
                         mean_confidence=float(confidence[selected].mean()) if n else None,
                         accuracy=float(correct[selected].mean()) if n else None))
    ece = sum(b['crops'] * abs(b['mean_confidence'] - b['accuracy']) for b in bins if b['crops']) / len(y)
    onehot = np.eye(5)[y]
    diagnostics = dict(scope='GT crops and unreviewed supplied labels only; not patient-independent calibration',
                       crops=len(y), mean_top1_confidence=float(confidence.mean()), accuracy=float(correct.mean()),
                       top1_ece_equal_width_10_bins=float(ece),
                       multiclass_brier_sum_squared=float(((p - onehot) ** 2).sum(1).mean()),
                       negative_log_likelihood=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
                       bins=bins, fitted_calibration=False)
    (EVAL / 'classifier_confidence_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    populated = [b for b in bins if b['crops']]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Confidence = accuracy')
    ax[0].plot([b['mean_confidence'] for b in populated], [b['accuracy'] for b in populated], marker='o', label='Supplied GT crop labels')
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean top-1 confidence', ylabel='Accuracy', title=f'10-bin ECE = {ece:.3f}')
    ax[0].legend(fontsize=8)
    ax[1].bar([(b['low'] + b['high']) / 2 for b in bins], [b['crops'] for b in bins], width=.08)
    ax[1].set(xlim=(0, 1), xlabel='Top-1 confidence', ylabel='Crop count', title='Confidence distribution')
    fig.suptitle('Exploratory diagnostics: unreviewed labels, no verified patient groups', fontsize=10)
    fig.tight_layout()
    fig.savefig(EVAL / 'classifier_confidence_diagnostics.png', dpi=150)
    plt.close(fig)
    receipt = dict(passed=True, checks=['checkpoint hashes unchanged', 'protocol and dataset hashes unchanged',
        'GT crop probabilities and confusion matrix', 'accuracy from saved predictions',
        'binary TP/FP/FN totals and P/R', 'raw prediction offsets and threshold counts'],
        metrics_sha256=sha(EVAL / 'metrics.json'), classifier_predictions_sha256=sha(EVAL / 'classifier_gt_crop_predictions.csv'))
    (EVAL / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    b = metrics['phase1_binary_detection']
    c = metrics['phase2_gt_crop_classifier']
    pipeline = metrics['phase1_plus_phase2_pipeline']
    percent = lambda v: f'{100*v:.2f}%'
    text = ['<!-- FROZEN_RESULTS -->', '## Kết quả chạy cố định và kiểm tra lại', '',
            '| Phần đánh giá | Chỉ số | Kết quả |', '| --- | --- | ---: |',
            f'| G1 phát hiện mụn | AP50 / AP50–95, theo Ultralytics | {percent(b["mAP50"])} / {percent(b["mAP50_95"])} |',
            f'| G1 tại ngưỡng cố định, IoU 0,5 | Precision / Recall / F1 | {percent(op["precision"])} / {percent(op["recall"])} / {percent(op["f1"])} |',
            f'| G1 tại ngưỡng cố định | TP / FP / FN | {op["tp"]} / {op["fp"]} / {op["fn"]} |',
            f'| G2 trên 18.846 crop nhãn | Accuracy / Macro-F1 | {percent(c["accuracy"])} / {percent(c["macro_f1"])} |']
    for threshold in ['iou0.3', 'iou0.5']:
        v = pipeline[threshold]['micro']
        text.append(f'| G1+G2 đúng lớp, {threshold} | Micro Precision / Recall / F1 | {percent(v["precision"])} / {percent(v["recall"])} / {percent(v["f1"])} |')
    text += ['', 'G2 trên crop nhãn được cung cấp:', '', '| Lớp | Crop | Precision | Recall | F1 |', '| --- | ---: | ---: | ---: | ---: |']
    for name in CLASSES:
        v = c['per_class'][name]
        text.append(f'| {name} | {v["support"]} | {percent(v["precision"])} | {percent(v["recall"])} | {percent(v["f1"])} |')
    text += ['', 'Pipeline phát hiện rồi phân loại, ghép đúng lớp tại IoU 0,5:', '',
             '| Lớp | TP | FP | FN | F1 |', '| --- | ---: | ---: | ---: | ---: |']
    for name in CLASSES:
        v = pipeline['iou0.5']['per_class'][name]
        text.append(f'| {name} | {v["tp"]} | {v["fp"]} | {v["fn"]} | {percent(v["f1"])} |')
    negatives = [r for r in binary if int(r['gt']) == 0]
    text += ['', f'Trên {len(negatives)} ảnh không có box mụn trong nhãn nguồn, model dự đoán {sum(int(r["pred"]) for r in negatives)} box ở ngưỡng cố định. Không diễn giải đây là specificity trên da sạch vì chưa xác nhận nhãn đầy đủ.', '',
        f'Confidence top-1 trung bình trên crop: **{percent(diagnostics["mean_top1_confidence"])}**, accuracy **{percent(c["accuracy"])}**; ECE 10 bin **{ece:.4f}**, multiclass Brier (tổng bình phương năm lớp) **{diagnostics["multiclass_brier_sum_squared"]:.4f}**, NLL **{diagnostics["negative_log_likelihood"]:.4f}**. Đây là chẩn đoán mức phù hợp của confidence với nhãn được cung cấp, không phải xác nhận confidence lâm sàng hoặc kết quả sau hiệu chỉnh.', '',
        'Kiểm tra lại đã đạt: hash checkpoint/dataset, tổng số crop, xác suất hợp lệ, confusion matrix và accuracy từ CSV; tổng TP/FP/FN và P/R; thứ tự ảnh, offsets và số box vượt ngưỡng từ NPZ. Bằng chứng trong `frozen_evaluation/verification.json`.', '',
        'Các điểm trên chưa tách được lỗi model khỏi nhãn sai/thiếu hoặc khác định nghĩa lớp. Không dùng kết quả này làm bằng chứng rằng bộ dữ liệu hoặc model đã đạt chuẩn lâm sàng.', '<!-- END_FROZEN_RESULTS -->']
    report = OUT / 'audit_report.md'
    old = report.read_text(encoding='utf-8')
    start = old.find('<!-- FROZEN_RESULTS -->')
    if start >= 0:
        end = old.index('<!-- END_FROZEN_RESULTS -->', start) + len('<!-- END_FROZEN_RESULTS -->')
        old = old[:start] + '\n'.join(text) + old[end:]
    else:
        anchor = '## Bước tiếp theo để có đánh giá đáng tin cậy hơn'
        if anchor in old:
            old = old.replace(anchor, '\n'.join(text) + '\n\n' + anchor, 1)
        else:
            old += '\n' + '\n'.join(text) + '\n'
    report.write_text(old, encoding='utf-8')
    print(json.dumps(dict(verification='PASS', phase1=b, phase2=c, confidence_diagnostics=diagnostics,
                         pipeline=pipeline['iou0.5']['micro']), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
