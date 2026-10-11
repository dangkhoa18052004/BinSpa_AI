"""Collect existing, verified test plots and summarize their recorded results.

No model is loaded. Counts are visually transcribed PNG labels with source hashes,
not estimates reconstructed from benchmark precision/recall.
"""
import ast
import csv
import shutil
import sys
from pathlib import Path

from common import ROOT, REPORTS, DATASET, digest, read_json, write_csv, write_json


MODELS = ('yolo26n', 'yolo26s', 'yolo11s')
NAMES = dict(zip(MODELS, ('YOLO26n', 'YOLO26s', 'YOLO11s')))
PLOTS = ('confusion_matrix.png', 'confusion_matrix_normalized.png',
         'PR_curve.png', 'P_curve.png', 'R_curve.png', 'F1_curve.png')
OUT = REPORTS / 'phase1_charts'


def existing_plot(directory, name):
    candidates = (directory / name, directory / ('Box' + name))
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise ValueError(f'Missing plot: {directory / name} (also checked Box prefix)')
    return found


def installed_cm_iou():
    path = ROOT / 'venv/Lib/site-packages/ultralytics/utils/metrics.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'ConfusionMatrix')
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'process_batch')
    defaults = dict(zip([arg.arg for arg in method.args.args[-len(method.args.defaults):]],
                        method.args.defaults))
    return float(ast.literal_eval(defaults['iou_thres'])), path


def main():
    # Windows redirected output may default to a code page without Vietnamese.
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    benchmark_path = REPORTS / 'metrics/phase1_test_benchmark.csv'
    with benchmark_path.open(encoding='utf-8-sig', newline='') as handle:
        benchmark = list(csv.DictReader(handle))
    by_model = {}
    for row in benchmark:
        model = row.get('architecture')
        if model in MODELS:
            if model in by_model:
                raise ValueError(f'Duplicate benchmark model: {model}')
            by_model[model] = row
    if set(by_model) != set(MODELS):
        raise ValueError(f'Missing benchmark model: {set(MODELS) - set(by_model)}')
    for field in ('conf', 'iou', 'imgsz', 'max_det', 'batch'):
        if len({float(row[field]) for row in by_model.values()}) != 1:
            raise ValueError(f'Models use different {field}; cannot compare counts as one protocol')

    observations_path = OUT / 'confusion_matrix_observations.json'
    observations = read_json(observations_path)
    manifest_path = DATASET / 'manifest.json'
    manifest = read_json(manifest_path)
    test_records = [row for row in manifest['images'] if row['split'] == 'test']
    if len(test_records) != 176 or manifest['counts']['test'] != 176:
        raise ValueError('Expected the existing 176-image test split; no split was changed')
    gt_boxes = sum(len((DATASET / 'labels/test' / (Path(row['file_name']).stem + '.txt'))
                       .read_text(encoding='utf-8').splitlines()) for row in test_records)
    cm_iou, cm_code = installed_cm_iou()
    inputs = [benchmark_path, manifest_path, observations_path, cm_code,
              ROOT / 'venv/Lib/site-packages/ultralytics/models/yolo/detect/val.py']
    jobs, summaries = [], []
    for model in MODELS:
        row = by_model[model]
        if row['split'] != 'test' or int(row['images']) != len(test_records):
            raise ValueError(f'{model}: benchmark does not describe the test split')
        if row['manifest_sha256'] != digest(manifest_path):
            raise ValueError(f'{model}: test manifest hash differs from benchmark')
        weights = Path(row['weights'])
        if digest(weights) != row['weights_sha256']:
            raise ValueError(f'{model}: weights differ from benchmark')
        directory = REPORTS / 'phase1_evaluation' / row['model'] / 'test'
        metrics_path = directory / 'metrics.json'
        metrics = read_json(metrics_path)
        for field in ('model', 'split', 'weights_sha256', 'manifest_sha256'):
            if metrics[field] != row[field]:
                raise ValueError(f'{model}: metrics.json disagrees with CSV field {field}')
        for field in ('images', 'imgsz', 'batch', 'conf', 'iou', 'max_det',
                      'precision', 'recall', 'mAP50', 'mAP50_95'):
            if float(metrics[field]) != float(row[field]):
                raise ValueError(f'{model}: metrics.json disagrees with CSV field {field}')
        inputs.append(metrics_path)
        for name in PLOTS:
            source = existing_plot(directory, name)
            inputs.append(source)
            jobs.append((source, OUT / model / name))
        counts = observations['models'][model]
        count_source = ROOT / counts['source']
        if count_source.resolve() != (directory / 'confusion_matrix.png').resolve():
            raise ValueError(f'{model}: counts came from a different confusion matrix')
        if digest(count_source) != counts['source_sha256']:
            raise ValueError(f'{model}: PNG changed; visually verify counts again')
        if any(type(counts[key]) is not int or counts[key] < 0 for key in ('tp', 'fp', 'fn')):
            raise ValueError(f'{model}: invalid observed counts')
        tp, fp, fn = (counts[key] for key in ('tp', 'fp', 'fn'))
        if tp + fn != gt_boxes:
            raise ValueError(f'{model}: TP + FN differs from test ground-truth box count')
        summaries.append(dict(
            model=model, split='test', images=len(test_records), ground_truth_boxes=gt_boxes,
            tp=tp, fp=fp, fn=fn, cm_confidence=float(row['conf']), cm_match_iou=cm_iou,
            cm_precision=tp / (tp + fp) if tp + fp else 0,
            cm_recall=tp / (tp + fn) if tp + fn else 0,
            precision=float(row['precision']), recall=float(row['recall']),
            mAP50=float(row['mAP50']), mAP50_95=float(row['mAP50_95']),
            cm_source=str(count_source.relative_to(ROOT)),
            benchmark_source=str(benchmark_path.relative_to(ROOT))))

    # Check every input before copying; preserve originals and previous charts.
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in inputs}
    copied = []
    for source, target in jobs:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise ValueError(f'Copy verification failed: {target}')
        copied.append(dict(source=str(source.relative_to(ROOT)),
                           destination=str(target.relative_to(ROOT)), sha256=digest(target)))

    write_csv(OUT / 'evaluation_comparison.csv', summaries, list(summaries[0]))
    winners = {}
    for field, maximum in (('tp', True), ('fp', False), ('fn', False),
                           ('precision', True), ('recall', True),
                           ('mAP50', True), ('mAP50_95', True)):
        best = (max if maximum else min)(row[field] for row in summaries)
        winners[field] = [row['model'] for row in summaries if row[field] == best]

    table = ['## So sánh trên tập test', '',
             f'Tập test: {len(test_records)} ảnh, {gt_boxes} box ground truth; '
             f'SHA256 manifest: `{digest(manifest_path)}`.', '',
             '### Số đếm đọc từ confusion matrix có sẵn', '',
             f'Confidence = {summaries[0]["cm_confidence"]}; IoU ghép box > {cm_iou}. '
             'Số đếm được chép trực tiếp từ nhãn nguyên trên PNG và lưu kèm SHA256 '
             'trong `confusion_matrix_observations.json`; không suy ngược số đếm từ Precision/Recall.', '',
             '| Model | TP | FP | FN | Precision từ CM | Recall từ CM |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        table.append(f'| {NAMES[row["model"]]} | {row["tp"]} | {row["fp"]} | {row["fn"]} '
                     f'| {row["cm_precision"]:.2%} | {row["cm_recall"]:.2%} |')
    table.extend(['', '### Các metric trong benchmark CSV', '',
                  'Precision/Recall dùng IoU 0.5 tại confidence cho mean F1 đã làm mượt cao nhất '
                  '(mỗi model có thể có confidence khác nhau). Đây không phải Precision/Recall '
                  'tính từ confusion matrix bên trên. CSV không lưu chính xác confidence tại điểm này.', '',
                  '| Model | Precision | Recall | mAP50 | mAP50-95 |',
                  '|---|---:|---:|---:|---:|'])
    for row in summaries:
        table.append(f'| {NAMES[row["model"]]} | {row["precision"]:.2%} | {row["recall"]:.2%} '
                     f'| {row["mAP50"]:.2%} | {row["mAP50_95"]:.2%} |')
    table.extend(['', '### Model dẫn đầu theo từng chỉ tiêu', '',
                  '| Chỉ tiêu | Model |', '|---|---|'])
    descriptions = ('TP cao nhất (CM)', 'FP thấp nhất (CM)', 'FN thấp nhất (CM)',
                    'Precision cao nhất (benchmark)', 'Recall cao nhất (benchmark)',
                    'mAP50 cao nhất (benchmark)', 'mAP50-95 cao nhất (benchmark)')
    for (field, models), description in zip(winners.items(), descriptions):
        table.append(f'| {description} | {", ".join(NAMES[model] for model in models)} |')
    table.extend(['', 'Các kết luận chỉ áp dụng cho test split và điều kiện đánh giá được ghi ở trên. '
                  'Chưa có bảng TP/FP/FN cho cả ba model tại cùng confidence với IoU 0.5. '
                  'Không diễn giải thứ hạng hiện tại thành kết quả trên ảnh ngoài dataset.', ''])
    (OUT / 'evaluation_comparison.md').write_text('\n'.join(table), encoding='utf-8')
    for path, expected in hashes.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'Input changed: {path}')
    write_json(OUT / 'evaluation_manifest.json', dict(
        split='test', images=len(test_records), ground_truth_boxes=gt_boxes,
        count_extraction=observations['method'],
        confusion_matrix_match_iou=cm_iou,
        confusion_matrix_confidence={row['model']: row['cm_confidence'] for row in summaries},
        threshold_provenance='Inferred from recorded metrics.json and inspected installed '
                             'Ultralytics validator / ConfusionMatrix.process_batch defaults; '
                             'PNG itself does not encode these settings.',
        source_sha256=hashes, copied_files=copied, winners=winners,
        retrained=False, inference_rerun=False, source_files_unchanged=True))
    print(f'Copied and SHA256-verified {len(copied)} existing test plots:')
    for item in copied:
        print(item['destination'])
    print('\n' + '\n'.join(table))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, StopIteration, csv.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
