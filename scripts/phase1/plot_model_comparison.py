"""Plot existing phase 1 benchmarks without loading or training models.

Usage: venv/Scripts/python.exe scripts/phase1/plot_model_comparison.py
Scores are displayed as percentages; CSV values remain unchanged.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

from common import REPORTS, digest, write_json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


MODELS = ('yolo26n', 'yolo26s', 'yolo11s')
LABELS = ('YOLO26n', 'YOLO26s', 'YOLO11s')
COLORS = ('#2563eb', '#ea580c', '#059669')
SCORES = ('precision', 'recall', 'mAP50', 'mAP50_95')
SCORE_LABELS = ('Precision', 'Recall', 'mAP50', 'mAP50-95')
FIELDS = (*SCORES, 'inference_ms', 'model_mb')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def model_id(row):
    architecture = str(row.get('architecture', '')).lower()
    if architecture in MODELS:
        return architecture
    match = re.search(r'(?<![a-z0-9])(yolo26n|yolo26s|yolo11s)(?![a-z0-9])',
                      str(row.get('model', '')).lower())
    return match.group(1) if match else None


def metric_records(rows, split, source):
    records = {}
    for number, row in enumerate(rows, start=2):
        architecture = model_id(row)
        if architecture is None:
            continue
        if row.get('split') != split:
            raise ValueError(f'{source}, row {number}: expected split {split!r}, '
                             f"got {row.get('split')!r}")
        if architecture in records:
            raise ValueError(f'{source}: duplicate records for {architecture}')
        parsed = {}
        for field in FIELDS:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{source}, {architecture}: missing/invalid {field}') from exc
            if not math.isfinite(value):
                raise ValueError(f'{source}, {architecture}: non-finite {field}')
            if field in SCORES and not 0 <= value <= 1:
                raise ValueError(f'{source}, {architecture}: {field} must be within [0, 1]')
            if field not in SCORES and value <= 0:
                raise ValueError(f'{source}, {architecture}: {field} must be positive')
            parsed[field] = value
        records[architecture] = parsed
    missing = [model for model in MODELS if model not in records]
    if missing:
        raise ValueError(f'{source}: missing model metrics: {", ".join(missing)}')
    return records


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)


def save_figure(fig, output, filename, dpi, source):
    fig.text(0.01, 0.015, f'Source: {source.name}', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = output / filename
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)
    return path


def plot_split(records, split, output, dpi, source):
    title_split = 'Test' if split == 'test' else 'Validation'
    prefix = 'test' if split == 'test' else 'validation'
    files = []

    # One figure for the four detection metrics.
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.19
    metric_colors = ('#2563eb', '#ea580c', '#059669', '#9333ea')
    for index, (field, label) in enumerate(zip(SCORES, SCORE_LABELS)):
        values = [records[model][field] * 100 for model in MODELS]
        positions = [x + (index - 1.5) * width for x in range(len(MODELS))]
        bars = ax.bar(positions, values, width=width, color=metric_colors[index], label=label)
        ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=4, fontsize=9)
    ax.set_xticks(range(len(MODELS)), LABELS)
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric (%)')
    ax.set_ylim(0, 100)
    ax.set_title(f'{title_split}: detection metrics comparison')
    ax.legend(ncols=4, loc='upper center')
    style_axis(ax)
    files.append(save_figure(fig, output, f'{prefix}_detection_metrics.png', dpi, source))

    # Each single-metric comparison gets its own figure and model legend.
    specifications = (
        ('inference_ms', 'Inference time (ms/image)', 'inference time',
         'inference_time', 1, ' ms'),
        ('model_mb', 'Model size (MiB)', 'model size', 'model_size', 1, ' MiB'),
        ('mAP50_95', 'mAP50-95 (%)', 'mAP50-95', 'map50_95', 100, '%'),
    )
    for field, ylabel, title, suffix, scale, unit in specifications:
        fig, ax = plt.subplots(figsize=(9, 6))
        values = [records[model][field] * scale for model in MODELS]
        bars = ax.bar(LABELS, values, color=COLORS, width=0.55)
        ax.bar_label(bars, labels=[f'{v:.2f}{unit}' for v in values], padding=5)
        ax.set_xlabel('Model')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title_split}: {title} comparison')
        ax.set_ylim(0, 100 if scale == 100 else max(values) * 1.3)
        ax.legend(handles=[Patch(facecolor=color, label=label)
                           for color, label in zip(COLORS, LABELS)], loc='upper left')
        style_axis(ax)
        files.append(save_figure(fig, output, f'{prefix}_{suffix}.png', dpi, source))

    fig, ax = plt.subplots(figsize=(10, 7))
    precisions = [records[model]['precision'] * 100 for model in MODELS]
    recalls = [records[model]['recall'] * 100 for model in MODELS]
    # Stagger labels to keep nearby model points legible.
    offsets = ((-150, -75), (20, 65), (25, -60))
    for label, color, precision, recall, offset in zip(
            LABELS, COLORS, precisions, recalls, offsets):
        ax.scatter(precision, recall, color=color, s=100, label=label, zorder=3)
        ax.annotate(f'{label}\nP: {precision:.2f}% | R: {recall:.2f}%',
                    (precision, recall), xytext=offset, textcoords='offset points',
                    fontsize=9, color=color,
                    arrowprops={'arrowstyle': '-', 'color': color, 'alpha': 0.6},
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})
    ax.set_xlabel('Precision (%)')
    ax.set_ylabel('Recall (%)')
    ax.set_title(f'{title_split}: precision vs recall')
    # Fixed percentage axes make test and validation visually comparable.
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')
    style_axis(ax)
    ax.grid(axis='x', alpha=0.2)
    files.append(save_figure(fig, output, f'{prefix}_precision_recall.png', dpi, source))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-dir', type=Path, default=REPORTS,
                        help='Input reports directory; benchmarks are read only')
    parser.add_argument('--output-dir', type=Path, default=REPORTS / 'phase1_charts')
    parser.add_argument('--split', choices=('test', 'val', 'both'), default='both')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 200:
        parser.error('--dpi must be at least 200')

    sources = {
        'status': args.reports_dir / 'phase1_training_status.json',
        'test': args.reports_dir / 'metrics' / 'phase1_test_benchmark.csv',
        'val': args.reports_dir / 'metrics' / 'phase1_validation_benchmark.csv',
        'ranking': args.reports_dir / 'metrics' / 'phase1_validation_ranking.csv',
    }
    missing = []
    loaded = {}
    source_hashes = {}
    for key, path in sources.items():
        if not path.is_file():
            missing.append(str(path))
            print(f'WARNING: missing input file: {path}', file=sys.stderr)
            continue
        source_hashes[str(path)] = digest(path)
        loaded[key] = (json.loads(path.read_text(encoding='utf-8-sig'))
                       if key == 'status' else read_csv(path))
        print(f'Read: {path}')

    splits = ('test', 'val') if args.split == 'both' else (args.split,)
    comparisons = {}
    # Validate all requested inputs before writing any charts. No zero filling,
    # averaging across splits, or metrics extracted by evaluating weights.
    for split in splits:
        if split not in loaded:
            raise ValueError(f'Cannot plot {split}: benchmark file is missing: {sources[split]}')
        comparisons[split] = metric_records(loaded[split], split, sources[split])
    ranking = None
    if 'ranking' in loaded:
        ranked = metric_records(loaded['ranking'], 'val', sources['ranking'])
        ranking = [model_id(row) for row in loaded['ranking'] if model_id(row)]
        if 'val' in comparisons and ranked != comparisons['val']:
            print('WARNING: validation ranking differs from validation benchmark; '
                  'charts use phase1_validation_benchmark.csv.', file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for split, records in comparisons.items():
        files.extend(plot_split(records, split, args.output_dir, args.dpi, sources[split]))
    # Guard the read-only input contract and record exact source provenance.
    for path, expected in source_hashes.items():
        if digest(path) != expected:
            raise ValueError(f'Input changed during plotting: {path}')
    status = loaded.get('status', {})
    write_json(args.output_dir / 'chart_manifest.json', {
        'dpi': args.dpi,
        'source_sha256': source_hashes,
        'missing_input_files': missing,
        'training_state': status.get('state'),
        'selected_model': status.get('selected_model'),
        'validation_ranking': ranking,
        'model_order': list(MODELS),
        'metrics': comparisons,
        'score_display_unit': 'percent (CSV fraction multiplied by 100)',
        'inference_display_unit': 'ms/image (inference only)',
        'size_display_unit': 'MiB (benchmark model_mb = bytes / 1024**2)',
        'png_files': [str(path.resolve()) for path in files],
    })
    print(f'\nCreated {len(files)} PNG files at {args.dpi} dpi:')
    for path in files:
        print(path.resolve())


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, csv.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
