"""Figures for the repository README (docs/figures/). Reads saved reports only; no model runs.

- dataset_stats.png       lesions/image, lesion size at 640 px, images per severity level, per split
- forest_vs_yolo26s.png   paired-bootstrap differences vs YOLO26s on test (95% CI)
- recall_by_density.png   test recall per lesion-density bucket (heatmap)
- copies of the comparison charts and the YOLO26s test PR curve
No face images are exported (privacy + ACNE04 academic-use licence).
Usage: venv/Scripts/python.exe scripts/phase1/make_readme_figures.py
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from common import DATASET, REPORTS, ROOT, read_json
from evaluate_detector_common import ADD

OUT = ROOT / 'docs' / 'figures'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # categorical slots 1-4 (validated)
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SPLITS = ('train', 'val', 'test')


def style(ax, ylabel='', xlabel=''):
    ax.set_facecolor(SURFACE)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xlabel(xlabel, color=MUTED)


def dataset_stats():
    manifest = read_json(DATASET / 'manifest.json')
    rows = []
    for r in manifest['images']:
        s = 640 / max(r['width'], r['height'])
        for line in (DATASET / 'labels' / r['split'] / (Path(r['file_name']).stem + '.txt')).read_text().splitlines():
            _, _, _, w, h = map(float, line.split())
            rows.append(dict(split=r['split'], size=np.sqrt(w * r['width'] * s * h * r['height'] * s)))
    boxes = pd.DataFrame(rows)
    images = pd.DataFrame([dict(split=r['split'], lesions=r['lesions'], level=int(r['file_name'][5]))
                           for r in manifest['images']])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    bins = np.arange(0, 180, 5)
    for i, split in enumerate(SPLITS):
        axes[0].hist(images.loc[images.split == split, 'lesions'].clip(upper=175), bins=bins, density=True,
                     histtype='step', linewidth=2, color=SERIES[i], label=f'{split} (n={sum(images.split == split)})')
    axes[0].set_title('(a) Lesions per image', loc='left', color=INK)
    style(axes[0], 'density', 'lesions per image (>175 clipped)'); axes[0].legend(frameon=False, labelcolor=INK)
    bins = np.arange(0, 40, 1)
    for i, split in enumerate(SPLITS):
        axes[1].hist(boxes.loc[boxes.split == split, 'size'].clip(upper=39), bins=bins, density=True,
                     histtype='step', linewidth=2, color=SERIES[i], label=split)
    med = boxes['size'].median()
    axes[1].axvline(med, color=MUTED, linestyle='--', linewidth=1)
    axes[1].text(med + 0.5, axes[1].get_ylim()[1] * 0.9, f'median {med:.1f} px', color=INK, fontsize=9)
    axes[1].set_title('(b) Lesion size at 640 px input', loc='left', color=INK)
    style(axes[1], 'density', '√(box area) in pixels (>39 clipped)'); axes[1].legend(frameon=False, labelcolor=INK)
    counts = images.groupby(['level', 'split']).size().unstack()[list(SPLITS)]
    x = np.arange(len(counts))
    for i, split in enumerate(SPLITS):
        rects = axes[2].bar(x + (i - 1) * 0.27, counts[split], 0.25, color=SERIES[i], label=split)
        axes[2].bar_label(rects, padding=2, fontsize=8, color=MUTED)
    axes[2].set_xticks(x, [f'level {l}' for l in counts.index], color=INK)
    axes[2].set_title('(c) Images per ACNE04 severity level', loc='left', color=INK)
    style(axes[2], 'images'); axes[2].legend(frameon=False, labelcolor=INK)
    fig.tight_layout(); fig.savefig(OUT / 'dataset_stats.png', dpi=150, facecolor=SURFACE); plt.close(fig)
    return images, boxes


def forest():
    data = read_json(ADD / 'common_eval/bootstrap_test_vs_yolo26s.json')['results']
    labels = {r['model']: r['label'] for _, r in pd.read_csv(REPORTS / 'metrics/phase1_all_models_comparison.csv').iterrows()}
    models = [m for m in ('rtdetr', 'yolo11s_full100', 'yolov5s', 'faster_rcnn_full50', 'yolo26n')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), facecolor=SURFACE, sharey=True)
    for ax, (metric, title) in zip(axes, [('mAP50_95', 'Δ mAP50-95'), ('recall', 'Δ Recall'), ('f1', 'Δ F1')]):
        for y, m in enumerate(models):
            d, lo, hi = (data[m][f'{metric}_diff_vs_ref'], data[m][f'{metric}_diff_ci_low'], data[m][f'{metric}_diff_ci_high'])
            significant = hi < 0 or lo > 0
            ax.plot([lo, hi], [y, y], color=SERIES[0] if significant else MUTED, linewidth=2.5, solid_capstyle='round')
            ax.plot(d, y, 'o', markersize=8, color=SERIES[0] if significant else SURFACE,
                    markeredgecolor=SERIES[0] if significant else MUTED, markeredgewidth=2)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_title(title + ' vs YOLO26s', loc='left', color=INK)
        style(ax, xlabel='difference (95% CI)'); ax.xaxis.grid(True, color=GRID); ax.yaxis.grid(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    axes[0].set_yticks(range(len(models)), [labels[m] for m in models], color=INK)
    axes[0].invert_yaxis()
    fig.text(0.01, 0.01, 'Filled = CI excludes 0 (significant); hollow = not distinguishable from YOLO26s. '
             'Paired bootstrap, 1000 resamples of the 176 test images.', color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / 'forest_vs_yolo26s.png', dpi=150, facecolor=SURFACE); plt.close(fig)


def density():
    comp = pd.read_csv(REPORTS / 'metrics/phase1_all_models_comparison.csv')
    buckets = [(1, 10), (11, 30), (31, 60), (61, 1000)]
    names = ['1–10', '11–30', '31–60', '> 60']
    grid, n_images = [], None
    for _, r in comp.iterrows():
        d = pd.read_csv(ADD / 'common_eval' / r['model'] / 'eval_test/per_image_counts.csv')
        grid.append([d.loc[d['gt'].between(a, b), 'tp'].sum() / d.loc[d['gt'].between(a, b), 'gt'].sum() for a, b in buckets])
        n_images = [int(d['gt'].between(a, b).sum()) for a, b in buckets]
    grid = np.array(grid)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.imshow(grid, cmap='Blues', vmin=0.25, vmax=0.55, aspect='auto')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f'{grid[i, j]:.3f}', ha='center', va='center', fontsize=10,
                    color='white' if grid[i, j] > 0.45 else INK, fontweight='bold' if grid[i, j] == grid[:, j].max() else 'normal')
    ax.set_xticks(range(len(names)), [f'{n}\n({k} images)' for n, k in zip(names, n_images)], color=INK)
    ax.set_yticks(range(len(comp)), comp['label'], color=INK)
    ax.set_xlabel('ground-truth lesions per test image', color=MUTED)
    ax.set_title('Test recall by lesion density (bold = best per column)', loc='left', color=INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / 'recall_by_density.png', dpi=150, facecolor=SURFACE); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images, boxes = dataset_stats()
    forest()
    density()
    for chart in (ADD / 'charts').glob('*.png'):
        shutil.copy2(chart, OUT / chart.name)
    shutil.copy2(ADD / 'common_eval/yolo26s/eval_test/PR_curve.png', OUT / 'yolo26s_test_PR_curve.png')
    shutil.copy2(ADD / 'common_eval/yolo26s/eval_test/F1_curve.png', OUT / 'yolo26s_test_F1_curve.png')
    summary = dict(images=len(images), lesions=len(boxes), lesion_size_median=float(boxes['size'].median()),
                   per_level=images.groupby(['level', 'split']).size().unstack().to_dict())
    (OUT / 'dataset_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
