"""Charts for the 6-model test comparison (reads reports/metrics/phase1_all_models_comparison.csv).

Model identity is carried by axis/direct labels; colour encodes the metric only
(fixed categorical order, validated palette). Output: reports/phase1_additional_models/charts/
Usage: venv/Scripts/python.exe scripts/phase1/plot_all_models_comparison.py
"""
import csv

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import REPORTS
from evaluate_detector_common import ADD

OUT = ADD / 'charts'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # categorical slots 1-4, fixed order
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=MUTED)


def figure(title, width=10):
    fig, ax = plt.subplots(figsize=(width, 5), facecolor=SURFACE)
    ax.set_title(title, loc='left', color=INK, fontsize=13, pad=14)
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def bars(ax, labels, groups, names, fmt):
    x = np.arange(len(labels))
    width = 0.8 / len(groups)
    for i, (values, name) in enumerate(zip(groups, names)):
        rects = ax.bar(x + (i - (len(groups) - 1) / 2) * width, values, width * 0.9, color=SERIES[i], label=name)
        if len(groups) <= 2 or name in ('mAP50-95',):
            ax.bar_label(rects, [fmt(v) for v in values], padding=2, fontsize=8, color=MUTED)
    ax.set_xticks(x, labels, color=INK)


def main():
    with (REPORTS / 'metrics/phase1_all_models_comparison.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    OUT.mkdir(parents=True, exist_ok=True)
    labels = [r['label'].replace(' R50-FPN', '\nR50-FPN') for r in rows]
    num = lambda k: [float(r[k]) for r in rows]
    pct = lambda v: f'{v:.3f}'

    fig, ax = figure('Test quality (176 images) — P/R at the validation-chosen threshold, IoU 0.5', 12)
    bars(ax, labels, [num('precision'), num('recall'), num('mAP50'), num('mAP50_95')],
         ['Precision', 'Recall', 'mAP50', 'mAP50-95'], pct)
    style(ax, 'score'); ax.set_ylim(0, 1); ax.legend(frameon=False, ncol=4, loc='upper right', labelcolor=INK)
    save(fig, '1_precision_recall_map.png')

    order = np.argsort(num('mAP50_95'))[::-1]
    fig, ax = figure('Test mAP50-95 with 95% bootstrap CI (1000 resamples of 176 images)')
    low, high = num('mAP50_95_ci_low'), num('mAP50_95_ci_high')
    values = [num('mAP50_95')[i] for i in order]
    err = [[values[j] - low[i] for j, i in enumerate(order)], [high[i] - values[j] for j, i in enumerate(order)]]
    rects = ax.bar([labels[i] for i in order], values, 0.55, color=SERIES[0], yerr=err, capsize=5,
                   error_kw=dict(ecolor=MUTED, elinewidth=1.2))
    ax.bar_label(rects, [f'{v:.4f}' for v in values], padding=3, color=INK, label_type='center', fontsize=9)
    style(ax, 'mAP50-95'); ax.tick_params(axis='x', colors=INK)
    save(fig, '2_map50_95.png')

    # Dedicated benchmark (benchmark_latency.py): 30 test photos x 5 interleaved rounds, Best-performance power mode.
    fig, ax = figure('End-to-end latency per photo (median), batch 1, FP32, RTX 5050 Laptop')
    bottom = np.zeros(len(rows))
    for i, (k, name) in enumerate([('e2e_preprocess_ms', 'preprocess'), ('e2e_inference_ms', 'inference'), ('e2e_postprocess_ms', 'postprocess')]):
        ax.bar(labels, num(k), 0.55, bottom=bottom, color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=2)
        bottom += num(k)
    total, p10, p90 = np.array(num('e2e_total_ms_median')), np.array(num('e2e_total_ms_p10')), np.array(num('e2e_total_ms_p90'))
    ax.errorbar(range(len(rows)), total, yerr=[total - p10, p90 - total], fmt='none', ecolor=INK, elinewidth=1.2, capsize=5)
    for x, (t, g) in enumerate(zip(total, num('gflops'))):
        ax.text(x + 0.3, t, f'{t:.1f} ms\n{g:.0f} GFLOPs', ha='left', va='center', color=INK, fontsize=8)
    ax.text(0, -0.2, 'Photo array -> boxes (decode excluded); 30 test photos x 5 interleaved rounds; whiskers = p10-p90.',
            transform=ax.transAxes, color=MUTED, fontsize=8)
    style(ax, 'ms / photo'); ax.tick_params(axis='x', colors=INK); ax.legend(frameon=False, labelcolor=INK)
    save(fig, '3_inference_time.png')

    fig, ax = figure('Checkpoint size on disk (FP16 weights; lower is lighter)')
    rects = ax.bar(labels, num('model_mb'), 0.55, color=SERIES[0])
    ax.bar_label(rects, [f'{v:.1f} MB\n{p:.1f}M params' for v, p in zip(num('model_mb'), num('params_m'))], padding=3, color=INK, fontsize=9)
    style(ax, 'MB'); ax.tick_params(axis='x', colors=INK); ax.set_ylim(0, max(num('model_mb')) * 1.25)
    save(fig, '4_model_size.png')

    fig, ax = figure('Precision vs Recall on test (validation-chosen threshold)', 8)
    ax.scatter(num('recall'), num('precision'), s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    for r, p, name in zip(num('recall'), num('precision'), [r['label'] for r in rows]):
        ax.annotate(name, (r, p), xytext=(6, 4), textcoords='offset points', color=INK, fontsize=9)
    style(ax, 'Precision'); ax.set_xlabel('Recall', color=MUTED); ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    lo = min(num('recall') + num('precision')) - 0.05
    hi = max(num('recall') + num('precision')) + 0.05
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    save(fig, '5_precision_vs_recall.png')

    fig, ax = figure(f'TP / FP / FN on test ({rows[0]["match_iou"]} IoU, validation-chosen threshold; {int(num("tp")[0] + num("fn")[0])} GT lesions)', 12)
    bars(ax, labels, [num('tp'), num('fp'), num('fn')], ['TP', 'FP', 'FN'], lambda v: f'{int(v)}')
    for container in ax.containers:
        ax.bar_label(container, [f'{int(v)}' for v in container.datavalues], padding=2, fontsize=8, color=MUTED)
    style(ax, 'boxes'); ax.set_ylim(0, max(num('tp') + num('fp') + num('fn')) * 1.12)
    ax.legend(frameon=False, ncol=3, labelcolor=INK, loc='lower right', bbox_to_anchor=(1, 1.0))
    save(fig, '6_tp_fp_fn.png')
    print('charts ->', OUT)


if __name__ == '__main__':
    main()
