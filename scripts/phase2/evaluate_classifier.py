"""Select the phase-2 classifier on validation, lock it, then evaluate on test once.

Rule (fixed before any test result): per architecture, the configuration with the highest
validation macro-F1; between architectures, the higher validation macro-F1, and if the two are
within 0.005 the lighter EfficientNet-B0. The choice is written to selection.json before the test
split is read. Test metrics: accuracy, balanced accuracy, macro and per-class precision/recall/F1,
confusion matrix (counts and row-normalised), one-vs-rest ROC-AUC, and a paired bootstrap over
source images (1000 resamples) of the macro-F1 difference between the two architectures.
Installs the chosen weights as models/phase2/best_acne_type.pth.
Usage: venv/Scripts/python.exe scripts/phase2/evaluate_classifier.py
"""
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import torch

from common2 import ACNE_CLASSES, MODELS, REPORTS, RUNS, digest, read_json, write_csv, write_json
from train_classifier import ARCH, CONFIGS, CropSet, load_split, macro_f1, predict, transforms

OUT = REPORTS / 'phase2_evaluation'
TIE = 0.005


def load_model(path):
    import timm
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = timm.create_model(ckpt['timm_name'], pretrained=False, num_classes=len(ckpt['classes']))
    model.load_state_dict(ckpt['model'])
    return model.cuda().eval(), ckpt


def metrics(y, prob):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)
    pred = prob.argmax(1)
    labels = list(range(len(ACNE_CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    return dict(accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_precision=float(p.mean()), macro_recall=float(r.mean()), macro_f1=float(f.mean()),
                roc_auc_ovr_macro=float(roc_auc_score(y, prob, multi_class='ovr', average='macro', labels=labels)),
                per_class={c: dict(precision=float(p[k]), recall=float(r[k]), f1=float(f[k]), support=int(s[k]))
                           for k, c in enumerate(ACNE_CLASSES)},
                confusion_matrix=cm.tolist())


def plot_cm(cm, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm = np.array(cm, dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, data, fmt, sub in ((axes[0], cm, '.0f', 'counts'), (axes[1], norm, '.2f', 'row-normalised (recall)')):
        ax.imshow(data, cmap='Blues')
        for i in range(len(ACNE_CLASSES)):
            for j in range(len(ACNE_CLASSES)):
                ax.text(j, i, format(data[i, j], fmt), ha='center', va='center',
                        color='white' if data[i, j] > data.max() * .6 else 'black', fontsize=9)
        ax.set_xticks(range(len(ACNE_CLASSES)), ACNE_CLASSES, rotation=30, ha='right')
        ax.set_yticks(range(len(ACNE_CLASSES)), ACNE_CLASSES)
        ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(f'{title} — {sub}')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for arch in ARCH:
        for cfg in CONFIGS:
            done = RUNS / f'{arch}_{cfg}' / 'completed.json'
            if not done.exists():
                raise RuntimeError(f'{done.parent} not completed; finish all 6 runs first')
            runs.append(dict(arch=arch, config=cfg, **read_json(done)))
    write_csv(OUT / 'validation_runs.csv', runs, list(runs[0]))
    best = {a: max((r for r in runs if r['arch'] == a), key=lambda r: r['val_macro_f1']) for a in ARCH}
    conv, eff = best['convnext_tiny'], best['efficientnet_b0']
    chosen = eff if conv['val_macro_f1'] - eff['val_macro_f1'] < TIE else conv
    selection = dict(rule=f'max val macro-F1 per architecture; between architectures max val macro-F1, '
                          f'EfficientNet-B0 if within {TIE}', representatives=best, chosen=chosen,
                     locked_before_test=True, locked_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT / 'selection.json', selection)

    # ---- test (first and only read of the test split)
    cache, y, rows = load_split('test')
    groups = np.array([int(r['source_id']) for r in rows])
    results, probs = {}, {}
    for arch, rep in best.items():
        path = RUNS / f'{arch}_{rep["config"]}' / 'best.pth'
        model, ckpt = load_model(path)
        loader = torch.utils.data.DataLoader(CropSet(cache, y, transforms(False, ckpt['mean'], ckpt['std'])), 64)
        predict(model, loader)  # warm-up
        torch.cuda.synchronize(); t = time.perf_counter()
        prob = predict(model, loader)
        torch.cuda.synchronize()
        m = metrics(y, prob)
        m.update(config=rep['config'], val_macro_f1=rep['val_macro_f1'], weights_sha256=digest(path),
                 params_m=sum(p.numel() for p in model.parameters()) / 1e6, model_mb=path.stat().st_size / 1024 ** 2,
                 ms_per_crop_batch64=(time.perf_counter() - t) / len(y) * 1000)
        results[arch], probs[arch] = m, prob
        plot_cm(m['confusion_matrix'], f'{arch} test', OUT / f'confusion_matrix_{arch}.png')
        write_csv(OUT / f'test_predictions_{arch}.csv',
                  [dict(crop_path=r['crop_path'], source_id=r['source_id'], true=r['cls'],
                        pred=ACNE_CLASSES[int(p.argmax())], **{f'p_{c}': float(p[k]) for k, c in enumerate(ACNE_CLASSES)})
                   for r, p in zip(rows, prob)],
                  ['crop_path', 'source_id', 'true', 'pred'] + [f'p_{c}' for c in ACNE_CLASSES])
        del model; torch.cuda.empty_cache()

    # ---- paired bootstrap over source images
    rng = np.random.default_rng(42)
    ids = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in ids}
    diffs, f_c, f_e = [], [], []
    for _ in range(1000):
        sel = np.concatenate([idx_of[g] for g in rng.choice(ids, len(ids))])
        a = macro_f1(y[sel], probs['convnext_tiny'][sel].argmax(1))
        b = macro_f1(y[sel], probs['efficientnet_b0'][sel].argmax(1))
        f_c.append(a); f_e.append(b); diffs.append(a - b)
    boot = dict(unit='source image', n_boot=1000, seed=42,
                convnext_macro_f1_ci=np.percentile(f_c, [2.5, 97.5]).tolist(),
                efficientnet_macro_f1_ci=np.percentile(f_e, [2.5, 97.5]).tolist(),
                diff_convnext_minus_efficientnet=float(np.mean(diffs)), diff_ci=np.percentile(diffs, [2.5, 97.5]).tolist())
    write_json(OUT / 'test_metrics.json', dict(selection=selection, test=results, bootstrap=boot))

    # ---- install chosen model
    MODELS.mkdir(parents=True, exist_ok=True)
    src = RUNS / f'{chosen["arch"]}_{chosen["config"]}' / 'best.pth'
    shutil.copy2(src, MODELS / 'best_acne_type.pth')
    write_json(MODELS / 'model_metadata.json', dict(
        status='final', architecture=chosen['arch'], config=chosen['config'], classes=ACNE_CLASSES, input=224,
        weights_sha256=digest(MODELS / 'best_acne_type.pth'), validation_macro_f1=chosen['val_macro_f1'],
        test=results[chosen['arch']], dataset='ACNE-DET Roboflow mirror v1, master split datasets/acnedet_master_split.csv',
        report='reports/phase2_summary.md', disclaimer='Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da.'))
    for arch, m in results.items():
        print(arch, m['config'], 'test macro-F1', round(m['macro_f1'], 4), 'acc', round(m['accuracy'], 4),
              {c: round(v['recall'], 3) for c, v in m['per_class'].items()})
    print('chosen', chosen['arch'], chosen['config'], 'bootstrap', boot)


if __name__ == '__main__':
    main()
