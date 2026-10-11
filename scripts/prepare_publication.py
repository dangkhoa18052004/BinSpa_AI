"""Export aggregate research artifacts, without faces or per-image records.

Requires the private, frozen evaluation reports. Does not train, infer, or select
models. Usage: python scripts/prepare_publication.py
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'reports/publication'
FIGURES = ROOT / 'docs/figures/research'
MATRICES = FIGURES / 'confusion_matrices'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def export_metrics():
    scope = read('project_scope.json')
    assert scope['active_stages'] == ['phase1', 'phase2'] and not scope['scar_detection_enabled']
    matrices = read('reports/confusion_matrices/matrices.json')['matrices']
    assert len(matrices) == 18 and all(not item['key'].startswith('phase3_') for item in matrices)
    phase1 = []
    for item in matrices:
        if not item['key'].startswith('phase1_'):
            continue
        source = ROOT / item['source']
        data = read((source.parent.parent / 'common_eval_summary.json').relative_to(ROOT))
        op = data['test']['operating_point']
        phase1.append(dict(key=item['key'], model=item['model'], imgsz=data['imgsz'],
            confidence=op['conf'], test_images=data['test']['images'], gt_boxes=data['test']['gt_boxes'],
            mAP50=data['test']['mAP50'], mAP50_95=data['test']['mAP50_95'],
            precision=op['precision'], recall=op['recall'], f1=op['f1'],
            tp=op['tp'], fp=op['fp'], fn=op['fn'], checkpoint_sha256=item['checkpoint_sha256']))
    p2 = read('reports/phase2_evaluation/test_metrics.json')
    phase2 = {key: {field: data[field] for field in ['accuracy', 'balanced_accuracy', 'macro_precision',
        'macro_recall', 'macro_f1', 'roc_auc_ovr_macro', 'per_class', 'confusion_matrix', 'config',
        'val_macro_f1', 'weights_sha256', 'params_m', 'model_mb', 'ms_per_crop_batch64']}
        for key, data in p2['test'].items()}
    external = read('reports/detectdataskin2_audit/frozen_evaluation/metrics.json')
    result = dict(status='Exploratory engineering study; not a clinically validated system',
        datasets=dict(acne04=dict(images=1167, split_images=dict(train=816,val=175,test=176),
            test_lesions=5046, patient_ids_verified=False),
            acnedet=dict(images=275, split_images=dict(train=190,val=40,test=45),
                inferred_groups=50, patient_ids_verified=False, test_crops=2537),
            detectdataskin2=dict(source_images=3462, screened_images=1275, acne_labels=18974,
                supported_type_crops=18846, labels_expert_verified=False, patient_ids_verified=False)),
        phase1=phase1, phase2=phase2, phase2_bootstrap=p2['bootstrap'],
        external=dict(phase1=external['phase1_binary_detection'],
            phase2=external['phase2_gt_crop_classifier'], pipeline=external['phase1_plus_phase2_pipeline'],
            confidence_diagnostics=read('reports/detectdataskin2_audit/frozen_evaluation/classifier_confidence_diagnostics.json')))
    write(PUBLIC / 'metrics.json', result)
    safe_matrices = []
    for item in matrices:
        safe = {k: item[k] for k in ['key','phase','model','dataset','matrix_true_by_predicted','labels',
            'display_labels','detection','images','ground_truth_labels','support','predictions',
            'checkpoint_sha256','matching']}
        safe.update({k:item[k] for k in ['confidence','iou','accuracy','macro_f1'] if k in item})
        source = ROOT / 'reports/confusion_matrices' / item['figure']
        target = MATRICES / item['figure']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        safe['figure'] = target.relative_to(ROOT).as_posix()
        safe_matrices.append(safe)
    write(PUBLIC / 'confusion_matrices.json', dict(orientation='rows=true, columns=predicted',
        normalization='true-label row percentages; detector background row undefined', matrices=safe_matrices))
    for name in ['all_models_confusion_matrices.pdf','deployed_original_test.png','phase1_640_comparison.png',
                 'phase1_1280_comparison.png','phase2_comparison.png',
                 'detectdataskin2_exploratory.png']:
        shutil.copyfile(ROOT / 'reports/confusion_matrices' / name, MATRICES / name)
    shutil.copyfile(ROOT / 'reports/detectdataskin2_audit/frozen_evaluation/classifier_confidence_diagnostics.png',
                    FIGURES / 'external_confidence_diagnostics.png')
    return result


def charts(data):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for col, size in enumerate([640, 1280]):
        rows = [r for r in data['phase1'] if r['imgsz']==size]
        names = [r['model'].replace(f' @{size}', '') for r in rows]
        for ax, metric, title in [(axes[0,col],'mAP50_95','mAP50–95'),(axes[1,col],'recall','Recall @ IoU 0,5')]:
            bars = ax.barh(names, [r[metric] for r in rows], color=['#167d8d' if r['key']=='phase1_rtdetr_ft' else '#3274a1' for r in rows])
            ax.invert_yaxis()
            ax.set_xlim(0, .24 if metric=='mAP50_95' else .65)
            ax.set_title(f'{size}px — {title}')
            ax.set_xlabel('Giá trị (0–1)')
            ax.bar_label(bars, fmt='%.3f', padding=4)
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('GĐ1 — 14 checkpoint trên 176 ảnh test ACNE04\nRecall dùng ngưỡng validation riêng của mỗi model', fontsize=16)
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(FIGURES/'phase1_benchmark.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,5.5))
    keys=['closed_comedo','open_comedo','papule','pustule','nodule']
    x=np.arange(5)
    for offset,key,label,color in [(-.18,'efficientnet_b0','EfficientNet-B0','#3274a1'),(.18,'convnext_tiny','ConvNeXt-Tiny','#167d8d')]:
        vals=[data['phase2'][key]['per_class'][c]['f1'] for c in keys]
        bars=ax.bar(x+offset,vals,width=.36,label=label,color=color)
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    ax.set_xticks(x,['Đầu trắng\nn=971','Đầu đen\nn=473','Sẩn\nn=867','Mủ\nn=192','Nốt\nn=34'])
    ax.set_ylim(0,1.05)
    ax.set_ylabel('F1 theo lớp')
    ax.set_title('GĐ2 — crop theo box nhãn, cùng 2.537 crop test ACNEDET')
    ax.legend(loc='lower left')
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES/'phase2_class_f1.png',dpi=180)
    plt.close(fig)


def readme_tables(data):
    text = (ROOT/'README.md').read_text(encoding='utf-8')
    p1 = ['| Model | mAP50 | mAP50–95 | Precision | Recall | F1 |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in data['phase1']:
        name = row['model'] + (' **(đang dùng)**' if row['key']=='phase1_rtdetr_ft' else '')
        p1.append('| '+name+' | '+' | '.join(f'{row[k]:.4f}' for k in ['mAP50','mAP50_95','precision','recall','f1'])+' |')
    p2 = ['| Model | Accuracy | Macro-F1 [CI 95%] | Balanced accuracy |',
          '| --- | ---: | --- | ---: |']
    for key, name, ci_key in [('efficientnet_b0','EfficientNet-B0 **(đang dùng)**','efficientnet_macro_f1_ci'),
                             ('convnext_tiny','ConvNeXt-Tiny','convnext_macro_f1_ci')]:
        row=data['phase2'][key]
        lo,hi=data['phase2_bootstrap'][ci_key]
        p2.append(f'| {name} | {row["accuracy"]:.2%} | {row["macro_f1"]:.4f} [{lo:.4f}; {hi:.4f}] | {row["balanced_accuracy"]:.4f} |')
    for number,rows in enumerate([p1,p2],1):
        start=f'<!-- PHASE{number}_TABLE -->'
        end=f'<!-- /PHASE{number}_TABLE -->'
        replacement=start+'\n\n'+'\n'.join(rows)+'\n\n'+end
        if end in text:
            text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:replacement,text,flags=re.S)
        else:
            assert start in text
            text=text.replace(start,replacement)
    (ROOT/'README.md').write_text(text,encoding='utf-8',newline='\n')


def main():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    FIGURES.mkdir(parents=True,exist_ok=True)
    data=export_metrics()
    charts(data)
    readme_tables(data)
    rows=['# Kết quả tổng hợp công khai','',
        'Các file ở đây chỉ chứa thống kê tổng hợp, checkpoint hash và ma trận. Ảnh khuôn mặt, ID ảnh/người, tọa độ box, dự đoán từng ảnh, tài liệu cá nhân và đường dẫn máy cá nhân không được xuất.','',
        '- [Báo cáo chính và các hình hiển thị trực tiếp](../../README.md)',
        '- [Số liệu các thực nghiệm](metrics.json)',
        '- [Ma trận dưới dạng số nguyên](confusion_matrices.json)',
        '- [PDF 18 trang](../../docs/figures/research/confusion_matrices/all_models_confusion_matrices.pdf)','',
        '## Ma trận từng checkpoint','',
        '| Giai đoạn | Model | Tập dữ liệu | Hình |','| --- | --- | --- | --- |']
    for item in read('reports/publication/confusion_matrices.json')['matrices']:
        rows.append(f'| {item["phase"]} | {item["model"]} | {item["dataset"]} | [Mở](../../{item["figure"]}) |')
    rows += ['', '## Tái lập và phạm vi', '',
        '`python scripts/prepare_publication.py` xuất lại các artifact này từ báo cáo đánh giá lưu riêng. Script không chạy model hay chọn lại ngưỡng. Muốn chấm lại từ đầu cần dữ liệu được cấp quyền, manifest chia tập và checkpoint tương ứng; các đầu vào đó không nằm trong bản công khai.', '',
        'Dữ liệu DetectDataSkin-2 là thăm dò với mapping nhãn tạm thời. Không diễn giải kết quả này như kiểm định lâm sàng hoặc kiểm định độc lập theo bệnh nhân.', '']
    (PUBLIC/'README.md').write_text('\n'.join(rows),encoding='utf-8',newline='\n')
    artifacts=sorted([p for p in FIGURES.rglob('*') if p.is_file() and 'phase3' not in p.parts and 'phase3' not in p.name] +
                     [PUBLIC/'metrics.json',PUBLIC/'confusion_matrices.json',PUBLIC/'README.md'])
    write(PUBLIC/'artifact_manifest.json',dict(privacy='Aggregate figures and metrics only; no per-image records',
        artifacts=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in artifacts]))
    print(f'Exported {len(artifacts)} aggregate publication artifacts')


if __name__=='__main__':
    main()
