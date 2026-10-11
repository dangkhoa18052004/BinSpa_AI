"""Render reproducible confusion matrices for all available test checkpoints.

Rows are true labels, columns predicted labels. Detection background/background
is undefined and masked; the background row is not normalized as a false-positive
rate. Only acne detection and acne-type classification are active tasks.
No training, threshold selection, or checkpoint promotion occurs.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/confusion_matrices'
CLASSES = ['closed_comedo', 'open_comedo', 'papule', 'pustule', 'nodule']
DISPLAY = ['Đầu trắng', 'Đầu đen', 'Sẩn', 'Mủ', 'Nốt']
MODEL_NAMES = {
    'faster_rcnn': 'Faster R-CNN @640', 'faster_rcnn_full50': 'Faster R-CNN @640, full50',
    'rtdetr': 'RT-DETR-L @640', 'rtdetr_full100': 'RT-DETR-L @640, full100',
    'yolo11s': 'YOLO11s @640', 'yolo11s_full100': 'YOLO11s @640, full100',
    'yolo26n': 'YOLO26n @640', 'yolo26s': 'YOLO26s @640', 'yolov5s': 'YOLOv5su @640',
    'rtdetr_1280': 'RT-DETR-L @1280', 'yolo11s_1280': 'YOLO11s @1280',
    'yolo26s_1280': 'YOLO26s @1280', 'rtdetr_ft': 'RT-DETR-L @1280, fine-tune',
    'yolo26s_ft': 'YOLO26s @1280, fine-tune',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def entry(key, phase, model, dataset, matrix, labels, display, detection, images, source, **extra):
    matrix = np.asarray(matrix, dtype=np.int64)
    assert matrix.shape == (len(labels), len(labels)) and (matrix >= 0).all()
    if detection:
        assert matrix[-1, -1] == 0
        support = matrix[:-1].sum(1)
        predictions = int(matrix[:, :-1].sum())
    else:
        support = matrix.sum(1)
        predictions = int(matrix.sum())
    return dict(key=key, phase=phase, model=model, dataset=dataset, matrix_true_by_predicted=matrix.tolist(),
                labels=labels, display_labels=display, detection=detection, images=images,
                ground_truth_labels=int(support.sum()), support=support.tolist(), predictions=predictions,
                source=str(source.relative_to(ROOT)), source_sha256=sha(source), **extra)


def acne_matrices():
    paths = sorted((ROOT/'reports/phase1_additional_models/common_eval').glob('*/common_eval_summary.json'))
    paths += sorted((ROOT/'reports/phase1_additional_models').glob('*_1280/common_eval_summary.json'))
    paths += sorted((ROOT/'reports/phase1_combined').glob('*/common_eval_summary.json'))
    result = []
    for path in paths:
        summary = read(path)
        source = path.parent/'eval_test/confusion_matrix.json'
        data = read(source)
        matrix = np.asarray(data['matrix_pred_by_true'], dtype=int).T
        op = summary['test']['operating_point']
        assert data['conf'] == summary['val']['operating_point']['conf'] == op['conf']
        assert matrix[0,0] == op['tp'] and matrix[0,1] == op['fn'] and matrix[1,0] == op['fp']
        assert matrix[0].sum() == summary['test']['gt_boxes'] == 5046
        result.append(entry('phase1_'+summary['model'], 'GĐ1', MODEL_NAMES[summary['model']],
                    'ACNE04 — test', matrix, ['acne','background'], ['Mụn','Nền'], True,
                    summary['test']['images'], source, confidence=data['conf'], iou=data['match_iou'],
                    checkpoint_sha256=summary['weights_sha256'],
                    matching='Existing project binary IoU matching; original unrounded aggregate counts'))
    assert len(result) == 14
    return result


def classification_matrix(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    y = np.array([CLASSES.index(r['true']) for r in rows])
    pred = np.array([CLASSES.index(r['pred']) for r in rows])
    probabilities = np.array([[float(r['p_'+c]) for c in CLASSES] for r in rows])
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1, atol=1e-6)
    assert np.array_equal(probabilities.argmax(1), pred)
    return np.bincount(y*5+pred, minlength=25).reshape(5,5)


def classifier_matrices():
    metrics = read(ROOT/'reports/phase2_evaluation/test_metrics.json')
    result = []
    for key, name in [('efficientnet_b0','EfficientNet-B0'),('convnext_tiny','ConvNeXt-Tiny')]:
        source = ROOT/f'reports/phase2_evaluation/test_predictions_{key}.csv'
        matrix = classification_matrix(source)
        assert matrix.tolist() == metrics['test'][key]['confusion_matrix'] and matrix.sum() == 2537
        assert abs(np.trace(matrix)/matrix.sum() - metrics['test'][key]['accuracy']) < 1e-12
        result.append(entry('phase2_'+key, 'GĐ2', name, 'ACNEDET — crop test', matrix, CLASSES, DISPLAY,
                    False, 45, source, checkpoint_sha256=metrics['test'][key]['weights_sha256'],
                    accuracy=metrics['test'][key]['accuracy'], macro_f1=metrics['test'][key]['macro_f1'],
                    matching='Ground-truth crops; top-1 argmax, not full-face pipeline accuracy'))
    return result




def external_matrices():
    source = ROOT/'reports/detectdataskin2_audit/frozen_evaluation/metrics.json'
    metrics = read(source)
    binary = metrics['phase1_binary_detection']
    op = binary['operating_point']
    result = [entry('external_phase1_rtdetr', 'GĐ1', 'RT-DETR-L @1280, fine-tune',
              'DetectDataSkin-2 — thăm dò', [[op['tp'],op['fn']],[op['fp'],0]], ['acne','background'],
              ['Mụn','Nền'], True, binary['images'], source, confidence=op['conf'], iou=op['iou'],
              checkpoint_sha256=metrics['protocol']['detector_sha256'],
              matching='Existing frozen binary detector evaluation; unreviewed source labels')]
    path = ROOT/'reports/detectdataskin2_audit/frozen_evaluation/classifier_gt_crop_predictions.csv'
    matrix = classification_matrix(path)
    classifier = metrics['phase2_gt_crop_classifier']
    assert matrix.tolist() == classifier['confusion_matrix'] and matrix.sum()==18846
    result.append(entry('external_phase2_efficientnet', 'GĐ2', 'EfficientNet-B0',
                  'DetectDataSkin-2 — crop thăm dò', matrix, CLASSES, DISPLAY, False, binary['images'], path,
                  checkpoint_sha256=metrics['protocol']['classifier_sha256'], accuracy=classifier['accuracy'],
                  macro_f1=classifier['macro_f1'], matching='GT crops; provisional label mapping; labels unreviewed'))
    return result


def values(item, normalized):
    matrix = np.array(item['matrix_true_by_predicted'], dtype=float)
    mask = np.zeros(matrix.shape, bool)
    if item['detection']:
        mask[-1,-1] = True
        if normalized:
            mask[-1,:] = True  # no verified number of true background instances
    if normalized:
        matrix = np.divide(matrix, matrix.sum(1,keepdims=True), out=np.zeros_like(matrix),
                           where=matrix.sum(1,keepdims=True)!=0)
        zero_support = np.array(item['matrix_true_by_predicted']).sum(1)==0
        mask[zero_support,:] = True
    return np.ma.array(matrix, mask=mask)


def panel(ax, item, normalized, compact=False):
    data = values(item, normalized)
    color = plt.colormaps['Blues'].copy()
    color.set_bad('#e9e9e9')
    maximum = 1 if normalized else max(float(data.max()),1)
    ax.imshow(data,cmap=color,vmin=0,vmax=maximum,interpolation='nearest')
    n = len(item['labels'])
    font = 9 if compact else 11
    ax.set_xticks(range(n),item['display_labels'],rotation=25 if n>2 else 0,ha='right' if n>2 else 'center',fontsize=font)
    ax.set_yticks(range(n),item['display_labels'],fontsize=font)
    ax.set_xlabel('Dự đoán',fontsize=font+1)
    ax.set_ylabel('Nhãn thật',fontsize=font+1)
    for row in range(n):
        for col in range(n):
            if np.ma.getmaskarray(data)[row,col]:
                text='N/A'; text_color='#666666'
            else:
                v=float(data[row,col])
                text=(f'{100*v:.1f}%'.replace('.',',')) if normalized else f'{int(v):,}'.replace(',','.')
                text_color='white' if v>maximum*.55 else '#102a43'
            ax.text(col,row,text,ha='center',va='center',fontsize=font if compact else 13,color=text_color)
    extra = f' | conf={item["confidence"]:.5f}, IoU={item["iou"]:.2f}' if item['detection'] else ''
    title = ('Tỷ lệ theo nhãn thật' if normalized else 'Số lượng') + extra
    if compact:
        title = item['model']+'\n'+title
    ax.set_title(title,fontsize=font+1,pad=10)
    ax.spines[['top','right','bottom','left']].set_visible(False)


def pair(item):
    fig, axes = plt.subplots(1,2,figsize=(12.8,5.8))
    for ax,normalized in zip(axes,[False,True]):
        panel(ax,item,normalized)
    fig.suptitle(f'{item["phase"]} — {item["model"]}\n{item["dataset"]} | {item["images"]} ảnh, {item["ground_truth_labels"]:,} nhãn',fontsize=15)
    foot = 'Detector: Nền ở cột = bỏ sót; Nền ở hàng = báo dư. N/A không phải 0 âm tính đúng.' if item['detection'] else 'Mỗi hàng tỷ lệ cộng thành 100%. Crop theo box nhãn; không phải accuracy của pipeline trên ảnh mặt.'
    fig.text(.5,.015,foot,ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.90))
    return fig


def overview(items, name, title):
    rows = len(items)
    fig, axes = plt.subplots(rows,2,figsize=(13,rows*4.5),squeeze=False)
    for row,item in enumerate(items):
        for col,normalized in enumerate([False,True]):
            panel(axes[row,col],item,normalized,compact=True)
        axes[row,0].set_title(item['model']+'\n'+item['dataset']+' — số lượng',fontsize=11,pad=10)
    fig.suptitle(title,fontsize=17,y=.995)
    fig.tight_layout(rect=(0,.015,1,.97))
    fig.savefig(OUT/name,dpi=180)
    plt.close(fig)


def render(items):
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    with PdfPages(OUT/'all_models_confusion_matrices.pdf') as pdf:
        for item in items:
            print('Plot',item['key'],flush=True)
            fig=pair(item)
            relative=Path(item['phase'].replace('GĐ','phase'))/f'{item["key"]}.png'
            target=OUT/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(target,dpi=220)
            pdf.savefig(fig)
            item['figure']=relative.as_posix()
            plt.close(fig)
    original=[r for r in items if r['key'] in ['phase1_rtdetr_ft','phase2_efficientnet_b0']]
    overview(original,'deployed_original_test.png','Hai model đang triển khai — các tập test gốc')
    overview([r for r in items if r['key'].startswith('phase2_')], 'phase2_comparison.png','GĐ2 — hai classifier, cùng 2.537 crop test ACNEDET')
    overview([r for r in items if r['key'].startswith('external_')], 'detectdataskin2_exploratory.png','DetectDataSkin-2 — nhãn chưa xác minh, đánh giá thăm dò')
    for size in ['640','1280']:
        subset=[r for r in items if r['phase']=='GĐ1' and r['key'].startswith('phase1_') and '@'+size in r['model']]
        cols=3
        rows=(len(subset)+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(15,rows*4.3),squeeze=False)
        for ax,item in zip(axes.flat,subset):
            panel(ax,item,False,compact=True)
        for ax in list(axes.flat)[len(subset):]:
            ax.axis('off')
        fig.suptitle(f'GĐ1 @{size} — ACNE04 test; mỗi model dùng ngưỡng validation riêng',fontsize=15)
        fig.tight_layout(rect=(0,.025,1,.95))
        fig.savefig(OUT/f'phase1_{size}_comparison.png',dpi=180)
        plt.close(fig)


def documentation(items):
    lines=['# Ma trận nhầm lẫn các model BinSpa_AI','',
        'Đã vẽ 18 ma trận: 16 checkpoint trên tập test gốc, cùng hai ma trận của checkpoint đang triển khai trên DetectDataSkin-2. Mỗi ảnh có hai phần: số lượng và tỷ lệ theo hàng nhãn thật. Không train, đổi ngưỡng mặc định hoặc lựa chọn lại model.','',
        '## Mở nhanh','',
        '- [PDF toàn bộ 18 ma trận](all_models_confusion_matrices.pdf)',
        '- [Hai model đang triển khai trên test gốc](deployed_original_test.png)',
        '- [GĐ1 @640](phase1_640_comparison.png)',
        '- [GĐ1 @1280 và fine-tune](phase1_1280_comparison.png)',
        '- [Hai classifier GĐ2](phase2_comparison.png)',
        '- [Kết quả thăm dò DetectDataSkin-2](detectdataskin2_exploratory.png)','',
        '## Cách đọc','',
        '- Hàng = nhãn thật; cột = dự đoán. Đường chéo = đúng lớp, ngoài đường chéo = nhầm lớp.',
        '- GĐ2: phần trăm là tỷ lệ trong từng lớp nhãn thật, tương ứng recall trên đường chéo. Đầu trắng/đầu đen/sẩn/mủ/nốt lần lượt là closed_comedo/open_comedo/papule/pustule/nodule.',
        '- Detector: cột Nền là tổn thương bị bỏ sót; hàng Nền là dự đoán không ghép được với tổn thương có nhãn. Ô Nền–Nền không xác định vì không đếm được mọi vùng nền âm tính đúng; ghi N/A.',
        '- Phần tỷ lệ của detector chỉ chuẩn hóa các hàng có nhãn tổn thương. Toàn bộ hàng Nền ghi N/A vì không có mẫu số nền đã xác minh; không suy ra specificity hoặc false-positive rate từ hàng này.',
        '- Mỗi detector dùng ngưỡng đã chọn từ validation riêng, IoU ghép 0,5; đây không phải so sánh ở cùng một confidence. Confidence của model mụn đang triển khai vẫn là 0,48048.',
        '- GĐ1: chuyển vị ma trận số nguyên đã lưu, đối chiếu TP/FP/FN với artifact tương ứng; không dùng metadata test cũ của checkpoint fine-tune hoặc đếm số từ ảnh PNG.',
        '- GĐ2: tính lại từ CSV dự đoán top-1 và kiểm tra khớp ma trận, accuracy đã công bố.',
        '- Hai ma trận DetectDataSkin-2 là thăm dò trên nhãn nguồn chưa được chuyên gia xác minh và chưa xác nhận tách người. GĐ2 chỉ gồm 18.846 crop có mapping hỗ trợ, không gồm Cysts.',
        '- Không dùng test gốc hoặc bộ thăm dò đã xem để khẳng định chất lượng trên người mới. Các tập dữ liệu và support khác nhau không so trực tiếp như cùng một benchmark.','',
        '## Ma trận từng model','',
        '| Model | Bộ dữ liệu | Số nhãn thật | Hình |',
        '| --- | --- | ---: | --- |']
    for item in items:
        lines.append(f'| {item["phase"]} {item["model"]} | {item["dataset"]} | {item["ground_truth_labels"]:,} | [Mở]({item["figure"]}) |')
    lines+=['','## Dữ liệu và tái lập','',
            '`matrices.json` chứa số nguyên, nhãn, trục, nguồn/hash, confidence và phương pháp ghép. `verification.json` ghi kiểm tra tổng nhãn và tổng dự đoán.','',
            '```powershell','.\\venv\\Scripts\\python.exe -X utf8 scripts\\plot_model_confusion_matrices.py','```']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'matrices.json', dict(orientation='rows=true, columns=predicted',
          normalization='row percentage; detection background row masked', matrices=items))
    write(OUT/'verification.json',dict(passed=True,matrices=len(items), phase1_matrices=14,phase2_matrices=2,
          external_matrices=2,checks=['source aggregate counts match operating point',
          'classifier CSV argmax, confusion matrix and accuracy match',
          'undefined background masked'],
          matrices_sha256=sha(OUT/'matrices.json')))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    items=acne_matrices()+classifier_matrices()+external_matrices()
    assert len(items)==18 and len({r['key'] for r in items})==18
    render(items)
    documentation(items)
    print(f'Saved {len(items)} paired confusion matrices, overview figures, PDF and verified data to',OUT,flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
