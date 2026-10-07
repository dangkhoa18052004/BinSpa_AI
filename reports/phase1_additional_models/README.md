# Giai đoạn 1 — Train & benchmark thêm YOLOv5s, Faster R-CNN, RT-DETR

Tài liệu này ghi lại **cách** 3 detector mới được train và đánh giá, để so sánh công bằng với YOLO26n/YOLO26s/YOLO11s đã có. Kết quả so sánh và kết luận nằm ở [final_comparison.md](final_comparison.md).

## 1. Dataset và split (không thay đổi)

| Mục | Giá trị |
|---|---|
| Dataset | `datasets/phase1_acne_detection` (`acne04v2-clean-v1`), 1 class `acne` |
| Split | train **816** / val **175** / test **176** ảnh (seed 42, tạo trước đây; không chia lại) |
| Lesion | train 21.193 / val 5.259 / test 5.046 |
| Kích thước lesion | trung vị ~7,8 px khi cạnh dài ảnh = 640 (p5 3,7 px; p95 18,7 px) |
| Ảnh dày đặc nhất | `levle3_89.jpg` (test) có **434** lesion |

- Mọi script gọi `verify_dataset()` (so SHA256 ảnh + nhãn với `manifest.json`) trước khi train/eval; không có file nào của dataset bị sửa.
- `data.yaml` trong dataset vẫn trỏ tới `D:/BinSpa_AI` (đường dẫn cũ). Không sửa; dùng bản sao `data_portable.yaml` trỏ tới **cùng các thư mục split**.
- Ultralytics tự tạo lại file cache `labels/*.cache` (chỉ là index do đường dẫn đổi D: → E:); ảnh và nhãn `.txt` không đổi.
- Faster R-CNN đọc trực tiếp `images/<split>` + `labels/<split>` theo danh sách file của manifest, đổi YOLO `cx cy w h` (chuẩn hoá) → `xmin ymin xmax ymax` (pixel) **trong bộ nhớ**; file nhãn chỉ được mở để đọc.
- Test chỉ được dùng **sau khi** lựa chọn đã khoá bằng validation (`selection.json`, `locked_before_test: true`).

## 2. Môi trường

Python 3.12.7 · torch 2.11.0+cu128 · torchvision 0.26.0+cu128 · ultralytics 8.4.173 · CUDA 12.8 · NVIDIA GeForce RTX 5050 Laptop GPU 8 GB · Windows 11. AMP bật cho mọi lần train (chính sách AMP giống 3 model cũ: dùng lại chứng nhận từ smoke run vì self-test AMP của Ultralytics bị treo trên GPU này).

## 3. Giao thức tuning (ngân sách bằng nhau cho 3 model mới)

1. **Screening**: tối đa **3 cấu hình**/model, train ngắn (30 epoch cho Ultralytics, 15 epoch cho Faster R-CNN).
2. Xếp hạng bằng **validation** với evaluator chung: mAP50-95 (chênh < 0,002 coi là hoà) → Recall → latency.
3. Train đầy đủ cấu hình thắng; checkpoint best theo **val mAP50-95**.
4. Khoá lựa chọn (`selection.json`) → mới đánh giá val → test.

**Run kiểm soát "đủ lịch train"**: 3 model đã dừng sớm do patience (YOLO11s cũ 77/100, Faster R-CNN 29/50, RT-DETR 78/100) được train lại với **cùng cấu hình, không early stopping**. Mỗi kiến trúc dùng **bản có val mAP50-95 cao hơn** làm đại diện trong bảng chính (không dùng test để chọn).

3 model cũ **không** được tune và **không** được train lại (trừ run kiểm soát YOLO11s đủ 100 epoch, tạo run mới, không ghi đè run cũ).

## 4. Từng model

### 4.1 YOLOv5s

- Weights: `yolov5su.pt` (Ultralytics, COCO). Đây là backbone/neck YOLOv5s với **head anchor-free của Ultralytics**; YOLOv5s gốc (anchor-based) cần repo `ultralytics/yolov5` riêng, không chạy được trong package `ultralytics` mà 3 model cũ dùng.
- Chung: imgsz 640, batch 8, seed 42, deterministic, patience 20, close_mosaic 10, cache RAM, workers 0, max_det 500, augment mặc định Ultralytics (mosaic, scale, HSV, flip).

| Cấu hình | Optimizer / LR | Val mAP50-95 (30 ep) | Val recall |
|---|---|---|---|
| cfg_a (giống hệt 3 model cũ) | AdamW 0,001 | 0,1422 | 0,3978 |
| **cfg_b** ✅ | SGD 0,01, momentum 0,937 | 0,1427 | 0,3986 |
| cfg_c | AdamW 0,0005 | 0,1349 | 0,3776 |

cfg_a và cfg_b hoà (< 0,002) → Recall chọn cfg_b. Train chính 100/100 epoch, **best epoch 82**, VRAM train 2,8 GB. Weights: `models/phase1/yolov5s_best.pt`.

### 4.2 Faster R-CNN ResNet50-FPN

- torchvision `fasterrcnn_resnet50_fpn`, weights `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`; head thay bằng `FastRCNNPredictor(2)` (background + acne).
- Đầu vào: cạnh dài **640** (`min_size=max_size=640`), giống YOLO. Ảnh được thu nhỏ trước bằng `INTER_AREA` về cạnh dài 1024 (cả lúc train và lúc suy luận).
- SGD momentum 0,9, weight decay 1e-4, warm-up 300 iter + cosine, AMP, grad-norm clip 10, batch 4 (batch 8 cần ~8–9 GB, vượt VRAM), trainable backbone layers 3.
- Augmentation: random crop 0,6–1,0 (p=0,5), lật ngang, độ sáng/tương phản. Nhẹ hơn YOLO (không mosaic/HSV).
- Suy luận: score ≥ 0,001, tối đa 500 box/ảnh, NMS IoU 0,5 (mặc định torchvision), RPN 2000/1000 proposal.

| Cấu hình | Anchor (px) | LR | Val mAP50-95 (15 ep) | Val recall |
|---|---|---|---|---|
| cfg_a | 16–256 | 0,005 | 0,1172 | 0,3765 |
| cfg_b | 8–128 | 0,005 | 0,1215 | 0,3835 |
| **cfg_c** ✅ | 8–128 | 0,01 | 0,1306 | 0,3991 |

Anchor mặc định 32–512 px không phù hợp lesion ~8 px nên kích thước anchor là một trục tuning.

| Run train chính | Epoch chạy | Best epoch | Val mAP50-95 | |
|---|---|---|---|---|
| patience 10 | 29/50 | 19 | 0,1393 | |
| **đủ 50 epoch** ✅ (đại diện) | 50/50 | 19 | 0,1405 | VRAM train 4,6 GB |

Run đủ 50 epoch bị gián đoạn ở epoch 24 (tắt máy) và được tiếp tục từ `last.pth` (khôi phục trọng số, optimizer, scheduler, scaler; chuỗi augmentation ngẫu nhiên không khôi phục). Weights FP16: `models/phase1/faster_rcnn_full50_best.pth` (bản 29 epoch: `faster_rcnn_best.pth`).

> **Sửa lỗi đánh giá:** phiên bản đầu của predictor đưa ảnh gốc ~3000 px thẳng vào bộ resize bilinear (không antialias) của torchvision, làm mất lesion nhỏ (val mAP50-95 0,100 thay vì 0,117). Đã sửa để khớp pipeline train (`INTER_AREA`), chấm lại toàn bộ; thứ hạng cấu hình không đổi. Kết quả cũ giữ trong `faster_rcnn/eval_*_stale_bilinear_predictor/` để đối chiếu, **không dùng**.

### 4.3 RT-DETR

- `rtdetr-l.pt` (Ultralytics, COCO; backbone HGNetv2, 32M tham số). Ultralytics không phát hành bản R18 pretrained; RT-DETR-L là bản nhỏ nhất có sẵn, vừa 8 GB VRAM ở batch 8.
- Công thức RT-DETR: AdamW, weight decay 1e-4, `warmup_bias_lr=0`, imgsz 640, batch 8, seed 42, patience 20, close_mosaic 10, cache RAM, max_det 300 (= số object query).

| Cấu hình | LR | Augment | Val mAP50-95 (30 ep) | Val recall |
|---|---|---|---|---|
| cfg_a | 1e-4 | mặc định (có mosaic) | 0,1177 | 0,4005 |
| **cfg_b** ✅ | 2e-4 | mặc định | 0,1351 | 0,4280 |
| cfg_c | 1e-4 | không mosaic | 0,1230 | 0,3993 |

| Run train chính | Epoch chạy | Best epoch | Val mAP50-95 |
|---|---|---|---|
| **patience 20** ✅ (đại diện) | 78/100 | 58 | 0,1557 |
| đủ 100 epoch | 100/100 | 51 | 0,1534 |

VRAM train ~6,9 GB (gần giới hạn 8 GB). Weights: `models/phase1/rtdetr_best.pt` (bản 100 epoch: `rtdetr_full100_best.pt`).

### 4.4 YOLO11s — run kiểm soát đủ 100 epoch

Giữ nguyên toàn bộ tham số của `train_phase1.py` (AdamW 0,001, batch 8, 640, seed 42, close_mosaic 10, cache RAM), chỉ đổi `patience=100`. Run mới `runs/phase1/benchmark_yolo11s_e100_s42_b8_w0_cram_p100`; run cũ và `models/phase1/best_acne_detector.pt` không bị đụng tới. Best epoch 90, val mAP50-95 0,1566 (bản cũ: best epoch 57, 0,1560) → bản 100 epoch là đại diện.

## 5. Đánh giá chung (cùng quy tắc cho cả 6 kiến trúc)

`scripts/phase1/evaluate_detector_common.py` quy mọi framework về dự đoán theo pixel ảnh gốc rồi tính:

| Chỉ số | Cách tính |
|---|---|
| mAP50, mAP50-95 | `ap_per_class` của Ultralytics + matching tham lam theo IoU của Ultralytics; IoU 0,50:0,05:0,95; dự đoán thô conf ≥ 0,001, tối đa 500 box (RT-DETR: 300) |
| P, R, F1, TP, FP, FN | tại **ngưỡng confidence chọn trên VALIDATION** (điểm F1 cao nhất, quy tắc smoothed max-F1 của Ultralytics), IoU 0,5, ghép 1-1; cùng ngưỡng đó áp lên test |
| Phụ | P/R kiểu Ultralytics (max-F1 trên chính split), TP/FP/FN ở conf cố định 0,25 |
| Khoảng tin cậy | bootstrap ghép cặp 1000 lần trên 176 ảnh test, cùng mẫu cho mọi model, percentile 95% |
| Confusion matrix | 1 class: [TP, FP; FN, —] tại ngưỡng vận hành, IoU 0,5 |

Đối chiếu: evaluator chung cho YOLO26s val mAP50-95 **0,1557** so với 0,1574 khi Ultralytics tự chấm (chênh do predict letterbox từng ảnh vs val rect batch). mAP của các model YOLO cũ trên test lệch ≤ 0,0015 so với số trong `reports/metrics/phase1_test_benchmark.csv`.

**Latency** (`scripts/phase1/benchmark_latency.py`, chế độ nguồn Windows *Best performance*, cắm sạc): batch 1, FP32.
- *End-to-end*: mảng ảnh gốc → box cuối (không tính giải mã JPEG); 30 ảnh test cố định × 5 vòng xen kẽ giữa các model; median.
- *GPU forward*: CUDA event, 10 vòng × 20 lần; YOLO/RT-DETR trên tensor 1×3×640×640, Faster R-CNN trên 1 ảnh thật.
- Lần đo đầu (chế độ *Balanced*) bị nhiễu do xung nhịp laptop (YOLO26n đo chậm hơn YOLO26s) nên **không dùng**; lưu ở `reports/metrics/phase1_all_models_latency_rounds.csv` để tham khảo.
- Không so trực tiếp với `inference_ms` trong CSV cũ (đo theo batch 8 trên ảnh đã resize).

## 6. Artifact cho mỗi model

`reports/phase1_additional_models/<model>/` (model = `yolov5s`, `faster_rcnn`, `rtdetr`, `yolo11s_full100`, `faster_rcnn_full50`, `rtdetr_full100`):

| File | Nội dung |
|---|---|
| `train_config.yaml` / `train_config.json`, `selection.json` | cấu hình, best epoch, lựa chọn đã khoá |
| `train_log_results.csv`, `train_results.png` / `train_curves.png` | log theo epoch |
| `screening_val.csv`, `screening_selection.json` | kết quả screening trên val |
| `eval_val/`, `eval_test/` | `metrics.json`, `confusion_matrix(_normalized).png`, `PR/P/R/F1_curve.png`, `per_image_counts.csv`, `predictions.json` |
| `predictions_test/` | 20 ảnh test có vẽ box (xanh lá = nhãn, cyan = dự đoán khớp, đỏ = dự đoán không khớp) |
| `failure_cases_test/` | 5 ảnh sót nhiều nhất + 5 ảnh báo sai nhiều nhất |

So sánh chung: `common_eval/<model>/` (cả 9 run, gồm 3 model cũ), `common_eval/bootstrap_test*.json`, `charts/`, `reports/metrics/phase1_*.csv`. Log: `logs/`.

## 7. Tái lập

```powershell
venv\Scripts\python.exe -u scripts\phase1\train_yolov5s.py
venv\Scripts\python.exe -u scripts\phase1\train_faster_rcnn.py            # thêm --full-schedule cho bản 50 epoch
venv\Scripts\python.exe -u scripts\phase1\train_rtdetr.py                 # thêm --full-schedule cho bản 100 epoch
venv\Scripts\python.exe -u scripts\phase1\train_yolo11s_full100.py
venv\Scripts\python.exe -u scripts\phase1\benchmark_additional_models.py
venv\Scripts\python.exe -u scripts\phase1\benchmark_latency.py
venv\Scripts\python.exe scripts\phase1\plot_all_models_comparison.py
venv\Scripts\python.exe scripts\phase1\check_additional_progress.py      # xem tiến độ
```
Run đã hoàn tất được dùng lại, run dở dang không bị xoá ngầm; `launch_additional.ps1` chạy chuỗi script ở nền.

## 8. Hạn chế

- **Ngân sách tuning không đều**: 3 model mới có 3 cấu hình screening, 3 model cũ không. Với họ YOLO, lợi thế này gần như bằng 0 (YOLOv5s: cấu hình cũ 0,1422 vs cấu hình thắng 0,1427).
- **Một seed** cho mỗi model; chênh lệch nhỏ hơn ~0,005 mAP50-95 nằm trong dao động ngẫu nhiên (CI test rộng ±0,015).
- Faster R-CNN dùng augmentation nhẹ hơn YOLO và NMS 0,5 mặc định (không tune để giữ ngân sách).
- RT-DETR giới hạn 300 dự đoán/ảnh: ít nhất 134/5.046 lesion test (2,7%, đều ở `levle3_89.jpg`) không thể phát hiện.
- YOLOv5s là biến thể `yolov5su` (head anchor-free), không phải YOLOv5s gốc.
- Không có ID bệnh nhân; split theo SHA256 chưa đảm bảo độc lập theo người. Chưa có ảnh camera thực tế ngoài dataset.
- Nhãn có thể sót lesion: một phần FP có thể là lesion thật chưa được gán nhãn (chưa review thủ công).
- Latency đo trên một laptop; GPU/CPU server sẽ cho số khác, thứ tự tương đối đáng tin hơn giá trị tuyệt đối.
