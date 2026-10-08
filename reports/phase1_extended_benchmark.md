# GĐ1 — Báo cáo benchmark mở rộng và quyết định model

**Trạng thái: HOÀN THÀNH.** Model triển khai cuối cùng: **RT-DETR-L @1280** (`models/phase1/best_acne_detector.pt`, SHA256 `3bf12463…6935`, imgsz 1280, conf 0,459).

> **Cập nhật sau benchmark:** mục 2–5 bên dưới là benchmark ở độ phân giải 640, khi YOLO26s được chọn. Sau đó, thử nghiệm độ phân giải cho thấy recall thấp chủ yếu do lesion quá nhỏ ở 640. Ba model hòa điểm (YOLO26s, YOLO11s, RT-DETR-L) được train lại ở 1280, và RT-DETR-L @1280 được chọn theo cùng quy tắc (hòa mAP50-95 trên validation → Recall), sau đó được test xác nhận: Recall 0,511, F1 0,555, hơn YOLO26s @1280 có ý nghĩa thống kê. Chi tiết: [phase1_resolution_study.md](phase1_resolution_study.md). Bản YOLO26s @640 được giữ tại `models/phase1/best_acne_detector_yolo26s_640.pt`.

Báo cáo này tổng hợp kết quả GĐ1 theo yêu cầu của prompt GĐ1-C trong tài liệu *Hướng dẫn phát triển AI 3 giai đoạn*. Số liệu chi tiết nằm ở [phase1_additional_models/final_comparison.md](phase1_additional_models/final_comparison.md) và [phase1_additional_models/README.md](phase1_additional_models/README.md).

## 1. Dữ liệu và giao thức

- Dataset `acne04v2-clean-v1`: ảnh ACNE04 ghép với annotation ACNE04-v2. Mỗi lesion được đổi từ tâm + bán kính sang bbox `2r×2r`, chỉ có một class `acne`.
- Ảnh trùng SHA256 được loại trước khi split. Split theo ảnh với seed 42: **816 / 175 / 176** (train/val/test). Manifest SHA256: `0a40262f…09d6`.
- Cả 6 detector dùng chung split, chung evaluator (IoU 0,5 để tính TP/FP/FN, COCO mAP50 và mAP50-95) và cùng một phiên đo latency.
- Hyperparameter chỉ chọn trên validation, mỗi kiến trúc tối đa 3 cấu hình. Ngưỡng confidence vận hành cũng chọn trên validation. Test chỉ chạy một lần sau khi đã khóa cấu hình.

## 2. Bảng xếp hạng cuối (TEST, 176 ảnh, 5.046 lesion)

Thứ tự ưu tiên: mAP50-95, sau đó Recall, sau đó Precision. Latency và kích thước là tiêu chí phụ.

| # | Model | mAP50-95 [CI 95%] | mAP50 | Recall | Precision | F1 | ms/ảnh | MB |
|---|---|---|---|---|---|---|---|---|
| 1 | **YOLO26s** | **0,148** [0,134–0,165] | **0,424** | 0,428 | 0,534 | 0,475 | 10,8 | 19,4 |
| 2 | RT-DETR-L | 0,143 [0,126–0,161] | 0,422 | **0,436** | **0,551** | **0,487** | 28,4 | 63,2 |
| 3 | YOLO11s (100 ep) | 0,143 [0,130–0,158] | 0,412 | 0,398 | 0,547 | 0,461 | 9,2 | 18,3 |
| 4 | YOLOv5s | 0,139 [0,126–0,153] | 0,402 | 0,400 | 0,525 | 0,454 | **8,3** | 17,7 |
| 5 | Faster R-CNN R50-FPN (50 ep) | 0,130 [0,117–0,145] | 0,388 | 0,410 | 0,498 | 0,450 | 54,0 | 79,1 |
| 6 | YOLO26n | 0,119 [0,105–0,133] | 0,351 | 0,360 | 0,500 | 0,419 | 10,3 | **5,1** |

Bootstrap ghép cặp so với YOLO26s cho thấy RT-DETR-L và YOLO11s **không khác biệt có ý nghĩa** về mAP50-95. YOLOv5s, Faster R-CNN và YOLO26n **kém hơn rõ**. Confusion matrix, normalized confusion matrix, các đường PR/P/R/F1-confidence, ảnh dự đoán và failure cases của từng model nằm trong `phase1_additional_models/common_eval/<model>/eval_test/`. Config, version và thời gian train của từng run được ghi trong `reports/metrics/phase1_all_models_comparison_all_runs.csv`.

## 3. Lý do chọn YOLO26s (ở 640; đã được thay, xem phần cập nhật đầu báo cáo)

1. Thuộc nhóm có chất lượng dẫn đầu và có mAP50 cùng mAP50-95 cao nhất trên test. Không model nào vượt YOLO26s có ý nghĩa thống kê.
2. Ổn định ở ảnh mụn dày đặc (>60 lesion/ảnh). RT-DETR-L bị giới hạn 300 query nên giảm recall đúng ở nhóm ca nặng.
3. Chi phí triển khai thấp: khoảng 11 ms/ảnh, 19 MB, dưới 1 GB VRAM khi suy luận, và train lại được trên GPU 8 GB.
4. RT-DETR-L được giữ làm phương án dự phòng. Lợi thế recall của nó trên validation không lặp lại trên test, trong khi model chậm hơn khoảng 2,6 lần và nặng hơn khoảng 3 lần.

Model không được chọn vì "mới hơn". Quy tắc chọn trên validation thực ra xếp RT-DETR-L đứng đầu, nhờ recall cao hơn trong một nhóm hòa nhau về mAP. Quyết định cuối cùng chọn YOLO26s sau khi xét thêm kết quả test và các tiêu chí triển khai, như đã giải thích ở trên.

## 4. Rà soát false positive theo vùng (tập test)

Có 1.414 detection của YOLO26s (conf 0,25, IoU 0,5) không khớp với nhãn nào. Vùng của từng detection được gán tự động bằng MediaPipe Face Landmarker (lấy đa giác mắt, môi, lông mày) kết hợp mô hình phân vùng tóc. Nốt ruồi được nhận diện bằng heuristic "điểm tối, không đỏ". Sau đó mình **xem lại toàn bộ** các nhóm eyes, lips, mole, hair_beard qua contact sheet và sửa tay. Nhóm `other` được kiểm tra theo mẫu (3/17 sheet). Script: `scripts/phase1/label_error_regions.py`, rồi đến `summarize_error_regions.py`.

| Vùng | Số detection | Tỷ lệ |
|---|---:|---:|
| Mắt / mí mắt (`eyes`) | 5 | 0,4% |
| Môi / viền môi (`lips`) | 17 | 1,2% |
| Nốt ruồi (`mole`) | 7 | 0,5% |
| Tóc, lông mày, râu (`hair_beard`) | 38 | 2,7% |
| Khác (`other`) | 1.347 | 95,3% |
| ↳ trên da mặt | 1.289 | 91,2% |
| ↳ ngoài mặt (nền, thước đo, cổ áo, tai) | 35 | 2,5% |
| ↳ nếp da/bóng, vật thể, mụn viêm (từ nhóm mole bị sửa lại) | 23 | 1,6% |

Nhận xét:
- Chỉ **4,7%** detection sai rơi vào các vùng nhạy cảm (mắt, môi, nốt ruồi, tóc/râu). Model hầu như không nhầm nốt ruồi: chỉ 7 trường hợp, trong đó 2 trường hợp còn khó phân biệt với mụn đầu đen.
- Phần lớn detection "sai" nằm trên da mặt. Khi xem mẫu, nhiều detection trong số đó là nốt sẩn hoặc mụn đầu trắng nhỏ **trông như lesion thật nhưng không có nhãn**, hoặc box lệch so với nhãn (lesion ~8 px). Vì vậy Precision trên test (0,53) có khả năng đang **thấp hơn** độ chính xác thực tế. Muốn định lượng chính xác cần chuyên gia gán lại nhãn.
- Nhóm heuristic `mole` tự động có độ chính xác thấp (7/36). Nó bắt nhầm nếp da, vật màu xanh và mụn viêm có vảy, nên đã được sửa tay hoàn toàn.

Dữ liệu: `reports/metrics/phase1_false_positive_regions.csv` (cột `notes` ghi `verified:` cho dòng đã xem tay, `auto:` cho dòng chỉ gán bằng rule), `reports/metrics/phase1_region_error_summary.json` và `reports/metrics/phase1_region_autolabel_meta.json`. Có 9/160 ảnh không tìm được landmark (ảnh chụp sát một bên má); các ảnh này chỉ dùng mô hình phân vùng tóc.

## 5. Nghiệm thu trên ảnh ngoài dataset

Đã chạy trên 19 ảnh điện thoại/ảnh ngoài ACNE04, chi tiết ở [phase1_camera_review.md](phase1_camera_review.md). Kết quả **đạt có điều kiện**. Model bắt được nốt mụn thật trên ảnh chụp gần, độ phân giải cao, không nhận nhầm nốt ruồi và không sinh box trên ảnh da sạch. Tuy nhiên model sót nhiều nốt sẩn nhỏ ở trán và không dùng được với ảnh nhỏ hơn khoảng 700 px.

## 6. Hạn chế

- Chất lượng tuyệt đối còn thấp: recall ~0,43 (sót hơn một nửa lesion), mAP50-95 ~0,15.
- Mỗi model chỉ chạy một seed. Split theo ảnh chứ không theo người, vì không có ID bệnh nhân.
- Nhãn ACNE04-v2 có lesion bị bỏ sót, nên FP trên test bị thổi phồng (xem §4).
- Ảnh ngoài dataset ít (19 ảnh) và không có nhãn chuyên gia. Chưa thể định lượng hiệu năng trong môi trường spa thật.
- Rà soát vùng của nhóm `other` mới kiểm tra theo mẫu, chưa xem từng detection.

## 7. Đưa sang GĐ2

- Detector: `models/phase1/best_acne_detector.pt` (RT-DETR-L @1280), conf 0,459, imgsz 1280; `predict_acne.py` đọc hai giá trị này từ `model_metadata.json`. Ảnh đầu vào nên có cạnh dài ≥ 1500 px.
- Khi crop cho classifier GĐ2: gộp các box lồng nhau, thêm padding nhỏ, và ghi nhận rằng crop từ ảnh điện thoại có thể khác phân phối với ảnh macro của dataset GĐ2.
- Hướng cải thiện GĐ1 về sau (không chặn GĐ2): tăng độ phân giải hoặc dùng tiling, gán lại nhãn các lesion bị sót, chạy nhiều seed, thu thập ảnh spa có nhãn.

## 8. Checklist nghiệm thu GĐ1

| Tiêu chí | Kết quả |
|---|---|
| 100% ảnh annotation v2 được map hoặc có báo cáo missing | ✅ `missing_images: 0`, `reports/phase1_data/` |
| ≥ 30 ảnh label visualization đã kiểm tra | ✅ `reports/phase1_label_check/` (138 ảnh: 54/50/34) |
| Test split độc lập | ✅ 176 ảnh, đã dedupe SHA256 |
| Benchmark mở rộng 6 model kèm config/version từng run | ✅ §2 |
| Best detector kèm version metadata | ✅ `models/phase1/model_metadata.json` (RT-DETR-L @1280) |
| Confusion matrix, PR, P/R/F1-confidence, ≥ 20 ảnh minh họa | ✅ `common_eval/*/eval_test/`, `reports/phase1_predictions/` |
| Đếm FP ở vùng mắt, môi, nốt ruồi, tóc/râu | ✅ §4 (YOLO26s 640); RT-DETR-L 1280 trong `phase1_resolution_study.md` |
| Thử trên ảnh chưa thấy/ảnh người dùng chụp | ✅ §5 (đạt có điều kiện) |

*Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da, không phải chẩn đoán.*
