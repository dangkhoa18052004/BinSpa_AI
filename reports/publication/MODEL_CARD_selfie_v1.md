# Model card — BinSpa AI soi mụn, bản `hyb_d1` (selfie_v1)

Ngày nghiệm thu: 04/11/2026 (P11). Phạm vi: **demo nội bộ trên localhost của spa**. Không phải thiết bị y tế, không chẩn đoán.

## Chức năng

Nhận một ảnh khuôn mặt (chụp bằng điện thoại hoặc ảnh lâm sàng), khoanh các nốt nghi là mụn và hiển thị theo tầng:

- **Tầng A – tên loại:** mụn đầu trắng, mụn đầu đen, mụn sẩn, mụn mủ. Không bao giờ hiện tên "mụn nốt" vì đo chưa đủ.
- **Tầng B – nhóm:** mụn viêm / không viêm. Chỉ áp dụng cho ảnh giống ảnh lâm sàng.
- **Tầng C – vị trí:** chỉ khoanh vị trí, ghi là nghi là mụn.

Ảnh nào cũng có kết quả. Ảnh mờ, tối, mặt nhỏ hoặc khác kiểu ảnh huấn luyện chỉ kèm cảnh báo, và nhiều nốt sẽ xuống tầng thấp hơn.

## Thành phần (bundle `models/candidates/selfie_v1/hyb_d1/`, sha trong `bundle.json`)

| Phần | File | sha256 (16 ký tự đầu) |
| --- | --- | --- |
| Detector RT-DETR-L @1280 (D1): fine-tune từ detector phase 1 trên 4.674 ảnh trộn gồm ảnh lâm sàng, bản giả lập điện thoại và DetectDataSkin-2 | `detector.pt` | `675ad7517814c548` |
| Classifier C1 EfficientNet-B0: dùng cho ảnh "rủi ro cao", tức ảnh điện thoại, ảnh lạ hoặc ảnh kém | `classifier.pth` | `6e6f212327a28afb` |
| Classifier phase 2: dùng cho ảnh "rủi ro thấp", tức ảnh giống ảnh lâm sàng | `models/phase2/best_acne_type_robust.pth` | `5dd4daffaeea7d2a` |
| Thống kê miền (Mahalanobis) theo đặc trưng C1 | `domain_stats.npz` | trong `bundle.json` |
| Ngưỡng và hiệu chỉnh theo tầng: chọn trên val, không chỉnh theo test | `display_policy.json` | `3e2905f04ab32056` |

## Dữ liệu

- **Train/val:**
  - ACNE-DET, nhãn của tác giả;
  - ACNE04, kèm loại mụn ghép từ kbm0q (chủ dự án duyệt 30/30 vị trí);
  - DetectDataSkin-2 và xnbio, nhãn Roboflow chưa xác minh.
- **Cách chia:** chia theo lineage group, các bản augmentation và ảnh trùng nằm chung nhóm. Không có ID bệnh nhân, nên nhóm chỉ là suy đoán.
- **Tập kiểm tra khoá từ P03:**
  - T1: ACNE-DET, nhãn bác sĩ.
  - T2: DetectDataSkin-2, proxy ảnh thường, nhãn chưa xác minh. Chỉ dùng một lần ở P11.
  - T3: ảnh T1 giả lập điện thoại.
  - T4: SCIN đầu/cổ, nhãn bác sĩ ở mức ca.
  - T5: chủ dự án duyệt 287 box trên 20 selfie và 54 ảnh SCIN.

## Kết quả nghiệm thu (IoU 0,3; CI 95% bootstrap theo lineage group)

| Hạng mục | T1 (nhãn bác sĩ) | T2 (ảnh thường, nhãn chưa xác minh) | Mục tiêu T1 / T2 |
| --- | --- | --- | --- |
| Tên loại hiển thị đúng (tầng A) | **91,9%** [87,9–97,3], 136 box | **83,5%** [79,2–87,6], 650 box | ≥ 90% / ≥ 80% ✅ |
| Nhóm hiển thị đúng (tầng B) | 94,3% [88,2–98,2], 105 box | không hiển thị | ≥ 92% ✅ |
| Box hiển thị nằm đúng nốt | 74,3% | 75,2% | ≥ 70% / ≥ 60% ✅ |
| Tỉ lệ nốt thật được khoanh (recall) | 40,4% | 39,2% | ≥ 50% / ≥ 45% ❌ |
| Tỉ lệ nốt tìm đúng có loại hoặc nhóm | 22,6% | 39,7% | ≥ 60% / ≥ 40% ❌ |
| Hiệu chỉnh xác suất (ECE) | 0,058 | 0,034 | ≤ 0,05 ❌ / ≤ 0,08 ✅ |

Các hạng mục khác:
- **T3, giả lập điện thoại:** 4/7 kiểu biến đổi đạt. Không đạt ở thu nhỏ 1080 px (tầng A giảm 6,2 điểm), JPEG chất lượng 35 (giảm 7,0 điểm) và kiểu tổng hợp xấu (giảm 20,9 điểm; tầng A còn 71%). Mức cho phép là giảm ≤ 5 điểm.
- **T4, SCIN mức ca:** AUC 0,662 [0,572–0,744] trên 27 ca mụn và 362 ca không mụn; mục tiêu ≥ 0,70 ❌.
- **T5, chủ dự án duyệt:** 3/287 box (1,0%) rõ ràng sai chỗ, cả ba trên ảnh SCIN; 0 lỗi xử lý ✅.
- **Regression:** macro-F1 box+loại trên T1 là 0,090, so với 0,347 ở baseline P05 ❌. Lý do: chỉ 13% nốt được ghi tên loại, so với mọi nốt "confident" ở bản cũ.
- **Dịch vụ:** 1.589/1.589 ảnh có kết quả. Thời gian mỗi ảnh p50 0,35 s, p95 1,40 s (RTX 5050 Laptop) ✅.

Tên loại theo từng loại trên T1, theo dạng (số box hiện tên / độ đúng / số nốt thật):
- đầu trắng: 12 / 92% / 971;
- đầu đen: 1 / – / 473;
- sẩn: 94 / 92% / 867;
- mủ: 29 / 93% / 192.

Trên T2, theo cùng dạng:
- đầu trắng: 0 / – / 125;
- đầu đen: 19 / 74% / 330;
- sẩn: 177 / 78% / 990;
- mủ: 454 / 86% / 2.238.

## Giới hạn cần nói rõ khi demo

1. AI bỏ sót khoảng 60% nốt và chỉ ghi tên loại cho phần nhỏ trong số nốt tìm được. **Mụn đầu trắng và mụn đầu đen gần như không được gọi tên.**
2. Ảnh rất xấu (mờ, nén và tối cùng lúc) làm tên loại kém đúng hẳn: chỉ đúng khoảng 71% trên ảnh giả lập.
3. Con số trên ảnh thường dựa trên nhãn chưa được chuyên gia xác minh. **Chưa kiểm định trên selfie có nhãn chuyên gia.**
4. Ảnh chụp nằm, mặt xoay ngang thường không nhận ra mặt, nên phần lớn nốt chỉ được khoanh vị trí.
5. Không đánh giá độ nặng, sẹo hay phác đồ điều trị.

## Câu được phép công bố (mục 6.3)

"Trên dữ liệu kiểm tra có nhãn, loại mụn được hiển thị đúng khoảng 92% (bộ ảnh lâm sàng, nhãn bác sĩ) và 84% (bộ ảnh thường, nhãn chưa xác minh). Chưa kiểm định trên selfie có nhãn chuyên gia. Kết quả chỉ để tham khảo chăm sóc da."

Câu này lấy từ `configs/selfie_v1/serving.json`, vốn gắn với sha của policy. Nếu đổi sang policy khác, trang tự trở về câu "đang được đánh giá".

## Quyết định và quay lại

- **Quyết định:** `release_for_internal_demo` (xem `release_decision.json`).
  - Mọi hạng mục về độ đúng của nội dung được hiển thị đều đạt trên T1 và T2; T5 đạt; dịch vụ đạt.
  - Các mục không đạt ở trên được ghi là giới hạn.
  - Chủ dự án đã chọn đổi sang bản này ở điểm quyết định 2.
- **Quay lại bản chặng 1:** trong `configs/selfie_v1/serving.json`, đặt `"policy": "models/candidates/selfie_v1/baseline_display_policy.json"` rồi khởi động lại `api/server.py`.
- **Quay về bản demo cũ (định dạng mà web Spa đang đọc):** chạy `api/server.py --engine legacy --demo`.

## Tái lập

```
venv\Scripts\python.exe scripts\evaluation_v2\freeze_bundle.py --tag hyb_d1
venv\Scripts\python.exe -u scripts\evaluation_v2\final_evaluation.py --run
venv\Scripts\python.exe scripts\evaluation_v2\make_t5_review.py
venv\Scripts\python.exe scripts\data_v2\review_server.py --task t5_boxes
venv\Scripts\python.exe scripts\evaluation_v2\final_evaluation.py --score
```
