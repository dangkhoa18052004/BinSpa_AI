# GĐ1 — Thử nghiệm độ phân giải: YOLO26s 640 so với 1280

**Câu hỏi:** recall thấp (~0,43) có phải do lesion quá nhỏ ở input 640 hay không?
**Cách làm:** train lại YOLO26s với **imgsz 1280** và giữ nguyên mọi thứ khác: dataset, split 816/175/176, seed 42, AdamW 1e-3, 100 epoch, patience 20, các augmentation. Chỉ khác batch: 4 thay vì 8 để vừa VRAM 8 GB. Nhờ cộng dồn gradient (nbs = 64), batch hiệu dụng vẫn là 64. Ngưỡng confidence được chọn trên validation, test chỉ chạy một lần.
**Script:** `scripts/phase1/train_1280.py --model yolo26s`, sau đó `scripts/phase1/compare_resolution.py`.

## Kết quả trên TEST (176 ảnh, 5.046 lesion)

| | YOLO26s 640 | YOLO26s 1280 | Chênh lệch [CI 95%] |
|---|---:|---:|---|
| mAP50-95 | 0,148 | **0,194** | **+0,046** [+0,037; +0,054] |
| mAP50 | 0,424 | **0,507** | **+0,083** [+0,064; +0,100] |
| Recall | 0,427 | **0,480** | **+0,053** [+0,038; +0,068] |
| Precision | 0,534 | **0,594** | **+0,060** [+0,039; +0,082] |
| F1 | 0,475 | **0,531** | **+0,056** [+0,042; +0,071] |
| TP / FP / FN | 2157 / 1879 / 2889 | **2424 / 1655 / 2622** | +267 TP, −224 FP |
| Ngưỡng conf (chọn trên val) | 0,212 | 0,272 | |
| Latency end-to-end | 12,3 ms | 24,7 ms | ×2 |
| VRAM khi train | — | 5,9 GB (batch 4) | |

Chênh lệch được tính bằng bootstrap ghép cặp, 1.000 lần lấy mẫu lại 176 ảnh test. **Mọi chỉ số đều tốt hơn có ý nghĩa thống kê**, vì không khoảng tin cậy nào chứa 0. Lần train dừng sớm ở epoch 86; epoch tốt nhất là 66 (val mAP50-95 0,202).

### Recall theo kích thước lesion (cạnh dài sau khi thu ảnh về 640)

| Kích thước | Số lesion | 640 | 1280 |
|---|---:|---:|---:|
| < 5 px | 874 | 0,124 | **0,236** |
| 5–8 px | 1.793 | 0,403 | **0,499** |
| 8–12 px | 1.360 | 0,555 | 0,553 |
| 12–20 px | 789 | 0,582 | 0,577 |
| ≥ 20 px | 230 | 0,487 | 0,509 |

Kết quả khớp đúng giả thuyết: toàn bộ phần cải thiện đến từ **lesion nhỏ hơn 8 px**, trong đó nhóm dưới 5 px tăng gần gấp đôi. Recall của lesion lớn gần như không đổi.

### Recall theo mật độ mụn (số lesion/ảnh)

| Lesion/ảnh | 640 | 1280 |
|---|---:|---:|
| 1–10 | 0,470 | **0,537** |
| 11–30 | 0,417 | **0,476** |
| 31–60 | 0,445 | **0,492** |
| > 60 | 0,409 | **0,462** |

Recall tăng đều ở mọi mức độ mụn.

## Nhận xét

- Tăng độ phân giải giúp **tăng recall, đồng thời giảm báo sai**: số FP giảm 224, nên precision cũng tăng.
- Recall vẫn chưa tới 50%. Lesion dưới 5 px vẫn chủ yếu bị sót, và nhãn bị thiếu vẫn kéo các con số đo được xuống. Muốn tăng tiếp có thể thử tiling (giữ nguyên độ phân giải gốc) hoặc gán lại nhãn.
- Chi phí: latency tăng gấp đôi (khoảng 25 ms/ảnh), nhưng vẫn rất nhỏ so với thời gian upload ảnh. Kích thước model không đổi (cùng kiến trúc).
- **Giới hạn của so sánh:** chỉ YOLO26s được train ở 1280. Ở 640, YOLO11s và RT-DETR-L ngang YOLO26s về mặt thống kê, nên chưa thể kết luận YOLO26s vẫn là kiến trúc tốt nhất ở 1280. Muốn kết luận điều đó cần chạy bước 2.

Dữ liệu: `reports/phase1_additional_models/yolo26s_1280/` (`resolution_comparison.json`, `common_eval_summary.json`, `eval_val/`, `eval_test/`, `train_log_results.csv`).

---

# Bước 2 — So sánh 3 model hòa điểm, cùng ở 1280

Ở 640, YOLO26s, YOLO11s và RT-DETR-L không khác biệt có ý nghĩa thống kê. Vì vậy cả ba được train lại ở 1280. Mỗi model giữ đúng cấu hình đã đại diện cho nó ở 640 (được chọn trên validation), patience 20, seed 42 và cùng split. Script: `scripts/phase1/train_1280.py`, sau đó `scripts/phase1/compare_1280.py`.

| | YOLO26s | YOLO11s | RT-DETR-L |
|---|---|---|---|
| Cấu hình | AdamW 1e-3 | AdamW 1e-3 | AdamW 2e-4, wd 1e-4 (cfg_b) |
| Batch (VRAM 8 GB) | 4 | 4 | **2** (batch 4 cần 10,7 GB) |
| Epoch chạy / epoch tốt nhất | 86 / 66 | 100 / 86 | 78 / 58 |
| VRAM khi train | 5,9 GB | 4,9 GB | 6,2 GB |

## Chọn trên VALIDATION (quy tắc đặt trước, chưa xem test)

| Model | Val mAP50-95 | Val Recall |
|---|---:|---:|
| YOLO26s @1280 | **0,2017** | 0,482 |
| RT-DETR-L @1280 | 0,2016 | **0,524** |
| YOLO11s @1280 | 0,2002 | 0,495 |

Ba model nằm trong khoảng 0,002 mAP50-95 so với model tốt nhất, nên được coi là hòa. Theo quy tắc, khi hòa thì xét Recall → chọn **RT-DETR-L @1280**.

## Kiểm chứng trên TEST (176 ảnh, 5.046 lesion)

| Model | mAP50-95 [CI 95%] | mAP50 | Recall | Precision | F1 | TP / FP / FN | Latency* | MB |
|---|---|---:|---:|---:|---:|---|---:|---:|
| **RT-DETR-L @1280** | 0,191 [0,172–0,211] | **0,528** | **0,511** | **0,607** | **0,555** | 2581 / 1671 / 2465 | 78 ms | 63,4 |
| YOLO26s @1280 | **0,194** [0,179–0,210] | 0,507 | 0,480 | 0,594 | 0,531 | 2424 / 1655 / 2622 | 24 ms | 19,5 |
| YOLO11s @1280 | 0,187 [0,172–0,203] | 0,509 | 0,486 | 0,597 | 0,536 | 2453 / 1653 / 2593 | 22 ms | 18,4 |
| YOLO26s @640 (bản cũ) | 0,148 [0,134–0,165] | 0,424 | 0,427 | 0,534 | 0,475 | 2157 / 1879 / 2889 | 10 ms | 19,4 |

\*Latency end-to-end, trung vị trên 30 ảnh test × 5 vòng xen kẽ, batch 1, FP32, RTX 5050 Laptop.

**Chênh lệch so với YOLO26s @1280** (bootstrap ghép cặp, 1.000 lần lấy mẫu):

| | Δ mAP50-95 | Δ Recall | Δ Precision | Δ F1 |
|---|---|---|---|---|
| RT-DETR-L @1280 | −0,003 [−0,012; +0,007] | **+0,032 [+0,007; +0,056]** | +0,013 [−0,005; +0,032] | **+0,024 [+0,006; +0,042]** |
| YOLO11s @1280 | **−0,007 [−0,013; −0,000]** | +0,006 [−0,007; +0,019] | +0,003 [−0,013; +0,020] | +0,005 [−0,008; +0,018] |

- Về mAP50-95, ba model vẫn **hòa nhau**.
- RT-DETR-L có **Recall và F1 cao hơn YOLO26s một cách có ý nghĩa thống kê**. Như vậy lựa chọn trên validation **được test xác nhận**, khác với ở 640 (lúc đó lợi thế recall của RT-DETR-L không lặp lại trên test).
- YOLO11s không hơn YOLO26s ở chỉ số nào và kém hơn một chút về mAP50-95.

### Recall theo kích thước lesion và theo mật độ mụn (test)

| Kích thước (@640) | YOLO26s 1280 | YOLO11s 1280 | RT-DETR-L 1280 |
|---|---:|---:|---:|
| < 5 px | 0,236 | 0,227 | **0,241** |
| 5–8 px | 0,499 | 0,504 | **0,516** |
| 8–12 px | 0,553 | 0,573 | **0,621** |
| 12–20 px | 0,577 | 0,583 | **0,627** |
| ≥ 20 px | **0,509** | 0,491 | 0,452 |

| Lesion/ảnh | YOLO26s 1280 | YOLO11s 1280 | RT-DETR-L 1280 |
|---|---:|---:|---:|
| 1–10 | 0,537 | 0,533 | **0,581** |
| 11–30 | 0,476 | 0,475 | **0,533** |
| 31–60 | 0,492 | 0,494 | **0,527** |
| > 60 | 0,462 | **0,480** | 0,457 |

RT-DETR-L hơn ở hầu hết các nhóm, trừ lesion rất to (≥ 20 px) và ảnh mụn dày đặc (> 60 lesion). Giới hạn 300 query của nó vẫn còn, nhưng ở ngưỡng vận hành, mỗi ảnh tối đa chỉ có 164 box, nên giới hạn này không còn là điểm nghẽn rõ rệt như ở 640.

## Ảnh ngoài dataset (19 ảnh, đánh giá bằng mắt)

Tổng số box: bản 640 có 75, YOLO26s 1280 có 61, RT-DETR-L 1280 có 50.

- RT-DETR-L tự tin hơn trên nốt mụn thật: nốt sẩn ở test6 đạt 0,77, cụm ở trán test13 đạt 0,6–0,7. Nó **không còn báo sai** trên ảnh da sạch (test14: 0 box, bản 640 có 4 box) và bỏ các box tin cậy thấp ở mép hàm (test11, test12).
- Đổi lại, với **ảnh độ phân giải thấp** (test15: 501 px), RT-DETR-L chỉ bắt được 1 nốt trong khi YOLO26s 1280 bắt được 5. Ảnh nhỏ phóng lên 1280 khác xa dữ liệu train. Ứng dụng cần yêu cầu ảnh có cạnh dài ≥ 1500 px (điều kiện này đã có trong khuyến nghị chụp ảnh).

## Rà soát FP theo vùng (test, RT-DETR-L @1280, conf 0,459)

Có 1.671 detection không khớp nhãn: mắt 17, môi 10, nốt ruồi 7, tóc/lông mày/râu 58, khác 1.579 (trong đó 1.524 nằm trên da mặt). Vùng nhạy cảm chiếm **5,5%**, so với 4,7% của bản 640. Số FP ở vùng mắt tăng từ 5 lên 17, phần lớn là các chấm nhỏ ngay dưới hoặc trên mí mắt. Các nhóm mắt, môi, nốt ruồi, tóc/râu đã được xem tay toàn bộ (heuristic tự động gắn 34 box là nốt ruồi, nhưng chỉ 7 box là nốt ruồi thật). Dữ liệu: `reports/metrics/phase1_false_positive_regions_rtdetr1280.csv`.

## Quyết định: RT-DETR-L @1280

1. **Đúng quy tắc chọn đặt trước:** hòa về mAP50-95 trên validation, Recall cao nhất.
2. **Được test xác nhận:** Recall +3,2 điểm và F1 +2,4 điểm so với YOLO26s @1280, đều có ý nghĩa thống kê. So với bản 640 đang dùng: Recall +8,5 điểm, Precision +7,3 điểm.
3. **Chi phí chấp nhận được** với luồng "chụp → upload → phân tích": 78 ms/ảnh, 63 MB, dưới 1 GB VRAM khi suy luận. Tài liệu hướng dẫn xếp latency và kích thước model là tiêu chí phụ.

Rủi ro đã biết:
- Train lại cần batch 2 trên GPU 8 GB, nên BatchNorm ước lượng thống kê trên 2 ảnh; muốn tăng batch phải dùng GPU lớn hơn.
- Model kém trên ảnh độ phân giải thấp.
- Recall ở ảnh mụn dày đặc (> 60 lesion) ngang YOLO26s, không hơn.
- Chỉ train 1 seed cho mỗi model.

**Phương án dự phòng:** YOLO26s @1280 (`models/phase1/yolo26s_1280_best.pt`). Nhanh hơn 3 lần, nhẹ hơn 3 lần, mAP50-95 ngang RT-DETR-L, recall thấp hơn khoảng 3 điểm. Nên dùng nếu cần chạy trên CPU hoặc máy yếu.

**Triển khai:** `models/phase1/best_acne_detector.pt` giờ là RT-DETR-L @1280 (SHA256 `3bf12463…6935`), dùng imgsz 1280 và conf 0,459; `predict_acne.py` tự đọc hai giá trị này từ `model_metadata.json`. Bản cũ được giữ tại `models/phase1/best_acne_detector_yolo26s_640.pt` cùng `model_metadata_yolo26s_640.json`. Script: `scripts/phase1/promote_detector.py`.

*Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da.*
