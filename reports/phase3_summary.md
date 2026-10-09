# Báo cáo Giai đoạn 3: phát hiện nhóm sẹo mụn

**Trạng thái:** đã train, chọn model trên validation, đánh giá test một lần, chạy bootstrap và kiểm tra bằng mắt trên ảnh selfie.
**Model triển khai:** `models/phase3/best_scar_detector.pt`, YOLO11s @1280 (cfg `c1280`), conf 0,122 (ngưỡng max-F1 trên validation).
**Model thay thế ngang hàng:** YOLO26s @1280 (`runs/phase3/yolo26s_c1280/weights/best.pt`).

## 1. Dữ liệu

- **Nguồn:** bản mirror ACNE-DET trên Roboflow (giống GĐ2). Dùng **cùng master split** `datasets/acnedet_master_split.csv` (190/40/45 ảnh, chia theo nhóm ảnh gốc), nên không có ảnh nào nằm ở hai split.
- **Hai lớp, giữ đúng tên trong ACNE-DET:** `atrophic_scar` (sẹo lõm) và `hypertrophic_scar` (sẹo phì đại/lồi). Không suy ra rolling, icepick, boxcar hay keloid.
- Các lớp khác (mụn, melasma, nevus, other) **không được gán nhãn** trong bài toán 2 lớp này, và cũng không bị đổi thành sẹo. Có 1 box bị cắt bớt cho vừa biên ảnh. Đã xem 30 ảnh kiểm tra nhãn (`reports/phase3_label_check/`): box nằm đúng các vết lõm ở má và vết lồi ở cằm, hàm.

| Split | Ảnh | atrophic_scar | hypertrophic_scar | Ảnh có hypertrophic |
|---|---:|---:|---:|---:|
| train | 190 | 6.228 | 520 | 110 |
| val | 40 | 1.233 | 130 | 28 |
| test | 45 | 1.271 | 115 | 27 |

Kích thước trung vị của sẹo trên ảnh gốc khoảng 3456×5184 px: atrophic 50 px, hypertrophic 68 px. Khi đưa ảnh về 640 px, sẹo chỉ còn khoảng 6–8 px.

## 2. Huấn luyện và chọn model

- **Cấu hình chung**, giống công thức GĐ1: pretrained COCO, AdamW, lrf 0,01, wd 5e-4, 100 epoch, patience 20, close_mosaic 10, seed 42, AMP, max_det 500.
- **Ngân sách tuning:** 3 cấu hình cho mỗi kiến trúc: `c640` (batch 8, lr 1e-3), `c1280` (batch 4, lr 1e-3), `c1280_lr5` (batch 4, lr 5e-4).
- **Quy tắc chọn (đặt trước khi xem test):** mAP50-95 trên validation. Nếu hai model chênh không quá 0,002 thì xét recall của `hypertrophic_scar` trên validation, sau đó mới xét latency.

| Run | Val mAP50-95 | Val mAP50 | Val recall | Val recall HS | Epoch tốt nhất / số epoch đã chạy | VRAM khi train |
|---|---:|---:|---:|---:|---|---:|
| YOLO26s c640 | 0,033 | 0,119 | 0,205 | 0,075 | 73 / 93 | 3,1 GB |
| YOLO26s c1280 | **0,099** | 0,291 | 0,316 | 0,185 | 51 / 71 | 6,7 GB |
| YOLO26s c1280_lr5 | 0,090 | 0,274 | 0,321 | 0,206 | 40 / 60 | 6,7 GB |
| YOLO11s c640 | 0,032 | 0,120 | 0,170 | 0,054 | 89 / 100 | 2,7 GB |
| **YOLO11s c1280** | 0,098 | **0,294** | 0,321 | **0,223** | 61 / 81 | 5,6 GB |
| YOLO11s c1280_lr5 | 0,093 | 0,282 | 0,306 | 0,185 | 45 / 65 | 5,6 GB |

- **Độ phân giải là yếu tố quyết định:** ở 1280, mAP50-95 cao gấp khoảng 3 lần so với 640, ở cả hai kiến trúc.
- YOLO26s c1280 và YOLO11s c1280 chênh nhau 0,001, nên được coi là hòa. Khi hòa, quy tắc xét recall của sẹo phì đại, và YOLO11s cao hơn (0,223 so với 0,185) nên được chọn. Lựa chọn được khóa trong `reports/phase3_evaluation/selection.json` trước khi chạy test.

## 3. Kết quả trên TEST (45 ảnh)

| Model | mAP50-95 [CI 95%] | mAP50 | Precision | Recall | Latency* | MB |
|---|---|---:|---:|---:|---:|---:|
| **YOLO11s @1280** (đã chọn) | 0,103 [0,076–0,130] | 0,315 | 0,413 | 0,360 | 22 ms | 18,4 |
| YOLO26s @1280 | 0,119 [0,089–0,150] | 0,351 | 0,453 | 0,363 | 24 ms | 19,5 |

\*Trung vị trên 3 lượt chạy qua ảnh test, batch 1, tính từ ảnh gốc tới box.

Kết quả theo từng lớp (ngưỡng conf chọn trên validation, IoU 0,5):

| Lớp | Model | P | R | AP50 | AP50-95 | TP / FP / FN |
|---|---|---:|---:|---:|---:|---|
| atrophic_scar | YOLO11s | 0,408 | 0,408 | 0,361 | 0,128 | 554 / 1035 / 717 |
| | YOLO26s | 0,404 | 0,422 | 0,371 | 0,128 | 514 / 692 / 757 |
| hypertrophic_scar | YOLO11s | 0,419 | 0,313 | 0,268 | 0,078 | 35 / 55 / 80 |
| | YOLO26s | 0,503 | 0,304 | 0,330 | 0,110 | 39 / 30 / 76 |

**Bootstrap ghép cặp** (1.000 lần lấy mẫu lại theo ảnh test), tính YOLO26s trừ YOLO11s:

| Chỉ số | Chênh lệch | CI 95% |
|---|---:|---|
| mAP50-95 | +0,016 | [−0,008; +0,040] |
| AP50-95 atrophic | +0,000 | [−0,009; +0,009] |
| AP50-95 hypertrophic | +0,031 | [−0,014; +0,079] |
| mAP50 | +0,050 | [−0,002; +0,096] |

→ **Không có chênh lệch nào có ý nghĩa thống kê.** Giống GĐ1 và GĐ2, model được validation chọn không đứng đầu trên test, nhưng với 45 ảnh test thì hai model không phân biệt được. Mình giữ lựa chọn theo quy tắc. Nếu muốn ưu tiên sẹo phì đại, YOLO26s @1280 là phương án thay thế hợp lý: nó nhỉnh hơn ở mAP50 (p = 0,97) và ít báo sai hơn.

Ma trận nhầm lẫn, các đường PR/P/R/F1 và ảnh dự đoán minh họa nằm trong `reports/phase3_evaluation/test_{yolo11s,yolo26s}/`. Ảnh khuôn mặt không được đưa lên git.

## 4. Kiểm tra trên ảnh selfie (`test_img`, đánh giá bằng mắt)

- Độ tin cậy thấp, chỉ khoảng 0,1–0,6. Một ảnh (test18) có tới 18 box sẹo lõm, nhưng phần lớn rơi vào **lỗ chân lông to** ở má.
- Ở ảnh có mụn đang viêm (test6), nhiều box rơi vào **vết thâm hoặc nốt đỏ** trên trán chứ không phải sẹo.
- Kết luận: model **lệch miền nặng** khi chuyển từ ảnh VISIA sang ảnh điện thoại. Ở tầng hiển thị chỉ nên coi đây là "dấu hiệu", kèm độ tin cậy, và không nên đếm sẹo một cách tuyệt đối.

## 5. Hạn chế

- **Chất lượng tuyệt đối còn thấp:** mAP50-95 khoảng 0,10–0,12, recall khoảng 0,36. Sẹo lõm có ranh giới mờ, rất khó khoanh box chính xác.
- **Test chỉ có 45 ảnh và 115 sẹo phì đại**, nên khoảng tin cậy rộng. Mỗi cấu hình chỉ chạy 1 seed.
- **Ảnh VISIA được chụp chuẩn hóa**, trong khi ảnh selfie thực tế cho kết quả kém hơn nhiều.
- **Không có ID bệnh nhân.** Việc gom nhóm dựa trên số thứ tự liền nhau và độ giống ảnh (xem báo cáo GĐ2).
- **Chưa benchmark Faster R-CNN.** Tài liệu ghi đây là tùy chọn khi còn tài nguyên.

## 6. Tái lập

```powershell
.\venv\Scripts\python.exe scripts\phase3\build_scar_dataset.py
.\venv\Scripts\python.exe -u scripts\phase3\train_scar.py --model yolo11s --config c1280   # x 2 kiến trúc x 3 cấu hình
.\venv\Scripts\python.exe scripts\phase3\evaluate_scar.py
.\venv\Scripts\python.exe scripts\phase3\bootstrap_scar.py
.\venv\Scripts\python.exe scripts\phase3\predict_scar.py --source test_img\test6.jpg
```

*Kết quả chỉ là dấu hiệu nhóm sẹo trên ảnh, mang tính tham khảo chăm sóc da, không phải chẩn đoán.*
