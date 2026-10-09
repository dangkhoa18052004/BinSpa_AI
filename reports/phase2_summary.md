# Báo cáo Giai đoạn 2: phân loại loại tổn thương mụn

**Trạng thái:** đã train, chọn model trên validation, đánh giá test một lần và kiểm tra pipeline GĐ1 → GĐ2.
**Model triển khai:** `models/phase2/best_acne_type.pth` (EfficientNet-B0, cfg_a, input 224).
**Model thay thế ngang hàng:** ConvNeXt-Tiny cfg_a (`runs/phase2/convnext_tiny_cfg_a/best.pth`).

## 1. Dữ liệu

- **Nguồn:** bản mirror ACNE-DET trên Roboflow Universe (`detect-4fwbi/acnedet`, v1, export COCO, không augmentation). Bản này khớp với paper DSDH (Wei et al., KBS 2024): ảnh VISIA chụp chính diện, 10 lớp, nhãn do 6 bác sĩ da liễu gán. Vì là mirror cộng đồng, nguồn gốc dữ liệu chưa được xác minh bằng checksum của tác giả. Người upload trên Roboflow ghi license MIT, còn tác giả gốc không ghi license, nên chỉ dùng cho mục đích học thuật.
- **Audit** (`reports/phase2_data/audit.json`): 275 ảnh khoảng 3456×5184 px, 31.502 annotation. Không có ảnh trùng SHA256, không có box lỗi.
- **Lớp dùng cho GĐ2:** đúng tên gốc trong ACNE-DET là `closed_comedo`, `open_comedo`, `papule`, `pustule`, `nodule`. Không đổi nodule thành cyst. Các lớp scar, melasma, nevus và other không dùng ở GĐ2.
- **Master split dùng chung cho GĐ2 và GĐ3** (`datasets/acnedet_master_split.csv`):
  - Mirror không có ID bệnh nhân. Xem ảnh cho thấy các lần khám lại của cùng một người có số thứ tự liền nhau.
  - Vì vậy ảnh được gom thành khối 5 số liên tiếp; hai khối được nối lại nếu ảnh ở ranh giới giống nhau (cosine ≥ 0,95).
  - Kết quả là 50 nhóm, chia train/val/test = **190/40/45 ảnh** với seed 42. Tỷ lệ từng lớp lệch so với mục tiêu tối đa 2%.
- **Crop** (`datasets/phase2_acne_type/crop_manifest.csv`):
  - Mỗi nốt được cắt một crop vuông, cạnh bằng 2 × max(w, h) và tối thiểu 32 px, từ ảnh gốc độ phân giải đầy đủ.
  - Split của crop lấy theo split của ảnh gốc, nên không có crop nào của cùng một ảnh nằm ở hai split.

| Lớp | Train | Val | Test | Kích thước nốt (trung vị, px gốc) |
|---|---:|---:|---:|---:|
| closed_comedo | 4.127 | 776 | 971 | 26 |
| open_comedo | 2.309 | 511 | 473 | 22 |
| papule | 3.773 | 899 | 867 | 48 |
| pustule | 879 | 196 | 192 | 40 |
| nodule | 143 | 32 | 34 | 167 |
| **Tổng** | 11.231 | 2.414 | 2.537 | |

## 2. Huấn luyện và chọn model

- **Cấu hình chung:** timm, pretrained ImageNet-1k, input 224, AdamW (wd 0,05), 1 epoch warm-up rồi cosine, AMP, batch 32, tối đa 30 epoch, dừng sớm sau 6 epoch không cải thiện macro-F1 trên validation, seed 42.
- **Augmentation** (chỉ áp dụng cho tập train): random resized crop (scale 0,75–1), lật ngang, xoay tối đa 15°, thay đổi độ sáng, độ tương phản và độ bão hòa tối đa 0,1.
- **Ngân sách tuning:** 3 cấu hình cho mỗi kiến trúc.
  - cfg_a: learning rate gốc, cross-entropy với label smoothing 0,1.
  - cfg_b: thêm sampler cân bằng lớp, trọng số theo 1/√tần suất.
  - cfg_c: learning rate bằng một nửa, cross-entropy có trọng số theo 1/√tần suất.
- **Quy tắc chọn (đặt trước khi xem test):** mỗi kiến trúc lấy cấu hình có macro-F1 trên validation cao nhất. Giữa hai kiến trúc, lấy model có macro-F1 trên validation cao hơn; nếu chênh dưới 0,005 thì lấy EfficientNet-B0 vì nhẹ hơn.

| Run | Val macro-F1 | Val acc | Epoch tốt nhất / số epoch đã chạy | Thời gian train | VRAM |
|---|---:|---:|---|---:|---:|
| ConvNeXt-Tiny cfg_a | 0,834 | 0,871 | 5 / 11 | 12 phút | 2,4 GB |
| ConvNeXt-Tiny cfg_b | 0,832 | 0,875 | 6 / 12 | 23 phút | 2,4 GB |
| ConvNeXt-Tiny cfg_c | 0,828 | 0,878 | 7 / 13 | 23 phút | 2,4 GB |
| **EfficientNet-B0 cfg_a** | **0,845** | 0,875 | 12 / 18 | 34 phút | 1,5 GB |
| EfficientNet-B0 cfg_b | 0,828 | 0,862 | 11 / 17 | 22 phút | 1,5 GB |
| EfficientNet-B0 cfg_c | 0,828 | 0,871 | 12 / 18 | 12 phút | 1,5 GB |

→ Theo quy tắc, **EfficientNet-B0 cfg_a** được chọn: macro-F1 trên validation hơn ConvNeXt-Tiny 0,011. Lựa chọn được khóa trong `reports/phase2_evaluation/selection.json` trước khi chạy test.

## 3. Kết quả trên TEST (2.537 crop từ 45 ảnh)

| Model | Macro-F1 [CI 95%] | Accuracy | Balanced acc | Macro P / R | ROC-AUC OvR | Tham số | MB | ms/crop* |
|---|---|---:|---:|---|---:|---:|---:|---:|
| **EfficientNet-B0** (đã chọn) | 0,805 [0,757–0,843] | 0,866 | 0,806 | 0,805 / 0,806 | 0,963 | 4,0 M | 16 | 1,3 |
| ConvNeXt-Tiny | 0,832 [0,804–0,862] | 0,868 | 0,831 | 0,842 / 0,831 | 0,978 | 27,8 M | 106 | 8,7 |

\*Đo theo batch 64 trên RTX 5050, không tính thời gian đọc ảnh.

Bootstrap ghép cặp theo ảnh gốc (1.000 lần lấy mẫu): macro-F1 của ConvNeXt-Tiny trừ EfficientNet-B0 = +0,027, khoảng tin cậy [−0,012; +0,074]. Khoảng này chứa 0, nên chênh lệch **không có ý nghĩa thống kê**. Giống GĐ1 ở 640: model được validation chọn không đứng đầu trên test, nhưng hai model không phân biệt được với 45 ảnh test.

Kết quả theo từng lớp trên test (P / R / F1):

| Lớp | EfficientNet-B0 | ConvNeXt-Tiny | Số mẫu |
|---|---|---|---:|
| closed_comedo | 0,91 / 0,87 / 0,89 | 0,96 / 0,81 / 0,88 | 971 |
| open_comedo | 0,94 / 0,90 / 0,92 | 0,91 / 0,94 / 0,92 | 473 |
| papule | 0,83 / 0,89 / 0,86 | 0,83 / 0,93 / 0,87 | 867 |
| pustule | 0,71 / 0,70 / 0,71 | 0,62 / 0,75 / 0,68 | 192 |
| nodule | 0,64 / 0,68 / 0,66 | 0,89 / 0,74 / 0,81 | 34 |

Ma trận nhầm lẫn: `reports/phase2_evaluation/confusion_matrix_{efficientnet_b0,convnext_tiny}.png`. Nhầm lẫn chính là closed_comedo ↔ papule (75 crop) và closed_comedo ↔ pustule (31 crop), vì các nốt nhỏ nhô lên trông khá giống nhau. Nodule chỉ có 34 mẫu nên F1 của lớp này dao động mạnh.

## 4. Kiểm tra pipeline GĐ1 → GĐ2 trên ảnh test ACNE-DET

Script `scripts/phase2/eval_gd1_crops.py` chạy detector GĐ1 (train trên ACNE04-v2) trên 45 ảnh test của ACNE-DET, ghép box detector với nốt mụn thật (IoU ≥ 0,3), cắt crop **từ box của detector**, rồi đưa qua classifier GĐ2.

| | RT-DETR-L @1280 (đang dùng) | YOLO26s @1280 |
|---|---:|---:|
| Recall 5 lớp mụn (IoU ≥ 0,3) | **0,310** | 0,291 |
| closed_comedo | **0,115** | 0,089 |
| open_comedo | 0,047 | 0,049 |
| papule | **0,594** | 0,579 |
| pustule | **0,578** | 0,526 |
| nodule | **0,794** | 0,735 |
| Detection rơi vào một tổn thương có nhãn bất kỳ | 61% | 60% |
| Classifier: accuracy trên crop từ box detector / crop từ box nhãn | 0,825 / 0,861 | 0,839 / 0,859 |

Nhận xét:
- **RT-DETR-L tổng quát tốt hơn YOLO26s ở gần như mọi lớp** trên một dataset GĐ1 chưa từng thấy. Đây là bằng chứng ngoài ủng hộ lựa chọn của GĐ1.
- **Điểm nghẽn của pipeline là GĐ1 với comedo:** chỉ khoảng 11% closed comedo và 5% open comedo được phát hiện. Nguyên nhân là nốt quá nhỏ (khoảng 6 px khi đưa ảnh về 1280), và nhãn ACNE04-v2 có thể không bao quát comedo như ACNE-DET. Vì vậy phân bố loại mụn trong Skin Profile sẽ **lệch về nhóm viêm** (papule, pustule, nodule).
- Khi classifier nhận crop cắt từ box của detector thay vì box nhãn, accuracy giảm khoảng 2–4 điểm. Mức lệch miền do hình dạng crop như vậy là nhỏ.
- Chạy thử trên selfie (`test_img`): các nốt đỏ được phân loại là papule với độ tin cậy 0,8–0,96, nốt trắng nhỏ nhô lên là closed_comedo. Có một trường hợp nghi sai (test15). Đây chỉ là đánh giá bằng mắt, không có nhãn.

## 5. Hạn chế

- Chỉ có 45 ảnh test, nên khoảng tin cậy rộng (±0,04 macro-F1). Mỗi cấu hình chỉ chạy 1 seed.
- Không có ID bệnh nhân. Việc gom nhóm dựa trên số thứ tự liền nhau và độ giống ảnh, nên có thể vẫn sót trường hợp cùng một người.
- Ảnh VISIA được chụp chuẩn hóa, khác selfie điện thoại. Chưa có tập selfie có nhãn loại mụn để đo trực tiếp.
- Nodule chỉ có 143 mẫu train, nên kết quả lớp này kém ổn định.
- Recall comedo của GĐ1 thấp, nên GĐ2 nhận được rất ít crop comedo trong thực tế.

## 6. Bước tiếp theo đề xuất

1. **Cải thiện khả năng tìm comedo của GĐ1:** fine-tune detector trên ACNE04-v2 cộng thêm các box mụn của ACNE-DET (gộp thành một lớp `acne`, chỉ dùng ảnh train của master split), hoặc dùng tiling. Đây là cải thiện có tác động lớn nhất tới pipeline.
2. Nếu ưu tiên chất lượng hơn tốc độ, có thể dùng ConvNeXt-Tiny: nhỉnh hơn trên test, nhất là ở lớp nodule, nhưng chưa có ý nghĩa thống kê. Cần ghi rõ việc đổi model này là quyết định triển khai, không phải kết quả của quy tắc chọn.
3. Chuyển sang GĐ3 (phát hiện sẹo, 2 lớp) và dùng lại đúng master split này.

## 7. Tái lập

```powershell
.\venv\Scripts\python.exe scripts\phase2\audit_and_split.py
.\venv\Scripts\python.exe scripts\phase2\make_crops.py
.\venv\Scripts\python.exe -u scripts\phase2\train_classifier.py --model convnext_tiny --config cfg_a   # x 2 kiến trúc x 3 cấu hình
.\venv\Scripts\python.exe scripts\phase2\evaluate_classifier.py
.\venv\Scripts\python.exe scripts\phase2\eval_gd1_crops.py
.\venv\Scripts\python.exe scripts\phase2\predict_acne_type.py --face test_img\test6.jpg
```

*Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da, không phải chẩn đoán.*
