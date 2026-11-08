# Kết quả tổng hợp công khai

Các file ở đây chỉ chứa thống kê tổng hợp, checkpoint hash và ma trận. Ảnh khuôn mặt, ID ảnh/người, tọa độ box, dự đoán từng ảnh, tài liệu cá nhân và đường dẫn máy cá nhân không được xuất.

- [Báo cáo chính và các hình hiển thị trực tiếp](../../README.md)
- [Số liệu các thực nghiệm, gồm bench_v1 trong khóa `bench_v1`](metrics.json)
- [Ma trận dưới dạng số nguyên](confusion_matrices.json)
- [PDF 18 trang](../../docs/figures/research/confusion_matrices/all_models_confusion_matrices.pdf)

## Kết quả mới nhất

README chính trình bày benchmark `bench_v1` (3 detector × 2 classifier, kết quả qua seed 42/43/44) và đối chiếu với pipeline demo RT-DETR-L + 2×EfficientNet-B0. Pipeline chọn trên validation là YOLO26s + ConvNeXt-Tiny; cấu hình demo vẫn giữ RT-DETR-L + 2×EfficientNet-B0. Detector seed 43/44 chỉ train lại chặng C; T1–T4 là các bộ đánh giá phát triển đã được xem trước. Ma trận bên dưới thuộc các đợt lịch sử, không phải ma trận mới của bench_v1.

## Ma trận từng checkpoint — các đợt lịch sử

| Giai đoạn | Model | Tập dữ liệu | Hình |
| --- | --- | --- | --- |
| GĐ1 | Faster R-CNN @640 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_faster_rcnn.png) |
| GĐ1 | Faster R-CNN @640, full50 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_faster_rcnn_full50.png) |
| GĐ1 | RT-DETR-L @640 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_rtdetr.png) |
| GĐ1 | RT-DETR-L @640, full100 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_rtdetr_full100.png) |
| GĐ1 | YOLO11s @640 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_yolo11s.png) |
| GĐ1 | YOLO11s @640, full100 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_yolo11s_full100.png) |
| GĐ1 | YOLO26n @640 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_yolo26n.png) |
| GĐ1 | YOLO26s @640 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_yolo26s.png) |
| GĐ1 | YOLOv5su @640 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_yolov5s.png) |
| GĐ1 | RT-DETR-L @1280 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_rtdetr_1280.png) |
| GĐ1 | YOLO11s @1280 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_yolo11s_1280.png) |
| GĐ1 | YOLO26s @1280 | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_yolo26s_1280.png) |
| GĐ1 | RT-DETR-L @1280, fine-tune | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_rtdetr_ft.png) |
| GĐ1 | YOLO26s @1280, fine-tune | ACNE04 — test | [Mở](../../docs/figures/research/confusion_matrices/phase1/phase1_yolo26s_ft.png) |
| GĐ2 | EfficientNet-B0 | ACNEDET — crop test | [Mở](../../docs/figures/research/confusion_matrices/phase2/phase2_efficientnet_b0.png) |
| GĐ2 | ConvNeXt-Tiny | ACNEDET — crop test | [Mở](../../docs/figures/research/confusion_matrices/phase2/phase2_convnext_tiny.png) |
| GĐ1 | RT-DETR-L @1280, fine-tune | DetectDataSkin-2 — thăm dò | [Mở](../../docs/figures/research/confusion_matrices/phase1/external_phase1_rtdetr.png) |
| GĐ2 | EfficientNet-B0 | DetectDataSkin-2 — crop thăm dò | [Mở](../../docs/figures/research/confusion_matrices/phase2/external_phase2_efficientnet.png) |

## Tái lập và phạm vi

`python scripts/prepare_publication.py` xuất lại các artifact này từ báo cáo đánh giá lưu riêng. Script không chạy model hay chọn lại ngưỡng. Muốn chấm lại từ đầu cần dữ liệu được cấp quyền, manifest chia tập và checkpoint tương ứng; các đầu vào đó không nằm trong bản công khai.

Dữ liệu DetectDataSkin-2 là thăm dò với mapping nhãn tạm thời. Không diễn giải kết quả này như kiểm định lâm sàng hoặc kiểm định độc lập theo bệnh nhân.
