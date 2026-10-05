# Báo cáo Giai đoạn 1

## Trạng thái

Đã hoàn thành pipeline dữ liệu và chuẩn bị train/evaluate/inference. Smoke test YOLO26n 10 epoch đã hoàn tất; pipeline benchmark ba model đang được tiếp tục trong nền. Chưa nghiệm thu chất lượng model.

## Dữ liệu đã kiểm tra

- ACNE04: 3 shard Parquet, 1.457 ảnh. Kiểm tra toàn bộ ảnh cũ với nguồn khi chạy extract: existing=1457, exported=0, duplicated filename=0, failed=0. Giữ nguyên ảnh đã trích xuất.
- ACNE04-v2: 1.204 ảnh, 32.443 lesion. Mapping 1.204/1.204, missing=0; ảnh đọc được và kích thước khớp JSON.
- Loại 36 ảnh trùng SHA256, giữ filename đầu tiên theo thứ tự từ điển. Log khác biệt nhãn tại `reports/phase1_data/duplicate_images.csv`.
- Cách ly `levle2_130.jpg`: 49 annotation lỗi tọa độ ngoài ảnh, 18 annotation còn lại không đưa vào train để tránh nhãn thiếu. Không sửa nguồn, không đoán tọa độ.
- Clip 59 bbox hợp lệ vượt biên trước normalization.
- Dataset sạch: **1.167 ảnh / 31.498 lesion**. Train **816**, val **175**, test **176**, seed **42**.
- Dataset/version/hash: `datasets/phase1_acne_detection/manifest.json`. Kiểm tra hash ảnh/nhãn, bbox normalized, file thiếu/thừa và trùng nội dung xuyên split đã qua.
- 90 ảnh visualization (30/split), 3 contact sheets và 3 ảnh chi tiết đã được Codex xem trước train. Bbox không lệch hệ tọa độ có hệ thống. Có lesion rất nhỏ, ảnh macro/ảnh xoay/ảnh ghép và ánh sáng khác nhau; giữ các đặc điểm này trong phân tích hạn chế dataset.
- Review gắn hash manifest: `datasets/phase1_acne_detection/label_review.json`.

## Môi trường và kiểm thử

Python 3.12.7, torch 2.11.0+cu128, torchvision 0.26.0+cu128, ultralytics 8.4.173. RTX 5050 Laptop ~8 GB VRAM, CUDA available; phép nhân ma trận CUDA đã chạy thành công. `pip check` không có dependency lỗi. Phiên bản package được lưu trong `requirements-phase1-lock.txt`.

Sáu regression tests bảo vệ bbox clip, annotation tọa độ lỗi/radius lỗi, ảnh trùng, ảnh thiếu, mutation dataset và matching prediction/GT đã qua. `test_phase1.py` có fixture ảnh có một phần nhãn lỗi để đảm bảo cách ly cả ảnh.

## Train/evaluate đã triển khai

- Smoke: YOLO26n, imgsz 640, batch 8, 10 epoch.
- Đánh giá smoke gần nhất trên validation (175 ảnh): Precision 0.3833, Recall 0.3205, mAP50 0.2546, mAP50–95 0.0841. Chi tiết tại `reports/phase1_smoke.json`; đây chưa phải kết quả model cuối.
- Benchmark: YOLO26n, YOLO26s, YOLO11s; seed 42, imgsz 640, tối đa 100 epoch/model, patience 20, AdamW lr0=.001, AMP, workers=0. OOM giảm batch 4/2/1 và ghi cấu hình thực tế.
- Lần smoke với workers=2 dừng tại epoch đầu do OpenCV thiếu bộ nhớ CPU khi đọc ảnh lớn; đổi workers=0 và chạy lại. Run lỗi được giữ nguyên để kiểm tra, không tính là kết quả train.
- Benchmark khởi động lần đầu bị kẹt tại FP16 convolution trong kiểm tra AMP lặp lại (xác nhận bằng py-spy). Run đó được lưu với hậu tố `_amp_stalled`. Benchmark hiện dùng certificate AMP từ smoke 10 epoch cùng GPU/phiên bản thư viện; không đổi AMP training và dừng nếu validation loss/metrics không finite. Chi tiết tại `phase1_amp_certificate.json`, `phase1_amp_check_policy.json`.
- Job đã chạy qua các epoch đầu của benchmark YOLO26n trong nền. Log hiện hành: `reports/logs/phase1_20261005_124954.stdout.log`; tiến độ JSON là nguồn trạng thái cập nhật, không dùng số epoch ghi trong báo cáo này làm trạng thái trực tiếp.
- Chọn model trên validation bằng mAP50–95, Recall, latency; chốt metadata trước khi test cả 3 model. Test không dùng chọn hyperparameter/model.
- Khi train xong: model/config/version tại `models/phase1`; CSV tại `reports/metrics`; 20 prediction overlays và CSV kiểm tra lỗi tại `reports/phase1_predictions`. Chưa tạo metric hoặc model kết quả giả.
- Tiến độ thực tế: `reports/phase1_training_status.json`. Tài liệu chạy/resume: `scripts/phase1/README.md`.
- Script `predict_acne.py` đã chạy thử thành công trên một ảnh validation bằng checkpoint smoke, xuất JSON bbox/confidence/version/disclaimer và ảnh minh họa tại `reports/phase1_smoke_prediction`. Đây chưa phải model được chọn sau benchmark.

## Còn cần nghiệm thu

1. Hoàn thành benchmark, kiểm tra metric/prediction thật của ba model; smoke đã hoàn tất.
2. Kiểm tra unmatched predictions theo mắt/môi/nốt ruồi/tóc/râu. Pipeline xuất CSV và hỗ trợ tổng hợp vùng bằng `summarize_error_regions.py`; không tự gán vùng khi chưa xem ảnh.
3. Kiểm tra ảnh camera ngoài dataset. Chưa có đường dẫn ảnh thực tế được cung cấp.
4. Hash chỉ kiểm tra duplicate nội dung chính xác; nguồn không có patient ID nên chưa xác nhận split độc lập theo người, near-duplicate hoặc domain camera thực tế.

Chưa tự động triển khai GĐ2. Kết quả chỉ mang tính tham khảo chăm sóc da.
