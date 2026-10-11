# Giai đoạn 1: phát hiện vị trí mụn

Chạy PowerShell tại thư mục gốc của repository. Các lệnh bên dưới dùng trực tiếp venv; dữ liệu và checkpoint phải được chuẩn bị riêng theo README chính.

## Dữ liệu

- ACNE04: 3 shard Parquet, 1.457 ảnh; nguồn `ACNE04/data`, ảnh `raw/ACNE04_images`.
- ACNE04-v2: 1.204 filename, 32.443 annotation. Chỉ dùng nhãn v2, không cộng hai nguồn thành hai bộ ảnh độc lập.
- Map 100% filename, kiểm tra ảnh hỏng/kích thước trước conversion.
- Dedupe SHA256 trước split; mỗi nhóm giữ filename hợp lệ đầu tiên theo thứ tự từ điển. Không gộp nhãn; log khác biệt trong `duplicate_images.csv`.
- `levle2_130.jpg` có 49 annotation tọa độ ngoài ảnh và 18 annotation hợp lệ. Cách ly toàn bộ ảnh để tránh supervision thiếu, không đoán hệ số sửa tọa độ.
- Clip bbox vượt biên rồi tính lại tâm/kích thước và ghi log.
- Dataset sạch: 1.167 ảnh, 31.498 lesion; train/val/test **816/175/176**, seed 42, xấp xỉ 70/15/15.
- Manifest chứa hash ảnh/nhãn/version. Kiểm tra thiếu/thừa file, mutation, bbox và leakage trước train/eval.
- Không có patient ID; SHA256 chưa chứng minh độc lập theo người hoặc near-duplicate.
- Dataset đã tạo giữ cố định. Đổi nguồn/chính sách thì dùng output mới và review lại. `datasets/phase1_acne_detection_before_quarantine` là bản nháp trước cách ly, **không dùng train**.

## Môi trường

Python 3.12.7; CUDA torch đã kiểm tra trên RTX 5050. Cài CUDA trước:

```powershell
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements-phase1.txt
.\venv\Scripts\python.exe -m pip check
```

Phiên bản thực tế: `requirements-phase1-lock.txt`, `reports/phase1_environment.json`. Cài lại bằng CUDA index trước rồi file lock. Cache/settings nằm trong `.cache`.

## Chuẩn bị

```powershell
.\venv\Scripts\python.exe scripts\phase1\inspect_acne04_parquet.py
.\venv\Scripts\python.exe scripts\phase1\extract_acne04_images.py
.\venv\Scripts\python.exe scripts\phase1\check_v2_mapping.py
.\venv\Scripts\python.exe scripts\phase1\convert_v2_to_yolo.py
.\venv\Scripts\python.exe scripts\phase1\split_phase1.py
.\venv\Scripts\python.exe scripts\phase1\visualize_labels.py
.\venv\Scripts\python.exe scripts\phase1\test_phase1.py
```

Extract không ghi đè. Kiểm tra ảnh cũ bằng dimensions/thumbnail RGB (cho phép sai khác nhỏ do nén JPEG). Ảnh mới giữ bytes nguồn nếu format phù hợp. Xuất mapping shard/row/filename và exported/existing/duplicated/failed; lỗi trả exit code khác 0.

Xem 30 ảnh mỗi split và contact sheets tại `reports/phase1_label_check`. Chỉ ghi review sau khi thật sự xem ảnh:

```powershell
.\venv\Scripts\python.exe scripts\phase1\visualize_labels.py --reviewed-by TenNguoiKiemTra --notes 'Mo ta kiem tra bbox thuc te'
```

Train từ chối nếu thiếu review hoặc manifest thay đổi. Annotation gốc chỉ đọc; log tại `reports/phase1_data`.

## Train

```powershell
# Smoke 10 epoch; chỉ validation.
.\venv\Scripts\python.exe -u scripts\phase1\train_phase1.py --smoke
# Ba model, tối đa 100 epoch/model, patience=20.
.\venv\Scripts\python.exe -u scripts\phase1\train_phase1.py
# Hoặc tuần tự hai bước trong nền, cửa sổ ẩn, có log:
.\scripts\phase1\launch_phase1.ps1
# Nếu smoke standalone đang chạy, xếp pipeline chờ nó hoàn tất rồi benchmark:
.\scripts\phase1\launch_phase1.ps1 -WaitForTraining
```

Cùng seed 42, imgsz 640, AdamW lr0=.001, AMP, batch 8, workers=0 (tránh nhân bản bộ nhớ Python/PyTorch trên Windows). OOM thử 4/2/1 và ghi batch thực tế, kể cả khi Ultralytics tự giảm batch. Pretrained: `models/phase1/pretrained`; checkpoints/config: `runs/phase1`. Early stopping có thể làm số epoch thực tế khác nhau.

Smoke không cache. Benchmark mặc định `--cache auto`: cache ảnh đã resize trong RAM khi ước tính đủ bộ nhớ và còn dự phòng 4 GiB; quyết định một lần cho cả ba model, log tại `reports/phase1_cache_policy.json`. Có thể ép `--cache none`. Ultralytics tự kiểm tra RAM lần nữa; cache thực tế lưu trong completed.json. Seed cố định không bảo đảm kết quả bitwise giống nhau giữa môi trường/phiên bản thư viện, đặc biệt khi RAM cache.

AMP được kiểm tra ở smoke. Benchmark có thể dùng lại certificate khi GPU/Python/torch/torchvision/Ultralytics khớp chính xác; vẫn train bằng AMP và dừng nếu validation loss/metrics không finite. Trên máy này lần kiểm tra AMP lặp lại bị kẹt ở FP16 convolution dù smoke đã qua 10 epoch; certificate lưu tại `reports/phase1_amp_certificate.json`, quyết định tại `phase1_amp_check_policy.json`. `--recheck-amp` buộc kiểm tra lại. Không sửa thư viện cài đặt, wrapper chỉ áp dụng trong tiến trình train và khôi phục sau mỗi model.

Tiến độ: `reports/phase1_training_status.json`; logs: `reports/logs`; PID: `reports/phase1_process.json`. Giữ máy hoạt động. Run hoàn thành khớp config/hash được dùng lại; run dở cần resume:

```powershell
.\venv\Scripts\python.exe -u scripts\phase1\train_phase1.py --resume 'runs\phase1\TEN_RUN\weights\last.pt'
.\venv\Scripts\python.exe -u scripts\phase1\run_phase1.py
```

Không resume khi job còn chạy. Nếu lỗi trước khi có last.pt, kiểm tra nguyên nhân và lưu lại/đổi tên thư mục trước khi chạy lại; script không tự xóa kết quả.

## Đánh giá

Chọn YOLO26n/YOLO26s/YOLO11s trên validation bằng mAP50–95, Recall, rồi inference latency. Chốt metadata trước test cả 3 model; không đổi winner/hyperparameter bằng test.

- CSV tại `reports/metrics`: validation benchmark/ranking và test benchmark.
- Model/data.yaml/train_config.yaml/hash/version tại `models/phase1`.
- mAP conf=.001; đếm lỗi conf=.25, matching IoU=.5, mỗi GT tối đa một prediction. Latency là metrics.speed trên GPU/imgsz thực tế, chưa phải deployment CPU.
- Benchmark/evaluation/inference dùng max_det=500 vì dataset có ảnh chứa 434 lesion. Smoke đầu tiên dùng max_det=300 (lưu trong args.yaml); không dùng smoke chọn winner.
- 20 ảnh prediction tại `reports/phase1_predictions`: xanh lá=GT, cyan=prediction khớp, đỏ=prediction chưa khớp nhãn.
- Gán vùng trong `false_positive_region_review.csv`: eyes/lips/mole/hair_beard/other. Annotation thiếu có thể tạo false positive biểu kiến.

```powershell
.\venv\Scripts\python.exe scripts\phase1\evaluate_phase1.py --split test --error-analysis
.\venv\Scripts\python.exe scripts\phase1\predict_acne.py --source 'test_img\selfie.jpg'
```

Inference trả bbox/confidence/lesion_count/version/disclaimer, không trả loại mụn/severity. Camera ngoài dataset và kiểm tra lỗi theo vùng là điều kiện nghiệm thu thực tế; không tự chuyển GĐ2.

Sau khi gán vùng trong CSV, tổng hợp bằng `python scripts/phase1/summarize_error_regions.py`; báo riêng các detection chưa được review.

## Nguồn

[PyTorch CUDA](https://pytorch.org/get-started/locally/), [YOLO26](https://docs.ultralytics.com/models/yolo26/), [Train](https://docs.ultralytics.com/modes/train/), [Val](https://docs.ultralytics.com/modes/val/).

README nguồn ACNE04 ghi academic/research use only; benchmark này chưa xác nhận quyền dùng thương mại.
