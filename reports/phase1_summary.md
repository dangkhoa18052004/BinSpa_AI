# Báo cáo Giai đoạn 1

**Trạng thái (2026-10-08): GĐ1 hoàn thành.** Benchmark 6 detector, rà soát FP theo vùng và nghiệm thu trên ảnh ngoài dataset đã xong; model triển khai cuối cùng là RT-DETR-L @1280 (xem [phase1_resolution_study.md](phase1_resolution_study.md)). Báo cáo cuối: [phase1_extended_benchmark.md](phase1_extended_benchmark.md). File này giữ lại kết quả vòng đầu (3 model) để tham khảo.

Dataset: acne04v2-clean-v1; split seed=42; {'train': 816, 'val': 175, 'test': 176}.
Manifest SHA256: `0a40262f24a967caccd4854df9649eddbb5790f73a1190c3fec51b05785e09d6`.

Ảnh trùng SHA256 được loại trước split; giữ filename đầu tiên theo thứ tự từ điển. Nhãn invalid được ghi log; cách ly toàn bộ ảnh có nhãn invalid để tránh nhãn thiếu. Không có ID bệnh nhân, nên chưa xác nhận được độc lập theo người.

| Model | Val mAP50–95 | Test precision | Test recall | Test mAP50 | Test mAP50–95 | Inference ms |
|---|---:|---:|---:|---:|---:|---:|
| yolo26s | 0.1574 | 0.5347 | 0.4217 | 0.4228 | 0.1498 | 6.26 |
| yolo11s | 0.1563 | 0.5132 | 0.4201 | 0.4111 | 0.1451 | 6.79 |
| yolo26n | 0.1308 | 0.4377 | 0.3823 | 0.3461 | 0.1189 | 3.00 |

Model chọn bằng validation: `D:\BinSpa_AI\models\phase1\best_acne_detector.pt`.
Đếm lỗi tại conf=0.25, IoU=0.5: {'true_positive': 1990, 'false_positive': 1414, 'false_negative': 3056}.

20 ảnh prediction: `reports/phase1_predictions`. Màu xanh lá: nhãn; cyan: prediction khớp; đỏ: prediction chưa khớp nhãn.

Đã rà soát `false_positive_region_review.csv` (bản lưu: `reports/metrics/phase1_false_positive_regions.csv`): mắt 5, môi 17, nốt ruồi 7, tóc/râu 38, khác 1.347 (phần lớn trên da mặt, nhiều nốt trông như lesion chưa gán nhãn). Nghiệm thu 19 ảnh ngoài dataset: [phase1_camera_review.md](phase1_camera_review.md).

Có thể chuyển sang GĐ2. Kết quả chỉ mang tính tham khảo chăm sóc da.
