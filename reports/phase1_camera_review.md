# GĐ1 — Nghiệm thu trên ảnh ngoài dataset

> **Cập nhật:** model triển khai cuối cùng là **RT-DETR-L @1280**. So sánh trên cùng 19 ảnh giữa bản 640, YOLO26s @1280 và RT-DETR-L @1280 nằm ở [phase1_resolution_study.md](phase1_resolution_study.md) (mục *Ảnh ngoài dataset*); kết quả lưu tại `reports/phase1_external_predictions_1280/`. Phần dưới đây là đánh giá cho model YOLO26s @640 cũ.

Model: `models/phase1/best_acne_detector.pt` (YOLO26s, version `aa67a569e8a95bc7`), ngưỡng confidence 0,212 (ngưỡng chọn trên validation), imgsz 640, max_det 500.
Nguồn: 19 ảnh trong `test_img/` (selfie điện thoại, ảnh tải về, 1 ảnh ghép meme), không có trong ACNE04.
Lệnh:

```
cd scripts/phase1
python predict_acne.py --source ../../test_img --conf 0.212
```

Kết quả: `reports/phase1_external_predictions/` (ảnh có box và `predictions.json`). Các ảnh này không được đưa vào `docs/` vì là ảnh khuôn mặt.

**Lưu ý về phương pháp:** các ảnh này không có nhãn chuyên gia, nên không tính được Precision/Recall. Bảng dưới đây là đánh giá bằng mắt từng ảnh. Nó giúp nhận ra các kiểu lỗi, nhưng không thay được số liệu trên tập test.

| Ảnh | Kích thước | Số box | Nhận xét |
|---|---|---:|---|
| test1 | 2560×1920 | 3 | 2 box nhỏ ở thái dương/má hợp lý; 1 box (0,26) rơi vào tay/tóc → FP ngoài mặt |
| test2 | 2560×1920 | 0 | Da ít mụn; vài chấm nhỏ ở má/hàm không được bắt |
| test3 | 2560×1920 | 0 | Da ít mụn, chính diện; không FP |
| test4 | 2560×1920 | 1 | Bắt 1 nốt trán; vài nốt nhỏ khác ở trán bị sót |
| test5 | 2560×1920 | 1 | 1 nốt giữa hai lông mày, hợp lý |
| test6 | 2560×1920 | 8 | Nốt sẩn đỏ ở má bắt đúng (0,71); trán có nhiều nốt sẩn đỏ nhỏ nhưng chỉ bắt 2 → sót nhiều |
| test7 | 2560×1920 | 7 | Cùng người với test6, cùng kiểu: má tốt, trán sót |
| test8 | 2560×1920 | 8 | Nốt sẩn đỏ bắt đúng (0,71); các box ở má hợp lý |
| test9 | 647×436 | 0 | Ảnh nhỏ, mặt chiếm ít diện tích → không phân giải được nốt mụn |
| test10 | 502×377 | 0 | Ảnh nhỏ và mờ → không có box |
| test11 | 2048×1536 | 8 | Cận mặt, các box rơi vào chấm nhỏ ở trán/má/hàm, hợp lý |
| test12 | 2048×1536 | 9 | Có 2 box chồng nhau ở má (0,67/0,59); nốt ruồi nâu gần tai **không** bị bắt nhầm |
| test13 | 2048×1536 | 10 | Cụm nốt nhỏ ở trán được bắt khá tốt; còn nhiều nốt nhỏ khác ở trán bị sót |
| test14 | 654×469 | 4 | Ảnh chân dung đã chỉnh, da mịn: 4 box confidence thấp (2 cặp chồng nhau) ở chân tóc và mép tai → FP |
| test15 | 501×399 | 9 | Ảnh nhỏ nhưng mặt chiếm gần hết khung; các box ở mũi/má trùng các chấm đỏ |
| test16 | 2048×2048 | 0 | Ảnh ghép meme có chữ, ngoài miền dữ liệu; không có box nào (đúng như mong đợi) |
| test17 | 2560×1920 | 4 | 3 box lồng nhau ở dưới mắt/cạnh mũi (trùng lặp, khả năng là bóng) → FP; nốt ruồi ở dái tai không bị bắt |
| test18 | 2560×1920 | 0 | Da sạch, chính diện; không FP |
| test19 | 2560×1920 | 3 | 1 box (0,46) ở khóe mắt trong → FP vùng mắt; 2 box ở má hợp lý |

Tổng cộng 75 box. Bằng mắt thấy rõ khoảng 8 box sai: 1 ở tay/tóc, 4 ở ảnh da mịn, 1 ở khóe mắt và 2 box trùng lặp.

## Kết luận

- **Đạt có điều kiện.** Model phát hiện được nốt mụn thật trên ảnh điện thoại chưa từng thấy, khi ảnh chụp gần, đủ sáng, cạnh dài ≥ ~1500 px và mặt chiếm phần lớn khung hình. Nốt ruồi nâu (test9, test12, test17, test18) không bị nhận nhầm. Ảnh da sạch (test3, test18) và ảnh ngoài miền dữ liệu (test16) không sinh FP.
- **Sót nhiều nốt sẩn nhỏ ở trán** (test4, test6, test7, test13). Điều này khớp với recall ~0,43 trên test và mAP50-95 thấp, vì lesion trong ảnh chỉ khoảng 8 px khi input là 640.
- **Ảnh độ phân giải thấp hoặc chụp xa** (test9, test10) không cho kết quả. Ứng dụng cần hướng dẫn người dùng chụp gần, đủ sáng và giữ ảnh gốc độ phân giải cao.
- **Box trùng hoặc lồng nhau** (test12, test17): khi ghép với GĐ2 nên gộp các box lồng nhau trước khi crop, để không phân loại cùng một tổn thương hai lần.
- FP ở ngoài mặt và ở vùng mắt có xảy ra nhưng ít. Kết quả này nhất quán với phần rà soát vùng FP trên tập test (`reports/phase1_extended_benchmark.md` §4).

*Kết quả phân tích hình ảnh chỉ mang tính tham khảo chăm sóc da, không phải chẩn đoán.*
