# Kế hoạch tăng độ tin cậy — chỉ phát hiện và phân loại mụn

## Mục tiêu và phạm vi

Pipeline hiện tại: **ảnh → phát hiện vị trí mụn → crop tổn thương → loại mụn**. Giữ năm lớp `closed_comedo`, `open_comedo`, `papule`, `pustule`, `nodule`. Phát hiện sẹo được bỏ khỏi phạm vi hiện tại. Không có bước huấn luyện mới nào được thực hiện chỉ vì lập kế hoạch này.

Bạn không phải tự chụp hoặc thu thập ảnh. Hướng chính là khai thác dữ liệu đã gán nhãn, xác minh nguồn và đánh giá đúng cách. Cần phân biệt: **chứng minh đáng tin cậy hơn** bằng kiểm tra tốt hơn; **làm model tốt hơn** bằng cải tiến được kiểm chứng. Thêm dữ liệu hoặc train lâu hơn không tự bảo đảm cả hai.

## Điểm xuất phát

| Hạng mục | Bằng chứng hiện có | Điểm cần xử lý |
| --- | --- | --- |
| GĐ1 RT-DETR-L fine-tune | Test ACNE04: P 0,6018; R 0,5224; F1 0,5593 | Bỏ sót nhiều tổn thương nhỏ; báo dư còn đáng kể |
| GĐ2 EfficientNet-B0 | 2.537 crop ACNEDET: accuracy 86,60%; macro-F1 0,8053 | Crop nhãn khác crop từ detector; chỉ 34 mẫu nodule trên test |
| DetectDataSkin-2 | 1.275 ảnh đã sàng lọc: GĐ1 F1 0,1906; GĐ2 accuracy 22,35% | Khác miền ảnh và ontology; nhãn chưa được kiểm tra chuyên môn |
| Confidence GĐ2 trên tập thăm dò | Trung bình 72,59%; ECE 0,5024 | Dự đoán sai vẫn có thể rất tự tin |
| Độc lập theo người | Có nhóm suy đoán; không có ID người được xác minh | Không được gọi là kiểm định trên bệnh nhân độc lập |

## Tiến độ (cập nhật 15/10/2026)

| Hạng mục | Trạng thái | Kết quả chính |
| --- | --- | --- |
| Benchmark lại cả pipeline | Xong | 5-fold CV theo nhóm trên 275 ảnh ACNEDET: pipeline 2 giai đoạn hơn detector 5 lớp (macro-F1 0,401 so với 0,363, CI không chứa 0) |
| Chống trùng | Xong | pHash cộng 7 biến thể xoay/lật: không có cặp trùng giữa test ACNEDET và ACNE04 |
| Cải thiện GĐ2 với ảnh kém chất lượng | Xong | Classifier robust: ngang cfg_a trên ảnh sạch; mAP50 trên ảnh mờ/nhiễu/JPEG tăng từ 0,099 lên 0,160 |
| Hiệu chỉnh confidence và từ chối | Xong, đã triển khai | Temperature T = 0,80, ngưỡng τ = 0,87 (`type_status` confident/uncertain). CV: ECE 0,062 → 0,013; hiển thị loại cho 64% nốt, đúng 94,7% [93,7–95,8]. **Không có tác dụng ngoài miền:** DetectDataSkin-2 ECE 0,51 → 0,57, phần hiển thị chỉ đúng 23,7% |
| Kiểm tra ảnh đầu vào | Xong, đã triển khai (cập nhật 20/10) | `analysis_status` ok / retake_photo / location_only. Bản khóa trước không đạt: kiểm tra nhiễu chặn nhầm 20% ảnh test sạch. Bản triển khai nới ngưỡng nhiễu (post-hoc, ngưỡng lấy từ val): chặn nhầm 0,5% trên 190 ảnh độc lập; trên test, F1 ảnh được cho qua 0,33 so với ảnh bị chặn 0,16. Mọi selfie có ≥ 3 nốt bị đánh giá là khác miền |
| Xác minh ACNEDET gốc, xin AcneSCU | Chưa làm | Cần người dùng tự tải hoặc gửi yêu cầu |
| Nhiều seed, model card | Chưa làm | |

## Bảng kế hoạch thực hiện

Ước tính 3–5 tuần làm việc bán thời gian, chưa tính thời gian xin dữ liệu. Đây là lịch dự kiến, không phải cam kết sẽ đạt một mức accuracy cụ thể.

| Ưu tiên / thời gian | Công việc | Cách làm khi không tự thu thập ảnh | Sản phẩm và điều kiện chuyển bước |
| --- | --- | --- | --- |
| P0 · Ngày 1 | Chốt hai giai đoạn và lưu baseline | Giữ checkpoint và confidence hiện tại; ghi hash, split và cách crop | Có `project_scope.json`; số liệu baseline tái kiểm tra được; chưa đổi model |
| P1 · Ngày 2–4 | Xác minh nguồn ACNEDET | Lấy bản LabelMe từ repo tác giả; so tên lớp, ảnh, polygon/box và số annotation với mirror đang dùng | Có bảng đối chiếu nguồn. Bản gốc trùng ảnh chỉ giúp xác minh nhãn, không là test độc lập |
| P1 · Ngày 3–7 | Xin hoặc chọn dữ liệu bổ sung | Ưu tiên xin AcneSCU từ tác giả; khảo sát nguồn khác có box và nhãn loại. Không mua/tải hàng loạt chỉ vì số ảnh lớn | Bảng nguồn, quyền truy cập, nhãn, kiểu camera, patient ID và nguy cơ chồng dữ liệu; loại nguồn không phù hợp |
| P1 · Tuần 1 | Chống trùng và khóa cách chia tập | Kiểm tra hash, ảnh gần trùng, biến thể xoay/lật và nguồn gốc; gom cùng người nếu có ID. Augmentation chỉ sau chia tập | Không có trùng đã phát hiện giữa các tập. Thiếu patient ID thì ghi giới hạn, không gọi nhóm suy đoán là người thật |
| P1 · Tuần 1–2 | Kiểm tra ontology và nhãn | Đối chiếu nhãn gốc trên một pilot, gợi ý 50–100 ảnh và 30–50 tổn thương mỗi lớp nếu có; kiểm tra cả mẫu chọn ngẫu nhiên và ca nghi lỗi | Mapping năm lớp có căn cứ. Mẫu nhỏ chỉ giúp tìm lỗi, không chứng minh toàn bộ dataset đúng |
| P2 · Tuần 2 | Benchmark lại cả pipeline | Chạy GĐ1 → crop từ box dự đoán → GĐ2; ghép box đúng lớp, IoU ≥ 0,5; báo thêm IoU 0,3 như phân tích độ nhạy | Có TP/FP/FN, AP và F1 theo lớp, recall, confusion matrix; không chỉ accuracy của crop nhãn |
| P2 · Tuần 2–3 | Cải thiện GĐ1 có kiểm soát | Thử từng thay đổi trên train/validation: tiling, giữ độ phân giải tổn thương, fine-tune dữ liệu đã xác minh, thêm âm tính thật có nhãn | So sánh với baseline trên cùng validation, cùng evaluator. Recall tăng phải xét đồng thời FP và latency |
| P2 · Tuần 2–3 | Cải thiện GĐ2 và lệch kiểu crop | Train thêm crop từ box detector ghép với nhãn thật **chỉ ở train**; thử độ rộng ngữ cảnh, loss/sampler lớp hiếm và augmentation nhẹ | Báo macro-F1 và F1 từng lớp. Không tự đổi Cysts thành nodule; không lấy dự đoán của model làm nhãn test |
| P2 · Tuần 3 | Kiểm tra độ ổn định | Với cấu hình đã chọn trên validation, chạy 3 seed nếu tài nguyên cho phép; dùng group CV hoặc nested group CV trên phần phát triển | Có trung bình/độ lệch và bootstrap theo người nếu có ID, nếu không theo nhóm/ảnh với giới hạn ghi rõ |
| P2 · Tuần 3–4 | Hiệu chỉnh confidence và từ chối ca khó | Fit temperature scaling trên tập calibration riêng; chọn ngưỡng chấp nhận theo validation, đánh giá accuracy/precision **cùng coverage** | Có reliability diagram, ECE, Brier/NLL và tỷ lệ từ chối. Không đặt ngưỡng tùy tiện 0,8 rồi coi là xác suất đúng 80% |
| P3 · Tuần 4–5 | Kiểm định cuối và quyết định dùng | Chỉ dùng một cohort mới chưa xem để xác nhận nếu xin được; nếu không, báo kết quả hiện có là nghiên cứu/thăm dò | Có model card nêu miền sử dụng, lớp chưa đủ bằng chứng và giới hạn. Không tuyên bố đã đáng tin trên selfie nếu chưa có test phù hợp |

## Các nguồn dữ liệu nên dùng thế nào

| Nguồn | Cách tiếp cận | Vai trò hợp lý | Giới hạn |
| --- | --- | --- | --- |
| ACNE04 + ACNE04-v2 hiện có | Giữ bộ đã làm sạch; dùng các script hiện có | Baseline detector, dữ liệu train bổ sung khi đúng quyền sử dụng | Đã dùng train/test; không hỗ trợ trực tiếp năm loại mụn |
| ACNEDET từ tác giả | [Repo DSDH](https://github.com/INno-Vation/Decoupled-Sequential-Detection-Head) liên kết bản LabelMe trên Baidu, mã truy cập `foa7` | Kiểm tra nguồn và nhãn của mirror; train/phân tích trong miền ảnh gốc | Không tự động là dữ liệu mới hoặc được phép dùng thương mại; chuyển polygon → box phải giữ mapping đúng |
| AcneSCU | [Repo tác giả](https://github.com/pingguokiller/acnedetection) hướng dẫn liên hệ `jiangxian@scu.edu.cn` | Ứng viên bổ sung cho phân loại tổn thương nếu được cấp quyền và nhãn tương thích | Cần xin truy cập; tác giả giới hạn phi thương mại. Phải kiểm tra có trùng cohort ACNEDET không |
| DetectDataSkin-2 hiện có | Giữ bản thăm dò đã lọc; rà soát nhãn và nguồn trước khi cân nhắc train | Tìm lỗi chuyển miền, xây dựng câu hỏi kiểm tra ontology | Đã xem kết quả trên toàn bộ tập đã lọc. Chia lại sau đó không tạo một test mới chưa từng xem |
| Dataset Roboflow khác | Ưu tiên ảnh gốc không augmentation, nhãn box + loại tổn thương, nguồn gốc và người gán nhãn rõ | Ứng viên cần audit trước sử dụng | Có thể là mirror hoặc biến thể của cùng ACNE04/ACNEDET; nhiều ảnh không đồng nghĩa nhiều người |

Các đường dẫn trên được kiểm tra tại thời điểm lập kế hoạch. Bài DSDH giới thiệu ACNEDET và dẫn repo tác giả: [Wei et al., Knowledge-Based Systems](https://www.sciencedirect.com/science/article/pii/S0950705123010535). Việc có link không bảo đảm bạn tải được ngay hoặc có quyền tái phân phối ảnh.

## Nếu không quen chuyên gia và không có ngân sách

1. Ưu tiên nhãn của nguồn nghiên cứu gốc có mô tả gán nhãn. Việc đối chiếu mirror giúp phát hiện đổi tên lớp, mất annotation và conversion sai; không thay thế một bác sĩ xác minh.
2. Bạn có thể tự kiểm tra ảnh hỏng, trùng ảnh, box lệch hoặc mapping sai. Với loại mụn khó phân biệt, giữ nhãn gốc hoặc đánh dấu chưa xác minh; không coi đánh giá bằng mắt là ground truth chuyên môn.
3. Chạy cross-validation theo nhóm, nhiều seed, kiểm tra crop và đo pipeline trên dữ liệu đang có. Những bước này tăng bằng chứng kỹ thuật và khả năng tái lập, không tạo bằng chứng độc lập trên selfie mới.
4. Hiệu chỉnh confidence chỉ trên nhãn đủ tin cậy. Nếu không có nhãn miền selfie đã xác minh, không khẳng định calibration từ ảnh VISIA còn đúng cho selfie.
5. Với ca không đủ bằng chứng, thiết kế đầu ra “chưa đủ tin cậy để phân loại”, hoặc chỉ trả vị trí nghi mụn. Confidence thấp/độ bất đồng model có thể giúp sàng lọc, nhưng không phát hiện hết ca sai tự tin cao.

Nếu có ngân sách nhỏ, ưu tiên thuê người có chuyên môn kiểm tra pilot và hướng dẫn nhãn trên **ảnh đã được cấp quyền**, trước khi thuê gán nhãn hàng nghìn ảnh. Người thao tác công cụ annotation thông thường có thể khoanh box nhưng không tự bảo đảm phân biệt đúng loại mụn.

Mẫu yêu cầu dữ liệu (bạn tự gửi, dự án không gửi tự động):

> Dear authors, I am developing a research prototype for acne lesion detection and five-category classification. I would like to request access to AcneSCU for non-commercial research. Could you share the access procedure, annotation definitions, subject identifiers for splitting, and usage conditions? I will cite your work and will not redistribute face images.

## Giao thức tránh đánh giá quá lạc quan

- Train học trọng số; validation chọn cấu hình, checkpoint và detector confidence. Calibration hoặc phần validation dành riêng học bộ hiệu chỉnh và ngưỡng từ chối. Test cuối chỉ đo cấu hình đã khóa.
- Giữ toàn bộ ảnh của một người trong một tập nếu có patient ID. [GroupKFold/StratifiedGroupKFold](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) hỗ trợ chia theo nhóm; dùng nhóm suy đoán thì phải ghi rõ mức bằng chứng đó.
- Không chọn model, ghép lớp hoặc ngưỡng bằng cách xem điểm test rồi báo lại như test chưa từng xem. Khi dùng DetectDataSkin-2 đã khảo sát để phát triển, kết quả tiếp theo trên nguồn đó là thăm dò.
- Không train rồi chấm trên ảnh augmentation của cùng ảnh gốc; không để crop từ cùng ảnh đi sang nhiều split. [Hướng dẫn chẩn đoán của Roboflow](https://github.com/roboflow/computer-vision-skills/blob/main/skills/roboflow-training-and-evaluation/model-diagnosis.md) nhấn mạnh near-duplicate và test đại diện cho miền sử dụng.
- Ảnh không có annotation chưa chắc là da sạch. Cần nhãn âm tính đáng tin trước tính tỷ lệ báo sai trên ảnh da sạch hoặc train lớp background.
- Pseudo-label và ảnh tạo sinh chỉ có thể là đầu vào thử nghiệm cho train, không làm ground truth cho test.
- Đánh giá confidence cần dữ liệu riêng với model đã fit; [tài liệu calibration](https://scikit-learn.org/stable/modules/calibration.html) giải thích vì sao fit calibrator trên train có thể lạc quan.

## Điều kiện để nói “đáng tin cậy hơn”

Không có một ngưỡng accuracy chung tự động biến model thành đáng tin. Trước thử nghiệm, phải đặt tiêu chí gắn với mức bỏ sót/báo dư bạn chấp nhận và nguồn ảnh định dùng.

| Khía cạnh | Bằng chứng phải báo |
| --- | --- |
| Phát hiện | Precision, recall, F1/AP; lỗi theo kích thước nốt và theo nguồn ảnh |
| Loại mụn | Macro-F1, P/R từng lớp và support; không dùng accuracy tổng để che lớp hiếm |
| Pipeline | Ghép đúng box và đúng lớp; tính cả nốt bỏ sót và box báo dư |
| Confidence | Reliability diagram, ECE, Brier/NLL; không diễn giải Brier/ECE đơn lẻ như chứng nhận độ tin cậy |
| Từ chối | Accuracy/precision của phần được chấp nhận **và coverage trên toàn bộ mẫu đủ điều kiện**; phân bố lớp bị từ chối |
| Ổn định | Kết quả nhiều seed; CI lấy mẫu theo người hoặc đơn vị độc lập phù hợp |
| Tổng quát | Cohort khác có nhãn phù hợp và kiểm tra chồng dữ liệu; nếu không có thì nêu giới hạn miền sử dụng |

Ví dụ mục tiêu thử nghiệm nội bộ: precision ≥ 0,80 cho các dự đoán được hiển thị **và** báo coverage, lớp hiếm, CI và F1 toàn pipeline. Đây là mục tiêu phải chốt trước trên validation, không phải chuẩn y khoa, không bảo đảm đạt được và không cho phép bỏ gần hết mẫu khó rồi gọi model là tốt.

## Việc nên làm ngay

**Ba việc đầu tiên:** xác minh bản ACNEDET gốc; viết báo cáo lỗi của pipeline hiện tại theo từng lớp; chọn dữ liệu bổ sung có nguồn và nhãn phù hợp. Chưa fine-tune toàn bộ DetectDataSkin-2 khi mapping và nhãn còn nghi vấn. Không đổi model hoặc ngưỡng hiện tại chỉ vì đang lập kế hoạch.
