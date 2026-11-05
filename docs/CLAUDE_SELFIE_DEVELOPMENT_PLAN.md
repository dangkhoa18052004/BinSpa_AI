# Kế hoạch phát triển BinSpa AI cho demo selfie nội bộ

Ngày lập: 04/11/2026, sửa lại cùng ngày theo ràng buộc thực tế.

## 1. Bối cảnh và mục tiêu

**Ràng buộc thật, không giả định khác:**

- Hệ thống chỉ chạy **localhost**, phục vụ demo nội bộ của spa, không public ra ngoài.
- **Không có chuyên gia** gán hoặc duyệt nhãn. Chỉ dùng các bộ dữ liệu đã có tại máy (mục 3). Không chờ dữ liệu mới.
- Đầu vào phải **thoải mái như chế độ demo hiện tại**: khách chụp bằng điện thoại bất kỳ, ánh sáng thường, góc tự nhiên. Không bắt chụp giống VISIA, không chặn ảnh vì "khác tập train" hoặc vì ít mụn.

**Mục tiêu:** khi demo, những gì AI **hiển thị** (vị trí nốt, nhóm mụn, loại mụn) phải đúng trong phạm vi sai số đã khóa trước (mục 6). Mọi con số về độ chính xác và độ tin cậy phải đúng với cách đo và nguồn nhãn, không phóng đại.

Đầu ra vẫn là vị trí nốt và năm loại `closed_comedo`, `open_comedo`, `papule`, `pustule`, `nodule`. Không mở rộng sang sẹo, độ nặng hoặc tư vấn điều trị.

## 2. Nguyên tắc cốt lõi: chỉ hiển thị điều đã đo được là đủ đúng

Không thể bảo đảm mọi nốt trên mọi selfie đều đúng loại. Cách làm là **dự đoán có chọn lọc theo tầng**: mỗi nốt chỉ được hiển thị ở tầng thông tin mà dữ liệu kiểm tra cho thấy đạt precision mục tiêu.

| Tầng | Hiển thị cho khách | Điều kiện |
| --- | --- | --- |
| A | Loại mụn cụ thể (1 trong 5 loại) | Xác suất loại đã hiệu chỉnh ≥ ngưỡng τ_A của nhóm rủi ro ảnh đó |
| B | Nhóm: mụn viêm (sẩn, mủ, nốt) / không viêm (đầu trắng, đầu đen) | Không đạt A, xác suất nhóm ≥ τ_B |
| C | "Nốt nghi mụn", chỉ vị trí | Không đạt B, confidence detector ≥ τ_C |
| — | Không hiển thị | Dưới τ_C |

- Ngưỡng τ_A, τ_B, τ_C chọn trên validation để **phần được hiển thị ở mỗi tầng** đạt precision mục tiêu ở **nguồn kém nhất** (mục 6), không lấy trung bình các nguồn.
- Ảnh không bị từ chối. Ảnh kém chất lượng hoặc khác miền chỉ làm tăng nhóm rủi ro, khiến nhiều nốt xuống tầng thấp hơn, kèm lời nhắc chụp lại cho kết quả chi tiết hơn.
- Như vậy đầu vào vẫn thoải mái, nhưng lỗi hiển thị bị giới hạn. Cái giá phải trả là **coverage**: trên ảnh khó, ít nốt được hiện loại hơn. Coverage phải được báo cáo, không được giấu.

**Giới hạn phải nói rõ:** không có selfie được chuyên gia gán nhãn, nên **không chứng minh được** độ chính xác trên selfie khách hàng. Các số đo được là trên: nhãn bác sĩ của ACNE-DET; nhãn chưa xác minh của các bộ Roboflow; ảnh mô phỏng điều kiện điện thoại; nhãn mức ca của bác sĩ trong SCIN; và duyệt mắt không chuyên của chủ dự án. Báo cáo và giao diện phải ghi đúng điều này.

## 3. Sự thật đã biết: đọc trước khi làm

Nguồn: `reports/dataset_audit_2026-11-04/` và `reports/new_datasets_audit/` (pHash ngưỡng ≤10, thống kê SCIN, ảnh mẫu). Không cần tải lại các bộ đã có.

| Nguồn | Bản chất đã xác minh | Vai trò trong kế hoạch này |
| --- | --- | --- |
| `data_raw/acnedet_author/` (bản tác giả) | 276 ảnh VISIA, 31.777 polygon, 10 lớp, nhãn do bác sĩ da liễu của nhóm tác giả. Khớp 275/275 ảnh mirror; tác giả có thêm ID 40 | **Nguồn nhãn đáng tin nhất.** Train và đánh giá trong miền (T1). Dùng thay mirror; mirror chỉ để truy vết split cũ |
| `data_raw/acnedet_roboflow/` (mirror) | 275 ảnh, 31.502 box; 7 box lệch >1 px so với tác giả, 1 box vượt biên | Không cộng thêm như nguồn thứ hai |
| `raw/ACNE04_images/`, `acne04v2/` | 1.457 ảnh, 1 lớp `acne`; bộ GĐ1 khóa 816/175/176 | Train detector; giữ val/test cũ để kiểm tra regression |
| `data_raw/acne-kbm0q_roboflow/extracted/` | 926 ảnh nguồn (1.255 file do lật ngang); khoảng 756–1.021 ảnh trùng ACNE04 theo pHash, đã thu nhỏ còn ≤416 px; nhãn 5 loại + `dark spot`; license `undefined` | **Nhãn loại mụn cho ảnh ACNE04.** Ghép box về ảnh ACNE04 gốc để train classifier. 279 ảnh khớp val/test ACNE04 đã khóa thì không được dùng train |
| `data_raw/acne-xnbio_roboflow/extracted/` | 346 ảnh nguồn (1.029 file augmentation), là tập con của DetectDataSkin-2; ảnh web, có watermark DermNet/Alamy; có lớp `Cysts` | Gộp với DetectDataSkin-2 theo lineage, không coi là nguồn độc lập |
| `data_raw/detectdataskin2/` | 1.434 ảnh nguồn (3.462 file), 640×640, nhãn chưa được chuyên gia kiểm tra; có `Bekas Jerawat` (sẹo), `Pori-Pori` (lỗ chân lông); 131 ảnh từ ACNE04; 411 ảnh nguồn nằm ở nhiều split | **Miền gần selfie nhất có nhãn từng nốt.** Train, validation và proxy-test (T2) chia theo lineage |
| `data_raw/scin/` | 10.379 ảnh điện thoại thật, nhãn bác sĩ ở **mức ca**, không có box. 101 ca / 213 ảnh có Acne trong chẩn đoán phân biệt, 53 ca có trọng số ≥0,5; nhiều ảnh là cổ/ngực, mặt bị che | Kiểm tra trên ảnh điện thoại thật bằng nhãn mức ca (T4) và để chủ dự án duyệt mắt (T5). Không dùng làm nhãn từng nốt |

**Baseline hiện tại** (phải truy vết lại đúng checkpoint ở P00):

- Trong miền, 5-fold CV trên ACNE-DET: pipeline mAP50 0,286, macro-F1 0,401 (box IoU ≥0,5 và đúng loại). Recall detector khoảng 0,48 (IoU ≥0,3).
- Hiển thị có chọn lọc trong miền: loại được hiện cho 64,3% nốt tìm đúng, đúng 94,7%. Gộp 2 nhóm đúng 92,1%.
- Ngoài miền, DetectDataSkin-2: detector F1 0,19; accuracy loại trên crop 22,35%; phần được hiển thị chỉ đúng 23,7%; ECE 0,51 → 0,57 sau hiệu chỉnh. **Hiệu chỉnh trên VISIA không có tác dụng ngoài miền.**
- Chế độ `--demo` hiện tại: luôn hiện loại cho mọi nốt, detector conf 0,30, kết quả kiểm tra ảnh chỉ để báo, không chặn. Đây là kiểu đầu vào chủ dự án muốn giữ, nhưng loại mụn hiển thị trên ảnh điện thoại chỉ đúng khoảng 21–28%.

Mã cần đọc: `api/server.py`, `api/demo.html`, `scripts/phase2/predict_acne_type.py`, `input_gate.py`, `make_crops.py`, `common2.py`, các script train/evaluate/CV/calibration; `models/phase1/model_metadata.json`, `models/phase2/pipeline_metadata.json`, `models/phase2/input_gate.json`.

## 4. Quy tắc thực hiện

1. Kiểm tra `git status`, giữ nguyên thay đổi của người dùng. Dữ liệu nguồn và checkpoint hiện tại không bị ghi đè; mọi loại bỏ, clip, đổi mapping và chia tập ghi vào bản dẫn xuất có version, lý do và hash nguồn.
2. Dùng Python trong `venv`. Kiểm tra GPU/VRAM thực tế (RTX 5050 8 GB) trước khi chọn cấu hình. Reuse mã hiện có khi đúng.
3. **Không có chuyên gia:** không tự gọi nhãn nào là "đã xác minh". Nhãn của bộ Roboflow ghi `unverified`. Duyệt của chủ dự án ghi `owner_review` (không chuyên) và chỉ dùng cho câu hỏi người thường trả lời được: box có nằm trên một nốt/đốm trên da không, ảnh có nhìn rõ da không. Không dùng dự đoán của model hoặc của Claude làm nhãn test.
4. Ảnh và nhãn từng ảnh giữ cục bộ theo `docs/PRIVACY.md`. Hệ thống chỉ là demo nội bộ trên localhost: ACNE04 (chỉ học thuật), kbm0q (license undefined), xnbio (có ảnh bản quyền) **không được dùng cho sản phẩm thương mại hoặc dịch vụ public**. Ghi điều này vào registry và model card.
5. Hoàn thành một mốc khi đầu ra đã kiểm tra được. Phân biệt `done`, `failed`, `inconclusive`. Không gọi xong chỉ vì đã viết script; smoke run không phải bằng chứng accuracy.

## 5. Vai trò dữ liệu và cách chia

| Tập | Thành phần | Dùng để |
| --- | --- | --- |
| **Train** | ACNE-DET tác giả (nhóm train), ACNE04 train, ACNE04 + nhãn loại kbm0q (chỉ ảnh không khớp val/test ACNE04), DetectDataSkin-2 + xnbio (nhóm lineage train) | Train detector và classifier |
| **Val** | Nhóm val tương ứng của từng nguồn | Chọn checkpoint, chọn τ_A/τ_B/τ_C, hiệu chỉnh |
| **T1 – trong miền** | ACNE-DET tác giả: nhóm test của master split, kèm 5-fold CV theo nhóm khi cần ước lượng ổn định | Đo với nhãn bác sĩ |
| **T2 – proxy ảnh thường** | Khoảng 20% nhóm lineage DetectDataSkin-2 + xnbio, khóa **ngay ở P03**, không bao giờ dùng train/chọn ngưỡng | Đo trên miền gần selfie; nhãn chưa xác minh |
| **T3 – mô phỏng điện thoại** | Ảnh T1 qua biến đổi: thu nhỏ còn 1.000–1.500 px, JPEG, nhiễu, mờ nhẹ, cân bằng trắng, sáng/tối, nghiêng nhẹ | Đo độ bền của phần hiển thị |
| **T4 – SCIN mức ca** | Ca SCIN vùng đầu/cổ (kể cả ca thiếu cờ, sau khi xem ảnh): ca có Acne trọng số ≥0,5 so với ca không có Acne | Đo AI có phân biệt ca mụn và ca bệnh da khác trên ảnh điện thoại thật không, bằng nhãn bác sĩ mức ca |
| **T5 – duyệt của chủ dự án** | 50–100 ảnh SCIN có mặt rõ, cộng ảnh thật của spa nếu có sự đồng ý; có thể thêm 10–20 selfie da ít mụn của nhân viên tự nguyện | Đếm box hiển thị rõ ràng sai chỗ (mắt, lỗ mũi, tóc, nền, trang sức); kiểm tra không lỗi trên ảnh thật |

- Chia **theo lineage group** (ảnh gốc, bản lật/xoay/resize, mirror, ảnh trùng pHash đã duyệt cùng một nhóm). Không có patient ID; nhóm là suy đoán và phải ghi rõ.
- DetectDataSkin-2 đã được đánh giá thăm dò một lần với checkpoint cố định. T2 khóa từ giờ là **proxy-test**, không phải test chưa từng thấy. Ghi rõ trong báo cáo.
- Mapping nhãn: `Whitehead/Whiteheads/whiteheads` → `closed_comedo`; `Blackhead/Blackheads/blackheads` → `open_comedo`; `Papule(s)` → `papule`; `Pustule(s)` → `pustule`; `Nodule(s)` → `nodule`. `Cysts`, `dark spot`, `Bekas Jerawat`, `Pori-Pori` và các lớp ngoài 5 loại của ACNE-DET → vùng `ignore` (không là mụn, không là nền) trong lần train đầu. Dùng chúng làm mẫu âm (lớp "không phải mụn") chỉ là thí nghiệm có điều kiện ở P07. Không đổi `Cysts` thành `nodule`.

## 6. Đánh giá và phạm vi sai số cho phép

### 6.1 Định nghĩa cố định

- **Ghép nốt:** một-một theo confidence. Chỉ số chính dùng **IoU ≥0,3**, vì nốt chỉ rộng 6–20 px và mục đích là chỉ vị trí cho khách xem. Luôn báo thêm IoU ≥0,5.
- **Precision tầng A:** số nốt hiện loại đúng (khớp GT và đúng loại) / tổng số nốt được hiện loại, **kể cả box hiện loại trên vùng không có GT**.
- **Precision tầng B:** tương tự, với nhóm viêm/không viêm.
- **Precision tầng C (vị trí):** số box được hiển thị (mọi tầng) khớp một GT mụn / tổng số box được hiển thị. Box rơi vào vùng `ignore` không tính đúng hay sai.
- **Recall vị trí:** số GT mụn được ít nhất một box hiển thị khớp / tổng GT mụn. **Coverage A, A+B:** tỷ lệ GT được hiện ở tầng đó.
- **Calibration:** ECE và reliability diagram của xác suất hiển thị, tách theo nguồn.
- **CI 95%:** bootstrap theo lineage group, 1.000 lần.
- Mọi ảnh đều tính vào mẫu số. Ảnh có cảnh báo vẫn được xử lý và tính điểm.

### 6.2 Mục tiêu khóa tại P03 (trước khi train và xem T2)

Đây là mục tiêu cho demo nội bộ, không phải chuẩn y khoa. Lưu vào `configs/selfie_v1/acceptance.yaml` trước khi train. Không hạ sau khi xem kết quả. Mục nào không đủ dữ liệu để đo thì ghi `inconclusive`.

| Hạng mục | T1 (nhãn bác sĩ) | T2 (proxy, nhãn chưa xác minh) | Điều kiện đi kèm |
| --- | --- | --- | --- |
| Precision tầng A (loại cụ thể) | ≥ 90% | ≥ 80% | Cận dưới CI ≥ mục tiêu − 5 điểm; báo từng loại và support |
| Precision tầng B (nhóm) | ≥ 92% | ≥ 85% | Như trên |
| Precision vị trí (box hiển thị) | ≥ 70% | ≥ 60% | Báo thêm FP trung bình/ảnh |
| Recall vị trí | ≥ 50% | ≥ 45% | Không thấp hơn baseline cùng evaluator |
| Coverage A+B (trên GT tìm đúng) | ≥ 60% | ≥ 40% | Không được đạt precision bằng cách gần như không hiển thị gì |
| Calibration xác suất hiển thị | ECE ≤ 0,05 | ECE ≤ 0,08 | Báo reliability diagram |

| Hạng mục khác | Mục tiêu |
| --- | --- |
| T3 mô phỏng điện thoại | Ở mỗi kiểu biến đổi, precision tầng A và B giảm không quá 5 điểm so với T1 sạch |
| T4 SCIN mức ca | AUC của điểm "mức mụn" do AI tính, ca có Acne so với ca không có Acne, ≥ 0,70; báo CI; nếu dưới 20 ca mỗi bên thì `inconclusive` |
| T5 duyệt của chủ dự án | ≤ 10% box hiển thị bị đánh giá "rõ ràng không nằm trên da hoặc trên một nốt/đốm"; 0 ảnh bị lỗi xử lý |
| Đầu vào | 100% ảnh đọc được đều có kết quả; không có trạng thái chặn phân tích |
| Regression | Macro-F1 box+loại trên T1 giảm không quá 0,03 so với baseline |
| Latency | p95 ≤ 5 giây/ảnh trên GPU của máy, đủ pipeline |

**Quy tắc hạ tầng khi không đạt:** nếu tầng A không đạt ở T2, demo chỉ hiện loại cụ thể cho ảnh có rủi ro thấp, còn lại dừng ở tầng B. Nếu tầng B cũng không đạt, demo chỉ hiện vị trí. Hạ tầng là kết quả hợp lệ, không phải lý do để sửa ngưỡng theo test. Lớp `nodule` có support thấp: nếu không đo được thì không hiện loại `nodule` ở tầng A, chỉ hiện nhóm "mụn viêm".

### 6.3 Con số được phép công bố

Trên giao diện và báo cáo chỉ ghi dạng: *"Trên dữ liệu kiểm tra có nhãn, loại mụn được hiển thị đúng khoảng X% (bộ ảnh lâm sàng, nhãn bác sĩ) và Y% (bộ ảnh thường, nhãn chưa xác minh). Chưa kiểm định trên selfie có nhãn chuyên gia. Kết quả chỉ để tham khảo chăm sóc da."* Không dùng crop accuracy 86,6% hoặc confidence của một nốt để nói "chính xác 90%".

## 7. Bảng thực hiện

### 7.1 Ba chặng và hai điểm quyết định

Làm theo chặng để luôn có một bản demo dùng được, và dừng sớm nếu cải thiện thêm không đáng công.

| Chặng | Mốc | Kết quả cuối chặng | Ước tính |
| --- | --- | --- | --- |
| **1. Demo an toàn với model hiện tại** | P00–P05, P08a, P09, P10 | Demo đầu vào thoải mái, hiển thị theo tầng A/B/C với checkpoint hiện tại; ngưỡng chọn trên val của cả nguồn ảnh thường; có số đo trung thực trên T1, T3, val | 8–11 ngày làm việc; gần như không cần train |
| ⛔ **Điểm quyết định 1** | — | Claude báo precision và coverage từng tầng trên val. Chủ dự án chọn: (a) dừng, sang chặng 3 với model hiện tại; hoặc (b) làm chặng 2 | Vài phút |
| **2. Train cải thiện** | P06, P07, P08b | Candidate detector + classifier train trên dữ liệu trộn, ngưỡng hiển thị mới | 7–10 ngày làm việc; chủ yếu là thời gian GPU, chạy qua đêm |
| ⛔ **Điểm quyết định 2** | — | So candidate với kết quả chặng 1 trên cùng val (ghép cặp, có CI). Không cải thiện rõ (CI chạm 0 hoặc coverage tăng < 5 điểm ở nguồn ảnh thường) thì giữ model chặng 1 | Vài phút |
| **3. Nghiệm thu** | P11 (+ P12 sau demo) | Đánh giá một lần trên T1–T4 với cấu hình đã chọn, chủ dự án duyệt T5, quyết định phát hành cho demo, model card, rollback | 2–3 ngày làm việc + 1–2 giờ duyệt của chủ dự án |

- Dừng ở điểm quyết định 1: tổng khoảng **10–14 ngày làm việc** (khoảng 2–3 tuần lịch).
- Làm đủ ba chặng: tổng khoảng **17–24 ngày làm việc** (khoảng 3,5–5 tuần lịch).
- Ước tính giả định máy bật để train qua đêm và chủ dự án trả lời các điểm quyết định trong 1–2 ngày. Dựa trên lần train trước: RT-DETR-L @1280 mất khoảng 3 giờ cho 190 ảnh ACNE-DET; với dữ liệu trộn dự kiến 6–10 giờ mỗi run. Đây là ước tính, không phải cam kết.
- Code tích hợp (P09, P10) viết một lần ở chặng 1. Chặng 2 chỉ thay checkpoint và `display_policy.json` qua cấu hình, không viết lại giao diện.

### 7.2 Việc chủ dự án cần làm

Claude làm toàn bộ code, xử lý dữ liệu, train, đánh giá và tích hợp. Chủ dự án chỉ làm các việc sau; Claude chuẩn bị sẵn trang duyệt chạy trên localhost, mỗi ảnh chỉ cần bấm chọn.

| Việc | Mốc | Thời gian | Nếu không làm |
| --- | --- | --- | --- |
| Xem khoảng 30 overlay nhãn kbm0q đã ghép về ảnh ACNE04, bấm Đúng/Sai vị trí | P02 | ~15 phút | Chỉ dùng ảnh ghép có sai số chiếu lại thấp; ghi `unreviewed` |
| Chọn hướng ở hai điểm quyết định | Sau chặng 1, sau chặng 2 | Vài phút mỗi lần | Claude dừng và chờ |
| Để máy bật khi train | Chặng 2 | Không tốn công | Run bị ngắt thì train tiếp từ `last.pt` |
| Duyệt 50–100 ảnh kết quả, đánh dấu box rõ ràng sai chỗ (mắt, lỗ mũi, tóc, nền, trang sức) | P11 (T5) | ~1–2 giờ | Mục T5 ghi `inconclusive` |
| *(Tuỳ chọn)* Chụp 10–20 selfie da ít mụn của nhân viên tự nguyện, có đồng ý | P11 (T5) | ~30 phút | Không đo được báo nhầm trên da ít mụn; ghi `inconclusive` |

Tổng công sức của chủ dự án khoảng 2–3 giờ, rải trong vài tuần.

### 7.3 Chi tiết từng mốc

Thời gian là ước tính công sức, không phải cam kết.

| ID | Phụ thuộc | Việc cụ thể | Đầu ra | Điều kiện hoàn thành | Ước tính |
| --- | --- | --- | --- | --- | --- |
| P00 | Không | Truy vết checkpoint, hash, ngưỡng detector/classifier, crop, EXIF, chế độ thường và `--demo` của CLI/API; lưu dự đoán baseline | `reports/selfie_v1/baseline_manifest.json`, `baseline_runtime.md` | Mỗi số gắn checkpoint, split, evaluator, ngưỡng; baseline chạy lại được | 0,5–1 ngày |
| P01 | P00 | Registry nguồn/version/license/mức tin cậy nhãn; mapping nhãn mục 5; định nghĩa `ignore` | `datasets/selfie_v1/source_registry.json`, `configs/selfie_v1/label_mapping.yaml` | Không cộng mirror/augmentation thành nguồn độc lập; mapping có lý do | 1 ngày |
| P02 | P01 | Làm sạch hình học: box suy biến, vượt biên, 742 ảnh lệch kích thước, 7 box tác giả/mirror; polygon → box. **Ghép nhãn kbm0q về ảnh ACNE04 gốc**: ước lượng phép thu nhỏ + lật bằng khớp đặc trưng, kiểm tra sai số chiếu lại, chủ dự án xem overlay 30 ảnh mẫu | `scripts/data_v2/clean_annotations.py`, `scripts/data_v2/map_kbm0q_to_acne04.py`, `reports/selfie_v1/geometry_audit.json` | 0 box sai schema; ảnh ghép không đạt sai số thì loại; mọi sửa có log | 2–4 ngày |
| P03 | P01–P02 | Lineage + quarantine + chia tập mục 5; **khóa T2 và `acceptance.yaml`** | `datasets/selfie_v1/lineage.jsonl`, `split_manifest.jsonl`, `quarantine.jsonl`, `configs/selfie_v1/acceptance.yaml` | Không nhóm nào nằm ở hai vai trò; ứng viên trùng chưa rõ thì cách ly; manifest có SHA256 | 1–2 ngày |
| P04 | P03 | Evaluator thống nhất theo mục 6.1: tầng A/B/C, `ignore`, IoU 0,3/0,5, coverage, ECE, bootstrap; test với fixture biết trước TP/FP/FN | `scripts/evaluation_v2/evaluate_pipeline.py`, `tests/` | Fixture đúng số; ảnh có cảnh báo vẫn trong mẫu số | 1–2 ngày |
| P05 | P04 | Chạy baseline (chế độ thường và `--demo`) trên T1, T3, T4 và **val** của nguồn ảnh thường; phân loại lỗi | `reports/selfie_v1/baseline/` | Có đủ chỉ số mục 6; chưa chạm T2 | 1 ngày |
| P08a | P05 | **Chặng 1:** áp dụng hiển thị theo tầng cho checkpoint hiện tại. Điểm rủi ro ảnh (chất lượng + domain score) chia nhóm thấp/cao; hiệu chỉnh nhiệt độ theo nguồn; chọn τ_A/τ_B/τ_C trên val để đạt mục tiêu ở nguồn kém nhất | `models/candidates/selfie_v1/baseline_display_policy.json`, `reports/selfie_v1/risk_coverage/baseline/` | Đường precision–coverage từng tầng trên val; ngưỡng ghi hash checkpoint. Là số liệu cho điểm quyết định 1 | 1–2 ngày |
| P06 | Điểm quyết định 1 chọn (b) | Fine-tune detector trên dữ liệu trộn: sampler theo nguồn, augmentation kiểu điện thoại, thử tiling cho ảnh lớn; tối đa 4–5 thí nghiệm | `configs/selfie_v1/`, `runs/selfie_v1/`, bảng ablation | Chọn trên val, recall tăng không kèm FP tăng mất kiểm soát | 3–6 ngày |
| P07 | P06 | Fine-tune classifier: crop GT + crop từ box detector (chỉ train), nhãn loại từ kbm0q/ACNE04 và DetectDataSkin-2, cân bằng lớp, augmentation điện thoại; thí nghiệm lớp "không phải mụn" | Bảng ablation classifier | Macro-F1 val tăng ở nguồn ảnh thường, không giảm >0,03 ở ACNE-DET val | 3–5 ngày |
| P08b | P07 | **Chặng 2:** lặp lại quy trình P08a cho candidate mới; so ghép cặp với P08a trên cùng val | `models/candidates/selfie_v1/calibration.json`, `display_policy.json`, `reports/selfie_v1/risk_coverage/` | Đường precision–coverage từng tầng trên val; ngưỡng ghi hash checkpoint. Là số liệu cho điểm quyết định 2 | 1–2 ngày |
| P09 | P08a | Đầu vào thoải mái: bỏ trạng thái chặn, thay bằng cảnh báo + nhóm rủi ro; bỏ điều kiện ≥3 nốt và mặt ≥1.003 px như luật chặn; sửa EXIF, ảnh nhỏ thì phóng/tiling; 0–2 nốt xử lý bình thường | `scripts/phase2/input_gate.py` (cập nhật), `predict_acne_type.py` | Mọi ảnh đọc được đều có kết quả; test EXIF và round-trip bbox | 1–2 ngày |
| P10 | P09 | Tích hợp vào API và `demo.html`: hiển thị theo tầng A/B/C, chú giải, câu công bố mục 6.3, cảnh báo ảnh. Nút "xem mọi dự đoán (kỹ thuật)" cho nhân viên, đánh dấu rõ, không dùng trong nghiệm thu | `api/server.py`, `api/demo.html` | API và evaluator cho cùng kết quả; ảnh sạch không lỗi | 2 ngày |
| P11 | P10 và điểm quyết định 1 (a) hoặc 2 | Đánh giá cuối trên T1, T2, T3, T4 một lần với cấu hình đã khóa; chủ dự án duyệt T5; quyết định giữ/thay model; phương án rollback | `reports/selfie_v1/final_evaluation/`, `release_decision.json`, model card | Mỗi mục ghi đạt/không đạt/`inconclusive`; không sửa ngưỡng theo test rồi đo lại | 2–3 ngày |
| P12 | Sau demo | Chủ dự án đánh dấu kết quả sai trong lúc demo (lưu cục bộ, có đồng ý); định kỳ xem lại và cập nhật T5 | Nhật ký phản hồi, báo cáo định kỳ | Phản hồi không chuyên không được dùng làm nhãn test | Liên tục |

## 8. Thiết kế thí nghiệm

1. Giữ hai giai đoạn (detector → classifier) làm baseline. Chọn checkpoint và ngưỡng bằng val, không bằng T1/T2.
2. Mỗi thí nghiệm ghi lý do, config, seed, hash dữ liệu. Chạy 1 seed khi sàng lọc; cấu hình chọn cuối chạy 3 seed nếu đủ tài nguyên.
3. Augmentation mô phỏng điện thoại (ánh sáng, cân bằng trắng, JPEG, nhiễu, độ phân giải, nghiêng nhẹ) dựa trên ảnh thật. Không blur/resize tới mức nốt biến mất mà vẫn giữ nhãn.
4. Nhãn chưa xác minh (DetectDataSkin-2, kbm0q) có thể nhiễu. Thử thêm phiên bản bỏ các box mà model train trên ACNE-DET và nhãn nguồn bất đồng mạnh, **chỉ trên train**. Không lọc T2 theo dự đoán của model.
5. Chỉ thử backbone lớn hơn, ensemble hoặc domain adaptation khi phân tích lỗi cho thấy có ích, và đo cả FP, calibration, latency.
6. Mọi cache prediction/crop gắn hash model + dữ liệu + transform.

## 9. Kiểm tra kỹ thuật bắt buộc

- Hình học: polygon → box, EXIF xoay, resize/tiling và round-trip bbox, box ở biên, box diện tích 0, metadata lệch 1–2 px, phép ghép kbm0q → ACNE04.
- Evaluator: fixture có TP/FP/FN biết trước cho tầng A/B/C, box dư, box thiếu, trùng dự đoán, vùng `ignore`, ảnh không có nốt.
- Chia tập: ảnh gốc + lật + resize + mirror cùng lineage, không nằm ở hai vai trò; ảnh trong quarantine không vào train.
- API: mọi ảnh đọc được đều có kết quả; nốt dưới ngưỡng tầng A không hiện tên loại; API và CLI cùng output; bundle sai hash không dùng ngưỡng cũ.

## 10. Tổ chức đầu ra

```text
configs/selfie_v1/          # label mapping, acceptance, train/eval/calibration config
scripts/data_v2/            # registry, geometry, kbm0q mapping, lineage, split
scripts/evaluation_v2/      # evaluator thống nhất
datasets/selfie_v1/         # manifest, lineage, quarantine (tham chiếu ảnh nguồn)
runs/selfie_v1/             # run thử, checkpoint
reports/selfie_v1/          # tiến độ, metrics, ablation, CI, quyết định
models/candidates/selfie_v1/  # checkpoint + calibration + display policy cùng version
tests/                      # hình học, leakage, evaluator, API
```

Mỗi mốc cập nhật `reports/selfie_v1/progress.md`: đã làm gì, lệnh tái lập, kết quả thật, kiểm tra đã chạy, điểm chưa đạt và mốc tiếp theo.

## 11. Tham khảo phương pháp

- [Group cross-validation — scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data): chia theo nhóm; nhóm suy đoán không phải người thật đã xác minh.
- [Probability calibration — scikit-learn](https://scikit-learn.org/stable/modules/calibration.html): độ tin cậy khác accuracy; hiệu chỉnh trên dữ liệu phù hợp.
- [Selective classification (Geifman & El-Yaniv, 2017)](https://arxiv.org/abs/1705.08500): chọn ngưỡng từ chối để đạt mức rủi ro mục tiêu, đánh đổi với coverage.
- [SAHI](https://arxiv.org/abs/2202.06934): chia ô cho vật thể nhỏ; không suy diễn mức tăng sang mụn trên selfie.
- [SCIN schema — Google](https://github.com/google-research-datasets/scin/blob/main/dataset_schema.md): nhãn mức ca và chẩn đoán phân biệt; không có box từng nốt.
