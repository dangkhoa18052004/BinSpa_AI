# Phạm vi công bố và dữ liệu giữ riêng

Bản công khai được xây dựng từ danh sách file cho phép, không dùng `git add .` trên thư mục chứa dữ liệu nghiên cứu.

## Nội dung được công bố

- Mã Python, PowerShell và hướng dẫn thực nghiệm.
- Yêu cầu môi trường và file lock.
- Số liệu tổng hợp, phân bố lớp, ma trận số nguyên và checkpoint hash.
- Biểu đồ PNG và PDF ma trận, không có ảnh khuôn mặt hay contact sheet.

Phạm vi hiện tại chỉ gồm phát hiện mụn và phân loại năm loại mụn. Mã và kết quả phát hiện sẹo cũ được giữ riêng tại máy, không nằm trong danh sách công bố hiện tại.

## Nội dung giữ riêng

- Ảnh nguồn, selfie, crop và mọi bản xuất ảnh khuôn mặt.
- Annotation/manifest/prediction theo từng ảnh, ID nguồn ảnh và tọa độ tổn thương.
- Archive dữ liệu, Parquet, checkpoint/model weights.
- File môi trường, thông tin đăng nhập, private key và cấu hình cá nhân.
- Tài liệu Word, bản trích xuất tài liệu, log máy và đường dẫn người dùng.

`.gitignore` chỉ ngăn file chưa theo dõi được thêm tự động. File đã theo dõi phải được bỏ khỏi cây commit công khai; dữ liệu tại máy vẫn được giữ để làm việc.

## Giới hạn của việc dọn phiên bản mới

Repository đã từng đưa dữ liệu nguồn và môi trường lên Git. Commit dọn dữ liệu ở nhánh chính không xóa các blob khỏi commit trước đó, nhánh cũ, pull request, fork hoặc bản clone của người khác. Không coi việc README không còn ảnh là bằng chứng toàn bộ lịch sử đã sạch.

Muốn gỡ khỏi lịch sử cần một đợt xử lý riêng: thống kê mọi ref, sao lưu riêng, lọc các đường dẫn dữ liệu/tài liệu/env/per-image report khỏi mọi lịch sử có liên quan, kiểm tra lại rồi cập nhật các ref trên GitHub. Việc này đổi commit ID và yêu cầu người cộng tác đồng bộ lại; không thực hiện bằng một push thông thường. Dữ liệu từng công khai không thể được đảm bảo thu hồi khỏi mọi bản sao.

## Kiểm tra trước công bố

```powershell
python scripts/check_publication.py
# Đối chiếu một cây Git cụ thể, không chỉ kiểm tra .gitignore:
python scripts/check_publication.py --tree COMMIT_SHA
```

Kiểm tra dùng danh sách path cho phép, tìm mẫu credential/đường dẫn máy trong văn bản và đối chiếu hash các artifact tổng hợp. Đây là kiểm tra kỹ thuật trên tập file được chọn, không phải bảo đảm nhận diện mọi bí mật hoặc bảo đảm dữ liệu cũ đã được thu hồi.
