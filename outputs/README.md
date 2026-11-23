# Đầu ra ứng dụng

- images/: PNG RGB do người dùng xử lý, tên có hậu tố riêng để tránh ghi đè.
- runs/: Một JSON cho mỗi đường dẫn ảnh kết quả, chứa kích thước, thời gian và hash model; ghi đè cùng kết quả sẽ cập nhật JSON đó.
- logs/: web.log và log kiểm chứng.
- checks/: kết quả smoke/integration check.

Không đặt dataset hoặc model ở đây. Các file là dữ liệu cục bộ; source và checkpoint không được sửa khi xử lý ảnh.
