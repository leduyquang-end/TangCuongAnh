# Bộ 100 ảnh test đã chốt

- Mở [index.html](index.html) để xem đủ 100 bộ ảnh tối → kết quả → GT, có tìm theo ID.
- Thư mục [images](images) có đúng 100 ảnh đầu ra PNG RGB, từ 00690_RGB.png đến 00789_RGB.png.
- [per_image.csv](per_image.csv): chỉ số từng ảnh; [metrics.json](metrics.json): số liệu gốc.
- [export_manifest.json](export_manifest.json): nguồn và SHA256 của từng ảnh sao chép.

Bộ này được sao chép từ experiments/ag_refinement/run_four_metric_curriculum/test. Không huấn luyện lại, không tinh chỉnh từ test. Ảnh tối/GT trong trang xem được tham chiếu tới dataset cục bộ; cần giữ dataset ở vị trí hiện tại. Bản PNG kết quả trong images có thể mang đi độc lập.
