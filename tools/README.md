# Công cụ dự án

| Nhóm | Vai trò | Entry point |
|---|---|---|
| training | Công thức 6 epoch từ bản hiện tại | python -m kind_enhance train --epochs 6 --output experiments/next_run |
| evaluation | Đo/kiểm chứng một selection đã khóa | evaluation/evaluate_four_metrics.py |
| inspection | Trace cấu trúc và 52 tầng bằng input tổng hợp | inspection/inspect_model_architecture.py |
| verification | Chạy app.py, kiểm tra HTTP và model thật | verification/smoke_local_web.py |
| research | Mã thử loss/teacher/calibration, không tự triển khai | research/refine_clean_details.py và các công cụ liên quan |

Dùng GUI/CLI trong kind_enhance cho ảnh người dùng. Công cụ nghiên cứu cần thư mục run/đầu vào tương ứng; các kết quả thử đã bị thay thế có thể đã được dọn. Không coi một script nghiên cứu là tính năng đã phát hành.

Sau khi đổi code, chạy unit test và kiểm chứng phù hợp. Model/config mặc định chỉ thay khi người dùng duyệt. Các source snapshot trong experiments giữ nguyên theo thời điểm đo, độc lập với cây tools hiện tại.

Xuất bộ 100 ảnh đã chốt để xem: `python -B tools/evaluation/export_test_gallery.py`. Ảnh và trang xem nằm ở outputs/test_100; đây là sao chép có kiểm tra hash, không huấn luyện/suy luận mới.

Kiểm chứng độc lập đủ 100 ảnh: `python -B tools/verification/audit_reported_metrics.py`. Không huấn luyện hoặc chọn tham số trong bước kiểm chứng.
