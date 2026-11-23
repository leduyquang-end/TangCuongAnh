# Slide 21: Suy luận và tài nguyên triển khai

## Nội dung hiển thị

Ảnh mới chỉ đi qua forward và hậu xử lý; không cần GT hoặc backward.

CPU 6 luồng. Hai ảnh 600×400 qua HTTP: 0,927 và 0,581 giây.
6 epoch cuối: 438,77 giây; peak working set 2.171,59 MiB.

| Khối phần mềm | Chức năng |
| --- | --- |
| app.py / local_web.py | Khởi động local HTTP, nhận ảnh |
| EnhancementService | Nạp checkpoint một lần, chuẩn hóa RGB, forward |
| inference/pipeline.py | Model + calibration + chroma + luminance |
| Xuất kết quả | PNG RGB; cạnh dài tối đa 1280 px, giữ tỷ lệ |

## Lời thuyết trình và chi tiết kỹ thuật

Khi đưa ảnh mới vào giao diện, app.py khởi động server local. EnhancementService nạp checkpoint một lần, đọc ảnh về RGB và chạy pipeline chung. Suy luận dùng model.eval và inference_mode, không có GT, không tính loss và không cập nhật trọng số. Ảnh lớn được giữ tỷ lệ, giới hạn cạnh dài 1280; ảnh benchmark 600×400 không bị resize bởi giới hạn này.

Môi trường ghi trong log là PyTorch bản CPU, dùng sáu luồng. Hai ảnh 600×400 trong smoke test HTTP có thời gian xử lý khoảng 0,927 và 0,581 giây. Đây chỉ là hai mẫu đo, không phải kết luận tốc độ trung bình của mọi ảnh. Sáu epoch tinh chỉnh cuối cộng khoảng 438,77 giây; peak working set lớn nhất trong log khoảng 2.171,59 MiB.

Em kết luận: dự án đã triển khai và huấn luyện mô hình ghép ba nhánh, có giải thích được cơ chế attention và curve, có đầu ra thực và bảng đo. Kết quả tăng MLI, CI, entropy so với mốc AWF, nhưng AG, độ trung thực và nhiễu còn là giới hạn cần cải thiện.



Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md


