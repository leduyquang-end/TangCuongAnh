# Slide 12: Backpropagation và Adam

## Nội dung hiển thị

Decomposition đóng băng. Restoration và adjustment có 1.982.087 tham số học.

zero_grad → forward → loss.backward → clip_norm(5) → Adam.step

Teacher dùng no_grad và target detach. Seed 827.
AG mục tiêu trong loss không phải kết quả AG đã đạt.

| Giai đoạn | Epoch | Learning rate | AG mục tiêu |
| --- | --- | --- | --- |
| ag_005 | 3 | 6×10⁻⁵ | 0,05 |
| ag_010 | 3 | 4×10⁻⁵ | 0,10 |

## Lời thuyết trình và chi tiết kỹ thuật

Ở lượt tinh chỉnh cuối, decomposition đóng băng. Restoration và adjustment cùng học với tổng cộng 1.982.087 tham số. Teacher là bản cha cố định, tính đầu ra bằng no_grad và không nhận cập nhật.

Vòng học gồm xóa gradient cũ, forward, tính loss, loss.backward, clip gradient norm tối đa 5 rồi Adam.step. Autograd tính đạo hàm qua hai nhánh và phép nhân tái hợp. Adam dùng các ước lượng moment để điều chỉnh bước cập nhật. Đây là cơ chế tối ưu, còn chất lượng ảnh phụ thuộc đồng thời kiến trúc, dữ liệu và loss.

Lượt cuối có hai giai đoạn, mỗi giai đoạn ba epoch. Learning rate lần lượt là 6 nhân 10 mũ âm 5 và 4 nhân 10 mũ âm 5. Mục tiêu AG trong loss lần lượt là 0,05 và 0,10. Cần nhấn mạnh 0,10 là giá trị đích để tính thiếu hụt trong loss, không phải AG test mà mô hình đã đạt.

https://arxiv.org/abs/1412.6980

Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md

https://arxiv.org/abs/1412.6980
