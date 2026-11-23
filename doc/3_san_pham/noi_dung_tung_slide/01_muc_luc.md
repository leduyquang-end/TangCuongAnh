# Slide 01: Nội dung trình bày

## Nội dung hiển thị

KinD Decomposition + Attention U-Net + Curve CNN

01   Tổng quan mô hình và nhánh phân rã

02   Attention U-Net: encoder, decoder và attention

03   Curve CNN: hệ số và phép biến đổi đường cong

04   Huấn luyện, loss và quy trình đánh giá

05   Kết quả và bảng so sánh các phương pháp

## Lời thuyết trình và chi tiết kỹ thuật

Em trình bày mô hình tăng cường ảnh thiếu sáng gồm ba nhánh: phân rã KinD, phục hồi phản xạ bằng Attention U-Net và điều chỉnh illumination bằng Curve CNN. Nội dung có năm phần. Trước hết em giải thích luồng dữ liệu và vai trò từng nhánh. Sau đó em đi sâu vào encoder, decoder và attention gate, rồi giải thích cách Curve CNN dự đoán hệ số và lặp đường cong. Tiếp theo là cách học, hàm mất mát và quy trình chọn mô hình. Cuối cùng em đối chiếu các chỉ số với bảng mốc của cuộc thi. Các URL ở cuối trang mở nghiên cứu gốc tương ứng; kích thước và tham số em trình bày lấy từ code đang chạy.

https://arxiv.org/abs/1905.04161
https://arxiv.org/abs/1804.03999
https://arxiv.org/abs/2001.06826

Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md

https://arxiv.org/abs/1905.04161

https://arxiv.org/abs/1804.03999

https://arxiv.org/abs/2001.06826
