# Slide 13: Hàm mất mát kết hợp

## Nội dung hiển thị

Mục tiêu do dự án định nghĩa, gồm khớp GT, biên có hỗ trợ và neo teacher.

SSIM trong code dùng Gaussian separable 11×11, σ=1,5.

| Thành phần | Trọng số |
| --- | --- |
| L1 RGB + (1 − SSIM) | 1 + 1 |
| Khớp biên có hướng ở hai tỷ lệ | 10 |
| Thiếu AG trong vùng GT hỗ trợ | 80 |
| Gradient tăng ngoài vùng hỗ trợ | 12 |
| Neo MLI / neo CI | 100 / 8 |
| Thiếu soft entropy / neo luminance teacher | 4 / 0,2 |

## Lời thuyết trình và chi tiết kỹ thuật

Loss tổng gồm các nhóm mục tiêu có vai trò khác nhau. L1 RGB và một trừ SSIM khuyến khích ảnh gần GT ở mức pixel và cấu trúc. Khớp biên có hướng có trọng số 10. Thiếu AG trong vùng GT hỗ trợ có trọng số 80. Phần phạt gradient tăng ngoài vùng hỗ trợ có trọng số 12.

Tiếp theo, neo mean có trọng số 100, neo độ lệch chuẩn có trọng số 8, thiếu soft entropy có trọng số 4 và L1 luminance với teacher có trọng số 0,2. Các hệ số nhằm cân bằng mục tiêu tăng cường với việc giữ một số thuộc tính của bản cha. Giá trị 80 lớn hơn 10 chưa đủ kết luận nhóm loss thứ nhất chi phối tám lần, vì các thành phần có thang đo khác nhau.

Phần SSIM dựa trên nghiên cứu của Wang và cộng sự. Cách phối hợp các loss và hệ số cụ thể là công thức của dự án. Loss là hàm mục tiêu để tối ưu, không phải bảo đảm toán học rằng mọi ảnh test đều tăng tất cả chỉ số.

https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html

Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md

https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html
