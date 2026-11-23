# Slide 20: Độ trung thực và gradient vùng phẳng

## Nội dung hiển thị

Ảnh test ID 00789: input, kết quả hiện tại và GT.

Input

Mô hình

GT

PSNR 13,9251 dB; SSIM 0,5134. So với bản cha: −6,49% và −22,96%.
AG vùng phẳng tăng từ 0,012712 lên 0,026423, cần kiểm tra khuếch đại nhiễu.

## Lời thuyết trình và chi tiết kỹ thuật

Ba ảnh cho thấy input, đầu ra và GT của cùng ID test. Vùng tối được nâng sáng rõ, nhưng đầu ra vẫn khác GT về màu và bề mặt. Trên toàn bộ 100 ảnh, PSNR của bản chốt là 13,9251 dB và SSIM khoảng 0,5134. So với bản cha trong log, hai chỉ số này giảm lần lượt khoảng 6,49% và 22,96%.

AG ở vùng GT phẳng tăng từ khoảng 0,012712 lên 0,026423. Đây là dấu hiệu phải kiểm tra khuếch đại nhiễu hoặc texture thừa, chứ không thể xem mọi gradient tăng là chi tiết được phục hồi. Những giá trị này cho thấy mục tiêu tăng chỉ số độ sáng và tương phản có đánh đổi về độ trung thực.

Đề không cung cấp PSNR và SSIM tương ứng của AWF, nên em không so trực tiếp hai số này với AWF. Hướng cải thiện tiếp theo cần cân bằng các mục tiêu trên validation và kiểm chứng độ trung thực, thay vì chỉ tăng AG hoặc Z.

https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html

Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md

https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html
