# Slide 14: Gradient theo biên được GT hỗ trợ

## Nội dung hiển thị

D là sai phân theo trục x hoặc y. Prediction p, GT t, teacher r; target và mask đều detach.

s = clip(|D t| / 0,025, 0, 1) · 1[(D t)(D r) > 0]

v = clip(D r · (1 + (gain − 1)·s), −0,25, 0,25)

Ledge = mean có trọng số s của |D p − v|

Dùng cả trục x/y, tại scale 1 và 2.
Ngoài vùng hỗ trợ: phạt ReLU(|D p| − |D r|) · (1 − s).

## Lời thuyết trình và chi tiết kỹ thuật

Để tăng gradient có định hướng, code tính sai phân theo từng trục x và y. Mask hỗ trợ tăng theo độ lớn biên GT, chuẩn hóa bởi 0,025. Mask chỉ giữ vùng mà gradient GT và teacher cùng dấu. Những pixel đó có căn cứ từ cả ảnh chuẩn và cấu trúc của bản cha.

Gradient đích bắt đầu từ gradient teacher và được tăng biên độ theo mask cùng hệ số gain. Đích được chặn trong khoảng âm 0,25 đến 0,25. Prediction chịu phạt nếu lệch gradient đích tại vùng có hỗ trợ. Phép tính thực hiện ở tỷ lệ gốc và tỷ lệ giảm hai lần.

Ngoài vùng hỗ trợ, code phạt phần gradient prediction tăng vượt teacher. Một thành phần khác đo thiếu hụt AG có hỗ trợ so với mục tiêu tuyệt đối. Nếu vùng hỗ trợ quá ít, nhánh không nhận yêu cầu tạo texture mới. Đây là cách thiết kế để hướng việc tăng AG vào cấu trúc có căn cứ. Tuy nhiên log vẫn cho thấy gradient vùng phẳng tăng, nên cơ chế đó chưa khử hết đánh đổi nhiễu.



Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md


