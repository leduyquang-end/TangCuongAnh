# Slide 11: Dữ liệu huấn luyện và augmentation

## Nội dung hiển thị

Dùng các cặp low/high trong LOL-v2-real cục bộ.

Low/high dùng chung tọa độ crop, số lần xoay 90° và lật.
Chuyển RGB uint8 thành float32 trong [0,1].

| Phân vùng | Số cặp | Cách dùng |
| --- | --- | --- |
| Train | 620 | Crop 80×80, batch 8, shuffle |
| Validation | 69 | Chọn checkpoint / tham số |
| Test | 100 | Đánh giá bản đã khóa |

## Lời thuyết trình và chi tiết kỹ thuật

Dữ liệu dùng các cặp ảnh thiếu sáng và ảnh đủ sáng trong LOL-v2-real đã có trên máy. Manifest của dự án chia 620 cặp train, 69 cặp validation và 100 cặp test. Validation tách từ dữ liệu train, còn ID test được giữ nguyên từ 00690 đến 00789.

Trong train, ảnh được crop 80×80 với batch size 8. Cặp low/high luôn dùng cùng tọa độ crop, cùng phép xoay 90 độ và cùng phép lật. Nhờ vậy pixel của input vẫn tương ứng với GT sau augmentation. RGB uint8 được đưa về float32 trong khoảng 0 đến 1, không chuẩn hóa theo mean/std ImageNet.

Nguồn dữ liệu được truy vết qua repository tác giả của công trình DRBN và manifest cục bộ. Các con số chia tập em đang nêu là split hiện tại của dự án. Em không sử dụng GT khi người dùng đưa một ảnh mới vào giao diện để suy luận.

https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_From_Fidelity_to_Perceptual_Quality_A_Semi-Supervised_Approach_for_Low-Light_CVPR_2020_paper.html

Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md

https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_From_Fidelity_to_Perceptual_Quality_A_Semi-Supervised_Approach_for_Low-Light_CVPR_2020_paper.html
