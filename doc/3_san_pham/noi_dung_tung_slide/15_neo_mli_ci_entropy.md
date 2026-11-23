# Slide 15: Neo MLI, CI và entropy khi học

## Nội dung hiển thị

Các ràng buộc dưới đây áp dụng trên patch luminance trong train.

Học dùng surrogate khả vi. Entropy báo cáo tính từ histogram uint8 256 bins.

| Đại lượng | Khoảng / mục tiêu theo teacher | Trọng số |
| --- | --- | --- |
| Mean patch | [1,005; 1,02] × mean(r) | 100 |
| Độ lệch chuẩn patch | [1,06; 1,15] × std(r) | 8 |
| Soft entropy 64 bins | Hsoft(p) ≥ Hsoft(r) + 0,003 | 4 |
| Luminance L1 | Giữ p gần r | 0,2 |

## Lời thuyết trình và chi tiết kỹ thuật

Ba ràng buộc neo dùng thống kê patch luminance. Mean của prediction được khuyến khích nằm trong khoảng 1,005 đến 1,02 lần teacher. Độ lệch chuẩn được neo trong khoảng 1,06 đến 1,15 lần teacher. Nếu nằm ngoài khoảng, hàm ReLU tạo phí thiếu hoặc vượt.

Soft entropy của prediction được khuyến khích cao hơn teacher ít nhất 0,003. Trong train, soft entropy dùng 64 bins với đóng góp mềm để có gradient. Trong báo cáo, entropy dùng histogram cứng 256 bins từ ảnh uint8. Vì vậy hai đại lượng có liên quan nhưng không thể gọi là hoàn toàn giống nhau.

Thành phần L1 luminance với teacher giữ đầu ra khỏi trôi quá xa. Các neo này là mục tiêu khả vi ở mức patch. Khi đánh giá toàn ảnh, mean và độ lệch chuẩn còn phụ thuộc phân bố không gian và hậu xử lý, nên vẫn phải đo trực tiếp trên validation.



Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md


