# Slide 19: Đối chiếu bốn chỉ số với mốc AWF

## Nội dung hiển thị

Trung bình 100 ảnh test của bản chốt, ID 00690–00789.

AWF là mốc công bố, chưa tái chạy cùng code. Mô hình vẫn thấp hơn AWF ở AG.

| Chỉ số | AWF trong đề | Mô hình | Chênh lệch |
| --- | --- | --- | --- |
| MLI | 0,331700 | 0,486328 | +46,62% |
| CI | 0,196300 | 0,227169 | +15,73% |
| E | 7,218400 | 7,644204 | +5,90% |
| AG | 0,105200 | 0,051272 | −51,26% |

## Lời thuyết trình và chi tiết kỹ thuật

Khi chỉ đối chiếu AWF, mô hình tăng MLI khoảng 46,62%, CI khoảng 15,73% và entropy khoảng 5,90%. AG thấp hơn khoảng 51,26%. Đây là các chênh lệch tính bằng mô hình chia AWF rồi trừ một. Như vậy kết luận đúng là có ưu thế về ba chỉ số no-reference và Z ước tính theo mốc gốc, còn độ lớn gradient chưa đạt mốc AWF.

Với mỗi chỉ số j, Z thành phần được tính bằng x trừ mean của năm phương pháp gốc rồi chia std của năm giá trị trung bình đó. Sau đó lấy trung bình bốn Z thành phần. Các giá trị thành phần của mô hình xấp xỉ 5,0103; 1,7237; 2,9208 và âm 0,2312. MLI rất cao góp phần lớn vào Z tổng hợp.

Nếu ban chấm tính lại mean/std cho cả sáu phương pháp, Z của mọi phương pháp sẽ thay đổi. Khi đó không thể giữ nguyên AWF 1,4479 rồi trộn với Z đã chuẩn hóa theo nhóm sáu. Ngoài ra, chưa có mã AWF gốc để chạy lại cùng điều kiện, nên bảng hiện tại là đối chiếu với mốc công bố.



Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md


