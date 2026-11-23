# Slide 18: So sánh với các phương pháp trong đề

## Nội dung hiển thị

Mean ± SD trên 100 ảnh test. Sắp xếp theo Z đối chiếu mốc gốc.

* Thứ tự đối chiếu, chưa phải xếp hạng chính thức. Z mô hình dùng μ/σ của 5 mốc gốc; không chuẩn hóa lại nhóm 6.

AWF giữ Z công bố 1,4479. Mô hình cao hơn về MLI, CI, E và Z ước tính, còn AG thấp hơn.

| Phương pháp | MLI | CI | E | AG | Z | Hạng |
| --- | --- | --- | --- | --- | --- | --- |
| Mô hình nhóm | 0,4863 ± 0,0461 | 0,2272 ± 0,0437 | 7,6442 ± 0,2369 | 0,0513 ± 0,0185 | ≈ +2,3559 | 1* |
| AWF | 0,3317 ± 0,0798 | 0,1963 ± 0,0482 | 7,2184 ± 0,4370 | 0,1052 ± 0,0424 | +1,4479 | 2 |
| AGCWD | 0,2259 ± 0,0706 | 0,2085 ± 0,0307 | 6,8703 ± 0,5093 | 0,0793 ± 0,0361 | +0,4826 | 3 |
| SDD | 0,2425 ± 0,0695 | 0,1192 ± 0,0313 | 6,5764 ± 0,3861 | 0,0240 ± 0,0143 | −0,5666 | 4 |
| CLAHE | 0,1948 ± 0,0748 | 0,1273 ± 0,0452 | 6,4399 ± 0,7977 | 0,0647 ± 0,0340 | −0,5687 | 5 |
| RRM | 0,2724 ± 0,0629 | 0,0922 ± 0,0262 | 6,2910 ± 0,3663 | 0,0207 ± 0,0138 | −0,7952 | 6 |
| Input (đề) | 0,0500 ± 0,0314 | 0,0443 ± 0,0286 | 4,5533 ± 0,7764 | 0,0129 ± 0,0067 | — | — |

## Lời thuyết trình và chi tiết kỹ thuật

Bảng này giữ cấu trúc của bảng trong đề: mỗi phương pháp có MLI, CI, E, AG, Z và thứ tự đối chiếu. Nhóm thêm mô hình hiện tại vào cùng AWF, AGCWD, SDD, CLAHE và RRM, rồi sắp xếp theo Z đối chiếu giảm dần. Input được giữ làm hàng tham khảo nhưng không tham gia chuẩn hóa Z.

Mô hình có MLI khoảng 0,4863, CI 0,2272, entropy 7,6442 và AG 0,0513. Ba chỉ số đầu cao nhất trong các mốc đang đối chiếu. AG thấp hơn AWF, AGCWD và CLAHE, cao hơn SDD và RRM. Do đó thứ tự Z đầu bảng không được diễn giải là thắng ở từng chỉ số.

AWF giữ điểm công bố 1,4479. Z mô hình khoảng 2,3559 được dự án ước tính bằng mean và std của năm mốc gốc, chưa chuẩn hóa lại theo nhóm sáu. Dấu sao ở hạng một nhắc đây là thứ tự đối chiếu tạm, chưa phải xếp hạng chính thức của cuộc thi. Các số mean±SD của phương pháp khác lấy từ đề; SD mô hình tính trên 100 ảnh. SD trong từng hàng không phải sigma dùng để chuẩn hóa Z.



Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md


