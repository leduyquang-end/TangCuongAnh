# Slide 16: Đo MLI, CI, E, AG và PSNR/SSIM

## Nội dung hiển thị

Bốn chỉ số no-reference dùng luminance g = (0,299R + 0,587G + 0,114B)/255.

AG cao có thể gồm cả gradient do nhiễu. CI cao không tự bảo đảm màu đúng.

| Chỉ số | Cách tính trong code |
| --- | --- |
| MLI / CI | Mean(g) / std(g), population |
| E | −Σ p(k)log₂p(k), histogram 256 bins |
| AG | mean(√((Δx² + Δy²)/2)), sai phân cùng gốc |
| PSNR | −10log₁₀(MSE), dải RGB [0,1] |
| SSIM | So cấu trúc với GT, Gaussian 11×11 |

## Lời thuyết trình và chi tiết kỹ thuật

Bốn chỉ số trong đề được tính từ luminance g bằng 0,299R cộng 0,587G cộng 0,114B, sau đó chia 255. MLI là trung bình g. CI là độ lệch chuẩn không gian của g. Entropy tính từ histogram 256 bins. AG lấy độ lớn gradient từ sai phân ngang và dọc có cùng pixel gốc, rồi trung bình.

PSNR và SSIM được đo thêm khi có GT để kiểm tra độ trung thực. Trong code, PSNR là âm 10 nhân log10 của MSE trên miền giá trị chuẩn hóa. SSIM dùng cửa sổ Gaussian 11×11. Các chỉ số này bổ sung thông tin mà bốn số no-reference chưa thể hiện hết.

MLI cao hơn không luôn có nghĩa ảnh đẹp hơn, vì ảnh có thể bị quá sáng. Entropy hoặc AG cao hơn cũng có thể liên quan đến nhiễu. Khi trình bày theo đề, nhóm vẫn đối chiếu đủ bốn chỉ số, đồng thời nêu các chỉ số tham chiếu và giới hạn thay vì chỉ chọn số có lợi.

https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html

Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md

https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html
