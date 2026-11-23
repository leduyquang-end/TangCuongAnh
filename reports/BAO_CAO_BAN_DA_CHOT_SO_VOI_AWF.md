# Báo cáo bản đã chốt so với AWF

## Kiểm chứng cập nhật — 23/11/2026

Chạy lại checkpoint hiện tại trên đủ 100 ảnh test: 100/100 PNG khớp từng pixel; bốn chỉ số tính độc lập khớp báo cáo với sai số dưới 1e-14. PSNR/SSIM cũng tái lập. Đây là kết quả của toàn pipeline, gồm hậu xử lý; lần audit không huấn luyện hoặc chọn tham số.

**Chưa xác minh được tương đương chuẩn chấm cuộc thi.** Ngay trên 100 ảnh Input chưa tăng sáng, phép đo hiện tại đã khác hàng Input trong đề:

| Chỉ số Input | Đo tại dự án | Input công bố trong đề |
|---|---:|---:|
| MLI | 0.056857 | 0.0500 |
| CI | 0.042298 | 0.0443 |
| E | 4.716085 | 4.5533 |
| AG | 0.009555 | 0.0129 |

Chưa có mã chấm hoặc mã AWF chính thức để xác định nguyên nhân. Các chênh lệch AWF ở bảng dưới là đối chiếu với số công bố, không phải AWF được chạy lại bằng cùng evaluator. Z 2.3559 chỉ là ước tính từ mốc năm phương pháp ban đầu, không phải điểm do ban tổ chức xác nhận. Test từng được xem kết quả trong quá trình phát triển; không gọi đây là kiểm định độc lập hoàn toàn.

[Bằng chứng audit và công thức đo](../outputs/checks/metric_audit_20261123/audit.json) · [CSV đối chiếu 100 ảnh](../outputs/checks/metric_audit_20261123/per_image.csv)


Giữ bản đã được người dùng duyệt trước đợt tinh chỉnh giảm nhiễu. Checkpoint mặc định: `checkpoints/model_current.pt`. Số liệu dưới đây lấy đúng báo cáo 100 ảnh test đã lưu của bản này; không dùng số của ứng viên tinh chỉnh sau đó.

## Kết quả 100 ảnh test và mốc AWF trong đề

| Chỉ số | Bản đã chốt: 100 test | AWF trong đề | Chênh lệch so với mốc AWF |
|---|---:|---:|---:|
| MLI | 0.486328 | 0.3317 | +46.62% |
| CI | 0.227169 | 0.1963 | +15.73% |
| E | 7.644204 | 7.2184 | +5.90% |
| AG | 0.051272 | 0.1052 | -51.26% |

MLI, CI và E của bản đã chốt cao hơn các mốc AWF trong đề. AG vẫn thấp hơn; chưa thể nói mô hình vượt AWF ở mọi chỉ số.

## Điểm Z đối chiếu mốc gốc của đề

**Z AWF công bố trong đề là 1,4479.** Bảng chính giữ nguyên giá trị này. Z của mô hình được ước tính bằng cùng công thức với mean/std cố định từ năm phương pháp gốc: AWF, AGCWD, SDD, CLAHE và RRM.

```text
z_j = (x_j - mean_j) / std_j
Z = (z_MLI + z_CI + z_E + z_AG) / 4
```

Mean/std được tính từ năm giá trị trung bình của từng cột trong bảng đề, dùng độ lệch chuẩn population (ddof=0). Mô hình được chấm bằng mốc cố định này; không thêm mô hình vào dữ liệu tính mean/std.

| Phương pháp | Z theo mốc gốc | Nguồn |
|---|---:|---|
| AWF | 1.4479 | Công bố trong đề |
| Bản đã chốt | ≈ 2.3559 | Ước tính từ số đo 100 test và bảng mốc đã làm tròn |

Chênh lệch so với Z AWF công bố: khoảng +0.9080. Đây là đối chiếu theo mốc gốc; Z mô hình là giá trị tính lại của dự án, chưa phải điểm được ban chấm xác nhận.

### Các thành phần để kiểm tra phép tính

| Thành phần | Giá trị mô hình | Mean nhóm gốc | Std nhóm gốc | z của mô hình |
|---|---:|---:|---:|---:|
| MLI | 0.48632756 | 0.25346000 | 0.04647772 | 5.01030519 |
| CI | 0.22716896 | 0.14870000 | 0.04552419 | 1.72367615 |
| E | 7.64420412 | 6.67920000 | 0.33038990 | 2.92080395 |
| AG | 0.05127219 | 0.05878000 | 0.03246742 | -0.23124136 |

Lấy trung bình bốn thành phần z ở trên thu được Z mô hình = 2.35588598.

Lưu ý độ chính xác: dùng lại các số MLI/CI/E/AG đã làm tròn trong bảng đề sẽ cho Z AWF ≈ 1.44768428, lệch 0.00021572 so với giá trị công bố 1,4479. Không ép điều chỉnh công thức để triệt tiêu chênh lệch làm tròn; báo cáo giữ AWF công bố và ghi rõ Z mô hình là ước tính.

Bảng trước đã tính lại mean/std khi thêm mô hình vào nhóm sáu, làm Z AWF thành 0,873736. Giá trị đó thuộc một phép chuẩn hóa lại khác; đã bỏ khỏi bảng đối chiếu chính với đề để tránh nhầm lẫn. Các số đo MLI/CI/E/AG, checkpoint và tham số suy luận giữ nguyên.

Z cao hơn trong đối chiếu này vẫn không có nghĩa mô hình vượt AWF ở mọi chỉ số: AG của mô hình vẫn thấp hơn mốc AG AWF.

## Phạm vi và giới hạn

Số mô hình là thực đo trên 100 test LOL-v2-real, giữ nguyên các ID 00690–00789. Số AWF lấy từ CuocThi_XLA_1.pdf, trang 4; chưa có mã AWF/chấm gốc để chạy lại trên cùng dữ liệu và xác nhận hoàn toàn cùng công thức AG. Vì vậy đây là đối chiếu với mốc đề, không phải benchmark AWF được chạy lại cùng điều kiện.
Chỉ số MLI/CI/E/AG đo từ RGB uint8. Các giá trị PSNR/SSIM của mô hình lần lượt là 13,925115 dB và 0,513449; đề không cho các số tương ứng của AWF để so sánh. Z hoặc AG cao không đủ chứng minh ảnh ít nhiễu hoặc chi tiết trung thực hơn.
Checkpoint được chọn bằng validation và được khóa trước lần đánh giá test gốc. Lịch sử dự án đã từng xem test nên không coi đây là tập kiểm chứng mới độc lập. Lần lập báo cáo này chỉ đọc lại số liệu gốc, không huấn luyện, đổi tham số hay chọn theo test.

## Dữ liệu và tệp kiểm tra

- SHA256 checkpoint: `902c100e7455dd23946f9de7891c2e4f730427149f3604f9dcdabb4e79c747d0`.
- [Bảng số đo gốc 100 test](../experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json)
- [CSV chỉ số từng ảnh](../experiments/ag_refinement/run_four_metric_curriculum/test/per_image.csv)
- [100 ảnh RGB đầu ra](../experiments/ag_refinement/run_four_metric_curriculum/test/images/)
- [Ảnh đối chiếu đầu vào / bản cha / bản đã chốt / GT](../experiments/ag_refinement/run_four_metric_curriculum/test/comparison.png)
- [Ảnh các thành phần R/L/R′/L′ và đầu ra](../experiments/ag_refinement/run_four_metric_curriculum/test/stages/)
- [Đề thi chứa bảng AWF](../CuocThi_XLA_1.pdf)
- [Cấu hình triển khai](../configs/model.json)

Kiến trúc giữ KinD Decomposition + Attention U-Net + Curve CNN, 2.217.963 tham số. Pipeline và các scalar của bản đã chốt giữ nguyên.
