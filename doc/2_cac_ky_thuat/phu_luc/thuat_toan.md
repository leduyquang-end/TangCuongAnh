# Thuật toán của bản đã duyệt

Checkpoint: `checkpoints/model_current.pt`; cấu hình: `configs/model.json`.

1. KinD Decomposition phân rã ảnh thành R ba kênh và L một kênh; giữ nguyên kiến trúc và đóng băng khi tinh chỉnh cuối.
2. Attention U-Net ghép R/L thành bốn kênh, dùng encoder/decoder năm mức, skip có attention gate và đầu RGB sigmoid để tạo R′.
3. Curve CNN dự đoán tám bản đồ hệ số tanh và thực hiện tám bước L_next = L + a*L*(1-L), tạo L′.
4. Nhân R′×L′, hiệu chỉnh contrast 1,08, detail 0,25 và sharpen 1,4.
5. Giữ màu RGB bằng chroma input làm trơn separable 15×15, blend 1,0 và gain 0,5; scale chroma vào gamut để giữ luminance. Vì blend=1, mạng cung cấp luminance, chroma cuối lấy từ input đã lọc.
6. Hiệu chỉnh luminance contrast 1,02 và lift 0,007; giữ hướng chroma và chặn gamut.
7. Adam, Autograd, clip norm 5, teacher đóng băng, mục tiêu biên có GT hỗ trợ ở hai tỷ lệ, loss L1/SSIM và neo mean/CI/entropy; xem [công thức loss](huan_luyen_va_loss.md).
8. Chọn checkpoint bằng validation theo bốn trung bình; test chỉ đánh giá sau khóa. AWF là số benchmark trong đề, không phải thuật toán ghép vào mạng.

Không dùng mô hình ngôn ngữ, Transformer, GAN hoặc mạng bổ sung. Ba mạng có tổng 2.217.963 tham số, 52 tầng convolution/transposed convolution. [Kiến trúc chi tiết](../2.1_architecture.md), [trace](model_trace.json), [báo cáo](../../../reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md).
