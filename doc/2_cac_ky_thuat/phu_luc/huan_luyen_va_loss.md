# Kiến trúc huấn luyện và loss của bản hiện tại

Checkpoint triển khai: `checkpoints/model_current.pt`. Công thức đã học và số liệu gốc ở `experiments/ag_refinement/run_four_metric_curriculum/`. Người dùng duyệt bản ngày 13/11/2026.

## Nhánh học và lan truyền ngược

Giữ nguyên KinD Decomposition, Attention U-Net đầu RGB sigmoid và Curve CNN. Trong lượt tinh chỉnh cuối, decomposition đóng băng; restoration và adjustment cùng học (1.982.087 tham số). Teacher là bản cha đóng băng tại thời điểm huấn luyện, dùng no_grad. Không thêm teacher vào mô hình suy luận.

PyTorch Autograd thực hiện loss.backward; clip norm gradient tối đa 5; Adam cập nhật trọng số. Hai giai đoạn, mỗi giai đoạn 3 epoch: mục tiêu AG 0,05, hệ số biên 2,5 và LR 6e-5; sau đó AG 0,1, hệ số biên 4 và LR 4e-5. Batch 8, crop 80×80, seed 827, crop/xoay/lật đồng bộ low/high. Có gradient hữu hạn ở restoration/adjustment, không có gradient ở decomposition.

## Công thức loss

```text
L = L1_RGB + (1 − SSIM)
  + 10 * khớp_biên_có_hướng
  + 80 * thiếu_hụt_AG_có_GT_hỗ_trợ
  + 12 * tăng_gradient_ngoài_hỗ_trợ
  + 100 * neo_mean
  + 8 * neo_CI
  + 4 * thiếu_hụt_soft_entropy
  + 0.2 * L1_luminance_với_teacher
```

Biên có hướng tính ở hai tỷ lệ 1/2. Chỉ khuếch đại hướng gradient teacher được GT xác nhận; support là độ lớn biên GT chuẩn hóa 0,025, chặn [0,1]. Mục tiêu có hướng chặn [-0,25;0,25]. Thành phần AG tuyệt đối yêu cầu phần thiếu tới đích chỉ trong vùng hỗ trợ và tích vô hướng gradient GT/teacher dương; patch GT phẳng không nhận yêu cầu tạo texture. Target và teacher được detach.

Mean patch được neo trong [1,005;1,02] lần teacher; CI patch trong [1,06;1,15] lần teacher. Soft entropy 64 bins phải ít nhất teacher+0,003. Đây là surrogate học; E báo cáo dùng histogram 256 bins từ RGB uint8. stable_calibration_features giữ giá trị forward nhưng tránh đạo hàm không hữu hạn tại norm gradient bằng 0.

Mã: `kind_enhance/training/losses/four_metrics.py`, `kind_enhance/training/losses/gradient_alignment.py`, `kind_enhance/training/losses/supported_edges.py`, `kind_enhance/training/losses/contrast_gradient.py`, `kind_enhance/training/engine.py`.

## Chọn và đánh giá

620 train, 69 validation tách từ train; test chính thức 100 ảnh. Giữ AG cao nhất trong các ứng viên có bốn trung bình MLI/CI/E/AG không giảm và CI >=0,22 trên validation. PSNR/SSIM, sai màu, vùng phẳng và clipping được ghi riêng, không giấu phần đánh đổi. DataGuard chặn test khi tìm tham số. Test mở sau khi khóa selection; không tinh chỉnh theo test.

Bản được duyệt có CI test 0,227169, AG 0,051272; AG chưa gần 0,1. PSNR/SSIM giảm và gradient vùng phẳng tăng. Việc người dùng chọn triển khai không thay đổi kết luận đo. Z được chuẩn hóa cả AWF và ứng viên trong cùng nhóm sáu ở báo cáo; không dùng Z để bù một chỉ số riêng giảm.

## Chạy lượt mới

Lệnh train hiện tại bắt đầu từ checkpoint được duyệt và dùng nó làm teacher; dataset/kiến trúc giữ nguyên. Bản cha lịch sử đã xóa theo yêu cầu nên không tuyên bố có thể tái lập từ đầu toàn bộ chuỗi huấn luyện cũ. Protocol, mã tại thời điểm chạy, log, metric và bản trọng số được duyệt vẫn giữ để kiểm tra.

`python -B -m kind_enhance train --epochs 6 --output experiments/next_run`

Thư mục output phải mới. Lệnh không tự thay checkpoint mặc định. Trường reference trong cấu hình cùng trỏ tới model_current.pt để dùng làm mốc cho lượt tiếp.
