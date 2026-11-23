# Vòng tinh chỉnh từ bản đã duyệt

Giữ KinD Decomposition + Attention U-Net + Curve CNN. Checkpoint hiện tại trong configs/model.json là mốc khởi tạo/teacher cho lượt mới. Không tự thay bản mặc định.

```powershell
python -B -m kind_enhance train --epochs 6 --output experiments/next_run
python -B tools/evaluation/evaluate_four_metrics.py --output experiments/next_run --split val
python -B tools/evaluation/evaluate_four_metrics.py --output experiments/next_run --split test
```

Output phải mới; có thể chạy tiếp một lượt bị ngắt bằng `python -B tools/training/refine_four_metrics.py --output experiments/next_run --resume`. Test chỉ được mở khi đã có selection khóa và ứng viên qua bốn điều kiện validation; không dùng test để điều chỉnh tham số.

Hai giai đoạn 3 epoch hướng tới AG 0,05 rồi 0,1. Chỉ cập nhật restoration/adjustment, decomposition đóng băng. Chọn AG cao nhất với trung bình MLI/CI/E/AG không giảm và CI >=0,22. Các cờ nhiễu, màu, PSNR/SSIM và clipping được giữ trong báo cáo. Không bảo đảm mỗi lượt hoặc mọi ảnh đều tăng. Chi tiết [loss](../../2_cac_ky_thuat/phu_luc/huan_luyen_va_loss.md).

Run đã chốt giữ metadata/mã nguồn/ảnh/metric; checkpoint và optimizer trung gian cũ đã xóa theo yêu cầu nên không dùng --resume để tái huấn luyện run lịch sử. Một lượt hoàn tất chỉ trả trạng thái khi dùng --resume. Chạy mới lấy bản hiện tại, không đòi checkpoint các phiên bản cũ.
