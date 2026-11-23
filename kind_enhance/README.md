# Mã nguồn kind_enhance

| Thư mục | Trách nhiệm |
|---|---|
| `application` | Quy trình xử lý ảnh và điều phối service |
| `ui` | Server localhost + giao diện HTML/CSS/JS; desktop tùy chọn |
| `models` | KinD Decomposition, Attention U-Net, Curve CNN |
| `inference` | Pipeline suy luận, calibration, RGB và luminance |
| `data` | Dataset cặp ảnh |
| `training` | Vòng học và curriculum |
| `training/losses` | Hàm mất mát, gom theo loại mục tiêu |
| `evaluation` | MLI/CI/E/AG, PSNR/SSIM, Z và đánh giá chất lượng |
| `engineering` | Điều phối thử nghiệm và bảo vệ split |
| `infrastructure` | Đọc/ghi ảnh, checkpoint, hash, log/artifact |
| `research` | Tiện ích nghiên cứu ngoài pipeline đã chốt |

Giao diện chạy từ `app.py` tại gốc dự án. Công cụ CLI: `python -m kind_enhance`. Chạy lệnh tại gốc dự án để Python tìm đúng package. Không chạy trực tiếp các file bên trong package.

[Cây thư mục và quy ước đầy đủ](../doc/1_tong_quan/1.1_cau_truc_du_an.md) · [Hướng dẫn sử dụng](../doc/1_tong_quan/1.2_huong_dan_su_dung.md)
