# Phụ lục: toàn bộ tầng có tham số — đo trực tiếp bằng forward hook

Sinh bởi `python -B tools/inspection/inspect_model_architecture.py`. Input tổng hợp, không đọc train/validation/test, không cập nhật trọng số.

Checkpoint SHA-256: `902c100e7455dd23946f9de7891c2e4f730427149f3604f9dcdabb4e79c747d0`.

Thứ tự bảng là thứ tự thực thi forward. Kích thước tại Conv là trước activation; ConvTranspose là trước thao tác cắt. Pooling, concat, activation dạng functional, nhân/clip không có tham số nên được mô tả trong tài liệu kiến trúc chính.

Quy ước shape N×C×H×W; k/s/p là kernel/stride/padding; mọi convolution có dilation=1, groups=1.

| # | Module đúng tên mã | Loại | k/s/p | Bias | Input ảnh đầy đủ | Output ảnh đầy đủ | Output batch crop | Output kích thước lẻ | Tham số |
|---:|---|---|---|---|---|---|---|---|---:|
| 1 | `decomposition.first` | Conv2d | 3×3/1×1/1×1 | True | 1×3×400×600 | 1×32×400×600 | 8×32×80×80 | 1×32×81×79 | 896 |
| 2 | `decomposition.second` | Conv2d | 3×3/1×1/1×1 | True | 1×32×200×300 | 1×64×200×300 | 8×64×40×40 | 1×64×41×40 | 18,496 |
| 3 | `decomposition.bottom` | Conv2d | 3×3/1×1/1×1 | True | 1×64×100×150 | 1×128×100×150 | 8×128×20×20 | 1×128×21×20 | 73,856 |
| 4 | `decomposition.up_second` | ConvTranspose2d | 2×2/2×2/0×0 | False | 1×128×100×150 | 1×64×200×300 | 8×64×40×40 | 1×64×42×40 | 32,768 |
| 5 | `decomposition.decode_second` | Conv2d | 3×3/1×1/1×1 | True | 1×128×200×300 | 1×64×200×300 | 8×64×40×40 | 1×64×41×40 | 73,792 |
| 6 | `decomposition.up_first` | ConvTranspose2d | 2×2/2×2/0×0 | False | 1×64×200×300 | 1×32×400×600 | 8×32×80×80 | 1×32×82×80 | 8,192 |
| 7 | `decomposition.decode_first` | Conv2d | 3×3/1×1/1×1 | True | 1×64×400×600 | 1×32×400×600 | 8×32×80×80 | 1×32×81×79 | 18,464 |
| 8 | `decomposition.light_features` | Conv2d | 3×3/1×1/1×1 | True | 1×32×400×600 | 1×32×400×600 | 8×32×80×80 | 1×32×81×79 | 9,248 |
| 9 | `decomposition.reflectance` | Conv2d | 1×1/1×1/0×0 | True | 1×32×400×600 | 1×3×400×600 | 8×3×80×80 | 1×3×81×79 | 99 |
| 10 | `decomposition.illumination` | Conv2d | 1×1/1×1/0×0 | True | 1×64×400×600 | 1×1×400×600 | 8×1×80×80 | 1×1×81×79 | 65 |
| 11 | `restoration.encoders.0.0` | Conv2d | 3×3/1×1/1×1 | True | 1×4×400×600 | 1×16×400×600 | 8×16×80×80 | 1×16×81×79 | 592 |
| 12 | `restoration.encoders.0.2` | Conv2d | 3×3/1×1/1×1 | True | 1×16×400×600 | 1×16×400×600 | 8×16×80×80 | 1×16×81×79 | 2,320 |
| 13 | `restoration.encoders.1.0` | Conv2d | 3×3/1×1/1×1 | True | 1×16×200×300 | 1×32×200×300 | 8×32×40×40 | 1×32×41×40 | 4,640 |
| 14 | `restoration.encoders.1.2` | Conv2d | 3×3/1×1/1×1 | True | 1×32×200×300 | 1×32×200×300 | 8×32×40×40 | 1×32×41×40 | 9,248 |
| 15 | `restoration.encoders.2.0` | Conv2d | 3×3/1×1/1×1 | True | 1×32×100×150 | 1×64×100×150 | 8×64×20×20 | 1×64×21×20 | 18,496 |
| 16 | `restoration.encoders.2.2` | Conv2d | 3×3/1×1/1×1 | True | 1×64×100×150 | 1×64×100×150 | 8×64×20×20 | 1×64×21×20 | 36,928 |
| 17 | `restoration.encoders.3.0` | Conv2d | 3×3/1×1/1×1 | True | 1×64×50×75 | 1×128×50×75 | 8×128×10×10 | 1×128×11×10 | 73,856 |
| 18 | `restoration.encoders.3.2` | Conv2d | 3×3/1×1/1×1 | True | 1×128×50×75 | 1×128×50×75 | 8×128×10×10 | 1×128×11×10 | 147,584 |
| 19 | `restoration.encoders.4.0` | Conv2d | 3×3/1×1/1×1 | True | 1×128×25×38 | 1×256×25×38 | 8×256×5×5 | 1×256×6×5 | 295,168 |
| 20 | `restoration.encoders.4.2` | Conv2d | 3×3/1×1/1×1 | True | 1×256×25×38 | 1×256×25×38 | 8×256×5×5 | 1×256×6×5 | 590,080 |
| 21 | `restoration.decoders.0.up` | ConvTranspose2d | 2×2/2×2/0×0 | True | 1×256×25×38 | 1×128×50×76 | 8×128×10×10 | 1×128×12×10 | 131,200 |
| 22 | `restoration.decoders.0.skip_projection` | Conv2d | 1×1/1×1/0×0 | True | 1×128×50×75 | 1×64×50×75 | 8×64×10×10 | 1×64×11×10 | 8,256 |
| 23 | `restoration.decoders.0.gate_projection` | Conv2d | 1×1/1×1/0×0 | True | 1×128×50×75 | 1×64×50×75 | 8×64×10×10 | 1×64×11×10 | 8,256 |
| 24 | `restoration.decoders.0.score` | Conv2d | 1×1/1×1/0×0 | True | 1×64×50×75 | 1×1×50×75 | 8×1×10×10 | 1×1×11×10 | 65 |
| 25 | `restoration.decoders.0.fuse.0` | Conv2d | 3×3/1×1/1×1 | True | 1×256×50×75 | 1×128×50×75 | 8×128×10×10 | 1×128×11×10 | 295,040 |
| 26 | `restoration.decoders.0.fuse.2` | Conv2d | 3×3/1×1/1×1 | True | 1×128×50×75 | 1×128×50×75 | 8×128×10×10 | 1×128×11×10 | 147,584 |
| 27 | `restoration.decoders.1.up` | ConvTranspose2d | 2×2/2×2/0×0 | True | 1×128×50×75 | 1×64×100×150 | 8×64×20×20 | 1×64×22×20 | 32,832 |
| 28 | `restoration.decoders.1.skip_projection` | Conv2d | 1×1/1×1/0×0 | True | 1×64×100×150 | 1×32×100×150 | 8×32×20×20 | 1×32×21×20 | 2,080 |
| 29 | `restoration.decoders.1.gate_projection` | Conv2d | 1×1/1×1/0×0 | True | 1×64×100×150 | 1×32×100×150 | 8×32×20×20 | 1×32×21×20 | 2,080 |
| 30 | `restoration.decoders.1.score` | Conv2d | 1×1/1×1/0×0 | True | 1×32×100×150 | 1×1×100×150 | 8×1×20×20 | 1×1×21×20 | 33 |
| 31 | `restoration.decoders.1.fuse.0` | Conv2d | 3×3/1×1/1×1 | True | 1×128×100×150 | 1×64×100×150 | 8×64×20×20 | 1×64×21×20 | 73,792 |
| 32 | `restoration.decoders.1.fuse.2` | Conv2d | 3×3/1×1/1×1 | True | 1×64×100×150 | 1×64×100×150 | 8×64×20×20 | 1×64×21×20 | 36,928 |
| 33 | `restoration.decoders.2.up` | ConvTranspose2d | 2×2/2×2/0×0 | True | 1×64×100×150 | 1×32×200×300 | 8×32×40×40 | 1×32×42×40 | 8,224 |
| 34 | `restoration.decoders.2.skip_projection` | Conv2d | 1×1/1×1/0×0 | True | 1×32×200×300 | 1×16×200×300 | 8×16×40×40 | 1×16×41×40 | 528 |
| 35 | `restoration.decoders.2.gate_projection` | Conv2d | 1×1/1×1/0×0 | True | 1×32×200×300 | 1×16×200×300 | 8×16×40×40 | 1×16×41×40 | 528 |
| 36 | `restoration.decoders.2.score` | Conv2d | 1×1/1×1/0×0 | True | 1×16×200×300 | 1×1×200×300 | 8×1×40×40 | 1×1×41×40 | 17 |
| 37 | `restoration.decoders.2.fuse.0` | Conv2d | 3×3/1×1/1×1 | True | 1×64×200×300 | 1×32×200×300 | 8×32×40×40 | 1×32×41×40 | 18,464 |
| 38 | `restoration.decoders.2.fuse.2` | Conv2d | 3×3/1×1/1×1 | True | 1×32×200×300 | 1×32×200×300 | 8×32×40×40 | 1×32×41×40 | 9,248 |
| 39 | `restoration.decoders.3.up` | ConvTranspose2d | 2×2/2×2/0×0 | True | 1×32×200×300 | 1×16×400×600 | 8×16×80×80 | 1×16×82×80 | 2,064 |
| 40 | `restoration.decoders.3.skip_projection` | Conv2d | 1×1/1×1/0×0 | True | 1×16×400×600 | 1×8×400×600 | 8×8×80×80 | 1×8×81×79 | 136 |
| 41 | `restoration.decoders.3.gate_projection` | Conv2d | 1×1/1×1/0×0 | True | 1×16×400×600 | 1×8×400×600 | 8×8×80×80 | 1×8×81×79 | 136 |
| 42 | `restoration.decoders.3.score` | Conv2d | 1×1/1×1/0×0 | True | 1×8×400×600 | 1×1×400×600 | 8×1×80×80 | 1×1×81×79 | 9 |
| 43 | `restoration.decoders.3.fuse.0` | Conv2d | 3×3/1×1/1×1 | True | 1×32×400×600 | 1×16×400×600 | 8×16×80×80 | 1×16×81×79 | 4,624 |
| 44 | `restoration.decoders.3.fuse.2` | Conv2d | 3×3/1×1/1×1 | True | 1×16×400×600 | 1×16×400×600 | 8×16×80×80 | 1×16×81×79 | 2,320 |
| 45 | `restoration.head` | Conv2d | 1×1/1×1/0×0 | True | 1×16×400×600 | 1×3×400×600 | 8×3×80×80 | 1×3×81×79 | 51 |
| 46 | `adjustment.stem` | Conv2d | 3×3/1×1/1×1 | True | 1×1×400×600 | 1×16×400×600 | 8×16×80×80 | 1×16×81×79 | 160 |
| 47 | `adjustment.encoder.0` | Conv2d | 3×3/1×1/1×1 | True | 1×16×400×600 | 1×16×400×600 | 8×16×80×80 | 1×16×81×79 | 2,320 |
| 48 | `adjustment.encoder.1` | Conv2d | 3×3/1×1/1×1 | True | 1×16×400×600 | 1×16×400×600 | 8×16×80×80 | 1×16×81×79 | 2,320 |
| 49 | `adjustment.encoder.2` | Conv2d | 3×3/1×1/1×1 | True | 1×16×400×600 | 1×16×400×600 | 8×16×80×80 | 1×16×81×79 | 2,320 |
| 50 | `adjustment.decoder.0` | Conv2d | 3×3/1×1/1×1 | True | 1×32×400×600 | 1×16×400×600 | 8×16×80×80 | 1×16×81×79 | 4,624 |
| 51 | `adjustment.decoder.1` | Conv2d | 3×3/1×1/1×1 | True | 1×32×400×600 | 1×16×400×600 | 8×16×80×80 | 1×16×81×79 | 4,624 |
| 52 | `adjustment.coefficients` | Conv2d | 3×3/1×1/1×1 | True | 1×32×400×600 | 1×8×400×600 | 8×8×80×80 | 1×8×81×79 | 2,312 |

Tổng: **52 tầng convolution/transposed convolution**, **2,217,963 tham số**.

[Tài liệu kiến trúc chính](../2.1_architecture.md) · [Dữ liệu trace đầy đủ](model_trace.json)
