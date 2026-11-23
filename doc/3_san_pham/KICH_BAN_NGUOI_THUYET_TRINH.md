# Kịch bản dành cho người thuyết trình

Khớp với bộ 21 slide kiến trúc đơn giản. Mỗi phần dưới đây có gợi ý chỉ hình, lời đọc trực tiếp và câu chuyển sang slide kế tiếp.

Cách dùng: mở file này bên cạnh màn hình trình chiếu, hoặc đọc cùng nội dung trong phần Notes của PowerPoint. Khi trình bày bằng Presenter View, người nói có thể nhìn ghi chú của slide hiện tại. Các dòng “Nhìn vào” là chỉ dẫn thao tác, không cần đọc thành lời.

## Mục lục

- [01. Nội dung trình bày](#slide-01)
- [02. Kiến trúc tổng quan](#slide-02)
- [03. Nhánh KinD Decomposition](#slide-03)
- [04. Nhánh Attention U-Net](#slide-04)
- [05. Attention Gate](#slide-05)
- [06. Đầu ra RGB của Attention U-Net](#slide-06)
- [07. Nhánh Curve CNN](#slide-07)
- [08. Tám bước điều chỉnh đường cong](#slide-08)
- [09. Tái tạo ảnh và hiệu chỉnh](#slide-09)
- [10. Giữ màu ở đầu ra](#slide-10)
- [11. Dữ liệu huấn luyện và augmentation](#slide-11)
- [12. Backpropagation và Adam](#slide-12)
- [13. Hàm mất mát kết hợp](#slide-13)
- [14. Gradient theo biên được GT hỗ trợ](#slide-14)
- [15. Neo MLI, CI và entropy khi học](#slide-15)
- [16. Đo MLI, CI, E, AG và PSNR/SSIM](#slide-16)
- [17. Chọn mô hình trên validation](#slide-17)
- [18. So sánh với các phương pháp trong đề](#slide-18)
- [19. Đối chiếu bốn chỉ số với mốc AWF](#slide-19)
- [20. Độ trung thực và gradient vùng phẳng](#slide-20)
- [21. Suy luận và tài nguyên triển khai](#slide-21)

<a id="slide-01"></a>

## Slide 01: Nội dung trình bày

**Nhìn vào:** Chỉ vào năm phần trong mục lục.

### Lời đọc trực tiếp

Em trình bày mô hình tăng cường ảnh thiếu sáng gồm ba nhánh: phân rã KinD, phục hồi phản xạ bằng Attention U-Net và điều chỉnh illumination bằng Curve CNN. Nội dung có năm phần. Trước hết em giải thích luồng dữ liệu và vai trò từng nhánh. Sau đó em đi sâu vào encoder, decoder và attention gate, rồi giải thích cách Curve CNN dự đoán hệ số và lặp đường cong. Tiếp theo là cách học, hàm mất mát và quy trình chọn mô hình. Cuối cùng em đối chiếu các chỉ số với bảng mốc của cuộc thi. Các URL ở cuối trang mở nghiên cứu gốc tương ứng; kích thước và tham số em trình bày lấy từ code đang chạy.

**Chuyển ý:** Trước hết, em trình bày luồng xử lý chung của mô hình.

### Tra cứu thêm

https://arxiv.org/abs/1905.04161

https://arxiv.org/abs/1804.03999

https://arxiv.org/abs/2001.06826

---

<a id="slide-02"></a>

## Slide 02: Kiến trúc tổng quan

**Nhìn vào:** Đi từ ảnh tối qua KinD, tách hai nhánh rồi ghép tại khối nhân.

### Lời đọc trực tiếp

Ảnh đầu vào I là RGB, có shape N×3×H×W. N là số ảnh trong batch, C là số kênh, H và W là chiều cao và chiều rộng. Với ảnh 600×400, tensor có kích thước 1×3×400×600. Nhánh phân rã tạo R ba kênh và L một kênh. R biểu diễn phản xạ, còn L biểu diễn chiếu sáng theo mô hình Retinex.

R và L cùng đi vào Attention U-Net để ước lượng R phẩy. Đồng thời, L đi vào Curve CNN để tạo L phẩy. Hai nhánh tăng cường xử lý song song về mặt cấu trúc; Curve CNN không nhận ảnh RGB sau U-Net làm đầu vào. Đầu ra mạng là R phẩy nhân L phẩy, trong đó L một kênh được broadcast trên ba kênh RGB.

Mô hình có 2.217.963 tham số và 52 tầng convolution hoặc transposed convolution. Ý tưởng tách phản xạ và chiếu sáng dựa trên KinD. Các cấu hình tầng cụ thể là triển khai hiện tại của dự án. Đầu ra mạng còn qua hậu xử lý trước khi lưu và tính chỉ số.

**Chuyển ý:** Em đi vào nhánh đầu tiên là phân rã KinD.

### Tra cứu thêm

https://arxiv.org/abs/1905.04161

---

<a id="slide-03"></a>

## Slide 03: Nhánh KinD Decomposition

**Nhìn vào:** Chỉ encoder, decoder và hai đầu ra R/L.

### Lời đọc trực tiếp

Nhánh phân rã có encoder và decoder nhỏ. Conv đầu tiên biến ảnh ba kênh thành 32 kênh đặc trưng. Sau hai lần max pooling, số kênh tăng thành 64 và 128, còn chiều cao và chiều rộng giảm. Với ảnh 400×600, ba mức lần lượt là 32×400×600, 64×200×300 và 128×100×150.

Decoder dùng transposed convolution kernel 2, stride 2 để tăng kích thước. Ở mỗi mức, đặc trưng upsample được ghép với skip từ encoder rồi đưa qua convolution. Khi kích thước lẻ, tensor được cắt theo shape của skip trước khi concat, tránh lỗi lệch kích thước.

Head R dùng Conv1×1 từ 32 xuống ba kênh và sigmoid. Head L ghép 32 kênh đặc trưng ánh sáng từ mức đầu với 32 kênh decoded, rồi Conv1×1 tạo một kênh và sigmoid. Nhánh này có 235.876 tham số. Trong lượt tinh chỉnh cuối, nhóm đóng băng nhánh phân rã để tập trung học hai nhánh tăng cường.

**Chuyển ý:** Sau khi có R và L, em trình bày nhánh phục hồi phản xạ.

### Tra cứu thêm

https://arxiv.org/abs/1905.04161

---

<a id="slide-04"></a>

## Slide 04: Nhánh Attention U-Net

**Nhìn vào:** Chỉ luồng chính, sau đó đường skip phía trên.

### Lời đọc trực tiếp

Attention U-Net phục hồi phản xạ. Đầu vào là concat R ba kênh với L một kênh, thành bốn kênh. Việc đưa thêm L cung cấp thông tin vùng đang thiếu sáng bên cạnh nội dung phản xạ. Đây là giải thích vai trò thiết kế, chưa phải kết quả ablation chứng minh riêng tác dụng của L.

Encoder có năm mức 16, 32, 64, 128 và 256 kênh. Mỗi mức gồm hai Conv3×3 đi cùng LeakyReLU. Bốn lần pooling tạo đặc trưng ở nhiều tỷ lệ. Với ảnh 400×600, các shape không gian lần lượt là 400×600, 200×300, 100×150, 50×75 và 25×38 vì pooling dùng ceil_mode.

Decoder đi ngược lại từ 256 về 16 kênh. Mỗi bước dùng transposed convolution, điều tiết skip bằng attention, concat rồi double convolution. Skip mang lại đặc trưng độ phân giải cao mà pooling đã làm giảm. Ý tưởng encoder–decoder và skip kế thừa U-Net, còn gate dựa trên Attention U-Net. Dự án áp dụng chúng cho hồi quy ảnh, trong khi bài gốc đánh giá trên phân đoạn y sinh.

**Chuyển ý:** Điểm quan trọng trong decoder là attention gate, em giải thích ở phần tiếp theo.

### Tra cứu thêm

https://arxiv.org/abs/1505.04597

https://arxiv.org/abs/1804.03999

---

<a id="slide-05"></a>

## Slide 05: Attention Gate

**Nhìn vào:** Đi theo hai nhánh Conv 1×1, khối cộng và sigmoid.

### Lời đọc trực tiếp

Attention gate trong code là additive spatial attention. Có hai đầu vào: skip từ encoder và up từ decoder ở cùng độ phân giải. Hai Conv1×1 riêng chiếu chúng từ C kênh xuống C/2 kênh. Sau đó cộng hai tensor, đi qua ReLU, rồi một Conv1×1 khác đưa về một kênh. Sigmoid tạo mask alpha trong khoảng từ 0 đến 1.

Mask có shape N×1×H×W, được broadcast khi nhân với skip N×C×H×W. Như vậy mask điều tiết từng vị trí không gian. Skip đã điều tiết mới concat với up để phục hồi đặc trưng.

Điểm cần phân biệt là gate này không phải self-attention trong Transformer. Không có Q, K, V hoặc ma trận liên hệ mọi cặp pixel. Gate cũng không tự đảm bảo loại sạch nhiễu; đó là một cơ chế học phụ thuộc dữ liệu và loss. Ví dụ tại mức 128 kênh, hai projection tạo 64 kênh, score tạo một kênh, concat lại thành 256 kênh trước khi fuse về 128.

**Chuyển ý:** Sau khi kết hợp đặc trưng, mạng cần tạo lại đầu ra phản xạ RGB.

### Tra cứu thêm

https://arxiv.org/abs/1804.03999

---

<a id="slide-06"></a>

## Slide 06: Đầu ra RGB của Attention U-Net

**Nhìn vào:** Chỉ lần lượt đặc trưng decoder, Conv 1×1, sigmoid và R′.

### Lời đọc trực tiếp

Head cuối của Attention U-Net dùng Conv1×1 từ 16 xuống ba kênh rồi sigmoid để tạo R phẩy RGB. Sigmoid giới hạn từng giá trị trong khoảng từ 0 đến 1, phù hợp miền phản xạ mà mô hình dùng. Tuy nhiên giới hạn miền giá trị không có nghĩa là mô hình tự khôi phục đúng màu ảnh chuẩn.

Màu còn phụ thuộc mục tiêu huấn luyện và các bước sau mạng. Ở bản cấu hình đã chốt, bước giữ chroma dùng blend bằng 1. Vì vậy chroma cuối lấy từ input đã làm trơn, trong khi luminance chịu ảnh hưởng của đầu ra mạng. Em không diễn giải đầu ra RGB là bằng chứng toàn bộ màu đều do U-Net học ra.

Điều này cũng giải thích vì sao tăng độ sáng hoặc tăng gradient chưa đảm bảo màu giống GT. Muốn kết luận về màu phải đo riêng sai màu hoặc đánh giá với ảnh chuẩn, thay vì chỉ dựa vào MLI, CI, E và AG.

**Chuyển ý:** Tiếp theo là nhánh thứ hai, điều chỉnh illumination bằng Curve CNN.

### Tra cứu thêm

https://arxiv.org/abs/1804.03999

---

<a id="slide-07"></a>

## Slide 07: Nhánh Curve CNN

**Nhìn vào:** Chỉ nhánh trên dự đoán hệ số, rồi nhánh dưới lặp curve.

### Lời đọc trực tiếp

Curve CNN nhận L một kênh và dự đoán các hệ số điều chỉnh ánh sáng theo pixel. Mạng có bảy lớp Conv3×3 với padding 1 nên giữ nguyên kích thước không gian. Stem đưa một kênh thành 16 kênh. Ba convolution encoder giữ 16 kênh và lưu các đặc trưng trung gian. Hai convolution decoder ghép skip để có 32 kênh đầu vào, rồi đưa về 16 kênh.

Lớp cuối ghép với skip từ stem, nhận 32 kênh và tạo tám kênh hệ số. Tanh giới hạn mỗi hệ số trong khoảng âm một đến một. Cả nhánh có 18.680 tham số. Tám map này sẽ dùng cho tám bước biến đổi illumination, không phải tám ảnh đầu ra độc lập.

Dự án kế thừa công thức đường cong từ Zero-DCE nhưng có điều chỉnh rõ ràng. Zero-DCE gốc sử dụng curve trên RGB và học không cần ảnh tham chiếu. Nhánh hiện tại tác động lên L một kênh và học cùng ảnh GT thông qua loss của mô hình ghép. Vì vậy em gọi đây là Curve CNN dựa trên ý tưởng Zero-DCE.

**Chuyển ý:** Sau khi có hệ số, illumination được thay đổi qua tám bước như sau.

### Tra cứu thêm

https://arxiv.org/abs/2001.06826

---

<a id="slide-08"></a>

## Slide 08: Tám bước điều chỉnh đường cong

**Nhìn vào:** Đi từ L₀ đến L′, dừng ở công thức cập nhật một bước.

### Lời đọc trực tiếp

Sau khi có tám bản đồ hệ số, illumination được cập nhật tám lần theo công thức L tiếp theo bằng L hiện tại cộng a nhân L nhân một trừ L. Nếu hệ số a dương, phép biến đổi làm sáng các giá trị trung gian. Nếu a âm, nó có thể giảm sáng. Mức biến đổi phụ thuộc cả hệ số lẫn giá trị pixel.

Do hệ số phụ thuộc vị trí, vùng tối và vùng sáng có thể nhận cách điều chỉnh khác nhau. Các hệ số được CNN dự đoán một lần rồi dùng lần lượt trong vòng lặp. Đầu ra L phẩy sau tám bước mới nhân với R phẩy.

Khi coi a là cố định, đạo hàm theo L là 1+a(1−2L), không âm trong miền đã nêu. Đây là tính chất của phép curve. Với toàn mạng, a lại phụ thuộc vào L nên đạo hàm tổng còn đi qua mạng dự đoán hệ số. Em phân biệt hai trường hợp này để tránh kết luận quá rộng rằng toàn bộ CNN là một ánh xạ đơn điệu. Tất cả các phép cập nhật đều cho phép Autograd tính gradient.

**Chuyển ý:** Hai nhánh đã tạo R phẩy và L phẩy; bước tiếp theo là tái tạo ảnh.

### Tra cứu thêm

https://arxiv.org/abs/2001.06826

---

<a id="slide-09"></a>

## Slide 09: Tái tạo ảnh và hiệu chỉnh

**Nhìn vào:** Chỉ R′ và L′ đi vào khối nhân, sau đó calibration.

### Lời đọc trực tiếp

Đầu ra trực tiếp của mạng là Yraw bằng R phẩy nhân L phẩy. Nhưng ảnh được lưu và chấm điểm trong dự án là ảnh sau hậu xử lý. Calibration cộng thêm ba thành phần vào Yraw: đặc trưng tương phản, chi tiết từ ảnh đầu vào và thành phần sharpen. Hệ số lần lượt là 0,08; 0,25 và 1,4 theo cấu hình đã duyệt.

Đặc trưng tương phản lấy luminance trừ mean không gian. Thành phần detail dùng input nâng lũy thừa 0,4 và các bộ lọc trung bình, có ngưỡng để chặn biên độ. Thành phần sharpen dựa vào phần chênh giữa prediction và ảnh blur, có trọng số theo gradient. Cuối bước có clamp về khoảng hợp lệ.

Các hệ số này không phải trọng số của một CNN bổ sung. Đây là hiệu chỉnh được dự án triển khai. Khi báo cáo kết quả, cần nói rõ số đo là của toàn pipeline, vì chỉ chạy ba mạng rồi dừng ở Yraw sẽ chưa tái tạo đúng ảnh cuối đã công bố.

**Chuyển ý:** Ảnh sau calibration còn đi qua bước giữ màu ở đầu ra.

### Tra cứu thêm

https://arxiv.org/abs/1905.04161

---

<a id="slide-10"></a>

## Slide 10: Giữ màu ở đầu ra

**Nhìn vào:** Chỉ nhánh input cung cấp chroma và nhánh Ycal cung cấp luminance.

### Lời đọc trực tiếp

Bước giữ màu bắt đầu từ luminance y của ảnh đã calibration. Input được làm trơn bằng average pooling ngang và dọc với kernel 15, nhằm lấy thông tin màu ít dao động hơn. Chroma nguồn là độ lệch RGB so với luminance của input đã lọc, được chuẩn hóa rồi nhân với luminance đầu ra và gain 0,5.

Ở cấu hình blend bằng 1, chroma nguồn thay phần chroma của prediction. Code còn trừ luminance còn dư trong chroma và co biên độ theo gamut để RGB nằm trong khoảng hợp lệ. Sau đó luminance được tăng contrast 1,02 quanh mean và lift 0,007, đồng thời tiếp tục giữ hướng chroma.

Ý nghĩa là tách điều chỉnh mức sáng khỏi hướng màu. Dù vậy màu vẫn có thể lệch GT, vì input tối đã chứa nhiễu hoặc thiếu thông tin màu. Các bước này giữ nguyên scalar trong bản triển khai; chúng không thay đổi kiến trúc ba nhánh.

**Chuyển ý:** Em chuyển sang dữ liệu và cách huấn luyện các nhánh này.

---

<a id="slide-11"></a>

## Slide 11: Dữ liệu huấn luyện và augmentation

**Nhìn vào:** Chỉ từng dòng train, validation và test.

### Lời đọc trực tiếp

Dữ liệu dùng các cặp ảnh thiếu sáng và ảnh đủ sáng trong LOL-v2-real đã có trên máy. Manifest của dự án chia 620 cặp train, 69 cặp validation và 100 cặp test. Validation tách từ dữ liệu train, còn ID test được giữ nguyên từ 00690 đến 00789.

Trong train, ảnh được crop 80×80 với batch size 8. Cặp low/high luôn dùng cùng tọa độ crop, cùng phép xoay 90 độ và cùng phép lật. Nhờ vậy pixel của input vẫn tương ứng với GT sau augmentation. RGB uint8 được đưa về float32 trong khoảng 0 đến 1, không chuẩn hóa theo mean/std ImageNet.

Nguồn dữ liệu được truy vết qua repository tác giả của công trình DRBN và manifest cục bộ. Các con số chia tập em đang nêu là split hiện tại của dự án. Em không sử dụng GT khi người dùng đưa một ảnh mới vào giao diện để suy luận.

**Chuyển ý:** Với các cặp dữ liệu đó, mô hình cập nhật trọng số như thế nào?

### Tra cứu thêm

https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_From_Fidelity_to_Perceptual_Quality_A_Semi-Supervised_Approach_for_Low-Light_CVPR_2020_paper.html

---

<a id="slide-12"></a>

## Slide 12: Backpropagation và Adam

**Nhìn vào:** Chỉ hai giai đoạn học và chuỗi cập nhật trọng số.

### Lời đọc trực tiếp

Ở lượt tinh chỉnh cuối, decomposition đóng băng. Restoration và adjustment cùng học với tổng cộng 1.982.087 tham số. Teacher là bản cha cố định, tính đầu ra bằng no_grad và không nhận cập nhật.

Vòng học gồm xóa gradient cũ, forward, tính loss, loss.backward, clip gradient norm tối đa 5 rồi Adam.step. Autograd tính đạo hàm qua hai nhánh và phép nhân tái hợp. Adam dùng các ước lượng moment để điều chỉnh bước cập nhật. Đây là cơ chế tối ưu, còn chất lượng ảnh phụ thuộc đồng thời kiến trúc, dữ liệu và loss.

Lượt cuối có hai giai đoạn, mỗi giai đoạn ba epoch. Learning rate lần lượt là 6 nhân 10 mũ âm 5 và 4 nhân 10 mũ âm 5. Mục tiêu AG trong loss lần lượt là 0,05 và 0,10. Cần nhấn mạnh 0,10 là giá trị đích để tính thiếu hụt trong loss, không phải AG test mà mô hình đã đạt.

**Chuyển ý:** Để xác định hướng cập nhật, nhóm sử dụng hàm mất mát sau.

### Tra cứu thêm

https://arxiv.org/abs/1412.6980

---

<a id="slide-13"></a>

## Slide 13: Hàm mất mát kết hợp

**Nhìn vào:** Chỉ từng nhóm loss và trọng số tương ứng.

### Lời đọc trực tiếp

Loss tổng gồm các nhóm mục tiêu có vai trò khác nhau. L1 RGB và một trừ SSIM khuyến khích ảnh gần GT ở mức pixel và cấu trúc. Khớp biên có hướng có trọng số 10. Thiếu AG trong vùng GT hỗ trợ có trọng số 80. Phần phạt gradient tăng ngoài vùng hỗ trợ có trọng số 12.

Tiếp theo, neo mean có trọng số 100, neo độ lệch chuẩn có trọng số 8, thiếu soft entropy có trọng số 4 và L1 luminance với teacher có trọng số 0,2. Các hệ số nhằm cân bằng mục tiêu tăng cường với việc giữ một số thuộc tính của bản cha. Giá trị 80 lớn hơn 10 chưa đủ kết luận nhóm loss thứ nhất chi phối tám lần, vì các thành phần có thang đo khác nhau.

Phần SSIM dựa trên nghiên cứu của Wang và cộng sự. Cách phối hợp các loss và hệ số cụ thể là công thức của dự án. Loss là hàm mục tiêu để tối ưu, không phải bảo đảm toán học rằng mọi ảnh test đều tăng tất cả chỉ số.

**Chuyển ý:** Em giải thích kỹ hơn thành phần loss liên quan đến gradient và AG.

### Tra cứu thêm

https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html

---

<a id="slide-14"></a>

## Slide 14: Gradient theo biên được GT hỗ trợ

**Nhìn vào:** Chỉ mask hỗ trợ, gradient đích và phí ngoài vùng hỗ trợ.

### Lời đọc trực tiếp

Để tăng gradient có định hướng, code tính sai phân theo từng trục x và y. Mask hỗ trợ tăng theo độ lớn biên GT, chuẩn hóa bởi 0,025. Mask chỉ giữ vùng mà gradient GT và teacher cùng dấu. Những pixel đó có căn cứ từ cả ảnh chuẩn và cấu trúc của bản cha.

Gradient đích bắt đầu từ gradient teacher và được tăng biên độ theo mask cùng hệ số gain. Đích được chặn trong khoảng âm 0,25 đến 0,25. Prediction chịu phạt nếu lệch gradient đích tại vùng có hỗ trợ. Phép tính thực hiện ở tỷ lệ gốc và tỷ lệ giảm hai lần.

Ngoài vùng hỗ trợ, code phạt phần gradient prediction tăng vượt teacher. Một thành phần khác đo thiếu hụt AG có hỗ trợ so với mục tiêu tuyệt đối. Nếu vùng hỗ trợ quá ít, nhánh không nhận yêu cầu tạo texture mới. Đây là cách thiết kế để hướng việc tăng AG vào cấu trúc có căn cứ. Tuy nhiên log vẫn cho thấy gradient vùng phẳng tăng, nên cơ chế đó chưa khử hết đánh đổi nhiễu.

**Chuyển ý:** Bên cạnh gradient, nhóm còn neo các thống kê độ sáng và tương phản.

---

<a id="slide-15"></a>

## Slide 15: Neo MLI, CI và entropy khi học

**Nhìn vào:** Chỉ khoảng neo mean, CI và mục tiêu soft entropy.

### Lời đọc trực tiếp

Ba ràng buộc neo dùng thống kê patch luminance. Mean của prediction được khuyến khích nằm trong khoảng 1,005 đến 1,02 lần teacher. Độ lệch chuẩn được neo trong khoảng 1,06 đến 1,15 lần teacher. Nếu nằm ngoài khoảng, hàm ReLU tạo phí thiếu hoặc vượt.

Soft entropy của prediction được khuyến khích cao hơn teacher ít nhất 0,003. Trong train, soft entropy dùng 64 bins với đóng góp mềm để có gradient. Trong báo cáo, entropy dùng histogram cứng 256 bins từ ảnh uint8. Vì vậy hai đại lượng có liên quan nhưng không thể gọi là hoàn toàn giống nhau.

Thành phần L1 luminance với teacher giữ đầu ra khỏi trôi quá xa. Các neo này là mục tiêu khả vi ở mức patch. Khi đánh giá toàn ảnh, mean và độ lệch chuẩn còn phụ thuộc phân bố không gian và hậu xử lý, nên vẫn phải đo trực tiếp trên validation.

**Chuyển ý:** Sau khi học, các chỉ số được đo trực tiếp theo các công thức sau.

---

<a id="slide-16"></a>

## Slide 16: Đo MLI, CI, E, AG và PSNR/SSIM

**Nhìn vào:** Đi lần lượt qua MLI, CI, E, AG rồi PSNR/SSIM.

### Lời đọc trực tiếp

Bốn chỉ số trong đề được tính từ luminance g bằng 0,299R cộng 0,587G cộng 0,114B, sau đó chia 255. MLI là trung bình g. CI là độ lệch chuẩn không gian của g. Entropy tính từ histogram 256 bins. AG lấy độ lớn gradient từ sai phân ngang và dọc có cùng pixel gốc, rồi trung bình.

PSNR và SSIM được đo thêm khi có GT để kiểm tra độ trung thực. Trong code, PSNR là âm 10 nhân log10 của MSE trên miền giá trị chuẩn hóa. SSIM dùng cửa sổ Gaussian 11×11. Các chỉ số này bổ sung thông tin mà bốn số no-reference chưa thể hiện hết.

MLI cao hơn không luôn có nghĩa ảnh đẹp hơn, vì ảnh có thể bị quá sáng. Entropy hoặc AG cao hơn cũng có thể liên quan đến nhiễu. Khi trình bày theo đề, nhóm vẫn đối chiếu đủ bốn chỉ số, đồng thời nêu các chỉ số tham chiếu và giới hạn thay vì chỉ chọn số có lợi.

**Chuyển ý:** Từ các chỉ số này, nhóm chọn checkpoint trên validation.

### Tra cứu thêm

https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html

---

<a id="slide-17"></a>

## Slide 17: Chọn mô hình trên validation

**Nhìn vào:** Chỉ điều kiện chọn checkpoint và lưu ý lịch sử test.

### Lời đọc trực tiếp

Checkpoint được chọn trên 69 ảnh validation. Điều kiện chính là bốn trung bình MLI, CI, E và AG không thấp hơn bản cha, đồng thời CI đạt ít nhất 0,22. Trong các ứng viên thỏa điều kiện, chọn AG cao nhất. Sự kiện được chọn trong log là giai đoạn ag_010, epoch 3.

DataGuard giới hạn phân vùng được đọc khi huấn luyện và tìm tham số. Sau khi khóa selection, nhóm đánh giá trên 100 ảnh test. Cập nhật trọng số bằng test là điều không thực hiện trong lượt cuối. Tuy nhiên chỉ không backward trên test chưa đủ để bảo đảm độc lập nếu nhìn test rồi chọn tham số.

Vì lịch sử dự án đã từng xem kết quả test, em ghi rõ rằng đây không phải một tập kiểm chứng hoàn toàn mới chưa từng tiếp xúc. Đây là giới hạn của quy trình thực nghiệm và cần được nêu khi bảo vệ kết quả.

**Chuyển ý:** Tiếp theo là bảng kết quả của checkpoint đã chọn và các mốc trong đề.

---

<a id="slide-18"></a>

## Slide 18: So sánh với các phương pháp trong đề

**Nhìn vào:** Chỉ hàng mô hình, hàng AWF, cột AG và cột Z.

### Lời đọc trực tiếp

Bảng này giữ cấu trúc của bảng trong đề: mỗi phương pháp có MLI, CI, E, AG, Z và thứ tự đối chiếu. Nhóm thêm mô hình hiện tại vào cùng AWF, AGCWD, SDD, CLAHE và RRM, rồi sắp xếp theo Z đối chiếu giảm dần. Input được giữ làm hàng tham khảo nhưng không tham gia chuẩn hóa Z.

Mô hình có MLI khoảng 0,4863, CI 0,2272, entropy 7,6442 và AG 0,0513. Ba chỉ số đầu cao nhất trong các mốc đang đối chiếu. AG thấp hơn AWF, AGCWD và CLAHE, cao hơn SDD và RRM. Do đó thứ tự Z đầu bảng không được diễn giải là thắng ở từng chỉ số.

AWF giữ điểm công bố 1,4479. Z mô hình khoảng 2,3559 được dự án ước tính bằng mean và std của năm mốc gốc, chưa chuẩn hóa lại theo nhóm sáu. Dấu sao ở hạng một nhắc đây là thứ tự đối chiếu tạm, chưa phải xếp hạng chính thức của cuộc thi. Các số mean±SD của phương pháp khác lấy từ đề; SD mô hình tính trên 100 ảnh. SD trong từng hàng không phải sigma dùng để chuẩn hóa Z.

**Chuyển ý:** Em tách riêng so sánh với AWF để làm rõ mức chênh lệch và cách tính Z.

---

<a id="slide-19"></a>

## Slide 19: Đối chiếu bốn chỉ số với mốc AWF

**Nhìn vào:** Chỉ các phần trăm chênh lệch, nhấn mạnh AG còn thấp.

### Lời đọc trực tiếp

Khi chỉ đối chiếu AWF, mô hình tăng MLI khoảng 46,62%, CI khoảng 15,73% và entropy khoảng 5,90%. AG thấp hơn khoảng 51,26%. Đây là các chênh lệch tính bằng mô hình chia AWF rồi trừ một. Như vậy kết luận đúng là có ưu thế về ba chỉ số no-reference và Z ước tính theo mốc gốc, còn độ lớn gradient chưa đạt mốc AWF.

Với mỗi chỉ số j, Z thành phần được tính bằng x trừ mean của năm phương pháp gốc rồi chia std của năm giá trị trung bình đó. Sau đó lấy trung bình bốn Z thành phần. Các giá trị thành phần của mô hình xấp xỉ 5,0103; 1,7237; 2,9208 và âm 0,2312. MLI rất cao góp phần lớn vào Z tổng hợp.

Nếu ban chấm tính lại mean/std cho cả sáu phương pháp, Z của mọi phương pháp sẽ thay đổi. Khi đó không thể giữ nguyên AWF 1,4479 rồi trộn với Z đã chuẩn hóa theo nhóm sáu. Ngoài ra, chưa có mã AWF gốc để chạy lại cùng điều kiện, nên bảng hiện tại là đối chiếu với mốc công bố.

**Chuyển ý:** Các chỉ số no-reference chưa thể hiện hết chất lượng, nên em kiểm tra thêm với GT.

---

<a id="slide-20"></a>

## Slide 20: Độ trung thực và gradient vùng phẳng

**Nhìn vào:** Chỉ ba ảnh input, mô hình và GT, sau đó số đo độ trung thực.

### Lời đọc trực tiếp

Ba ảnh cho thấy input, đầu ra và GT của cùng ID test. Vùng tối được nâng sáng rõ, nhưng đầu ra vẫn khác GT về màu và bề mặt. Trên toàn bộ 100 ảnh, PSNR của bản chốt là 13,9251 dB và SSIM khoảng 0,5134. So với bản cha trong log, hai chỉ số này giảm lần lượt khoảng 6,49% và 22,96%.

AG ở vùng GT phẳng tăng từ khoảng 0,012712 lên 0,026423. Đây là dấu hiệu phải kiểm tra khuếch đại nhiễu hoặc texture thừa, chứ không thể xem mọi gradient tăng là chi tiết được phục hồi. Những giá trị này cho thấy mục tiêu tăng chỉ số độ sáng và tương phản có đánh đổi về độ trung thực.

Đề không cung cấp PSNR và SSIM tương ứng của AWF, nên em không so trực tiếp hai số này với AWF. Hướng cải thiện tiếp theo cần cân bằng các mục tiêu trên validation và kiểm chứng độ trung thực, thay vì chỉ tăng AG hoặc Z.

**Chuyển ý:** Cuối cùng, em trình bày cách dùng mô hình và tài nguyên đã đo.

### Tra cứu thêm

https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html

---

<a id="slide-21"></a>

## Slide 21: Suy luận và tài nguyên triển khai

**Nhìn vào:** Chỉ luồng suy luận, thời gian và tài nguyên CPU.

### Lời đọc trực tiếp

Khi đưa ảnh mới vào giao diện, app.py khởi động server local. EnhancementService nạp checkpoint một lần, đọc ảnh về RGB và chạy pipeline chung. Suy luận dùng model.eval và inference_mode, không có GT, không tính loss và không cập nhật trọng số. Ảnh lớn được giữ tỷ lệ, giới hạn cạnh dài 1280; ảnh benchmark 600×400 không bị resize bởi giới hạn này.

Môi trường ghi trong log là PyTorch bản CPU, dùng sáu luồng. Hai ảnh 600×400 trong smoke test HTTP có thời gian xử lý khoảng 0,927 và 0,581 giây. Đây chỉ là hai mẫu đo, không phải kết luận tốc độ trung bình của mọi ảnh. Sáu epoch tinh chỉnh cuối cộng khoảng 438,77 giây; peak working set lớn nhất trong log khoảng 2.171,59 MiB.

Em kết luận: dự án đã triển khai và huấn luyện mô hình ghép ba nhánh, có giải thích được cơ chế attention và curve, có đầu ra thực và bảng đo. Kết quả tăng MLI, CI, entropy so với mốc AWF, nhưng AG, độ trung thực và nhiễu còn là giới hạn cần cải thiện.

**Chuyển ý:** Phần trình bày của em xin kết thúc ở đây. Em xin lắng nghe câu hỏi của thầy và các bạn.

---
