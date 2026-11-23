# Slide 17: Chọn mô hình trên validation

## Nội dung hiển thị

Lượt cuối kiểm tra điều kiện trên 69 ảnh validation, trước lần test gốc.

Điều kiện: cả MLI / CI / E / AG không thấp hơn bản cha; CI ≥ 0,22.

Trong các ứng viên đạt điều kiện, chọn AG lớn nhất.

Checkpoint được chọn: ag_010, epoch 3.

DataGuard chặn mở test khi train và tìm tham số.

Lịch sử dự án đã từng xem test, nên không coi 100 test là kiểm chứng độc lập hoàn toàn mới.

## Lời thuyết trình và chi tiết kỹ thuật

Checkpoint được chọn trên 69 ảnh validation. Điều kiện chính là bốn trung bình MLI, CI, E và AG không thấp hơn bản cha, đồng thời CI đạt ít nhất 0,22. Trong các ứng viên thỏa điều kiện, chọn AG cao nhất. Sự kiện được chọn trong log là giai đoạn ag_010, epoch 3.

DataGuard giới hạn phân vùng được đọc khi huấn luyện và tìm tham số. Sau khi khóa selection, nhóm đánh giá trên 100 ảnh test. Cập nhật trọng số bằng test là điều không thực hiện trong lượt cuối. Tuy nhiên chỉ không backward trên test chưa đủ để bảo đảm độc lập nếu nhìn test rồi chọn tham số.

Vì lịch sử dự án đã từng xem kết quả test, em ghi rõ rằng đây không phải một tập kiểm chứng hoàn toàn mới chưa từng tiếp xúc. Đây là giới hạn của quy trình thực nghiệm và cần được nêu khi bảo vệ kết quả.



Đối chiếu mã và dữ liệu: E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/kind_enhance/models/; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/configs/model.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/experiments/ag_refinement/run_four_metric_curriculum/test/metrics.json; E:/trenlop/XuLyAnh/Do_An_Tang_Cuong_Anh/reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md


