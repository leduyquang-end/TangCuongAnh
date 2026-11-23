# Hồ sơ bản đang triển khai

Checkpoint và pipeline đang dùng nằm trong `checkpoints/model_current.pt` và `configs/model.json`.

- [Báo cáo hiện tại so với AWF, mốc Z gốc](../../../reports/BAO_CAO_BAN_DA_CHOT_SO_VOI_AWF.md)
- [100 ảnh test RGB](test/images/)
- [Chỉ số từng ảnh](test/per_image.csv)
- [Ảnh thành phần](test/stages/)

Protocol, selection, source, log và metrics giữ nguyên như thời điểm đánh giá. File `best_four_metrics.pt` là bản nguồn có cùng SHA256 với model_current.pt, thuộc cùng phiên bản đang dùng. Các nhánh thử nghiệm khác và báo cáo bị thay thế đã dọn theo yêu cầu.
