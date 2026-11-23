"""Evaluate a sealed four-metric selection, keeping test out of search."""
from pathlib import Path
import argparse
import json
import statistics
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image, ImageDraw, ImageFont
from kind_enhance.training.losses.supported_edges import stable_edges
from kind_enhance.infrastructure.artifacts import pixels, save_image, write_csv
from kind_enhance.data.paired_dataset import PairedLowLightDataset
from kind_enhance.engineering.guard import DataGuard
from kind_enhance.engineering.policy import benchmark_report
from kind_enhance.inference.pipeline import load_model
from kind_enhance.inference.luminance import enhance_all_gain
from kind_enhance.inference.calibration import luma
from kind_enhance.infrastructure.runtime import resource_snapshot, sha256, write_json
from kind_enhance.evaluation.experiments import branch_hash, load_candidate
from tools.training.refine_four_metrics import OUT, PARENT, KEYS, primary_failures
from kind_enhance.evaluation.experiments import describe_color, means



def edge_diagnostics(prediction, target):
    actual, truth = luma(prediction), luma(target)
    ax, tx = [x[..., :-1, 1:] - x[..., :-1, :-1] for x in (actual, truth)]
    ay, ty = [x[..., 1:, :-1] - x[..., :-1, :-1] for x in (actual, truth)]
    norm_actual = (ax.square()+ay.square()).sqrt()
    norm_truth = (tx.square()+ty.square()).sqrt()
    support = (norm_truth > .01).float()
    cosine = ((ax*tx+ay*ty)/(norm_actual*norm_truth).clamp_min(1e-8)).clamp(-1,1)
    error = (ax-tx).abs()+(ay-ty).abs()
    flat = (stable_edges(truth)<.003).float()
    return dict(gradient_cosine=((cosine*support).sum()/support.sum().clamp_min(1)).item(),
                supported_gradient_mae=((error*support).sum()/(2*support.sum().clamp_min(1))).item(),
                GT_flat_fraction=flat.mean().item(),
                flat_AG_contribution=(stable_edges(actual)*flat).mean().item(),
                nonflat_AG_contribution=(stable_edges(actual)*(1-flat)).mean().item())

def create_report(selection):
    validation_path = OUT / 'validation_export/metrics.json'
    test_path = OUT / 'test/metrics.json'
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    test = json.loads(test_path.read_text()) if test_path.exists() else None
    lines = ['# Tinh chỉnh bốn chỉ số: CI >= 0,22 và AG hướng tới 0,1', '',
             '## Mục tiêu và kết quả lựa chọn',
             'Yêu cầu chính: trung bình MLI, CI, E, AG không giảm so với bản trước tinh chỉnh; CI >= 0,22; AG tiến gần 0,1. CI dùng độ lệch chuẩn luminance chuẩn hóa, vì vậy mốc người dùng đã xác nhận là 0,22.',
             f"Có ứng viên qua bốn điều kiện trên validation: {selection['passed_four_metric_floors']}.",
             f"AG validation đạt 0,1: {selection['target_AG_010_achieved']}; đạt mức gần 0,1 (>=0,095): {selection['near_AG_010_achieved']}.",
             'Không tự thay mô hình mặc định. Đạt bốn chỉ số không đồng nghĩa màu/chi tiết trung thực hơn; các chỉ số chất lượng phụ và ảnh được báo cáo riêng.', '',
             '## Validation: cùng 69 ảnh', '| Chỉ số | Mốc trước | Được giữ | Thay đổi |', '|---|---:|---:|---:|']
    for key in (*KEYS, 'PSNR', 'SSIM', 'chroma_mae', 'flat_ag'):
        b, m = selection['baseline'][key], selection['validation'][key]
        lines.append(f'| {key} | {b:.7f} | {m:.7f} | {(m/b-1)*100:+.3f}% |')
    lines += ['', f"Điều kiện chính chưa đạt: {selection['primary_failures']}.",
              'Bảng trên là checkpoint được giữ theo điều kiện chính. Nếu không có ứng viên đạt, bảng này chính là bản cha được giữ nguyên.',
              f"Checkpoint: {selection['checkpoint']}; SHA256={selection['sha256']}."]
    highest = selection['best_unconstrained_event']
    h = highest['metrics']
    lines += ['', '## Ứng viên AG cao nhất (có thể bị loại)',
              f"Giai đoạn {highest['plan']}, epoch {highest['epoch']}: MLI={h['MLI']:.7f}; CI={h['CI']:.7f}; E={h['E']:.7f}; AG={h['AG']:.7f}.",
              f"Điều kiện chính chưa đạt: {highest['primary_failures']}. Cờ chất lượng phụ: {highest['secondary_quality_flags']}."]
    if validation:
        lines += [f"Ảnh/CSV validation xuất từ checkpoint: {validation['checkpoint']}. Là ảnh chẩn đoán nếu diagnostic=True: {validation['diagnostic']}.",
                  f"Bốn điều kiện ở ảnh xuất: {validation['primary_failures']}."]
    if test:
        lines.insert(4, f"Kết quả cuối trên test: điều kiện bốn chỉ số chưa đạt={test['primary_failures']}; CI>=0,22={test['CI_022_achieved']}; AG>=0,095={test['near_AG_010_achieved']}.")
        lines += ['', '## 100 test sau khi khóa lựa chọn',
                  '| Chỉ số | Mốc trước | Ứng viên khóa | AWF trong đề | Thay đổi so mốc |',
                  '|---|---:|---:|---:|---:|']
        awf = dict(zip(KEYS, (.3317, .1963, 7.2184, .1052)))
        for key in (*KEYS, 'PSNR', 'SSIM', 'chroma_mae', 'flat_ag'):
            b, m = test['baseline'][key], test['candidate'][key]
            awf_cell = f'{awf[key]:.4f}' if key in awf else 'Không có'
            lines.append(f'| {key} | {b:.7f} | {m:.7f} | {awf_cell} | {(m/b-1)*100:+.3f}% |')
        lines += [f"Điều kiện chính chưa đạt trên test: {test['primary_failures']}.",
                  f"CI test >=0,22: {test['candidate']['CI']>=.22}; AG test >=0,095: {test['candidate']['AG']>=.095}.",
                  f"Z cùng nhóm sáu (chuẩn hóa lại cả AWF và ứng viên): {test['benchmark']['six_method_cohort']['composite_z']}.",
                  f"Số ảnh giảm theo từng chỉ số so với bản cha: {test['images_decreasing']}. Điều kiện chọn là trung bình tập, không bảo đảm từng ảnh đều tăng.",
                  'Test được chạy sau selection_locked, có test_lock.json ghi hash checkpoint/protocol/selection. Không đổi lựa chọn hay chạy tiếp tìm tham số dựa trên kết quả test này.']
    else:
        lines += ['', '## Test', 'Chưa mở test trong lượt này. Không dùng số validation để gọi là đã đạt mục tiêu trên test.']
    for title, result in (('validation', validation), ('test', test)):
        if result:
            lines += ['', f'## Biên đúng và vùng phẳng trên {title}',
                      '| Chẩn đoán | Mốc trước | Ứng viên |', '|---|---:|---:|']
            for key in ('gradient_cosine','supported_gradient_mae','GT_flat_fraction','flat_AG_contribution','nonflat_AG_contribution'):
                lines.append(f"| {key} | {result['baseline'][key]:.7f} | {result['candidate'][key]:.7f} |")
    lines += ['', '## Công thức và benchmark',
              'CSV bổ sung cosine hướng gradient trong vùng GT có norm >0,01 (cao hơn tốt hơn), MAE gradient vùng đó (thấp hơn tốt hơn), tỷ lệ vùng GT phẳng và phần AG đóng góp từ vùng phẳng/không phẳng. Các số này giúp kiểm tra tăng AG có đi cùng tăng biên đúng hay nhiễu; không đưa vào lựa chọn sau khi xem test.',
              'Y=(0,299R+0,587G+0,114B)/255 với RGB uint8. MLI=mean(Y); CI=std(Y) với mẫu số N; E=-sum(p*log2(p)) từ histogram 256 bins sau làm tròn Y*255; AG=mean(sqrt((dx^2+dy^2)/2)) với sai phân tiến ở miền chung (H-1)x(W-1).',
              'MLI/CI/E/AG được đo từ RGB uint8 giống PNG đã lưu. PSNR/SSIM, sai màu và flat_ag dùng tensor float; flat_ag là gradient đầu ra tại vùng GT có gradient <0,003. Chroma MAE không phải Delta E Lab.',
              'Z của đề được tính bằng chuẩn hóa từng cột trên cùng nhóm phương pháp rồi lấy trung bình bốn z-score, dùng độ lệch chuẩn population. Khi thêm ứng viên, báo cáo chuẩn hóa lại AWF và ứng viên trong cùng nhóm sáu. Không lấy Z để che một chỉ số riêng giảm.',
              'CuocThi_XLA_1.pdf chỉ ghi tên AG, chưa cho chi tiết công thức/chuyển xám/chuẩn hóa. Không có mã AWF/chấm gốc. Mốc AWF là số trong đề, chưa phải chạy lại trên cùng tập/metric của dự án; chưa thể kết luận so sánh tuyệt đối cùng điều kiện.', '',
              '## Kiến trúc và cách học',
              'Giữ ba mạng KinD Decomposition + Attention U-Net (đầu RGB sigmoid) + Curve CNN; 2.217.963 tham số. Decomposition đóng băng. Restoration và Curve CNN cùng học: 1.982.087 tham số. Hash decomposition được xác nhận sau mỗi validation.',
              'Khởi tạo từ checkpoint AG cao nhất của lượt trước; teacher là bản cha cố định. Giai đoạn 1: 3 epoch, mục tiêu AG 0,05, hệ số biên 2,5, Adam lr=6e-5. Giai đoạn 2: 3 epoch, mục tiêu AG 0,1, hệ số biên 4, lr=4e-5. Batch 8, crop 80x80, seed 827, clip norm 5.',
              'Loss = L1 RGB + (1-SSIM) + 10*khớp biên có hướng + 80*thiếu hụt AG có GT hỗ trợ + 12*tăng gradient ngoài hỗ trợ + 100*thiếu hụt mean + 8*thiếu hụt CI + 4*thiếu hụt soft entropy + 0,2*neo luminance teacher.',
              'Biên GT/teacher được detach. Chỉ yêu cầu tăng AG ở vùng có biên GT và hướng teacher/GT cùng chiều; vùng GT phẳng không nhận yêu cầu tạo texture. Mục tiêu mean patch nằm trong [1,005;1,02] lần teacher; CI patch [1,06;1,15] lần teacher; soft entropy 64 bins yêu cầu ít nhất teacher+0,003. Đây là surrogate học, không bảo đảm thống kê toàn ảnh đạt ngưỡng.',
              'Hậu xử lý giữ đúng cấu hình trước: contrast 1,08; detail 0,25; sharpen 1,4; RGB blend 1,0/gain 0,5; luminance contrast 1,02/lift 0,007. Với blend 1,0, chroma đầu ra lấy từ input RGB đã làm trơn; mạng cung cấp luminance. Không mô tả toàn bộ màu là học được từ sigmoid.',
              'Lưu mỗi epoch để không mất ứng viên đạt bốn chỉ số. Chọn AG cao nhất trong các ứng viên không giảm bốn chỉ số trên validation. Các ngưỡng PSNR/SSIM/sai màu/nhiễu/clipping của lượt cũ chỉ dùng ghi cờ ở lượt này theo mục tiêu chính được người dùng làm rõ; không giấu phần đánh đổi.', '',
              '## Dữ liệu, tài nguyên và giới hạn',
              'Dataset có sẵn trên E; không tải dữ liệu hay cài thư viện. 620 train, 69 validation từ train, giữ nguyên 100 test chính thức. Crop/rotation/flip đồng bộ low/high. DataGuard chặn test trong search. Manifest: '+selection['output_config']['manifest_sha256'],
              f"Audit search: {selection['data_audit']}",
              f"Train + validation cộng từ epoch log: {selection['seconds']:.1f} giây; RAM peak {selection['peak_RAM_MiB']:.1f} MiB; CPU 6 luồng, PyTorch 2.8.0+cpu, không GPU. Thời gian session gồm khởi tạo/đo nền: {selection['session_seconds']:.1f} giây.",
              'Đã qua kiểm tra loss vùng phẳng/biên ngược hướng, GT detach, gradient hữu hạn và smoke backward ở restoration+adjustment. Decomposition không có gradient. Kiến trúc và checkpoint mặc định được kiểm tra hash.',
              'Lịch sử dự án đã xem test ở các lượt cũ nên test không phải tập mới độc lập; lượt này không dùng test để huấn luyện/chọn. Một seed, huấn luyện ngắn trên CPU; tăng AG có thể đi kèm nhiễu/ringing, phải xem ảnh và các chỉ số phụ. Chưa chứng minh tăng AG đồng nghĩa thêm chi tiết đúng.',
              'Theo hàm đo này, GT train/validation có AG trung bình khoảng 0,01947/0,01898; mục tiêu 0,1 cao hơn khoảng 5 lần. Không sửa hàm AG hoặc tạo hoa văn nhằm làm đẹp số.', '',
              '## Tệp để kiểm tra',
              'protocol.json: cấu hình/đích chọn công bố trước. validation_events.json: từng epoch và cờ loại. ag_005/ và ag_010/: checkpoint từng epoch, optimizer, log và CSV validation. selection.json: lựa chọn khóa và hash.',
              'validation_export/ và test/ (nếu chạy): images/ lưu toàn bộ RGB, stages/ lưu input/R/L/R*L/R′/L′/raw/final/GT ở 4 ảnh theo vị trí cố định, comparison.png để xem đối chiếu; per_image.csv, baseline_per_image.csv và metrics.json.',
              'Không ghi đè configs/model.json hay checkpoints/model_current.pt. Chạy tiếp dùng python -B tools/training/refine_four_metrics.py --resume; run đã hoàn tất chỉ xác minh hash và trả trạng thái.']
    (OUT / 'BAO_CAO_BON_CHI_SO.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


@torch.inference_mode()
def evaluate(split):
    selection_path = OUT / 'selection.json'
    selection = json.loads(selection_path.read_text())
    assert selection['selection_locked']
    assert sha256(selection['checkpoint']) == selection['sha256']
    if split == 'test' and not selection['passed_four_metric_floors']:
        create_report(selection)
        raise ValueError('No validation candidate meeting all four metric floors; test remains closed')
    existing = OUT / ('test' if split == 'test' else 'validation_export') / 'metrics.json'
    if existing.exists():
        record=json.loads(existing.read_text())
        frozen=record['selection_lock']
        assert frozen['selection_sha256']==sha256(selection_path)
        assert frozen['checkpoint_sha256']==sha256(selection['checkpoint'])
        assert frozen['protocol_sha256']==sha256(OUT/'protocol.json')
        print(json.dumps(dict(status='already_evaluated',split=split,report=str(ROOT/json.loads((ROOT/'configs/model.json').read_text())['evaluation_report']))),flush=True)
        return
    parent = json.loads(PARENT.read_text())
    config = parent['output_config']
    _, default_config, manifest = load_model()
    if split == 'test':
        assert len(manifest['splits']['test']) == 100
        assert {row['id'] for row in manifest['splits']['test']} == {f'{i:05d}' for i in range(690,790)}
    diagnostic = split == 'val' and not selection['passed_four_metric_floors']
    path = OUT/'highest_AG.pt' if diagnostic else Path(selection['checkpoint'])
    destination = OUT / ('test' if split == 'test' else 'validation_export')
    lock = dict(selection_sha256=sha256(selection_path), checkpoint_sha256=selection['sha256'],
                protocol_sha256=sha256(OUT/'protocol.json'),
                evaluated_checkpoint_sha256=sha256(path),
                metrics_source_sha256=sha256(ROOT/'kind_enhance/evaluation/competition.py'),
                evaluator_source_sha256=sha256(Path(__file__)), test_feedback_allowed=False)
    if (destination / 'metrics.json').exists():
        assert json.loads((destination / 'evaluation_lock.json').read_text()) == lock
        create_report(selection)
        print(json.dumps(dict(status='already_evaluated', split=split)), flush=True)
        return
    if destination.exists():
        assert json.loads((destination/'evaluation_lock.json').read_text()) == lock
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'evaluation_lock.json', lock)
    if split == 'test':write_json(destination/'test_lock.json', lock)
    before = load_candidate(parent['checkpoint'], config)
    after = load_candidate(path, config)
    assert branch_hash(after,'decomposition') == selection['frozen_decomposition_sha256']
    guard = DataGuard(manifest['source']['data_root'], manifest, (split,));guard.install()
    data = PairedLowLightDataset(manifest['source']['data_root'], manifest['splits'][split])
    images, stages = destination/'images', destination/'stages'
    images.mkdir(exist_ok=True);stages.mkdir(exist_ok=True)
    before_rows, rows = [], []
    started = time.perf_counter()
    chosen = [0,22,45,68] if split == 'val' else [0,33,66,99]
    canvas = Image.new('RGB',(1200,920),'white');draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    for index,(low,high,name) in enumerate(data):
        low,high = [x[None].contiguous(memory_format=torch.channels_last) for x in (low,high)]
        original = enhance_all_gain(before,low,config)
        prediction = enhance_all_gain(after,low,config)
        save_image(images/f'{name}_RGB.png',prediction)
        flat = (stable_edges(luma(high)) < .003).float()
        for value,record in ((original,before_rows),(prediction,rows)):
            row = {**describe_color(value,high,name), **edge_diagnostics(value,high)}
            row['flat_ag'] = ((stable_edges(luma(value))*flat).sum()/flat.sum().clamp_min(1)).item()
            record.append(row)
        if index in chosen:
            row_index = chosen.index(index)
            for column,(value,label) in enumerate(zip((low,original,prediction,high),('Input','Parent','Diagnostic' if diagnostic else 'Selected','GT'))):
                tile = Image.fromarray(pixels(value));tile.thumbnail((296,195))
                canvas.paste(tile,(column*300,row_index*230+28));draw.text((column*300+3,row_index*230+4),name+' '+label,font=font,fill='black')
            folder = stages/name;folder.mkdir(exist_ok=True)
            reflectance,illumination = after.decomposition(low)
            raw,restored,adjusted,_ = after.enhance(reflectance,illumination)
            values = dict(input_low=low,reflectance_R=reflectance,illumination_L=illumination,
                          reconstruction_RxL=reflectance*illumination,restored_reflectance=restored,
                          adjusted_illumination=adjusted,raw_output=raw,final_RGB=prediction,reference_GT=high)
            for label,value in values.items():save_image(folder/f'{label}.png',value)
        if (index+1)%25 == 0:print(f'{split}: {index+1}/{len(data)}',flush=True)
    canvas.save(destination/'comparison.png')
    baseline,candidate = means(before_rows),means(rows)
    for metrics,record in ((baseline,before_rows),(candidate,rows)):
        metrics['flat_ag'] = statistics.mean(row['flat_ag'] for row in record)
        for key in ('gradient_cosine','supported_gradient_mae','GT_flat_fraction','flat_AG_contribution','nonflat_AG_contribution'):
            metrics[key] = statistics.mean(row[key] for row in record)
    assert len(rows) == len(data) and len({row['id'] for row in rows}) == len(data)
    if split == 'val' and not diagnostic:
        assert all(abs(candidate[key]-selection['validation'][key])<1e-10 for key in selection['validation'])
    for row in rows:
        with Image.open(images/f"{row['id']}_RGB.png") as picture:assert picture.mode=='RGB' and picture.size==(600,400)
    assert sha256(selection_path) == lock['selection_sha256']
    assert sha256(ROOT/'configs/model.json') == selection['default_config_sha256']
    assert sha256(ROOT/default_config['model']) == selection['default_checkpoint_sha256']
    write_csv(destination/'baseline_per_image.csv',before_rows);write_csv(destination/'per_image.csv',rows)
    result = dict(split=split,checkpoint=str(path),checkpoint_sha256=sha256(path),diagnostic=diagnostic,
                  baseline=baseline,candidate=candidate,primary_failures=primary_failures(candidate,baseline),
                  CI_022_achieved=candidate['CI']>=.22,near_AG_010_achieved=candidate['AG']>=.095,
                  benchmark=benchmark_report(candidate),
                  images_decreasing={key:sum(new[key]<old[key] for old,new in zip(before_rows,rows)) for key in KEYS},
                  sample_std={key:statistics.stdev(row[key] for row in rows) for key in KEYS},
                  data_audit=guard.report(),selection_lock=lock,seconds=time.perf_counter()-started,
                  resources=resource_snapshot(torch.device('cpu')))
    write_json(destination/'metrics.json',result)
    create_report(selection)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['val','test'],default='val');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.output is not None:OUT=args.output.resolve()
    torch.set_num_threads(6)
    evaluate(args.split)
