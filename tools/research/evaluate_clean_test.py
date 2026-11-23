"""Report a single frozen clean-detail candidate on the official 100 test pairs."""
from pathlib import Path
import argparse,csv,json,shutil,statistics,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from PIL import Image,ImageDraw,ImageFont
from kind_enhance.infrastructure.artifacts import save_image,pixels,write_csv
from kind_enhance.data.paired_dataset import PairedLowLightDataset
from kind_enhance.evaluation.detail import edge_diagnostics,extra_detail_diagnostics,detail_quality_failures
from kind_enhance.evaluation.experiments import describe_color,means,load_candidate
from kind_enhance.inference.pipeline import load_model,enhance
from kind_enhance.evaluation.competition import no_reference_scores,METRIC_NAMES
from kind_enhance.engineering.guard import DataGuard
from kind_enhance.engineering.policy import benchmark_report
from kind_enhance.infrastructure.runtime import sha256,write_json,resource_snapshot


def summarize(rows):
    result=means(rows)
    for key in rows[0]:
        if key!='id' and key not in result:result[key]=statistics.mean(row[key] for row in rows)
    return result


def describe(prediction,truth,name):
    return {**describe_color(prediction,truth,name),**edge_diagnostics(prediction,truth),**extra_detail_diagnostics(prediction,truth)}


def write_report(directory,result):
    b,m=result['baseline'],result['candidate']
    lines=['# Báo cáo ứng viên giảm nhiễu trên 100 test','',
           'Đánh giá theo yêu cầu người dùng dù ứng viên chưa qua toàn bộ điều kiện validation. Chỉ đo đúng một checkpoint/cấu hình đã khóa; không dùng điểm test để chọn lại hoặc điều chỉnh tham số. Bản mặc định giữ nguyên.', '',
           '| Trung bình 100 test | Bản mặc định | Ứng viên đã khóa | Thay đổi |','|---|---:|---:|---:|']
    for key in ('MLI','CI','E','AG','PSNR','SSIM','chroma_mae','chromaticity_mae','saturation_mae','gradient_cosine','aligned_AG','supported_gradient_mae','flat_ag','flat_residual_rms','clipped_fraction'):
        lines.append(f'| {key} | {b[key]:.7f} | {m[key]:.7f} | {(m[key]/b[key]-1)*100:+.3f}% |')
    lines+=['',f"Điều kiện chưa đạt trên test: {result['failed_checks']}.",
            f"Số ảnh giảm theo từng chỉ số: {result['images_decreasing']}. Các điều kiện áp dụng cho trung bình tập.",
            'flat_ag là norm gradient dự đoán ở vùng GT có gradient <0,003. flat_residual_rms là RMS chênh highpass so với GT tại cùng vùng. Hai số thấp hơn biểu thị ít nhiễu/sai lệch ở vùng phẳng theo phép đo này, không thay thế đánh giá cảm nhận.',
            'MLI/CI/E/AG tính trên RGB uint8 đúng PNG xuất; PSNR/SSIM/sai màu/biên/nhiễu dùng tensor float như các lượt trước. Không thay công thức hoặc chuẩn hóa AG.',
            f"Cấu hình ứng viên: {result['candidate_output_config']}",
            f"Checkpoint SHA256: {result['candidate_checkpoint_sha256']}.",
            f"Z cùng nhóm sáu (mốc AWF lấy từ đề, chưa chạy lại cùng dataset): {result['benchmark']['six_method_cohort']['composite_z']}.",
            'Lịch sử dự án đã từng xem test nên không gọi đây là test mới độc lập. Lượt này không train, không chọn lại ứng viên và không tinh chỉnh theo test.',
            f"Thời gian: {result['seconds']:.1f} giây; tài nguyên: {result['resources']}.",
            f"Audit: {result['data_audit']}.", '',
            'images/: đủ 100 PNG RGB; stages/: 4 ảnh theo vị trí 0/33/66/99, mỗi ảnh có input/R/L/RxL/R′/L′/raw/final/GT. comparison.png: đối chiếu cố định. per_image.csv và baseline_per_image.csv: đủ 100 ID. test_lock.json và verification.json: khóa và kiểm chứng.']
    text='\n'.join(lines)+'\n'
    (directory/'BAO_CAO_TEST_100.md').write_text(text,encoding='utf-8')
    master=ROOT/'reports/BAO_CAO_TINH_CHINH_CHI_TIET_GIAM_NHIEU.md'
    if master.exists():
        previous=master.read_text(encoding='utf-8')
        marker='\n## Đánh giá 100 test theo yêu cầu sau khi khóa ứng viên\n'
        if marker in previous:previous=previous.split(marker)[0]
        master.write_text(previous+marker+'\n'+text,encoding='utf-8')


@torch.inference_mode()
def evaluate(selection_path):
    started=time.perf_counter()
    selection=json.loads(selection_path.read_text())
    assert selection['selection_locked']
    assert sha256(selection['checkpoint'])==selection['sha256']
    before,baseline_config,manifest=load_model()
    assert sha256(ROOT/'configs/model.json')==selection['parent_config_sha256']
    assert baseline_config['model_sha256']==selection['parent_checkpoint_sha256']
    assert len(manifest['splits']['test'])==100
    expected_ids={f'{i:05d}' for i in range(690,790)}
    assert {row['id'] for row in manifest['splits']['test']}==expected_ids
    candidate_config=selection['output_config']
    assert candidate_config['model_sha256']==selection['sha256']
    after=load_candidate(selection['checkpoint'],candidate_config)
    source_files=[Path(__file__),*sorted((ROOT/'kind_enhance').rglob('*.py'))]
    lock=dict(selection_sha256=sha256(selection_path),candidate_checkpoint_sha256=selection['sha256'],
              parent_checkpoint_sha256=baseline_config['model_sha256'],parent_config_sha256=sha256(ROOT/'configs/model.json'),
              manifest_sha256=baseline_config['manifest_sha256'],candidate_output_config=candidate_config,
              source_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in source_files},
              validation_passed=selection['passed_all_quality_gates'],test_report_explicitly_requested=True,
              test_feedback_allowed=False,automatic_promotion=False)
    directory=selection_path.parent/'test'
    if directory.exists():
        assert json.loads((directory/'test_lock.json').read_text())==lock,'Different frozen evaluation'
        if (directory/'metrics.json').exists():
            result=json.loads((directory/'metrics.json').read_text());write_report(directory,result)
            print(json.dumps(dict(status='already_evaluated',metrics=result['candidate'])),flush=True);return
    directory.mkdir(exist_ok=True)
    write_json(directory/'test_lock.json',lock)
    shutil.copyfile(Path(__file__),directory/'evaluation_source.py')
    guard=DataGuard(manifest['source']['data_root'],manifest,('test',));guard.install()
    data=PairedLowLightDataset(manifest['source']['data_root'],manifest['splits']['test'])
    images,stages=directory/'images',directory/'stages'
    images.mkdir(exist_ok=True);stages.mkdir(exist_ok=True)
    before_rows,rows=[],[]
    canvas=Image.new('RGB',(1200,920),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);fixed=[0,33,66,99]
    for index,(low,high,name) in enumerate(data):
        low,high=[x[None].contiguous(memory_format=torch.channels_last) for x in (low,high)]
        original=enhance(before,low,baseline_config)
        prediction=enhance(after,low,candidate_config)
        before_rows.append(describe(original,high,name));rows.append(describe(prediction,high,name))
        save_image(images/f'{name}_RGB.png',prediction)
        if index in fixed:
            row_index=fixed.index(index)
            for column,(value,label) in enumerate(zip((low,original,prediction,high),('Input','Current','Candidate','GT'))):
                tile=Image.fromarray(pixels(value));tile.thumbnail((296,195));left,top=column*300,row_index*230
                canvas.paste(tile,(left,top+28));draw.text((left+3,top+4),name+' '+label,font=font,fill='black')
            folder=stages/name;folder.mkdir(exist_ok=True)
            r,l=after.decomposition(low);raw,restored,adjusted,_=after.enhance(r,l)
            for label,value in dict(input=low,reflectance_R=r,illumination_L=l,reconstruction=r*l,restored_R=restored,
                                    adjusted_L=adjusted,raw=raw,final=prediction,GT=high).items():save_image(folder/f'{label}.png',value)
        if (index+1)%25==0:print(f'Test {index+1}/100',flush=True)
    canvas.save(directory/'comparison.png')
    baseline,candidate=summarize(before_rows),summarize(rows)
    known=json.loads((ROOT/'results/active_model/promotion.json').read_text())['test_metrics']
    assert all(abs(baseline[key]-known[key])<1e-10 for key in METRIC_NAMES),'Default baseline reproduction differs'
    failed,violation=detail_quality_failures(candidate,baseline)
    assert {row['id'] for row in rows}==expected_ids and len(rows)==100
    write_csv(directory/'baseline_per_image.csv',before_rows);write_csv(directory/'per_image.csv',rows)
    assert len(list(images.glob('*.png')))==100
    for row in rows:
        with Image.open(images/f"{row['id']}_RGB.png") as picture:
            assert picture.mode=='RGB' and picture.size==(600,400)
            actual=no_reference_scores(np.asarray(picture))
        assert all(abs(actual[key]-row[key])<1e-12 for key in METRIC_NAMES)
    assert len(list(stages.glob('*/*.png')))==36
    assert sha256(selection_path)==lock['selection_sha256']
    assert sha256(selection['checkpoint'])==lock['candidate_checkpoint_sha256']
    assert sha256(ROOT/'configs/model.json')==lock['parent_config_sha256']
    assert sha256(ROOT/baseline_config['model'])==lock['parent_checkpoint_sha256']
    result=dict(baseline=baseline,candidate=candidate,failed_checks=failed,violations=violation,
                candidate_checkpoint_sha256=selection['sha256'],candidate_output_config=candidate_config,
                benchmark=benchmark_report(candidate),test_lock=lock,validation_passed=selection['passed_all_quality_gates'],
                images_decreasing={key:sum(new[key]<old[key] for new,old in zip(rows,before_rows)) for key in METRIC_NAMES},
                sample_std={key:statistics.stdev(row[key] for row in rows) for key in METRIC_NAMES},
                data_audit=guard.report(),seconds=time.perf_counter()-started,resources=resource_snapshot(torch.device('cpu')),
                model_default_unchanged=True,no_training=True,no_parameter_selection_after_test=True)
    write_json(directory/'metrics.json',result)
    write_json(directory/'verification.json',dict(verified=True,RGB_images=100,unique_test_IDs=100,stage_images=36,
                                               PNG_metrics_match_CSV=True,default_baseline_matches_published_test=True,
                                               locks_unchanged=True,default_model_unchanged=True,data_audit=guard.report()))
    write_report(directory,result)
    print(json.dumps(dict(metrics=candidate,baseline=baseline,failed_checks=failed,verified=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selection',type=Path,default=ROOT/'experiments/clean_detail_calibration/selection.json');args=parser.parse_args()
    torch.set_num_threads(6);evaluate(args.selection.resolve())
