"""Validation-only calibration of a denoised checkpoint with unchanged quality gates."""
from pathlib import Path
import argparse,copy,csv,json,shutil,statistics,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image,ImageDraw,ImageFont
from kind_enhance.infrastructure.artifacts import pixels,save_image,write_csv
from kind_enhance.data.paired_dataset import PairedLowLightDataset
from kind_enhance.evaluation.detail import edge_diagnostics,extra_detail_diagnostics,detail_quality_failures
from kind_enhance.evaluation.experiments import describe_color,means,load_candidate
from kind_enhance.inference.pipeline import load_model,enhance
from kind_enhance.inference.calibration import calibrate,calibration_features
from kind_enhance.inference.color_preservation import preserve_color
from kind_enhance.inference.luminance import refine_luminance
from kind_enhance.engineering.guard import DataGuard
from kind_enhance.infrastructure.runtime import sha256,write_json,resource_snapshot
OUT=ROOT/'experiments/clean_detail_calibration'
SOURCE=ROOT/'experiments/clean_detail_refinement/trials/balanced_detail/epoch_01.pt'


def summarize(rows):
    result=means(rows)
    for key in rows[0]:
        if key!='id' and key not in result:result[key]=statistics.mean(row[key] for row in rows)
    return result


def describe(prediction,truth,name):
    return {**describe_color(prediction,truth,name),**edge_diagnostics(prediction,truth),**extra_detail_diagnostics(prediction,truth)}


def apply(raw,features,low,config):
    calibrated=calibrate(raw,features,config['calibration'])
    colored=preserve_color(calibrated,low,**config['rgb_preservation'])
    return refine_luminance(colored,**config['luminance_refinement'])


def changed_config(base,setting):
    cfg=copy.deepcopy(base)
    cfg['calibration'].update(contrast=setting['contrast'],detail=setting['detail'],sharpen=setting['sharpen'])
    cfg['rgb_preservation']['gain']=setting['gain']
    cfg['luminance_refinement']['lift']=setting['lift']
    return cfg


def report(result):
    b,m=result['baseline'],result['diagnostic']['metrics']
    lines=['# Hiệu chỉnh sau checkpoint giảm nhiễu','',
           f"Có ứng viên qua mọi điều kiện: {result['passed_all_quality_gates']}. Bản mặc định chưa thay.",
           'Checkpoint lấy từ epoch 1 balanced_detail đã học trên train. Chỉ tìm scalar hậu xử lý trên validation; không thêm mạng, không dùng GT trong forward, không sửa công thức AG.', '',
           '| 69 validation | Bản hiện tại | Ứng viên gần điều kiện nhất | Thay đổi |','|---|---:|---:|---:|']
    for key in ('MLI','CI','E','AG','PSNR','SSIM','chroma_mae','chromaticity_mae','saturation_mae','gradient_cosine','aligned_AG','flat_ag','flat_residual_rms','clipped_fraction'):
        lines.append(f'| {key} | {b[key]:.7f} | {m[key]:.7f} | {(m[key]/b[key]-1)*100:+.3f}% |')
    lines+=['',f"Điều kiện chưa đạt: {result['diagnostic']['failed_checks']}.",
            f"Cấu hình: {result['diagnostic']['setting']}.",
            'Điều kiện giữ nguyên từ nhánh clean_detail_refinement: bốn trung bình không giảm, PSNR/SSIM không giảm, biên đúng tăng, sai màu/clipping không tăng và hai phép đo nhiễu vùng phẳng giảm ít nhất 5%. Không dùng AG hoặc Z cao để bù điều kiện khác.',
            'Lưới được lưu trước khi đo. Sàng lọc trên 12 vị trí validation cách đều, sau đó xác nhận 6 cấu hình đứng đầu trên đủ 69 validation. Đây là tìm hyperparameter bằng validation; test không được mở trong search.',
            'GT chỉ dùng chấm chỉ số, mọi scalar là cấu hình cố định chung cho tất cả ảnh. Nếu không có ứng viên đạt, không triển khai hoặc mở test để tìm cách chọn.',
            f"Thời gian: {result['seconds']:.1f} giây; tài nguyên: {result['resources']}; audit: {result['data_audit']}.",
            'Số AWF không dùng để đổi định nghĩa AG. Lịch sử dự án đã xem test; không gọi đây là một holdout mới độc lập.',
            'validation_images/: đủ 69 ảnh ứng viên; comparison.png: input/current/candidate/GT tại vị trí 0,22,45,68. per_image.csv và baseline_per_image.csv: các chỉ số thực đo. Full_search và subset_search ghi mọi cấu hình và lý do loại.']
    (OUT/'BAO_CAO_HIEU_CHINH_GIAM_NHIEU.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


@torch.inference_mode()
def search():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    before,base,manifest=load_model()
    cfg_hash=sha256(ROOT/'configs/model.json');parent_hash=sha256(ROOT/base['model'])
    model=load_candidate(SOURCE,base)
    grid=[]
    def add(contrast,detail,sharp,lift,gain):
        setting=dict(contrast=contrast,detail=detail,sharpen=sharp,lift=lift,gain=gain)
        if setting not in grid:grid.append(setting)
    add(1.08,.25,1.4,.007,.5)
    for sharp in (1.6,2.,2.4):
        for lift in (.012,.017,.020):
            for gain in (.5,.65):add(1.08,.25,sharp,lift,gain)
    for sharp in (2.,2.4):
        for lift in (.017,.020):
            for gain in (.5,.65):
                add(1.10,.25,sharp,lift,gain);add(1.08,.10,sharp,lift,gain)
    indices=[round(i*68/11) for i in range(12)]
    protocol=dict(parent_config_sha256=cfg_hash,parent_checkpoint_sha256=parent_hash,
                  source_checkpoint=str(SOURCE),source_sha256=sha256(SOURCE),manifest_sha256=base['manifest_sha256'],
                  grid=grid,subset_indices=indices,full_confirmation_count=6,
                  selection='unchanged strict detail_quality_failures on all 69 validation images',
                  test_feedback_allowed=False,source_sha256_script=sha256(Path(__file__)))
    write_json(OUT/'protocol.json',protocol)
    shutil.copyfile(Path(__file__),OUT/'source.py')
    guard=DataGuard(manifest['source']['data_root'],manifest,('val',));guard.install()
    data=PairedLowLightDataset(manifest['source']['data_root'],manifest['splits']['val'])
    with (ROOT/'experiments/clean_detail_refinement/baseline_validation.csv').open(encoding='utf-8-sig',newline='') as handle:
        baseline_rows=[{key:value if key=='id' else float(value) for key,value in row.items()} for row in csv.DictReader(handle)]
    prior_protocol=json.loads((ROOT/'experiments/clean_detail_refinement/protocol.json').read_text())
    assert prior_protocol['parent_config_sha256']==cfg_hash and prior_protocol['parent_checkpoint_sha256']==parent_hash
    baseline=summarize(baseline_rows)
    assert baseline['AG']>.04
    write_csv(OUT/'baseline_per_image.csv',baseline_rows);write_json(OUT/'baseline.json',baseline)
    cached=[]
    for index,(low,gt,name) in enumerate(data):
        low,gt=[x[None].contiguous(memory_format=torch.channels_last) for x in (low,gt)]
        assert name==baseline_rows[index]['id']
        raw=model(low);cached.append((raw,calibration_features(low,raw),low,gt,name))
        if (index+1)%23==0:print(f'Validation raw cache {index+1}/69',flush=True)
    first=cached[0];check_config=changed_config(base,grid[-1])
    assert torch.allclose(apply(first[0],first[1],first[2],check_config),enhance(model,first[2],check_config),atol=1e-7,rtol=0)
    subset_base=summarize([baseline_rows[i] for i in indices])
    subset=[]
    for index,setting in enumerate(grid):
        cfg=changed_config(base,setting)
        rows=[describe(apply(cached[i][0],cached[i][1],cached[i][2],cfg),cached[i][3],cached[i][4]) for i in indices]
        metrics=summarize(rows);failed,violation=detail_quality_failures(metrics,subset_base)
        entry=dict(grid=index,setting=setting,metrics=metrics,failed_checks=failed,violations=violation)
        subset.append(entry);write_json(OUT/'subset_search.json',subset)
        if (index+1)%5==0:print(f'Validation screen {index+1}/{len(grid)}',flush=True)
    score=lambda e:(len(e['failed_checks']),sum(e['violations'].values()),-e['metrics']['SSIM'])
    confirmations=sorted(subset,key=score)[:6]
    full=[];chosen_rows={}
    for entry in confirmations:
        setting=entry['setting'];cfg=changed_config(base,setting)
        rows=[describe(apply(raw,features,low,cfg),gt,name) for raw,features,low,gt,name in cached]
        metrics=summarize(rows);failed,violation=detail_quality_failures(metrics,baseline)
        record=dict(grid=entry['grid'],setting=setting,metrics=metrics,failed_checks=failed,violations=violation)
        full.append(record);chosen_rows[entry['grid']]=rows
        write_csv(OUT/f"validation_grid_{entry['grid']:02d}.csv",rows);write_json(OUT/'full_search.json',full)
        print(json.dumps(dict(confirmation=entry['grid'],AG=metrics['AG'],SSIM=metrics['SSIM'],failed_checks=failed)),flush=True)
    passing=[entry for entry in full if not entry['failed_checks']]
    diagnostic=max(passing,key=lambda e:e['metrics']['AG']) if passing else min(full,key=score)
    config=changed_config(base,diagnostic['setting'])
    config.update(user_approved_as_final=False,selection_reason='Validation-only clean-detail quality constraints',training_provenance=str(SOURCE))
    path=OUT/'candidate.pt';shutil.copyfile(SOURCE,path)
    config.update(model=str(path),model_sha256=sha256(path))
    result=dict(selection_locked=True,passed_all_quality_gates=bool(passing),promoted_to_default=False,
                checkpoint=str(path),sha256=sha256(path),baseline=baseline,validation=diagnostic['metrics'],diagnostic=diagnostic,
                output_config=config,parent_config_sha256=cfg_hash,parent_checkpoint_sha256=parent_hash,
                seconds=time.perf_counter()-started,data_audit=guard.report(),resources=resource_snapshot(torch.device('cpu')),
                test_feedback_allowed=False)
    write_json(OUT/'selection.json',result);write_csv(OUT/'per_image.csv',chosen_rows[diagnostic['grid']])
    images=OUT/'validation_images';images.mkdir()
    canvas=Image.new('RGB',(1200,920),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    fixed=[0,22,45,68]
    for index,(raw,features,low,gt,name) in enumerate(cached):
        prediction=apply(raw,features,low,config);save_image(images/f'{name}_RGB.png',prediction)
        if index in fixed:
            old=enhance(before,low,base)
            for column,(value,label) in enumerate(zip((low,old,prediction,gt),('Input','Current','Candidate','GT'))):
                tile=Image.fromarray(pixels(value));tile.thumbnail((296,195));left,top=column*300,fixed.index(index)*230
                canvas.paste(tile,(left,top+28));draw.text((left+3,top+4),name+' '+label,font=font,fill='black')
    canvas.save(OUT/'comparison.png')
    report(result)
    assert sha256(ROOT/'configs/model.json')==cfg_hash and sha256(ROOT/base['model'])==parent_hash
    write_json(OUT/'data_audit.json',{**guard.report(),'status':'complete'})
    print(json.dumps(dict(selection=result)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(6);search()
