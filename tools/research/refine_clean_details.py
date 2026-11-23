"""Validation-only clean-detail refinement; keep the approved model protected."""
from pathlib import Path
import argparse,copy,json,shutil,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image,ImageDraw,ImageFont
from kind_enhance.infrastructure.artifacts import save_image,pixels,write_csv
from kind_enhance.training.losses.clean_detail import make_clean_detail_loss
from kind_enhance.data.paired_dataset import PairedLowLightDataset
from kind_enhance.evaluation.detail import measure_detail_quality,detail_quality_failures
from kind_enhance.engineering.guard import DataGuard
from kind_enhance.evaluation.experiments import branch_hash,load_candidate
from kind_enhance.inference.pipeline import load_model,enhance
from kind_enhance.infrastructure.runtime import sha256,read_checkpoint,save_checkpoint,write_json,resource_snapshot
from kind_enhance.training.engine import train
OUT=ROOT/'experiments/clean_detail_refinement'


def render_report(selection):
    diagnostic=selection['best_diagnostic']
    shown=selection['validation'] if selection['passed_all_quality_gates'] else diagnostic['metrics']
    baseline=selection['baseline']
    lines=['# Tinh chỉnh tăng chi tiết và giảm nhiễu', '',
           '## Kết quả',
           f"Ứng viên qua toàn bộ điều kiện: {selection['passed_all_quality_gates']}.",
           'Mô hình mặc định giữ nguyên. Bảng dưới là bản được chọn nếu qua điều kiện; nếu chưa qua thì là ứng viên chẩn đoán gần điều kiện nhất, không phải mô hình được triển khai.', '',
           '| Validation: 69 ảnh | Bản hiện tại | Ứng viên | Thay đổi |','|---|---:|---:|---:|']
    for key in ('MLI','CI','E','AG','PSNR','SSIM','chroma_mae','chromaticity_mae','saturation_mae','gradient_cosine','aligned_AG','supported_gradient_mae','flat_ag','flat_residual_rms','clipped_fraction'):
        b,m=baseline[key],shown[key]
        lines.append(f'| {key} | {b:.7f} | {m:.7f} | {(m/b-1)*100:+.3f}% |')
    failed,_=detail_quality_failures(shown,baseline)
    lines+=['',f'Các điều kiện chưa đạt: {failed}.',
            'Điều kiện chọn là trung bình tập; không bảo đảm mọi ảnh đều tăng. Chưa đạt thì không thay ngưỡng hoặc gọi đó là bản tốt hơn.', '',
            '## Loss và thuật toán',
            'Giữ nguyên KinD Decomposition + Attention U-Net sigmoid RGB + Curve CNN, 2.217.963 tham số. Đóng băng decomposition, học restoration/adjustment; teacher là checkpoint hiện tại đóng băng. Hậu xử lý RGB/luminance giữ nguyên.',
            'Mục tiêu chi tiết tạo từ GT: khớp mean/contrast với teacher có chặn tỷ lệ, tăng highpass chỉ nơi GT có biên, sau đó chuẩn hóa thống kê. Với crop GT phẳng, không tái tạo variance nhiễu của teacher.',
            'Loss biên dùng phép chiếu gradient dự đoán lên hướng GT. Chỉ thưởng tăng thành phần đúng hướng; phạt biên ngược hướng và thành phần vuông góc. Đây là thay đổi so với chỉ tăng norm gradient.',
            'Vùng GT ít biên chịu phạt gradient sai lệch và highpass residual so với GT. Loss gồm L1 luminance với mục tiêu GT, SSIM ảnh chuẩn, biên có hướng hai tỷ lệ, phép chiếu đúng hướng, phạt nhiễu và neo mean/CI/entropy/low-frequency teacher.',
            'CI/soft entropy chỉ neo ở crop GT có texture; mean vẫn neo ở mọi crop. Không ép tái tạo nhiễu để giữ contrast/entropy của crop GT phẳng. Thống kê toàn ảnh vẫn bị kiểm tra chặt trên validation.',
            'Adam, clip norm 5, seed 827, batch 8, patch 80, crop/xoay/lật cặp ảnh đồng bộ. Ba cấu hình bốn epoch được ghi trước trong protocol.json; mỗi cấu hình bắt đầu từ bản hiện tại hoặc ứng viên đã qua toàn bộ điều kiện.', '',
            '## Điều kiện chọn',
            'MLI/CI/E, PSNR/SSIM và cosine hướng gradient không giảm; AG tăng ít nhất 0,1%, SSIM tăng ít nhất 0,0001, cosine tăng ít nhất 0,001 và thành phần gradient đúng hướng tăng ít nhất 1%. Các margin này yêu cầu tiến bộ có ý nghĩa thay vì dao động số.',
            'Chroma/chromaticity/saturation MAE, MAE gradient hỗ trợ và clipping không tăng. Gradient vùng GT phẳng và highpass residual RMS phải giảm ít nhất 5%. Chỉ dùng sai số số học 1e-8, không cho phép chỉ số được giữ giảm 1%.',
            'Entropy cao có thể bao gồm nhiễu. Do đó giảm nhiễu và giữ E có thể xung đột; kết quả không qua điều kiện phải được báo rõ, không bỏ điều kiện hoặc tạo texture giả để bù E/AG.', '',
            '## Dữ liệu và kiểm chứng',
            '620 train / 69 validation từ train / 100 test chính thức giữ nguyên, dữ liệu có sẵn trên E. Không tải/cài gì thêm. DataGuard chặn test khi huấn luyện và chọn. Lịch sử dự án đã xem test nên không coi đây là test mới độc lập.',
            'MLI/CI/E/AG dùng cùng phép đo uint8 đã có, không đổi chuẩn hóa. PSNR/SSIM/sai màu/biên đúng/nhiễu vùng phẳng dùng tensor float. flat_ag là gradient ở vùng GT có gradient <0,003; flat_residual_rms đo residual highpass so với GT tại vùng đó.',
            'Nếu có ứng viên đủ điều kiện, test chỉ được mở bằng lệnh riêng sau khi khóa selection. Không dùng test để điều khiển thử tiếp. Nếu không có ứng viên, test vẫn đóng.',
            f"Thời gian train+validation từ log: {selection['seconds']:.1f} giây. Tài nguyên: {selection['resources']}.",
            f"Audit search: {selection['data_audit']}.",
            'Một seed, lịch học ngắn trên CPU; không bảo đảm đạt đồng thời toàn bộ mục tiêu hoặc khử hoàn toàn nhiễu. Không thay kiến trúc, dataset, checkpoint mặc định hoặc thông số suy luận đã duyệt.', '',
            '## Tệp',
            'protocol.json và source/: cấu hình trước khi học và mã nguồn chụp lại. validation_events.json: từng epoch/lý do loại. trials/: checkpoint/optimizer/log/CSV. selection.json: lựa chọn đã khóa. validation_preview/: 69 PNG, ảnh đối chiếu và các thành phần của 4 ảnh cố định.']
    if (OUT/'test/metrics.json').exists():
        test=json.loads((OUT/'test/metrics.json').read_text())
        lines+=['','## Test sau khóa lựa chọn',str(test)]
    (OUT/'BAO_CAO_CHI_TIET_VA_NHIEU.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


@torch.inference_mode()
def export_images(model,before,dataset,cfg,destination):
    destination.mkdir(parents=True,exist_ok=True)
    images=destination/'images';stages=destination/'stages'
    images.mkdir(exist_ok=True);stages.mkdir(exist_ok=True)
    chosen=[0,22,45,68] if len(dataset)==69 else [0,33,66,99]
    canvas=Image.new('RGB',(1200,920),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    for index,(low,high,name) in enumerate(dataset):
        low,high=[x[None].contiguous(memory_format=torch.channels_last) for x in (low,high)]
        prediction=enhance(model,low,cfg);save_image(images/f'{name}_RGB.png',prediction)
        if index not in chosen:continue
        old=enhance(before,low,cfg)
        for column,(value,label) in enumerate(zip((low,old,prediction,high),('Input','Current','Candidate','GT'))):
            tile=Image.fromarray(pixels(value));tile.thumbnail((296,195))
            left,top=column*300,chosen.index(index)*230
            canvas.paste(tile,(left,top+28));draw.text((left+3,top+4),name+' '+label,font=font,fill='black')
        folder=stages/name;folder.mkdir(exist_ok=True)
        r,l=model.decomposition(low);raw,restored,adjusted,_=model.enhance(r,l)
        for label,value in dict(input=low,reflectance_R=r,illumination_L=l,reconstruction=r*l,
                                restored_R=restored,adjusted_L=adjusted,raw=raw,final=prediction,GT=high).items():
            save_image(folder/f'{label}.png',value)
    canvas.save(destination/'comparison.png')


def search(resume=False):
    tick=time.perf_counter();OUT.mkdir(parents=True,exist_ok=resume)
    teacher,cfg,manifest=load_model();teacher.requires_grad_(False).eval()
    cfg_hash=sha256(ROOT/'configs/model.json');checkpoint=ROOT/cfg['model'];checkpoint_hash=sha256(checkpoint)
    plans=[dict(name='balanced_detail',epochs=4,lr=3e-5,detail_gain=1.5,aligned_gain=1.04,signed_weight=8.,projection_weight=8.,noise_weight=20.),
           dict(name='strong_detail',epochs=4,lr=3e-5,detail_gain=3.,aligned_gain=1.08,signed_weight=10.,projection_weight=12.,noise_weight=30.),
           dict(name='strong_clean_edges',epochs=4,lr=2e-5,detail_gain=5.,aligned_gain=1.12,signed_weight=12.,projection_weight=16.,noise_weight=40.)]
    sources=[Path(__file__).resolve(),*sorted((ROOT/'kind_enhance').rglob('*.py'))]
    source_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in sources}
    protocol=dict(parent_config_sha256=cfg_hash,parent_checkpoint_sha256=checkpoint_hash,
                  manifest_sha256=cfg['manifest_sha256'],output_config=cfg,plans=plans,
                  seed=827,patch=80,batch=8,source_hashes=source_hashes,
                  selection='all preservation/AG/GT-detail/noise constraints on full validation',
                  training_branches=['restoration','adjustment'],frozen='decomposition',
                  test_feedback_allowed=False,prior_project_test_exposure=True)
    if resume:assert json.loads((OUT/'protocol.json').read_text())==protocol,'Protocol changed'
    else:
        write_json(OUT/'protocol.json',protocol)
        for p in sources:
            dest=OUT/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    guard=DataGuard(manifest['source']['data_root'],manifest,('train','val'));guard.install()
    training=PairedLowLightDataset(manifest['source']['data_root'],manifest['splits']['train'],patch=80,cache=True)
    validation=PairedLowLightDataset(manifest['source']['data_root'],manifest['splits']['val'],cache=True)
    baseline,rows=measure_detail_quality(teacher,validation,cfg)
    write_json(OUT/'baseline.json',baseline);write_csv(OUT/'baseline_validation.csv',rows)
    frozen=branch_hash(teacher,'decomposition')
    event_path=OUT/'validation_events.json'
    events=json.loads(event_path.read_text()) if resume and event_path.exists() else []
    if resume:
        kept=[];discarded=[]
        for event in events:
            last=OUT/'trials'/event['plan']/'last.pt'
            completed=read_checkpoint(last,'cpu',cfg['manifest_sha256'],'enhance')['epoch'] if last.exists() else 0
            (kept if event['epoch']<=completed else discarded).append(event)
        if discarded:write_json(OUT/'recovered_orphan_events.json',discarded)
        events=kept;write_json(event_path,events)
    feasible=max((e for e in events if not e['failed_checks']),key=lambda e:e['metrics']['AG'],default=None)
    diagnostic=min(events,key=lambda e:(len(e['failed_checks']),sum(e['violations'].values()),-e['metrics']['SSIM']),default=None)
    if feasible:shutil.copyfile(feasible['checkpoint'],OUT/'best_feasible.pt')
    if diagnostic:shutil.copyfile(diagnostic['checkpoint'],OUT/'best_diagnostic.pt')
    status='running'
    try:
        for plan in plans:
            folder=OUT/'trials'/plan['name']
            last=read_checkpoint(folder/'last.pt','cpu',cfg['manifest_sha256'],'enhance') if resume and (folder/'last.pt').exists() else None
            if last and last['epoch']>=plan['epochs']:continue
            initial=OUT/'best_feasible.pt' if feasible else checkpoint
            model=load_candidate(initial,cfg)
            model.decomposition.requires_grad_(False);model.restoration.requires_grad_(True);model.adjustment.requires_grad_(True)
            recipe={**plan,'seed':827,'patch':80,'batch':8,'decomposition_lr':1e-6,'validate_every':1,'initial_sha256':sha256(initial)}
            if last:recipe={k:v for k,v in last['config'].items() if k not in ('loss_source_sha256','metrics_source_sha256')}
            torch.manual_seed(827)
            def validate(current,data,device):
                nonlocal feasible,diagnostic
                m,rows=measure_detail_quality(current,data,cfg)
                failed,violation=detail_quality_failures(m,baseline)
                assert branch_hash(current,'decomposition')==frozen
                assert sha256(ROOT/'configs/model.json')==cfg_hash and sha256(checkpoint)==checkpoint_hash
                epoch=1+sum(e['plan']==plan['name'] for e in events)
                path=folder/f'epoch_{epoch:02d}.pt'
                state=dict(model=current.state_dict(),stage='enhance',manifest=cfg['manifest_sha256'],epoch=epoch,config=recipe,validation=m)
                save_checkpoint(path,state)
                event=dict(plan=plan['name'],epoch=epoch,metrics=m,failed_checks=failed,violations=violation,checkpoint=str(path))
                events.append(event);write_json(event_path,events);write_csv(folder/f'validation_{epoch:02d}.csv',rows)
                if not failed and (feasible is None or m['AG']>feasible['metrics']['AG']):
                    feasible=event;shutil.copyfile(path,OUT/'best_feasible.pt')
                score=lambda e:(len(e['failed_checks']),sum(e['violations'].values()),-e['metrics']['SSIM'])
                if diagnostic is None or score(event)<score(diagnostic):
                    diagnostic=event;shutil.copyfile(path,OUT/'best_diagnostic.pt')
                return {**m,'failed_checks':failed},(not failed,m['AG'] if not failed else -sum(violation.values()),m['SSIM'])
            print(json.dumps(dict(plan=plan,initial=str(initial))),flush=True)
            train(model,training,validation,torch.device('cpu'),'enhance',recipe,plan['epochs'],folder,cfg['manifest_sha256'],
                  validation_scoring=validate,loss_function=make_clean_detail_loss(teacher,cfg))
        selected_path=OUT/'best_feasible.pt' if feasible else checkpoint
        selected=load_candidate(selected_path,cfg)
        selected_metrics,rows=measure_detail_quality(selected,validation,cfg);write_csv(OUT/'selected_validation.csv',rows)
        histories=[row for plan in plans for row in json.loads((OUT/'trials'/plan['name']/'history.json').read_text())]
        result=dict(checkpoint=str(selected_path),sha256=sha256(selected_path),baseline=baseline,validation=selected_metrics,
                    best_feasible=feasible,best_diagnostic=diagnostic,passed_all_quality_gates=feasible is not None,
                    selection_locked=True,promoted_to_default=False,parent_config_sha256=cfg_hash,parent_checkpoint_sha256=checkpoint_hash,
                    output_config={**cfg,'model':str(selected_path),'model_sha256':sha256(selected_path),'user_approved_as_final':False},
                    epochs=len(histories),seconds=sum(row['seconds'] for row in histories),session_seconds=time.perf_counter()-tick,
                    peak_RAM_MiB=max(row['resources'].get('peak_working_set_mib',0) for row in histories),
                    resources=resource_snapshot(torch.device('cpu')),data_audit=guard.report(),test_feedback_allowed=False)
        write_json(OUT/'selection.json',result)
        preview_path=selected_path if feasible else OUT/'best_diagnostic.pt'
        preview=load_candidate(preview_path,cfg)
        preview_metrics,rows=measure_detail_quality(preview,validation,cfg)
        write_json(OUT/'preview_metrics.json',preview_metrics);write_csv(OUT/'preview_per_image.csv',rows)
        export_images(preview,teacher,validation,cfg,OUT/'validation_preview')
        render_report(result)
        assert sha256(ROOT/'configs/model.json')==cfg_hash and sha256(checkpoint)==checkpoint_hash
        status='complete';print(json.dumps(dict(selection=result)),flush=True)
    finally:write_json(OUT/'data_audit.json',{**guard.report(),'status':status})


def finalize_test():
    selection=json.loads((OUT/'selection.json').read_text())
    assert selection['selection_locked'] and selection['passed_all_quality_gates'],'No quality-passing validation candidate'
    assert sha256(selection['checkpoint'])==selection['sha256']
    before,cfg,manifest=load_model()
    assert sha256(ROOT/'configs/model.json')==selection['parent_config_sha256']
    assert cfg['model_sha256']==selection['parent_checkpoint_sha256']
    directory=OUT/'test';directory.mkdir(exist_ok=False)
    lock=dict(selection_sha256=sha256(OUT/'selection.json'),protocol_sha256=sha256(OUT/'protocol.json'),
              checkpoint_sha256=selection['sha256'],test_feedback_allowed=False)
    write_json(directory/'test_lock.json',lock)
    guard=DataGuard(manifest['source']['data_root'],manifest,('test',));guard.install()
    data=PairedLowLightDataset(manifest['source']['data_root'],manifest['splits']['test'])
    after=load_candidate(selection['checkpoint'],cfg)
    baseline,before_rows=measure_detail_quality(before,data,cfg)
    metrics,rows=measure_detail_quality(after,data,cfg)
    failed,violation=detail_quality_failures(metrics,baseline)
    write_csv(directory/'baseline_per_image.csv',before_rows);write_csv(directory/'per_image.csv',rows)
    export_images(after,before,data,cfg,directory)
    write_json(directory/'metrics.json',dict(baseline=baseline,candidate=metrics,failed_checks=failed,violations=violation,
                                          data_audit=guard.report(),selection_lock=lock))
    assert sha256(OUT/'selection.json')==lock['selection_sha256']
    render_report(selection);print(json.dumps(dict(test=metrics,failed_checks=failed)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--resume',action='store_true');parser.add_argument('--finalize-test',action='store_true');args=parser.parse_args()
    if args.output:OUT=args.output.resolve()
    torch.set_num_threads(6)
    if args.finalize_test:finalize_test()
    elif args.resume and (OUT/'selection.json').exists() and (OUT/'validation_preview/comparison.png').exists():
        s=json.loads((OUT/'selection.json').read_text());assert sha256(s['checkpoint'])==s['sha256']
        print(json.dumps(dict(status='already_complete',selection=str(OUT/'selection.json'))),flush=True)
    else:search(args.resume)
