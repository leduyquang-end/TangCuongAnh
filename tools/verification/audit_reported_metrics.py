"""Audit reported metrics without training or selecting parameters on test."""
from pathlib import Path
import csv,json,math,statistics,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from kind_enhance.inference.pipeline import load_model,enhance
from kind_enhance.data.paired_dataset import read_rgb
from kind_enhance.infrastructure.artifacts import pixels
from kind_enhance.infrastructure.runtime import sha256,write_json
from kind_enhance.evaluation.assessment import describe
from kind_enhance.evaluation.competition import PDF_BENCHMARK,METRIC_NAMES,self_check
from kind_enhance.engineering.guard import DataGuard


def independent_scores(rgb):
    x=rgb.astype(np.float64)
    g=(.299*x[:,:,0]+.587*x[:,:,1]+.114*x[:,:,2])/255
    mean=float(np.sum(g)/g.size)
    ci=math.sqrt(float(np.sum((g-mean)**2)/g.size))
    _,counts=np.unique(np.rint(g*255).astype(np.uint8),return_counts=True)
    probabilities=counts/g.size
    entropy=float(-sum(float(p)*math.log2(float(p)) for p in probabilities))
    dx=g[:-1,1:]-g[:-1,:-1];dy=g[1:,:-1]-g[:-1,:-1]
    ag=float(np.mean(np.hypot(dx,dy)/math.sqrt(2)))
    return dict(zip(METRIC_NAMES,(mean,ci,entropy,ag)))


def zvalue(values,references):
    means=np.mean(references,axis=0);std=np.std(references,axis=0,ddof=0)
    return float(np.mean((np.asarray(values)-means)/std))


@torch.inference_mode()
def main():
    start=time.perf_counter();torch.set_num_threads(6);self_check()
    model,cfg,manifest=load_model()
    protected=[ROOT/cfg['model'],ROOT/'configs/model.json',ROOT/cfg['manifest']]
    before={p.relative_to(ROOT).as_posix():sha256(p) for p in protected}
    source=ROOT/'experiments/ag_refinement/run_four_metric_curriculum/test'
    reported=json.loads((source/'metrics.json').read_text())
    with (source/'per_image.csv').open(encoding='utf-8-sig',newline='') as handle:
        saved={r['id']:r for r in csv.DictReader(handle)}
    assert len(saved)==100 and len(manifest['splits']['test'])==100
    assert reported['checkpoint_sha256']==cfg['model_sha256']
    guard=DataGuard(manifest['source']['data_root'],manifest,('test',));guard.install()
    results=[];metric_rows=[];input_rows=[]
    for index,row in enumerate(manifest['splits']['test'],1):
        low_rgb=read_rgb(Path(manifest['source']['data_root'])/row['low'])
        high_rgb=read_rgb(Path(manifest['source']['data_root'])/row['high'])
        def tensor(rgb):
            return (torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float()[None]/255).contiguous(memory_format=torch.channels_last)
        prediction=enhance(model,tensor(low_rgb),cfg)
        output_rgb=pixels(prediction)
        with Image.open(source/'images'/f"{row['id']}_RGB.png") as opened:
            exported=np.asarray(opened).copy()
        equal=bool(np.array_equal(output_rgb,exported))
        assert equal,f"Output PNG differs: {row['id']}"
        independent=independent_scores(exported)
        measured=describe(prediction,tensor(high_rgb))
        delta=max(abs(independent[k]-float(saved[row['id']][k])) for k in METRIC_NAMES)
        assert delta<1e-10,(row['id'],delta)
        assert max(abs(independent[k]-measured[k]) for k in METRIC_NAMES)<1e-10
        psnr_error=abs(measured['PSNR']-float(saved[row['id']]['PSNR']))
        ssim_error=abs(measured['SSIM']-float(saved[row['id']]['SSIM']))
        assert psnr_error<1e-5 and ssim_error<1e-6,(row['id'],psnr_error,ssim_error)
        pred_np=prediction[0].permute(1,2,0).cpu().numpy().astype(np.float64)
        target_np=tensor(high_rgb)[0].permute(1,2,0).cpu().numpy().astype(np.float64)
        mse=np.mean((pred_np-target_np)**2)
        psnr_independent=-10*math.log10(max(float(mse),1e-12))
        assert abs(psnr_independent-measured['PSNR'])<1e-5
        png_mse=np.mean((exported.astype(np.float64)/255-high_rgb.astype(np.float64)/255)**2)
        png_psnr=-10*math.log10(max(float(png_mse),1e-12))
        results.append(dict(id=row['id'],PNG_identical=equal,MLI=independent['MLI'],CI=independent['CI'],E=independent['E'],AG=independent['AG'],PSNR=measured['PSNR'],SSIM=measured['SSIM'],max_four_metric_error=delta,PSNR_error=psnr_error,SSIM_error=ssim_error,PSNR_from_saved_PNG=png_psnr))
        metric_rows.append({**independent,'PSNR':measured['PSNR'],'SSIM':measured['SSIM'],'PSNR_from_saved_PNG':png_psnr})
        input_rows.append(independent_scores(low_rgb))
        if index%25==0:print(f'Audit {index}/100: outputs and independent metric calculations match',flush=True)
    means={k:statistics.mean(r[k] for r in metric_rows) for k in metric_rows[0]}
    for k in (*METRIC_NAMES,'PSNR','SSIM'):assert abs(means[k]-reported['candidate'][k])<1e-5
    original=np.array([PDF_BENCHMARK[k][0] for k in ('AWF','AGCWD','SDD','CLAHE','RRM')])
    candidate=[means[k] for k in METRIC_NAMES]
    six=np.vstack([original,candidate])
    six_z={name:zvalue(values,six) for name,values in zip(('AWF','AGCWD','SDD','CLAHE','RRM','Model'),six)}
    input_means={k:statistics.mean(r[k] for r in input_rows) for k in METRIC_NAMES}
    pdf_input=dict(zip(METRIC_NAMES,PDF_BENCHMARK['Input (PDF)'][0]))
    input_percent_difference={k:(input_means[k]/pdf_input[k]-1)*100 for k in METRIC_NAMES}
    assert all(sha256(ROOT/name)==value for name,value in before.items())
    result=dict(verified=True,verified_scope='Project numerical reproducibility only',contest_comparability_verified=False,published_input_matches_current_measurement=False,input_percent_difference=input_percent_difference,date='2026-11-23',images=100,checkpoint_sha256=cfg['model_sha256'],protected_files_sha256=before,
                full_test_PNGs_identical=sum(r['PNG_identical'] for r in results),independent_four_metric_recomputation=True,
                max_four_metric_error=max(r['max_four_metric_error'] for r in results),max_PSNR_report_error=max(r['PSNR_error'] for r in results),max_SSIM_report_error=max(r['SSIM_error'] for r in results),
                mean=means,sample_SD={k:statistics.stdev(r[k] for r in metric_rows) for k in METRIC_NAMES},
                input_mean=input_means,PDF_input_mean=dict(zip(METRIC_NAMES,PDF_BENCHMARK['Input (PDF)'][0])),
                Z_fixed_original_five=zvalue(candidate,original),AWF_Z_published=1.4479,Z_recomputed_common_six=six_z,
                four_metrics_domain='Saved RGB uint8 PNG; luma weights 0.299/0.587/0.114; normalized 0..1',
                PSNR_SSIM_domain='Float32 RGB prediction before PNG rounding; RGB GT normalized 0..1',
                scope='Full deployed pipeline including calibration, RGB preservation and luminance refinement',
                AWF_rerun=False,official_grader_code_available=False,training_performed=False,parameter_selection_performed=False,
                prior_project_test_exposure=True,data_audit=guard.report(),seconds=time.perf_counter()-start)
    out=ROOT/'outputs/checks/metric_audit_20261123';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'audit.json',result)
    with (out/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
