"""Learn restoration and illumination with four metric floors and no test feedback."""
from pathlib import Path
import argparse
import copy
import json
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from kind_enhance.infrastructure.artifacts import write_csv
from kind_enhance.data.paired_dataset import PairedLowLightDataset
from kind_enhance.engineering.guard import DataGuard
from kind_enhance.inference.pipeline import load_model
from kind_enhance.training.losses.four_metrics import four_metric_loss
from kind_enhance.infrastructure.runtime import read_checkpoint,resource_snapshot,sha256,write_json
from kind_enhance.training.engine import train
from kind_enhance.evaluation.experiments import branch_hash,load_candidate,measure,failures as secondary_failures

OUT=ROOT/'experiments/ag_refinement/run_four_metric_curriculum'
PARENT=ROOT/'results/active_model/selection.json'
INITIAL=ROOT/json.loads((ROOT/'configs/model.json').read_text())['model']
KEYS=('MLI','CI','E','AG')


def primary_failures(metrics,baseline):
    import math
    failed=[key for key in KEYS if not math.isfinite(metrics[key]) or metrics[key]<baseline[key]]
    if metrics['CI']<.22:failed.append('CI_absolute_0.22')
    if metrics['AG']<=baseline['AG']:failed.append('AG_no_progress')
    return failed


def run(resume=False):
    started=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=resume)
    parent=json.loads(PARENT.read_text())
    cfg=copy.deepcopy(parent['output_config'])
    assert sha256(parent['checkpoint'])==parent['sha256']
    plans=[dict(name='ag_005',ag_target=.05,edge_gain=2.5,epochs=3,lr=6e-5),
           dict(name='ag_010',ag_target=.10,edge_gain=4.,epochs=3,lr=4e-5)]
    protocol=dict(parent_selection_sha256=sha256(PARENT),manifest_sha256=cfg['manifest_sha256'],
                  initial_checkpoint=str(INITIAL),initial_sha256=sha256(INITIAL),output_config=cfg,
                  plans=plans,training_branches=['restoration','adjustment'],frozen_branch='decomposition',
                  user_targets=dict(CI_min=.22,AG_target=.10,no_decline=list(KEYS)),
                  near_AG_reporting_threshold=.095,
                  primary_selection='Max AG among candidates with all four validation means >= parent and CI>=.22',
                  secondary_quality='Report PSNR SSIM colour flat noise clipping separately; never call raised noise recovered detail',
                  test_feedback_allowed=False,prior_project_test_exposure=True)
    sources=['tools/training/refine_four_metrics.py','kind_enhance/training/losses/four_metrics.py','kind_enhance/training/losses/gradient_alignment.py',
             'kind_enhance/training/losses/supported_edges.py','kind_enhance/training/engine.py','kind_enhance/inference/color_preservation.py',
             'kind_enhance/inference/luminance.py','kind_enhance/evaluation/competition.py','kind_enhance/evaluation/experiments.py']
    if resume:
        assert json.loads((OUT/'protocol.json').read_text())==protocol
        for relative in sources:assert sha256(ROOT/relative)==sha256(OUT/'source'/relative)
    else:
        write_json(OUT/'protocol.json',protocol)
        for relative in sources:
            destination=OUT/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,destination)
    teacher=load_candidate(parent['checkpoint'],cfg).requires_grad_(False).eval()
    _,default_cfg,manifest=load_model()
    guard=DataGuard(manifest['source']['data_root'],manifest,('train','val'));guard.install()
    validation=PairedLowLightDataset(manifest['source']['data_root'],manifest['splits']['val'],cache=True)
    training=PairedLowLightDataset(manifest['source']['data_root'],manifest['splits']['train'],patch=80,cache=True)
    baseline,rows=measure(teacher,validation,cfg)
    write_json(OUT/'baseline.json',baseline);write_csv(OUT/'baseline_validation.csv',rows)
    frozen_decomposition=branch_hash(teacher,'decomposition')
    events_path=OUT/'validation_events.json'
    events=json.loads(events_path.read_text()) if resume and events_path.exists() else []
    current_path=INITIAL
    best_primary=max((e for e in events if not e['primary_failures']),key=lambda e:e['metrics']['AG'],default=None)
    status='running'
    default_hash=sha256(ROOT/'configs/model.json')
    try:
        for plan in plans:
            folder=OUT/plan['name']
            last=read_checkpoint(folder/'last.pt','cpu',cfg['manifest_sha256'],'enhance') if resume and (folder/'last.pt').exists() else None
            if last and last['epoch']>=plan['epochs']:
                current_path=OUT/'best_four_metrics.pt' if best_primary else folder/'last.pt'
                continue
            model=load_candidate(current_path,cfg)
            model.decomposition.requires_grad_(False)
            model.restoration.requires_grad_(True);model.adjustment.requires_grad_(True)
            assert branch_hash(model,'decomposition')==frozen_decomposition
            recipe={**plan,'patch':80,'batch':8,'seed':827,'decomposition_lr':1e-6,'validate_every':1,'initial_sha256':sha256(current_path)}
            if last:
                recipe={key:value for key,value in last['config'].items() if key not in ('loss_source_sha256','metrics_source_sha256')}
            torch.manual_seed(827)
            def validate(current,dataset,device):
                nonlocal best_primary
                metrics,rows=measure(current,dataset,cfg)
                assert branch_hash(current,'decomposition')==frozen_decomposition
                epoch=1+sum(e['plan']==plan['name'] for e in events)
                primary=primary_failures(metrics,baseline)
                secondary=[key for key in secondary_failures(metrics,baseline) if key not in ('MLI','E','CI','AG','brightness_stability','entropy_stability')]
                event=dict(plan=plan['name'],epoch=epoch,metrics=metrics,primary_failures=primary,secondary_quality_flags=secondary)
                write_csv(folder/f'validation_{epoch:02d}.csv',rows)
                events.append(event);write_json(events_path,events)
                state=dict(model=current.state_dict(),stage='enhance',manifest=cfg['manifest_sha256'],epoch=epoch,config=recipe,validation=metrics)
                # Preserve every validation checkpoint, including the best four-metric candidate.
                torch.save(state,folder/f'epoch_{epoch:02d}.pt')
                if metrics['AG']>=max(e['metrics']['AG'] for e in events):torch.save(state,OUT/'highest_AG.pt')
                if not primary and (best_primary is None or metrics['AG']>best_primary['metrics']['AG']):
                    best_primary=event;torch.save(state,OUT/'best_four_metrics.pt')
                metrics={**metrics,'primary_failures':primary,'secondary_quality_flags':secondary}
                return metrics,(not primary,metrics['AG'],metrics['SSIM'])
            print(json.dumps(dict(plan=plan,initial=str(current_path))),flush=True)
            train(model,training,validation,torch.device('cpu'),'enhance',recipe,plan['epochs'],folder,
                  cfg['manifest_sha256'],validation_scoring=validate,loss_function=four_metric_loss(teacher,cfg))
            current_path=OUT/'best_four_metrics.pt' if best_primary else folder/'last.pt'
        path=OUT/'best_four_metrics.pt' if best_primary else Path(parent['checkpoint'])
        selected=load_candidate(path,cfg)
        measured,rows=measure(selected,validation,cfg)
        write_csv(OUT/'selected_validation.csv',rows)
        assert not best_primary or not primary_failures(measured,baseline)
        assert sha256(ROOT/'configs/model.json')==default_hash
        assert sha256(ROOT/default_cfg['model'])==default_cfg['model_sha256']
        histories=[row for plan in plans for row in json.loads((OUT/plan['name']/'history.json').read_text())]
        result=dict(checkpoint=str(path),sha256=sha256(path),baseline=baseline,validation=measured,
                    best_primary_event=best_primary,best_unconstrained_event=max(events,key=lambda e:e['metrics']['AG']),
                    selection_locked=True,passed_four_metric_floors=bool(best_primary),
                    target_AG_010_achieved=measured['AG']>=.10,near_AG_010_achieved=measured['AG']>=.095,
                    CI_022_achieved=measured['CI']>=.22,primary_failures=primary_failures(measured,baseline),
                    output_config={**cfg,'model':str(path),'model_sha256':sha256(path),
                                   'training_provenance':str(OUT.relative_to(ROOT)),
                                   'selection_reason':'Highest AG subject to no decline of four validation means and CI>=0.22',
                                   'evaluation_report':str(OUT/'BAO_CAO_BON_CHI_SO.md')},
                    frozen_decomposition_sha256=frozen_decomposition,
                    changed_branches={name:branch_hash(selected,name)!=branch_hash(teacher,name) for name in ('restoration','adjustment')},
                    epochs=len(histories),seconds=sum(row['seconds'] for row in histories),session_seconds=time.perf_counter()-started,
                    peak_RAM_MiB=max(row['resources'].get('peak_working_set_mib',0) for row in histories),
                    data_audit=guard.report(),resources=resource_snapshot(torch.device('cpu')),
                    default_config_sha256=default_hash,default_checkpoint_sha256=default_cfg['model_sha256'],
                    promoted_to_default=False,test_feedback_allowed=False)
        write_json(OUT/'selection.json',result)
        print(json.dumps(dict(selection=result)),flush=True);status='complete'
    finally:
        write_json(OUT/'data_audit.json',{**guard.report(),'status':status})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.output is not None:OUT=args.output.resolve()
    torch.set_num_threads(6)
    if args.resume and (OUT/'selection.json').exists():
        result=json.loads((OUT/'selection.json').read_text())
        assert result['selection_locked'] and sha256(result['checkpoint'])==result['sha256']
        print(json.dumps(dict(status='already_complete',selection=str(OUT/'selection.json'))),flush=True)
    else:run(args.resume)
