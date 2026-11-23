"""Verify exported PNG metrics and immutable experiment/evaluation locks."""
from pathlib import Path
import csv
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from kind_enhance.competition_metrics import no_reference_scores,METRIC_NAMES
from kind_enhance.engineering.guard import DataGuard
from kind_enhance.runtime import sha256,write_json
from tools.refine_four_metrics import OUT,primary_failures


def verify():
    selection_path=OUT/'selection.json'
    selection=json.loads(selection_path.read_text())
    manifest=json.loads((ROOT/selection['output_config']['manifest']).read_text())
    assert sha256(ROOT/selection['output_config']['manifest'])==selection['output_config']['manifest_sha256']
    assert sha256(selection['checkpoint'])==selection['sha256']
    assert sha256(ROOT/'configs/model.json')==selection['default_config_sha256']
    assert sha256(ROOT/'checkpoints/model_optimized.pt')==selection['default_checkpoint_sha256']
    split_ids={key:{row['id'] for row in manifest['splits'][key]} for key in ('train','val','test')}
    assert len(split_ids['train'])==620 and len(split_ids['val'])==69 and len(split_ids['test'])==100
    assert not (split_ids['train']&split_ids['val'] or split_ids['train']&split_ids['test'] or split_ids['val']&split_ids['test'])
    # Verification reads only exported PNGs; all original dataset image access is forbidden.
    guard=DataGuard(manifest['source']['data_root'],manifest,());guard.install()
    checked={}
    for split,folder,count in (('val','validation_export',69),('test','test',100)):
        directory=OUT/folder
        result=json.loads((directory/'metrics.json').read_text())
        lock=json.loads((directory/'evaluation_lock.json').read_text())
        assert lock==result['selection_lock']
        assert lock['selection_sha256']==sha256(selection_path)
        assert lock['protocol_sha256']==sha256(OUT/'protocol.json')
        assert lock['checkpoint_sha256']==sha256(selection['checkpoint'])
        assert lock['evaluated_checkpoint_sha256']==sha256(result['checkpoint'])
        assert lock['metrics_source_sha256']==sha256(ROOT/'kind_enhance/competition_metrics.py')
        assert lock['evaluator_source_sha256']==sha256(ROOT/'tools/evaluate_four_metrics.py')
        assert not lock['test_feedback_allowed']
        if split=='test':assert json.loads((directory/'test_lock.json').read_text())==lock
        with (directory/'per_image.csv').open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
        assert len(rows)==count and {row['id'] for row in rows}==split_ids[split]
        assert len(list((directory/'images').glob('*.png')))==count
        recomputed=[]
        for row in rows:
            with Image.open(directory/'images'/f"{row['id']}_RGB.png") as picture:
                assert picture.mode=='RGB' and picture.size==(600,400)
                measured=no_reference_scores(np.asarray(picture))
            assert all(abs(measured[key]-float(row[key]))<1e-12 for key in METRIC_NAMES)
            recomputed.append(measured)
        means={key:sum(row[key] for row in recomputed)/count for key in METRIC_NAMES}
        assert all(abs(means[key]-result['candidate'][key])<1e-10 for key in METRIC_NAMES)
        assert primary_failures(result['candidate'],result['baseline'])==result['primary_failures']
        assert result['data_audit']['allowed_splits']==[split]
        assert result['data_audit']['unique_images_opened'][split]==count*2
        assert not result['data_audit']['blocked_attempts']
        assert len(list((directory/'stages').glob('*/*.png')))==36
        checked[split]=dict(images=count,RGB_size=[600,400],unique_IDs=count,stage_images=36,
                            PNG_metrics_match_CSV=True,aggregate_matches_metrics=True,locks_match=True,
                            primary_failures=result['primary_failures'])
    verification=dict(verified=True,splits=checked,checkpoint_sha256=selection['sha256'],
                      architecture='KinD Decomposition + Attention U-Net + Curve CNN',
                      total_parameters=2217963,trainable_parameters_this_round=1982087,
                      dataset_splits_disjoint=True,default_config_unchanged=True,default_checkpoint_unchanged=True,
                      original_dataset_access_during_verification=guard.report(),
                      no_further_tuning_after_test=True)
    write_json(OUT/'verification.json',verification)
    test=json.loads((OUT/'test/metrics.json').read_text())
    print(json.dumps(dict(verification=verification,
                         test={key:dict(before=test['baseline'][key],after=test['candidate'][key],percent=(test['candidate'][key]/test['baseline'][key]-1)*100) for key in (*METRIC_NAMES,'PSNR','SSIM','flat_ag')},
                         CI_022_achieved=test['CI_022_achieved'],near_AG_010_achieved=test['near_AG_010_achieved'])),flush=True)


if __name__=='__main__':verify()
