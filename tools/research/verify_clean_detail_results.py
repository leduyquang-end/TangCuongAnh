"""Verify the clean-detail branch without opening any original dataset images."""
from pathlib import Path
import argparse,csv,json,statistics,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from kind_enhance.evaluation.competition import no_reference_scores,METRIC_NAMES
from kind_enhance.evaluation.detail import detail_quality_failures
from kind_enhance.engineering.guard import DataGuard
from kind_enhance.infrastructure.runtime import sha256,write_json,read_checkpoint


def verify(out):
    protocol=json.loads((out/'protocol.json').read_text())
    selection=json.loads((out/'selection.json').read_text())
    baseline=json.loads((out/'baseline.json').read_text())
    cfg=json.loads((ROOT/'configs/model.json').read_text())
    manifest=json.loads((ROOT/cfg['manifest']).read_text())
    guard=DataGuard(manifest['source']['data_root'],manifest,());guard.install()
    assert sha256(ROOT/'configs/model.json')==protocol['parent_config_sha256']
    assert sha256(ROOT/cfg['model'])==protocol['parent_checkpoint_sha256']
    assert sha256(ROOT/cfg['manifest'])==protocol['manifest_sha256']
    assert all(sha256(ROOT/relative)==digest for relative,digest in protocol['source_hashes'].items())
    assert selection['selection_locked'] and not selection['promoted_to_default']
    assert sha256(selection['checkpoint'])==selection['sha256']
    audit=json.loads((out/'data_audit.json').read_text())
    assert audit['status']=='complete' and not audit['test_images_allowed']
    assert audit['allowed_splits']==['train','val'] and not audit['blocked_attempts']
    assert audit['unique_images_opened']=={'train':1240,'val':138}
    events=json.loads((out/'validation_events.json').read_text())
    assert len(events)==sum(plan['epochs'] for plan in protocol['plans'])
    for event in events:
        assert detail_quality_failures(event['metrics'],baseline)[0]==event['failed_checks']
        assert Path(event['checkpoint']).exists()
    passed=[event for event in events if not event['failed_checks']]
    assert bool(passed)==selection['passed_all_quality_gates']
    if passed:
        assert not detail_quality_failures(selection['validation'],baseline)[0]
    else:
        assert selection['sha256']==protocol['parent_checkpoint_sha256']
        assert not (out/'test').exists()
    metrics=json.loads((out/'preview_metrics.json').read_text())
    expected=selection['validation'] if passed else selection['best_diagnostic']['metrics']
    assert all(abs(metrics[key]-expected[key])<1e-10 for key in metrics)
    with (out/'preview_per_image.csv').open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==69 and {row['id'] for row in rows}=={row['id'] for row in manifest['splits']['val']}
    assert len(list((out/'validation_preview/images').glob('*.png')))==69
    assert len(list((out/'validation_preview/stages').glob('*/*.png')))==36
    measured=[]
    for row in rows:
        with Image.open(out/'validation_preview/images'/f"{row['id']}_RGB.png") as picture:
            assert picture.mode=='RGB' and picture.size==(600,400)
            actual=no_reference_scores(np.asarray(picture))
        assert all(abs(actual[key]-float(row[key]))<1e-12 for key in METRIC_NAMES)
        measured.append(actual)
    assert all(abs(statistics.mean(row[key] for row in measured)-metrics[key])<1e-10 for key in METRIC_NAMES)
    result=dict(verified=True,epochs=len(events),all_source_hashes_match=True,active_config_unchanged=True,
                active_checkpoint_unchanged=True,validation_RGB_images=69,stage_images=36,
                PNG_metrics_match_CSV=True,quality_gate_decisions_reproduced=True,
                passed_all_quality_gates=bool(passed),data_audit=audit,verification_original_image_access=guard.report())
    write_json(out/'verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'experiments/clean_detail_refinement');args=parser.parse_args();verify(args.output.resolve())
