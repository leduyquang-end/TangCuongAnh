"""Verify the approved default pipeline against the 100 already-exported test PNGs."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from PIL import Image
from kind_enhance.inference.pipeline import load_model,enhance
from kind_enhance.data.paired_dataset import PairedLowLightDataset,read_rgb
from kind_enhance.infrastructure.artifacts import pixels,save_image
from kind_enhance.engineering.guard import DataGuard
from kind_enhance.infrastructure.runtime import sha256,write_json

@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    model,config,manifest=load_model()
    run=ROOT/'experiments/ag_refinement/run_four_metric_curriculum'
    selection=json.loads((run/'selection.json').read_text())
    assert config['user_approved_as_final'] and config['model_sha256']==selection['sha256']
    guard=DataGuard(manifest['source']['data_root'],manifest,('test',));guard.install()
    matches=0
    for row in manifest['splits']['test']:
        low_path=Path(manifest['source']['data_root'])/row['low']
        low=torch.from_numpy(np.ascontiguousarray(read_rgb(low_path))).permute(2,0,1).float()[None]/255
        output=enhance(model,low.contiguous(memory_format=torch.channels_last),config)
        with Image.open(run/'test/images'/f"{row['id']}_RGB.png") as expected:
            assert np.array_equal(pixels(output),np.asarray(expected)),row['id']
        matches+=1
        if matches==1:first=low_path
        if matches%25==0:print(f'Default pipeline PNG match {matches}/100',flush=True)
    target=ROOT/'results/active_model/cli_output.png'
    completed=subprocess.run([sys.executable,'-B','-m','kind_enhance','infer','--input',str(first),'--output',str(target),'--overwrite'],cwd=ROOT,capture_output=True,text=True,check=True)
    with Image.open(target) as actual,Image.open(run/'test/images'/f"{manifest['splits']['test'][0]['id']}_RGB.png") as expected:
        assert np.array_equal(np.asarray(actual),np.asarray(expected))
    result=dict(verified=True,checkpoint=config['model'],checkpoint_sha256=sha256(ROOT/config['model']),
                full_test_PNGs_identical=matches,CLI_PNG_identical=True,architecture_parameters=sum(p.numel() for p in model.parameters()),
                source_selection_sha256=sha256(run/'selection.json'),data_audit=guard.report(),
                no_retraining=True,no_new_parameter_selection=True)
    write_json(ROOT/'results/active_model/pipeline_verification.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
