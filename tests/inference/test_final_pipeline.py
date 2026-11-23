import unittest
import torch
from kind_enhance.inference.pipeline import enhance,enhance_calibrated
from kind_enhance.inference.color_preservation import preserve_color
from kind_enhance.inference.luminance import enhance_all_gain,refine_luminance
from kind_enhance.inference.calibration import calibrate,calibration_features

class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.calls=0
    def forward(self,x):
        self.calls+=1
        return (.15+.7*x).clamp(0,1)

class FinalPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(827);torch.set_num_threads(2)
        self.low=torch.rand(1,3,31,29)
        self.config={'calibration':{'contrast':1.08,'detail':.25,'sharpen':1.4}}
    def test_legacy_calibration_without_optional_steps(self):
        model=FakeModel();raw=model(self.low)
        expected=calibrate(raw,calibration_features(self.low,raw),self.config['calibration'])
        actual=enhance(model,self.low,self.config)
        self.assertTrue(torch.equal(actual,expected))
    def test_current_pipeline_applies_rgb_and_luminance_once(self):
        config={**self.config,'rgb_preservation':{'blend':1.,'gain':.5},'luminance_refinement':{'contrast':1.02,'lift':.007}}
        model=FakeModel()
        expected=refine_luminance(preserve_color(enhance_calibrated(model,self.low,config),self.low,**config['rgb_preservation']),**config['luminance_refinement'])
        model.calls=0;actual=enhance(model,self.low,config)
        self.assertEqual(model.calls,1);self.assertTrue(torch.equal(actual,expected))
        model.calls=0;reference=enhance_all_gain(model,self.low,config)
        self.assertEqual(model.calls,1);self.assertTrue(torch.equal(actual,reference))
        self.assertTrue(torch.isfinite(actual).all())

if __name__=='__main__':unittest.main()
