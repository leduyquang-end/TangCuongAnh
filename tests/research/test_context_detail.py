import unittest
import torch
from tools.research.context_detail_support import context_student_output,redistribution_deficit
from kind_enhance.inference.pipeline import enhance
from kind_enhance.inference.calibration import luma,calibrate,calibration_features
from kind_enhance.inference.color_preservation import preserve_color

class IdentityModel(torch.nn.Module):
    def forward(self,x):return .15+.7*x

class ContextDetailTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(827)
    def test_context_forward_matches_canonical_for_full_image(self):
        x=torch.rand(2,3,32,40);model=IdentityModel()
        cfg=dict(calibration=dict(contrast=1.08,detail=.25,sharpen=1.4),rgb_preservation=dict(blend=1.,gain=.5),luminance_refinement=dict(contrast=1.02,lift=.007))
        raw=model(x)
        colored=preserve_color(calibrate(raw,calibration_features(x,raw),cfg['calibration']),x,**cfg['rgb_preservation'])
        output=context_student_output(model,x,cfg,luma(raw).mean((-2,-1),keepdim=True),luma(colored).mean((-2,-1),keepdim=True))
        self.assertTrue(torch.allclose(output,enhance(model,x,cfg),atol=1e-7,rtol=0))
    def test_redistribution_does_not_reward_perpendicular_noise(self):
        truth=torch.linspace(.2,.6,24).reshape(1,1,1,24).expand(1,1,24,24)
        vertical=torch.linspace(-.1,.1,24).reshape(1,1,24,1).expand_as(truth)
        noise=torch.tensor(.1,requires_grad=True)
        loss=redistribution_deficit(truth+noise*vertical,truth,truth*1.5)
        loss.backward();self.assertLess(abs(noise.grad.item()),1e-7)
    def test_flat_gt_has_no_redistribution_target(self):
        truth=torch.full((1,1,24,24),.4)
        actual=(truth+.01*torch.randn_like(truth)).requires_grad_()
        loss=redistribution_deficit(actual,truth,actual.detach())
        self.assertEqual(loss.item(),0);loss.backward()
        self.assertTrue(torch.isfinite(actual.grad).all())

if __name__=='__main__':unittest.main()
