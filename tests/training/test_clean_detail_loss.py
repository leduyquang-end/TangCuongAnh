import unittest
import torch
from kind_enhance.training.losses.clean_detail import clean_structure_terms,guided_detail_target
from kind_enhance.evaluation.detail import detail_quality_failures

class CleanDetailLossTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(827)
    def test_flat_gt_is_not_asked_to_reproduce_teacher_noise(self):
        truth=torch.full((2,1,24,24),.4,requires_grad=True)
        noisy=truth.detach()+torch.randn_like(truth)*.04
        goal=guided_detail_target(truth,noisy,3)
        self.assertLess(goal.std((-2,-1)).max().item(),1e-5)
        actual=noisy.clone().requires_grad_();terms=clean_structure_terms(actual,truth,noisy,1.05)
        self.assertEqual(terms['aligned_deficit'].item(),0)
        self.assertGreater(terms['flat_gradient'].item(),0)
        sum(terms.values()).backward()
        self.assertTrue(torch.isfinite(actual.grad).all());self.assertIsNone(truth.grad)
    def test_perpendicular_edges_are_penalized_not_rewarded_as_details(self):
        truth=torch.linspace(.2,.6,24).reshape(1,1,1,24).expand(1,1,24,24).clone()
        wrong=.4+torch.linspace(-.2,.2,24).reshape(1,1,24,1).expand_as(truth)
        correct=clean_structure_terms(truth,truth,truth,1.05)
        sideways=clean_structure_terms(wrong,truth,truth,1.05)
        self.assertGreater(sideways['aligned_deficit'].item(),correct['aligned_deficit'].item())
        self.assertGreater(sideways['perpendicular'].item(),correct['perpendicular'].item())
    def test_high_ag_cannot_offset_noise_or_ssim_regression(self):
        keys=('MLI','CI','E','AG','PSNR','SSIM','gradient_cosine','aligned_AG','chroma_mae','chromaticity_mae','saturation_mae','supported_gradient_mae','clipped_fraction','flat_ag','flat_residual_rms')
        baseline={key:1. for key in keys}
        candidate={**baseline,'AG':2.,'SSIM':.9,'flat_ag':1.1}
        failed,_=detail_quality_failures(candidate,baseline)
        self.assertIn('SSIM',failed);self.assertIn('flat_ag',failed)
        self.assertNotIn('AG',failed)

if __name__=='__main__':unittest.main()
