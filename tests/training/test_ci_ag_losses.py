import unittest
import torch
from kind_enhance.training.losses.contrast_gradient import ci_ag_terms,soft_entropy,stable_calibration_features
from kind_enhance.inference.calibration import calibrate,calibration_features


class CiAgTests(unittest.TestCase):
    def test_calibration_forward_matches_and_flat_gradient_is_finite(self):
        for image in (torch.full((1,3,32,32),0.5),torch.rand(1,3,32,32)):
            p=image.requires_grad_()
            low=torch.rand_like(p)*0.1
            settings={'contrast':1.05,'detail':0.0,'sharpen':0.8}
            old=calibrate(p.detach(),calibration_features(low,p.detach()),settings)
            new=calibrate(p,stable_calibration_features(low,p),settings)
            torch.testing.assert_close(old,new,atol=1e-7,rtol=1e-7)
            self.assertTrue(torch.isfinite(torch.autograd.grad(new.sum(),p)[0]).all())

    def test_identity_anchors_and_flat_edges(self):
        image=torch.full((1,3,32,32),0.5,requires_grad=True)
        terms=ci_ag_terms(image,image.detach(),image.detach())
        for key in ('brightness_anchor','entropy_anchor','supported_edges','reference_pixels'):
            self.assertLess(abs(terms[key].item()),1e-7)
        gradient=torch.autograd.grad(sum(terms.values()),image)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_only_prediction_receives_gradient(self):
        torch.manual_seed(20)
        p=torch.rand(2,3,32,32,requires_grad=True)
        ref=torch.rand_like(p,requires_grad=True)
        terms=ci_ag_terms(p,ref,torch.rand_like(p))
        gradients=torch.autograd.grad(sum(terms.values()),(p,ref),allow_unused=True)
        self.assertTrue(torch.isfinite(gradients[0]).all())
        self.assertIsNone(gradients[1])
        self.assertTrue(torch.isfinite(soft_entropy(p[:,0:1])).all())


if __name__=='__main__': unittest.main()
