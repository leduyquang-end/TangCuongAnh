import unittest
import torch
from kind_enhance.training.losses.contrast_gradient import band_gain_terms


class BandGainTests(unittest.TestCase):
    def test_mean_inside_band_has_no_anchor_gradient(self):
        ref=torch.full((1,3,32,32),.5)
        p=torch.full_like(ref,.502,requires_grad=True)
        anchor=band_gain_terms(p,ref,ref)['brightness_anchor']
        self.assertEqual(anchor.item(),0)
        self.assertEqual(torch.autograd.grad(anchor,p)[0].abs().sum().item(),0)

    def test_mean_outside_band_is_pulled_back(self):
        ref=torch.full((1,3,32,32),.5)
        p=torch.full_like(ref,.51,requires_grad=True)
        anchor=band_gain_terms(p,ref,ref)['brightness_anchor']
        self.assertGreater(anchor.item(),0)
        self.assertTrue((torch.autograd.grad(anchor,p)[0]>0).all())

    def test_all_terms_have_finite_gradients(self):
        torch.manual_seed(92)
        p=torch.rand(2,3,32,32,requires_grad=True)
        ref=torch.rand_like(p,requires_grad=True)
        terms=band_gain_terms(p,ref,torch.rand_like(p))
        gradient,teacher_gradient=torch.autograd.grad(sum(terms.values()),(p,ref),allow_unused=True)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertIsNone(teacher_gradient)


if __name__=='__main__':unittest.main()
