import unittest
import torch
from kind_enhance.training.losses.contrast_gradient import continued_gain_terms
from kind_enhance.inference.calibration import luma


class DirectGainTests(unittest.TestCase):
    def test_flat_image_has_no_forced_edges(self):
        image=torch.full((1,3,32,32),.5,requires_grad=True)
        terms=continued_gain_terms(image,image.detach(),image.detach())
        self.assertLess(terms['supported_edges'].item(),1e-7)
        self.assertTrue(torch.isfinite(torch.autograd.grad(sum(terms.values()),image)[0]).all())

    def test_supported_edges_receive_increasing_gradient(self):
        ref=torch.full((1,3,32,32),.4)
        ref[...,16:]=.6
        pred=ref.clone().requires_grad_()
        terms=continued_gain_terms(pred,ref,ref)
        self.assertGreater(terms['supported_edges'].item(),0)
        grad=torch.autograd.grad(terms['supported_edges'],pred)[0]
        updated=pred.detach()-grad
        self.assertGreater(luma(updated).diff(dim=-1).abs().mean().item(),luma(ref).diff(dim=-1).abs().mean().item())
        self.assertLess(abs(updated.mean().item()-ref.mean().item()),1e-6)


if __name__=='__main__':unittest.main()
