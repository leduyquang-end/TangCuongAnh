import unittest

import torch

from kind_enhance.training.losses.base import contrast_detail_loss, edge_alignment_loss, kind_gradient, refinement_loss
from kind_enhance.models import KinDAttentionCurve


class ConsistentLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_reference_has_zero_auxiliary_loss_and_gradient(self):
        flat = torch.full((1, 3, 32, 32), 0.5)
        step = flat.clone()
        step[..., 16:] = 0.8
        torch.manual_seed(31)
        for target in (flat, step, torch.rand_like(flat)):
            prediction = target.clone().requires_grad_()
            loss = contrast_detail_loss(prediction, target, {"detail_weight": 3, "contrast_weight": 1.2})
            loss = loss + edge_alignment_loss(prediction, target, 3)
            self.assertLess(abs(loss.item()), 1e-8)
            gradient = torch.autograd.grad(loss, prediction)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertLess(gradient.abs().max().item(), 1e-8)

    def test_noise_is_penalized_and_loss_is_batch_independent(self):
        torch.manual_seed(9)
        target = torch.rand(2, 3, 32, 32)
        prediction = (target + torch.randn_like(target) * 0.03).clamp(0, 1)
        def loss(p, t):
            return contrast_detail_loss(p, t, {"detail_weight": 1, "contrast_weight": 1}) + edge_alignment_loss(p, t, 1)
        combined = loss(prediction, target)
        separately = sum(loss(prediction[i:i+1], target[i:i+1]) for i in range(2)) / 2
        self.assertGreater(combined.item(), 0)
        torch.testing.assert_close(combined, separately)

    def test_gradient_has_no_artificial_border_or_batch_coupling(self):
        flat = torch.full((1, 1, 32, 32), 0.2)
        ramp = torch.linspace(0, 0.6, 32).expand(1, 1, 32, 32)
        for axis in (-1, -2):
            self.assertEqual(torch.count_nonzero(kind_gradient(flat, axis)).item(), 0)
            torch.testing.assert_close(kind_gradient(ramp, axis), kind_gradient(torch.cat((ramp, flat)), axis)[:1])
        self.assertTrue((kind_gradient(ramp, -1) > 0).all())

    def test_contrast_can_be_enabled_without_detail(self):
        target = torch.linspace(0.1, 0.9, 32).expand(1, 3, 32, 32)
        prediction = (target * 0.5).clone().requires_grad_()
        loss = contrast_detail_loss(prediction, target, {"detail_weight": 0, "contrast_weight": 1})
        self.assertGreater(loss.item(), 0)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, prediction)[0]).all())

    def test_all_three_networks_receive_finite_gradients(self):
        torch.manual_seed(4)
        model = KinDAttentionCurve()
        low, high = torch.rand(2, 3, 32, 32) * 0.1, torch.rand(2, 3, 32, 32)
        config = {"structure": 0.2, "gradient_weight": 0.25, "decomposition_weight": 0.05,
                  "detail_weight": 0.5, "contrast_weight": 0.25, "edge_alignment_weight": 0.5}
        loss = refinement_loss(model, low, high, config)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (model.decomposition, model.restoration, model.adjustment):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)


if __name__ == "__main__":
    unittest.main()
