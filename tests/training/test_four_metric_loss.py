import unittest
import torch
from kind_enhance.training.losses.four_metrics import supported_absolute_ag

class FourMetricLossTests(unittest.TestCase):
    def test_no_ag_request_when_ground_truth_is_flat(self):
        truth = torch.full((1,1,24,24),.4,requires_grad=True)
        actual = truth.detach().clone().requires_grad_()
        loss = supported_absolute_ag(actual, truth, truth, .1)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(actual.grad).all())
        self.assertIsNone(truth.grad)

    def test_ag_target_only_rewards_supported_aligned_edges(self):
        prior = torch.linspace(.2,.6,24).expand(1,1,24,24).clone().requires_grad_()
        actual = prior.detach().clone().requires_grad_()
        base = supported_absolute_ag(actual, prior, prior, .1)
        stronger = .4 + 1.3*(actual-.4)
        self.assertLess(supported_absolute_ag(stronger, prior, prior,.1).item(),base.item())
        self.assertEqual(supported_absolute_ag(actual, 1-prior, prior,.1).item(),0)
        base.backward()
        self.assertIsNone(prior.grad)
        self.assertTrue(torch.isfinite(actual.grad).all())

if __name__ == '__main__':
    unittest.main()
