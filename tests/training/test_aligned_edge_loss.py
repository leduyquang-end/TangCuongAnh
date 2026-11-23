import unittest
import torch
from kind_enhance.training.losses.gradient_alignment import aligned_edge_terms


class AlignedEdgeLossTests(unittest.TestCase):
    def test_flat_regions_do_not_request_texture(self):
        truth = torch.full((1, 1, 24, 24), .4, requires_grad=True)
        actual = truth.detach().clone().requires_grad_()
        terms = aligned_edge_terms(actual, truth, truth, 1.6)
        self.assertEqual(sum(terms.values()).item(), 0)
        sum(terms.values()).backward()
        self.assertIsNone(truth.grad)
        self.assertTrue(torch.isfinite(actual.grad).all())
        noisy = actual.detach() + .02 * torch.randn_like(actual)
        self.assertGreater(aligned_edge_terms(noisy, truth, truth, 1.6)['flat_noise'].item(), 0)

    def test_opposite_ground_truth_directions_are_not_amplified(self):
        prior = torch.linspace(.2, .6, 24).expand(1, 1, 24, 24).clone()
        opposite = 1 - prior
        current = prior.clone().requires_grad_()
        terms = aligned_edge_terms(current, opposite, prior, 1.6)
        self.assertEqual(terms['signed_edges'].item(), 0)
        self.assertEqual(terms['ag_deficit'].item(), 0)
        stronger = .4 + 1.3 * (current - .4)
        self.assertGreater(aligned_edge_terms(stronger, opposite, prior, 1.6)['flat_noise'].item(), 0)
        before = aligned_edge_terms(current, prior, prior, 1.6)['signed_edges']
        after = aligned_edge_terms(stronger, prior, prior, 1.6)['signed_edges']
        self.assertLess(after.item(), before.item())
        before.backward()
        self.assertTrue(torch.isfinite(current.grad).all())


if __name__ == '__main__':
    unittest.main()
