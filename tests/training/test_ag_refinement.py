import unittest,torch
from kind_enhance.training.losses.supported_edges import supported_edge_terms
class AGRefinementTests(unittest.TestCase):
 def test_flat_target_requests_no_edges_and_noise_is_penalized(self):
  t=torch.full((2,1,24,24),.4);p=t.clone().requires_grad_()
  terms=supported_edge_terms(p,t,t,4)
  self.assertEqual(sum(terms.values()).item(),0);sum(terms.values()).backward();self.assertTrue(torch.isfinite(p.grad).all())
  noisy=t+torch.randn_like(t)*.03
  self.assertGreater(supported_edge_terms(noisy,t,t,4)['flat_noise'].item(),0)
 def test_gt_targets_are_detached_and_signed_edges_gain(self):
  t=torch.linspace(.2,.6,24).expand(2,1,24,24).clone().requires_grad_();p=t.detach().clone().requires_grad_()
  before=supported_edge_terms(p,t,t,2)['signed_edges']
  stronger=.4+(p-.4)*1.5
  self.assertLess(supported_edge_terms(stronger,t,t,2)['signed_edges'].item(),before.item())
  before.backward();self.assertIsNone(t.grad);self.assertTrue(torch.isfinite(p.grad).all())
if __name__=='__main__':unittest.main()
