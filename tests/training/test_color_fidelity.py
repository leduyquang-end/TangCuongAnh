import unittest
import torch
from kind_enhance.training.losses.color_fidelity import fidelity_terms

class ColorFidelityTests(unittest.TestCase):
 def test_exact_target_and_flat_gradients(self):
  for value in (0.,.4,1.):
   p=torch.full((2,3,16,16),value,requires_grad=True)
   loss=sum(fidelity_terms(p,p.detach()).values());self.assertEqual(loss.item(),0.)
   loss.backward();self.assertTrue(torch.isfinite(p.grad).all())
 def test_grey_collapse_and_wrong_hue_penalized(self):
  t=torch.tensor([.8,.3,.1]).reshape(1,3,1,1).expand(1,3,8,8)
  for v in fidelity_terms(torch.full_like(t,.4),t).values():self.assertGreater(v.item(),0.)
  self.assertGreater(fidelity_terms(t.flip(1),t)['direction'].item(),.1)
 def test_batch_independent_and_target_detached(self):
  torch.manual_seed(31);p=torch.rand(2,3,12,12,requires_grad=True);t=torch.rand_like(p,requires_grad=True)
  loss=sum(fidelity_terms(p,t).values())
  separate=sum(sum(fidelity_terms(p[i:i+1],t[i:i+1]).values()) for i in range(2))/2
  torch.testing.assert_close(loss,separate);loss.backward()
  self.assertIsNone(t.grad);self.assertTrue(torch.isfinite(p.grad).all())
if __name__=='__main__':unittest.main()
