import unittest
import torch
from kind_enhance.training.losses.color import color_terms,chromaticity,make_color_loss
from kind_enhance.models import KinDAttentionCurve

class ColorLossTests(unittest.TestCase):
 def test_exact_target_zero_and_finite_flat_gradients(self):
  for value in (0.,.4,1.):
   pred=torch.full((2,3,16,16),value,requires_grad=True)
   target=pred.detach().clone().requires_grad_()
   loss=sum(color_terms(pred,target).values())
   self.assertEqual(loss.item(),0.)
   loss.backward();self.assertTrue(torch.isfinite(pred.grad).all());self.assertIsNone(target.grad)
 def test_gray_collapse_and_wrong_cast_penalized(self):
  target=torch.tensor([.8,.3,.1]).reshape(1,3,1,1).expand(1,3,8,8)
  gray=torch.full_like(target,.4);wrong=target.flip(1)
  for pred in (gray,wrong):self.assertGreater(sum(color_terms(pred,target).values()).item(),.1)
  torch.testing.assert_close(chromaticity(target),chromaticity(target*.5),rtol=.02,atol=.01)
 def test_batch_independence_and_small_step_decreases_error(self):
  torch.manual_seed(827)
  pred=torch.rand(2,3,12,12,requires_grad=True);target=torch.rand_like(pred)
  loss=sum(color_terms(pred,target).values())
  separate=sum(sum(color_terms(pred[i:i+1],target[i:i+1]).values()) for i in range(2))/2
  torch.testing.assert_close(loss,separate)
  loss.backward();updated=pred.detach()-.2*pred.grad
  self.assertLess(sum(color_terms(updated,target).values()).item(),loss.item())
if __name__=='__main__':unittest.main()
