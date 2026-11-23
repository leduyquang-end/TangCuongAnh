import unittest,torch
from kind_enhance.inference.color_preservation import preserve_color
from kind_enhance.inference.calibration import luma
class RGBPreservationTests(unittest.TestCase):
 def test_luminance_and_gamut_for_all_search_settings(self):
  torch.manual_seed(31);p=torch.rand(2,3,24,24);low=torch.rand_like(p)
  for blend in (.5,.75,1.):
   for gain in (.5,.75,1.):
    q=preserve_color(p,low,blend,gain)
    torch.testing.assert_close(luma(q),luma(p),atol=2e-7,rtol=1e-6)
    self.assertGreaterEqual(q.min().item(),0);self.assertLessEqual(q.max().item(),1)
 def test_disabled_identity_and_black_white_are_finite(self):
  p=torch.rand(1,3,12,12);torch.testing.assert_close(p,preserve_color(p,torch.zeros_like(p),blend=0),rtol=0,atol=0)
  for v in (0.,1.):
   q=preserve_color(torch.full_like(p,v),torch.zeros_like(p))
   self.assertTrue(torch.isfinite(q).all());torch.testing.assert_close(q,torch.full_like(p,v))
 def test_colour_signal_preserved_from_input_at_fixed_luma(self):
  p=torch.full((1,3,8,8),.4);low=torch.tensor([.1,.03,.01]).reshape(1,3,1,1).expand_as(p)
  q=preserve_color(p,low,blend=1,gain=.75)
  self.assertTrue((q[:,0]>q[:,1]).all() and (q[:,1]>q[:,2]).all())
  torch.testing.assert_close(luma(q),luma(p),atol=2e-7,rtol=1e-6)
if __name__=='__main__':unittest.main()

