import unittest,torch
from kind_enhance.inference.luminance import refine_luminance,all_gain_failures,METRICS
from kind_enhance.inference.calibration import luma
class AllGainTests(unittest.TestCase):
 def test_identity_and_chroma_preserved_away_from_boundaries(self):
  torch.manual_seed(31);x=torch.rand(2,3,24,24)*.4+.3
  torch.testing.assert_close(refine_luminance(x),x,rtol=0,atol=0)
  y=refine_luminance(x,1.02,.003)
  torch.testing.assert_close(y-luma(y),x-luma(x),atol=2e-7,rtol=1e-6)
 def test_black_white_and_random_stay_finite_in_gamut(self):
  for x in (torch.zeros(2,3,12,12),torch.ones(2,3,12,12),torch.rand(2,3,12,12)):
   y=refine_luminance(x,1.03,.003)
   self.assertTrue(torch.isfinite(y).all());self.assertGreaterEqual(y.min().item(),0);self.assertLessEqual(y.max().item(),1)
 def test_any_single_metric_reduction_or_tie_rejected(self):
  b=dict(zip(METRICS,[.4,.23,7.5,.026]));other={**b,'CI':.24}
  self.assertEqual(all_gain_failures({k:max(b[k],other[k])*1.01 for k in METRICS},[b,other]),[])
  for k in METRICS:
   m={key:max(b[key],other[key])*1.01 for key in METRICS};m[k]=b[k]
   self.assertIn(k,all_gain_failures(m,[b,other]))
if __name__=='__main__':unittest.main()
