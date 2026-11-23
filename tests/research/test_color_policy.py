import unittest
from kind_enhance.research.color_policy import metric_failures,METRICS
class ColorPolicyTests(unittest.TestCase):
 def test_unbounded_increases_and_one_percent_boundary(self):
  b=dict(zip(METRICS,[.4,.23,7.5,.026]))
  self.assertEqual(metric_failures({k:v*2 for k,v in b.items()},b),[])
  self.assertEqual(metric_failures({k:v*.99 for k,v in b.items()},b),[])
 def test_each_metric_individually_rejects_more_than_one_percent_loss(self):
  b=dict(zip(METRICS,[.4,.23,7.5,.026]))
  for k in METRICS:
   m={name:value*2 for name,value in b.items()};m[k]=b[k]*.989
   self.assertEqual(metric_failures(m,b),[k])
if __name__=='__main__':unittest.main()
