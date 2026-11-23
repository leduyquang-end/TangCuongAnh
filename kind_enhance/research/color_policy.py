"""User-requested one-sided floors: no upper cap, <=1% loss for each metric."""
import math
METRICS=('MLI','CI','E','AG')
def metric_failures(candidate,baseline):
 return [key for key in METRICS if not math.isfinite(candidate[key]) or candidate[key]+1e-12<.99*baseline[key]]
def color_quality_failures(m,b):
 failures=metric_failures(m,b)
 checks={'chroma_improvement':m['chroma_mae']<=b['chroma_mae']*.95,
 'chromaticity_improvement':m['chromaticity_mae']<=b['chromaticity_mae']*.98,
 'saturation_error_improvement':m['saturation_mae']<=b['saturation_mae']*.98,
 'PSNR':m['PSNR']>=b['PSNR']-.2,'SSIM':m['SSIM']>=b['SSIM']-.005,'clipping':m['clipped_fraction']<=.01}
 return failures+[k for k,v in checks.items() if not v]
