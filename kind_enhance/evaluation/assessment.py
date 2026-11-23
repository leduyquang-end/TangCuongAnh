import statistics
import numpy as np
from kind_enhance.infrastructure.artifacts import pixels
from kind_enhance.evaluation.competition import METRIC_NAMES,PDF_BENCHMARK,no_reference_scores,z_scores
from kind_enhance.evaluation.reference import score


def describe(prediction,target):
    rgb=pixels(prediction)
    psnr,ssim=score(prediction,target)
    return {**no_reference_scores(rgb),'PSNR':psnr,'SSIM':ssim,
            'clipped_fraction':float(np.mean((rgb==0)|(rgb==255)))}


def aggregate(rows):
    fields=(*METRIC_NAMES,'PSNR','SSIM','clipped_fraction')
    result={key:statistics.mean(row[key] for row in rows) for key in fields}
    reference=[row[0] for row in PDF_BENCHMARK.values() if row[2] is not None]
    candidate=[result[key] for key in METRIC_NAMES]
    result['z_fixed_pdf']=z_scores(candidate,reference)
    cohort=[*reference,candidate]
    result['awf_margin_six']=z_scores(candidate,cohort)-z_scores(PDF_BENCHMARK['AWF'][0],cohort)
    result['pairs']=len(rows)
    return result
