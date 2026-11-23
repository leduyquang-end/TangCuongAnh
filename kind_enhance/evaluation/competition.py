import math

import numpy as np


METRIC_NAMES = ("MLI", "CI", "E", "AG")
PDF_BENCHMARK = {
    "Input (PDF)": ((0.0500, 0.0443, 4.5533, 0.0129), (0.0314, 0.0286, 0.7764, 0.0067), None),
    "AWF": ((0.3317, 0.1963, 7.2184, 0.1052), (0.0798, 0.0482, 0.4370, 0.0424), 1.4479),
    "AGCWD": ((0.2259, 0.2085, 6.8703, 0.0793), (0.0706, 0.0307, 0.5093, 0.0361), 0.4826),
    "SDD": ((0.2425, 0.1192, 6.5764, 0.0240), (0.0695, 0.0313, 0.3861, 0.0143), -0.5666),
    "CLAHE": ((0.1948, 0.1273, 6.4399, 0.0647), (0.0748, 0.0452, 0.7977, 0.0340), -0.5687),
    "RRM": ((0.2724, 0.0922, 6.2910, 0.0207), (0.0629, 0.0262, 0.3663, 0.0138), -0.7952),
}


def no_reference_scores(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError("Expected an RGB uint8 image with at least two rows and columns")
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255
    histogram = np.bincount(np.rint(255 * gray).astype(np.uint8).ravel(), minlength=256)
    probabilities = histogram[histogram > 0] / gray.size
    origin = gray[:-1, :-1]
    horizontal = gray[:-1, 1:] - origin
    vertical = gray[1:, :-1] - origin
    gradient = np.sqrt((horizontal**2 + vertical**2) / 2)
    return dict(zip(METRIC_NAMES, map(float, (
        gray.mean(), gray.std(), -(probabilities * np.log2(probabilities)).sum(), gradient.mean(),
    ))))


def z_scores(method_means, reference_means):
    values = np.asarray(reference_means, dtype=np.float64)
    scale = values.std(axis=0, ddof=0)
    if np.any(scale == 0):
        raise ValueError("Each ranking column needs a nonzero standard deviation")
    standardized = (np.asarray(method_means) - values.mean(axis=0)) / scale
    return float(standardized.mean())


def self_check():
    constant = no_reference_scores(np.full((3, 3, 3), 128, dtype=np.uint8))
    assert math.isclose(constant["MLI"], 128 / 255)
    assert all(math.isclose(constant[key], 0, abs_tol=1e-12) for key in ("CI", "E", "AG"))
    ramp = no_reference_scores(np.repeat(np.array([[0, 255], [0, 255]], dtype=np.uint8)[..., None], 3, axis=2))
    assert all(math.isclose(ramp[key], value) for key, value in
               zip(METRIC_NAMES, (0.5, 0.5, 1, math.sqrt(0.5))))
    benchmark = [row[0] for row in PDF_BENCHMARK.values() if row[2] is not None]
    assert all(abs(z_scores(mean, benchmark) - published) < 0.0003
               for mean, _, published in PDF_BENCHMARK.values() if published is not None)
