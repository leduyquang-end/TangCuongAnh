"""Composite z ranking: one explicit cohort, equal metric weights, no Input."""
import numpy as np
from kind_enhance.evaluation.competition import METRIC_NAMES


def rank_algorithms(method_means):
    names = list(method_means)
    if len(names) < 2 or any(name.lower().startswith('input') for name in names):
        raise ValueError('Use at least two enhancement methods; exclude Input')
    values = np.asarray([method_means[name] for name in names], dtype=np.float64)
    if values.shape != (len(names), 4) or not np.isfinite(values).all():
        raise ValueError('Expected four finite metric means per algorithm')
    mu, sigma = values.mean(axis=0), values.std(axis=0, ddof=0)
    if np.any(sigma <= 0):
        raise ValueError('Cannot standardize a constant metric column')
    components = (values - mu) / sigma
    totals = components.mean(axis=1)
    return {'methods': names, 'metric_names': list(METRIC_NAMES), 'ddof': 0,
            'mean_across_methods': mu.tolist(), 'std_across_methods': sigma.tolist(),
            'per_metric_z': {name: row.tolist() for name, row in zip(names, components)},
            'composite_z': dict(zip(names, totals.tolist())),
            'rank': {name: 1 + int(np.sum(totals > score)) for name, score in zip(names, totals)}}
