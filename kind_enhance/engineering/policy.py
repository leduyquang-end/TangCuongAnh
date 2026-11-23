"""Fixed exam references and deterministic, validation-only decisions."""
from copy import deepcopy
import math
import numpy as np
from kind_enhance.evaluation.competition import METRIC_NAMES, PDF_BENCHMARK, z_scores
from kind_enhance.evaluation.ranking import rank_algorithms


REFERENCES = {name: list(row[0]) for name, row in PDF_BENCHMARK.items() if row[2] is not None}
REFERENCE_MEAN = np.array(list(REFERENCES.values())).mean(0)
REFERENCE_STD = np.array(list(REFERENCES.values())).std(0, ddof=0)


def exam_score(metrics):
    values = np.array([metrics[key] for key in METRIC_NAMES], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Four finite metric means are required')
    return float(((values - REFERENCE_MEAN) / REFERENCE_STD).mean())


def benchmark_report(metrics):
    values = [metrics[key] for key in METRIC_NAMES]
    cohort = rank_algorithms({**REFERENCES, 'Candidate': values})
    fixed_awf = z_scores(REFERENCES['AWF'], list(REFERENCES.values()))
    return {'fixed_reference_z': exam_score(metrics), 'fixed_reference_AWF_z': fixed_awf,
            'fixed_reference_margin_vs_AWF': exam_score(metrics) - fixed_awf,
            'published_AWF_z_rounded': PDF_BENCHMARK['AWF'][2],
            'passes_slide_5_threshold_1_45': exam_score(metrics) > 1.45,
            'six_method_cohort': cohort,
            'six_method_margin_vs_AWF': cohort['composite_z']['Candidate'] - cohort['composite_z']['AWF']}


def quality_failures(metrics, baseline, limits):
    names = (*METRIC_NAMES, 'PSNR', 'SSIM', 'clipped_fraction', 'chroma_mae')
    if any(not math.isfinite(metrics[key]) for key in names):
        return ['nonfinite']
    tests = {
        'PSNR': metrics['PSNR'] >= baseline['PSNR'] - limits['psnr_drop_db'],
        'SSIM': metrics['SSIM'] >= baseline['SSIM'] - limits['ssim_drop'],
        'chroma': metrics['chroma_mae'] <= baseline['chroma_mae'] * limits['chroma_ratio_max'],
        'clipping': metrics['clipped_fraction'] <= limits['clipping_max'],
    }
    return [key for key, passed in tests.items() if not passed]


def decision(metrics, incumbent, baseline, settings):
    failed = quality_failures(metrics, baseline, settings['quality_limits'])
    gain = exam_score(metrics) - exam_score(incumbent)
    keep = not failed and gain > settings['minimum_z_gain']
    return {'kept': keep, 'z_gain_vs_incumbent': gain, 'failed_quality_checks': failed,
            'reason': 'higher_validation_z' if keep else ('quality_guard' if failed else 'z_not_higher')}


def get_value(recipe, path):
    for part in path.split('.'):
        recipe = recipe[part]
    return recipe


def set_value(recipe, path, value):
    parts = path.split('.')
    for part in parts[:-1]:
        recipe = recipe[part]
    recipe[parts[-1]] = value


def propose(base_recipe, history, settings):
    """Adapt only to saved validation results; no test result is accepted here."""
    kept = [trial for trial in history if trial['decision']['kept']]
    recipe = deepcopy(kept[-1]['recipe'] if kept else base_recipe)
    recipe.setdefault('metric_reward_weight', 0.0)
    recipe.setdefault('metric_component_weights', [1.0] * 4)
    recipe.setdefault('flat_noise_weight', 0.0)
    if history:
        last = history[-1]
        if last['decision']['failed_quality_checks']:
            # Back off from the failed proposal, not repeatedly from the champion.
            champion = deepcopy(recipe)
            recipe = deepcopy(last['recipe'])
            recipe['lr'] *= .5
            recipe['decomposition_lr'] *= .5
            recipe['metric_reward_weight'] = max(recipe['metric_reward_weight'], settings['initial_metric_reward'] / 4) * .5
            recipe['extra_weights']['reference_pixels'] *= 1.4
            recipe['structure'] *= 1.2
            recipe['flat_noise_weight'] = min(4.0, max(1.0, recipe['flat_noise_weight']) * 1.5)
            for anchor in ('brightness_anchor', 'entropy_anchor'):
                recipe['extra_weights'][anchor] = max(recipe['extra_weights'][anchor],
                                                      champion['extra_weights'][anchor])
        else:
            recipe['metric_reward_weight'] = max(recipe['metric_reward_weight'], settings['initial_metric_reward'])
            recipe['flat_noise_weight'] = 1.0
            relaxation = .5 if len(history) == 1 else .9
            recipe['extra_weights']['brightness_anchor'] *= relaxation
            recipe['extra_weights']['entropy_anchor'] *= relaxation
            if len(history) > 1:
                recipe['metric_reward_weight'] *= 1.1
                changes = [(last['metrics'][key] - last['parent_metrics'][key]) / REFERENCE_STD[i]
                           for i, key in enumerate(METRIC_NAMES)]
                weakest = int(np.argmin(changes))
                weights = list(recipe['metric_component_weights'])
                weights[weakest] = min(2.0, weights[weakest] * 1.1)
                recipe['metric_component_weights'] = [4 * weight / sum(weights) for weight in weights]
    for path, (low, high) in settings['search_bounds'].items():
        set_value(recipe, path, round(min(high, max(low, get_value(recipe, path))), 9))
    recipe['validate_every'] = settings['epochs_per_trial']
    return recipe


def validate_settings(settings):
    if settings['max_trials'] < 1 or settings['epochs_per_trial'] < 1 or settings['patience'] < 1:
        raise ValueError('Positive trial, epoch and patience budgets are required')
    if settings['max_minutes'] <= 0 or settings['threads'] < 1 or settings['minimum_z_gain'] <= 0:
        raise ValueError('Invalid runtime budget or minimum gain')
    for name, (low, high) in settings['search_bounds'].items():
        if not math.isfinite(low + high) or low < 0 or low > high:
            raise ValueError(f'Invalid bounds: {name}')
    for name, (mean, _, published) in PDF_BENCHMARK.items():
        if published is not None and abs(exam_score(dict(zip(METRIC_NAMES, mean))) - published) >= .0003:
            raise ValueError('Published z-score reproduction failed')
