import copy
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from kind_enhance.evaluation.competition import PDF_BENCHMARK, METRIC_NAMES
from kind_enhance.engineering.guard import DataGuard
from kind_enhance.engineering.objective import metric_proxies, metric_reward
from kind_enhance.engineering.policy import (REFERENCES, benchmark_report, decision, exam_score,
    get_value, propose, validate_settings)
from kind_enhance.engineering.runner import output_path, validate_split, recover_state_checkpoint
from kind_enhance.inference.pipeline import ROOT
from kind_enhance.infrastructure.runtime import sha256


class EngineeringLoopTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((ROOT/'configs/engineering_loop.json').read_text())
        self.recipe = json.loads((ROOT/'configs/training.json').read_text())['config']
        self.base = dict(zip(METRIC_NAMES, PDF_BENCHMARK['AWF'][0]))
        self.base.update(PSNR=20., SSIM=.8, chroma_mae=.05, clipped_fraction=.002)

    def test_z_matches_all_five_published_rows(self):
        validate_settings(self.settings)
        for mean, _, published in PDF_BENCHMARK.values():
            if published is not None:
                self.assertAlmostEqual(exam_score(dict(zip(METRIC_NAMES, mean))), published, delta=.0003)

    def test_z_uses_equal_weights_and_frozen_population_scale(self):
        values = np.array(list(REFERENCES.values()))
        shifted = dict(zip(METRIC_NAMES, values.mean(0) + values.std(0, ddof=0)))
        self.assertAlmostEqual(exam_score(shifted), 1.0, places=12)
        report = benchmark_report(self.base)
        self.assertEqual(len(report['six_method_cohort']['methods']), 6)
        self.assertNotIn('Input', report['six_method_cohort']['methods'])

    def test_ties_lower_z_and_nonfinite_are_never_kept(self):
        self.assertFalse(decision(self.base, self.base, self.base, self.settings)['kept'])
        lower = {**self.base, 'CI': self.base['CI'] * .99}
        self.assertFalse(decision(lower, self.base, self.base, self.settings)['kept'])
        with self.assertRaises(ValueError):
            exam_score({**self.base, 'AG': float('nan')})

    def test_higher_z_must_also_pass_quality(self):
        better = {**self.base, 'CI': self.base['CI'] + .01}
        self.assertTrue(decision(better, self.base, self.base, self.settings)['kept'])
        bad = {**better, 'SSIM': .6}
        verdict = decision(bad, self.base, self.base, self.settings)
        self.assertFalse(verdict['kept'])
        self.assertIn('SSIM', verdict['failed_quality_checks'])

    def test_proposal_is_bounded_reproducible_and_uses_quality_feedback(self):
        original = copy.deepcopy(self.recipe)
        first = propose(self.recipe, [], self.settings)
        history = [{'recipe': first, 'metrics': self.base, 'parent_metrics': self.base,
                    'decision': {'kept': False, 'failed_quality_checks': ['SSIM']}}]
        next_recipe = propose(self.recipe, history, self.settings)
        self.assertLess(next_recipe['lr'], first['lr'])
        self.assertGreater(next_recipe['extra_weights']['reference_pixels'], first['extra_weights']['reference_pixels'])
        self.assertGreater(next_recipe['metric_reward_weight'], 0)
        self.assertEqual(next_recipe, propose(self.recipe, history, self.settings))
        self.assertEqual(self.recipe, original)
        for path, (low, high) in self.settings['search_bounds'].items():
            self.assertGreaterEqual(get_value(next_recipe, path), low)
            self.assertLessEqual(get_value(next_recipe, path), high)

    def test_repeated_quality_failures_reduce_steps_progressively(self):
        first = propose(self.recipe, [], self.settings)
        first['metric_reward_weight'] = .03
        history = []
        for i in range(2):
            history.append({'recipe': first, 'metrics': self.base, 'parent_metrics': self.base,
                            'decision': {'kept': False, 'failed_quality_checks': ['SSIM']}})
            smaller = propose(self.recipe, history, self.settings)
            self.assertLess(smaller['lr'], first['lr'])
            self.assertLess(smaller['decomposition_lr'], first['decomposition_lr'])
            self.assertLess(smaller['metric_reward_weight'], first['metric_reward_weight'])
            self.assertGreaterEqual(smaller['extra_weights']['brightness_anchor'],
                                    first['extra_weights']['brightness_anchor'])
            first = smaller

    def test_all_four_training_proxies_have_finite_gradients(self):
        torch.manual_seed(991)
        image = torch.rand(2,3,24,24,requires_grad=True)
        proxies = metric_proxies(image)
        self.assertEqual(tuple(proxies.shape), (2,4))
        for index in range(4):
            gradient = torch.autograd.grad(proxies[:,index].sum(), image, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_flat_proxy_gradient_finite_reference_detached_batch_independent(self):
        image = torch.full((2,3,24,24), .4, requires_grad=True)
        reference = torch.full_like(image, .5, requires_grad=True)
        reward = metric_reward(image, reference, [1,1,1,1])
        expected = sum(metric_reward(image[i:i+1], reference[i:i+1], [1]*4) for i in range(2))/2
        torch.testing.assert_close(reward, expected)
        gradient, target_gradient = torch.autograd.grad(reward, (image,reference), allow_unused=True)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertIsNone(target_gradient)

    def test_split_guards_reject_test_in_search_and_train_in_test(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = {'splits': {'train': [{'low':'Train/low/a.png','high':'Train/normal/a.png'}],
                'val':[{'low':'Train/low/b.png','high':'Train/normal/b.png'}],
                'test':[{'low':'Test/low/c.png','high':'Test/normal/c.png'}]}}
            guard = DataGuard(folder, manifest, ('train','val'))
            guard.check(Path(folder)/'Train/low/a.png')
            with self.assertRaises(PermissionError):
                guard.check(Path(folder)/'Test/low/c.png')
            with self.assertRaises(PermissionError):
                guard.check(Path(folder)/'Train/low/unknown.png')
            final = DataGuard(folder, manifest, ('test',))
            final.check(Path(folder)/'Test/low/c.png')
            with self.assertRaises(PermissionError):
                final.check(Path(folder)/'Train/low/a.png')

    def test_output_and_official_split_are_protected(self):
        with self.assertRaises(ValueError):
            output_path(ROOT/'checkpoints')
        manifest = json.loads((ROOT/'data/split-local.json').read_text())
        validate_split(manifest)
        manifest['splits']['test'][0]['id'] = '00000'
        with self.assertRaises(ValueError):
            validate_split(manifest)

    def test_crash_after_state_commit_recovers_only_matching_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out/'best.pt').write_bytes(b'old')
            (out/'next_best.pt').write_bytes(b'new')
            state = {'best_sha256': sha256(out/'next_best.pt')}
            recover_state_checkpoint(out, state)
            self.assertEqual((out/'best.pt').read_bytes(), b'new')
            self.assertFalse((out/'next_best.pt').exists())
            recover_state_checkpoint(out, state)

    def test_corrupt_checkpoint_cannot_be_promoted_during_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out/'best.pt').write_bytes(b'old')
            (out/'next_best.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                recover_state_checkpoint(out, {'best_sha256': '0'*64})
            self.assertEqual((out/'best.pt').read_bytes(), b'old')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
