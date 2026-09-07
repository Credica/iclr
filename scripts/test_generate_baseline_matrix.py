"""Checks for the two-machine 105-run baseline allocation."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_baseline_matrix import (  # noqa: E402
    assignments, teacher_prerequisites, METHODS, SEQUENCES)


class BaselineMatrixChecks(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]
        self.jobs = assignments(self.repo, Path('/tmp/baseline-artifacts'))

    def test_complete_factorial_and_unique_ids(self):
        expected = {(method, sequence, seed)
                    for method in METHODS for sequence in SEQUENCES
                    for seed in (1, 2, 3)}
        actual = {(job['method'], job['sequence'], job['seed'])
                  for job in self.jobs}
        self.assertEqual(expected, actual)
        self.assertEqual(105, len(self.jobs))
        self.assertEqual(105, len({job['run_id'] for job in self.jobs}))

    def test_balanced_disjoint_machine_assignment(self):
        counts = Counter(job['machine'] for job in self.jobs)
        self.assertEqual({1: 53, 2: 52}, dict(counts))
        positions = {machine: sum(len(job['tasks']) for job in self.jobs
                                  if job['machine'] == machine)
                     for machine in (1, 2)}
        self.assertEqual({1: 440, 2: 400}, positions)

    def test_recording_and_method_flags(self):
        for job in self.jobs:
            command = job['command']
            self.assertIn('--sac_optimizer adam', command)
            self.assertIn('--wandb True', command)
            self.assertIn('--bellman_probe True', command)
            self.assertIn('--bellman_spectral_stats True', command)
            self.assertIn('--no_stats False', command)
            self.assertIn('--num_evaluation_steps 10000', command)
            self.assertIn('--scalar_log_interval 1000', command)
            self.assertIn('--feature_stats_interval 10000', command)
            self.assertIn('--hessian_stats_interval 10000', command)
            self.assertIn('--bellman_probe_interval 100000', command)
            self.assertIn('--exact_sac_task_budget True', command)
        self.assertTrue(all(job['requires_teacher_artifacts']
                            for job in self.jobs if job['method'] == 'rnd'))
        self.assertTrue(all(not job['requires_teacher_artifacts']
                            for job in self.jobs if job['method'] != 'rnd'))

    def test_dmc_indices_and_occurrences(self):
        self.assertEqual((48, 49, 50, 48, 49, 50),
                         SEQUENCES['D-W6']['task_indices'])
        self.assertEqual((3, 5, 3, 5), SEQUENCES['D-C4']['task_indices'])
        self.assertEqual(6, len(SEQUENCES['D-W6']['tasks']))
        self.assertEqual(4, len(SEQUENCES['D-C4']['tasks']))

    def test_rnd_prerequisites_are_explicit_and_reusable(self):
        teacher_counts = {}
        for machine in (1, 2):
            selected = [job for job in self.jobs
                        if job['machine'] == machine]
            teachers = teacher_prerequisites(
                self.repo, Path('/tmp/baseline-artifacts'), selected)
            teacher_counts[machine] = len(teachers)
            self.assertGreater(len(teachers), 0)
            identities = {(job['env_type'], job['task'], job['seed'])
                          for job in teachers}
            self.assertEqual(len(teachers), len(identities))
            for teacher in teachers:
                self.assertIn('TEACHER_REUSE', teacher['command'])
                self.assertIn('--bellman_probe True', teacher['command'])
                self.assertIn('--sac_optimizer adam', teacher['command'])
                self.assertIn('--save_single_task_artifacts True',
                              teacher['command'])
                self.assertIn('--no_stats False', teacher['command'])
                self.assertTrue(teacher['model_path'].endswith('.pt'))
                self.assertTrue(teacher['rollout_path'].endswith('.pkl'))
        self.assertEqual({1: 48, 2: 15}, teacher_counts)


if __name__ == '__main__':
    unittest.main()
