"""Checks for the two-machine 105-run baseline allocation."""
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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
                self.assertEqual(Path(teacher['model_path']).parents[2],
                                 Path('/tmp/baseline-artifacts/teachers'))
                self.assertNotIn('complete_', teacher['command'])
                self.assertNotIn('completion_path', teacher)
        for job in self.jobs:
            if job['method'] == 'rnd':
                self.assertIn('--rd_teacher_root /tmp/baseline-artifacts/teachers ',
                              job['command'])
        self.assertEqual({1: 48, 2: 15}, teacher_counts)

    def test_teacher_cache_shell_reuses_pair_without_receipt(self):
        for present in (('model_path', 'rollout_path'), ('model_path',),
                        ('rollout_path',), ()):
            with self.subTest(present=present), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'artifacts with spaces'
                repo = Path(directory) / 'stub repo'
                repo.mkdir()
                # Exercise the actual shell branch without starting RL or W&B.
                (repo / 'main_garage.py').write_text(
                    "print('TEACHER_TRAIN_REQUESTED')\n")
                job = next(j for j in self.jobs if j['method'] == 'rnd')
                teacher = teacher_prerequisites(repo, root, [job])[0]
                for key in present:
                    path = Path(teacher[key])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'cached artifact')
                command = teacher['command'].replace(
                    'python -B -u ', shlex.quote(sys.executable) + ' -B -u ')
                result = subprocess.run(['bash', '-c', command],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                reuse = len(present) == 2
                self.assertEqual('TEACHER_REUSE' in result.stdout, reuse)
                self.assertEqual('TEACHER_TRAIN_REQUESTED' in result.stdout, not reuse)
                for key in present:
                    self.assertEqual(Path(teacher[key]).read_bytes(), b'cached artifact')

    def test_teacher_cache_does_not_match_other_seed_task_or_budget(self):
        for old_stem in ('metaworld_sac_button-press-v2_3000000_1',
                         'metaworld_sac_button-press-v2_1500000_2',
                         'metaworld_sac_reach-v2_1500000_1'):
            with self.subTest(old_stem=old_stem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'artifacts'
                repo = Path(directory) / 'stub'
                repo.mkdir()
                (repo / 'main_garage.py').write_text("print('TEACHER_TRAIN_REQUESTED')\n")
                job = next(j for j in self.jobs if j['method'] == 'rnd')
                teacher = teacher_prerequisites(repo, root, [job])[0]
                for key, name in (('model_path', 'policy_' + old_stem + '.pt'),
                                  ('rollout_path', 'rollouts_' + old_stem + '.pkl')):
                    path = Path(teacher[key]).with_name(name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'old artifact')
                command = teacher['command'].replace(
                    'python -B -u ', shlex.quote(sys.executable) + ' -B -u ')
                result = subprocess.run(['bash', '-c', command],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('TEACHER_TRAIN_REQUESTED', result.stdout)
                self.assertNotIn('TEACHER_REUSE', result.stdout)

    def test_both_launchers_prepare_and_dry_run_all_commands(self):
        from args import parse_args
        environment = dict(os.environ)
        environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment['PATH']
        parsed_count = 0
        for machine, job_count, teacher_count in ((1, 53, 48), (2, 52, 15)):
            with self.subTest(machine=machine), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'artifacts with spaces'
                result = subprocess.run(
                    ['bash', str(self.repo / 'scripts' /
                                 'run_baselines_machine_{}.sh'.format(machine)),
                     str(root), '--prepare-only'], env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                manifest = json.loads((root / 'manifests' /
                    'baseline_jobs_machine_{}.json'.format(machine)).read_text())
                self.assertEqual(len(manifest['jobs']), job_count)
                self.assertEqual(len(manifest['teacher_prerequisites']), teacher_count)
                self.assertEqual(manifest['teacher_cache_policy'],
                                 'existing_model_rollout_pair_no_receipt_required')
                for job in manifest['jobs'] + manifest['teacher_prerequisites']:
                    tokens = shlex.split(job['command'])
                    start = tokens.index(str(self.repo / 'main_garage.py'))
                    options = tokens[start + 1:]
                    if options[-1] == 'fi':
                        options = options[:-1]
                        options[-1] = options[-1].rstrip(';')
                    with patch.object(sys, 'argv', ['main_garage.py'] + options):
                        args = parse_args()
                    self.assertEqual(args.sac_optimizer, 'adam')
                    self.assertTrue(args.wandb)
                    if args.cl_method == 'rnd':
                        self.assertEqual(args.rd_teacher_root, str(root / 'teachers'))
                    parsed_count += 1
                for prefix, count in (('baseline_jobs', job_count),
                                      ('baseline_prerequisites', teacher_count)):
                    result = subprocess.run(
                        ['bash', str(self.repo / 'scripts' / 'run_two_per_gpu_queue.sh'),
                         '--jobs-file', str(root / 'manifests' /
                             '{}_machine_{}.txt'.format(prefix, machine)),
                         '--log-dir', str(root / 'dry run logs'), '--dry-run'],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        universal_newlines=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('SHARD 0/1: {} of {} jobs'.format(count, count),
                                  result.stdout)
                self.assertFalse((root / 'runs').exists())
                self.assertFalse((root / 'teachers').exists())
        self.assertEqual(parsed_count, 168)


if __name__ == '__main__':
    unittest.main()
