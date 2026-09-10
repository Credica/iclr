"""Check ablation coverage, exact event counts, CLI wiring and runnable queues."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from args import parse_args
from scripts.generate_clip_matrix import assignments
from scripts.generate_clip_ablations import ablation_assignments
from scripts.clip_sequences import SEQUENCES, DEFAULT_CLIP_SEQUENCES, paper_pair_names


REPO = Path(__file__).resolve().parents[1]


class ClipAblationQueueChecks(unittest.TestCase):
    def test_easy_hard_sequences_preserve_order_and_existing_defaults(self):
        self.assertEqual(DEFAULT_CLIP_SEQUENCES, ('F1', 'F2', 'F3', 'D-W6', 'D-C4'))
        self.assertEqual(len(assignments(REPO, Path('/tmp/artifacts'))), 15)
        expected = {'Easy': (19, 13, 5, 26, 48, 33, 24, 14),
                    'Hard': (19, 38, 47, 4, 49, 46, 7, 39)}
        jobs = assignments(REPO, Path('/tmp/artifacts'), sequences=['easy', 'HARD'])
        self.assertEqual(len(jobs), 6)
        self.assertEqual(sum(j['total_train_env_steps'] for j in jobs), 72000000)
        for job in jobs:
            self.assertEqual(job['task_indices'], list(expected[job['sequence']]))
            self.assertEqual(job['task_count'], 8)
            self.assertEqual(job['expected_clip_events_by_task'], [0] + [8] * 7)
            self.assertEqual([p['task_name'] for p in job['task_positions']],
                             list(SEQUENCES[job['sequence']]['tasks']))
            self.assertEqual([p['policy_head'] for p in job['task_positions']], list(range(8)))
        ablations, _ = ablation_assignments(REPO, Path('/tmp/artifacts'),
                                           sequences=['Easy', 'Hard'], groups=['sides'], seeds=[1])
        self.assertEqual(len(ablations), 4)

    def test_dmc_paper_pairs_cover_both_panels_without_duplicates(self):
        outgoing = paper_pair_names('dmc', 'from_reference')
        incoming = paper_pair_names('dmc', 'to_reference')
        both = paper_pair_names('dmc')
        self.assertEqual((len(outgoing), len(incoming), len(both)), (18, 18, 30))
        self.assertEqual(set(both), set(outgoing) | set(incoming))
        self.assertEqual(len(set(outgoing) & set(incoming)), 6)
        self.assertTrue(all(SEQUENCES[name]['pair_source'] != SEQUENCES[name]['pair_target'] for name in both))
        selected = paper_pair_names('dmc', sources=['cartpole-swingup'], targets=['fish-upright'])
        self.assertEqual(len(selected), 1)
        self.assertEqual(SEQUENCES[selected[0]]['task_indices'], (5, 18))
        self.assertEqual(paper_pair_names('dmc', sources=['ball-in-cup-catch'], targets=['finger-turn-easy']),
                         paper_pair_names('dmc', sources=['ball_in_cup-catch'], targets=['finger-turn_easy']))
        with self.assertRaises(ValueError):
            paper_pair_names('dmc', sources=['cartpole-swingup', 'typo'])
        jobs = assignments(REPO, Path('/tmp/artifacts'), sequences=both, seeds=[1], steps_per_task=1000000)
        self.assertEqual(len(jobs), 30)
        self.assertEqual(sum(j['total_train_env_steps'] for j in jobs), 60000000)
        self.assertTrue(all(j['task_count'] == 2 and j['expected_clip_events_by_task'] == [0, 6] for j in jobs))

    def test_paper_pair_launcher_prepares_filtered_direction(self):
        with tempfile.TemporaryDirectory(prefix='clip paper pairs ') as root:
            env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
            subprocess.run(['bash', str(REPO / 'scripts/run_clip_pairs.sh'), root, '0', '--prepare-only',
                            '--suite', 'dmc', '--sources', 'ball_in_cup-catch',
                            '--targets', 'finger-turn_easy', '--seeds', '1', '--steps-per-task', '1000000'],
                           check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            manifest = json.loads((Path(root) / 'manifests/clip_pair_jobs.json').read_text())
            self.assertEqual(manifest['total_runs'], 1)
            job = manifest['jobs'][0]
            self.assertEqual(job['task_indices'], [2, 16])
            self.assertEqual(job['pair_source'], 'ball_in_cup-catch')
            self.assertEqual(job['pair_target'], 'finger-turn_easy')
            self.assertFalse((Path(root) / 'runs').exists())

    def test_three_groups_reuse_controls_and_reference_interval(self):
        jobs, controls = ablation_assignments(
            REPO, Path('/tmp/artifacts with spaces'), sequences=['F1', 'D-W6'])
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len(controls), 12)
        self.assertEqual({j['variant'] for j in jobs},
                         {'lower_only', 'upper_only', 'entry_only', 'periodic_only',
                          'interval_100000', 'interval_400000'})
        self.assertEqual(len({j['run_id'] for j in jobs + controls}), 48)
        self.assertEqual(sum(j['total_train_env_steps'] for j in jobs), 432000000)
        self.assertTrue(all(j['expected_clip_events'] == 0 for j in controls if j['variant'] == 'ft'))
        for job in jobs + controls:
            command = shlex.split(job['command'])
            # Parse every actual training command, including paths with spaces.
            start = command.index(str(REPO / 'main_garage.py'))
            with patch('sys.argv', command[start:]):
                args = parse_args()
            self.assertEqual(args.singular_clip_mode, job['clip']['mode'])
            self.assertEqual(args.singular_clip_schedule, job['clip']['trigger_policy'])
            self.assertEqual(args.singular_clip_interval, job['clip']['interval_env_steps'])
            self.assertEqual(args.sac_singular_clip, job['method'] == 'clip')
            self.assertEqual(args.steps_per_task, 1500000)
            self.assertEqual(args.num_evaluation_episodes, 50)
            self.assertEqual(args.singular_clip_start_task, 1)
            self.assertEqual(args.task_seq_idx, job['task_indices'])
            subprocess.run(['bash', '-n', '-c', job['command']], check=True)

    def test_controls_are_queued_once_and_custom_parameters_propagate(self):
        jobs, controls = ablation_assignments(
            REPO, Path('/tmp/artifacts'), groups=['sides', 'schedule', 'interval'],
            intervals=[200000, 400000, 800000], include_controls=True,
            sequences=['D-C4'], seeds=[3], singular_clip_min=.125, singular_clip_max=8.,
            singular_clip_interval=400000, singular_clip_start_task=0)
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len(controls), 2)
        self.assertEqual([j['variant'] for j in jobs].count('ft'), 1)
        self.assertEqual([j['variant'] for j in jobs].count('full_clip'), 1)
        for job in jobs:
            self.assertEqual(job['clip']['lower'], .125)
            self.assertEqual(job['clip']['upper'], 8.)
            self.assertEqual(job['clip']['start_task_position'], 0)
        self.assertNotIn('interval_400000', {j['variant'] for j in jobs})

    def test_expected_counts_match_entry_period_and_final_endpoint_rules(self):
        for schedule, interval, expected in (
                ('entry_and_periodic', 200000, [0, 8, 8, 8]),
                ('entry_only', 200000, [0, 1, 1, 1]),
                ('periodic_only', 200000, [0, 7, 7, 7]),
                ('entry_and_periodic', 100000, [0, 15, 15, 16]),
                ('entry_and_periodic', 400000, [0, 4, 4, 4]),
                ('entry_and_periodic', 1500000, [0, 1, 1, 2])):
            with self.subTest(schedule=schedule, interval=interval):
                job = assignments(REPO, Path('/tmp/artifacts'), sequences=['D-C4'], seeds=[1],
                                  singular_clip_schedule=schedule,
                                  singular_clip_interval=interval)[0]
                self.assertEqual(job['expected_clip_events_by_task'], expected)
                self.assertEqual(job['expected_clip_events'], sum(expected))

    def test_changed_configs_have_distinct_ablation_run_ids(self):
        common = dict(sequences=['F1'], seeds=[1], groups=['sides'])
        first, _ = ablation_assignments(REPO, Path('/tmp/artifacts'), **common)
        second, _ = ablation_assignments(REPO, Path('/tmp/artifacts'),
                                         singular_clip_min=.5, **common)
        self.assertFalse({j['run_id'] for j in first} & {j['run_id'] for j in second})

    def test_invalid_queue_parameters_fail_before_launch(self):
        for kwargs in ({'seeds': [1, 1]}, {'seeds': [-1]}, {'sequences': []},
                       {'sequences': ['F1', 'F1']}, {'sequences': ['unknown']},
                       {'steps_per_task': 1500001}, {'singular_clip_start_task': 10},
                       {'singular_clip_interval': 9999}, {'singular_clip_interval': 10501},
                       {'singular_clip_min': 0}, {'singular_clip_max': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                assignments(REPO, Path('/tmp/artifacts'), **kwargs)
        for kwargs in ({'groups': []}, {'groups': ['sides', 'sides']},
                       {'groups': ['unknown']}, {'intervals': [200000]},
                       {'intervals': [100000, 100000]}, {'intervals': [0]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ablation_assignments(REPO, Path('/tmp/artifacts'), **kwargs)

    def test_shell_prepare_and_queue_dry_run_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='clip ablations ') as root:
            env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
            subprocess.run(['bash', str(REPO / 'scripts/run_clip_ablations.sh'), root,
                            '0,1', '--prepare-only', '--groups', 'interval',
                            '--sequences', 'F1', '--seeds', '1', '2', '3',
                            '--clip-intervals', '100000', '200000', '400000'],
                           check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            payload = json.loads((Path(root) / 'manifests/clip_ablation_jobs.json').read_text())
            self.assertEqual(payload['total_runs'], 6)
            self.assertFalse(payload['controls_queued'])
            self.assertEqual(len(payload['required_controls']), 3)
            self.assertEqual({j['variant'] for j in payload['required_controls']}, {'full_clip'})
            self.assertEqual(payload['status'], 'prepared_not_run')
            result = subprocess.run(
                ['bash', str(REPO / 'scripts/run_two_per_gpu_queue.sh'),
                 '--jobs-file', str(Path(root) / 'manifests/clip_ablation_jobs.txt'),
                 '--log-dir', str(Path(root) / 'queue'), '--gpus', '0,1', '--dry-run'],
                check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertIn('6 of 6 jobs', result.stdout)
            self.assertFalse((Path(root) / 'runs').exists())


if __name__ == '__main__':
    unittest.main()
