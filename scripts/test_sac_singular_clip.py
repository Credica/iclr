"""Clip schedule, exact environment budgets, and multi-occurrence SAC checks."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import akro
import numpy as np
import torch

from garage import EnvSpec, StepType
from garage.replay_buffer import PathBuffer
from garage.torch import as_torch_dict
from garage.torch.algos.sac_singular_clip import FinetuningSACSingularClip
from garage.torch.algos.bellman_spectral_stats import empirical_jacobian
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from scripts.generate_clip_matrix import assignments


def make_algo(spec=None, count=3, **overrides):
    if spec is None:
        spec = EnvSpec(akro.Box(-np.inf, np.inf, shape=(3,)),
                       akro.Box(-1., 1., shape=(2,)), max_episode_length=10)
    no_stats = overrides.get('no_stats', True)
    kwargs = dict(
        policy=TanhGaussianMLPPolicy(spec, n_tasks=count,
                                    hidden_sizes=(8, 8), no_stats=no_stats),
        qf1=ContinuousMLPQFunction(spec, hidden_sizes=(8, 8), no_stats=no_stats),
        qf2=ContinuousMLPQFunction(spec, hidden_sizes=(8, 8), no_stats=no_stats),
        env_spec=spec, sampler=None,
        replay_buffer=PathBuffer(capacity_in_transitions=30000),
        num_tasks=1, eval_env=[object()] * count,
        gradient_steps_per_itr=1, seed=7, no_stats=True, use_wandb=False,
        bellman_probe=False, bellman_spectral_stats=False,
        q_reset=False, policy_reset=False, task_names=['task'] * count,
        buffer_batch_size=8, fixed_alpha=0.01, multi_input=isinstance(spec, list))
    kwargs.update(overrides)
    return FinetuningSACSingularClip(**kwargs)


def samples(obs=3, act=2):
    return as_torch_dict(dict(
        observation=np.random.randn(8, obs).astype('float32'),
        next_observation=np.random.randn(8, obs).astype('float32'),
        action=np.random.uniform(-1, 1, (8, act)).astype('float32'),
        reward=np.random.randn(8, 1).astype('float32'),
        terminal=np.zeros((8, 1), dtype='float32')))


class SingularClipChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def test_all_later_entries_preserve_actor_bias_adam_and_sync_targets(self):
        algorithm = make_algo()
        algorithm.optimize_policy(samples(), seq_idx=0)
        actor = copy.deepcopy(algorithm.policy.state_dict())
        moments = copy.deepcopy(algorithm._qf1_optimizer.state_dict())
        parameter_ids = [id(p) for p in algorithm._qf1.parameters()]
        biases = {n: p.clone() for n, p in algorithm._qf1.named_parameters()
                  if n.endswith('bias')}
        algorithm._clip_at_task_start(0)
        self.assertEqual(algorithm._singular_clip_events, [])
        with torch.no_grad():
            for model in (algorithm._qf1, algorithm._qf2):
                for layer in model.modules():
                    if isinstance(layer, torch.nn.Linear):
                        layer.weight.mul_(100.)
        for previous in (0, 1):
            algorithm.global_env_step = (previous + 1) * 1500000
            algorithm.task_change(previous)
            algorithm._clip_at_task_start(previous + 1)  # duplicate is a no-op
        self.assertEqual([e['task'] for e in algorithm._singular_clip_events], [1, 2])
        self.assertEqual(parameter_ids, [id(p) for p in algorithm._qf1.parameters()])
        for name, value in actor.items():
            self.assertTrue(torch.equal(value, algorithm.policy.state_dict()[name]))
        for name, value in biases.items():
            self.assertTrue(torch.equal(value, algorithm._qf1.state_dict()[name]))
        after = algorithm._qf1_optimizer.state_dict()
        self.assertEqual(moments['param_groups'], after['param_groups'])
        for parameter, state in moments['state'].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, after['state'][parameter][key]))
        for online, target in ((algorithm._qf1, algorithm._target_qf1),
                               (algorithm._qf2, algorithm._target_qf2)):
            for name, value in online.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[name]))
            for layer in online.modules():
                if isinstance(layer, torch.nn.Linear):
                    values = torch.linalg.svdvals(layer.weight.double())
                    self.assertGreaterEqual(float(values.min()), .25 - 1e-5)
                    self.assertLessEqual(float(values.max()), 4. + 1e-5)

    def test_periodic_clock_uses_env_steps_not_updates_and_deduplicates(self):
        for updates in (50000, 190500, 200000):
            algorithm = make_algo()
            algorithm._task_env_start_step = 1500000
            algorithm.global_env_step = 1700000
            algorithm._critic_optimizer_steps = updates
            algorithm._is_last_collection_update = False
            algorithm._after_critic_update(None, 1)
            self.assertEqual(algorithm._singular_clip_events, [])
            algorithm._is_last_collection_update = True
            target = copy.deepcopy(algorithm._target_qf1.state_dict())
            algorithm._after_critic_update(None, 1)
            algorithm._after_critic_update(None, 1)
            self.assertEqual(len(algorithm._singular_clip_events), 1)
            event = algorithm._singular_clip_events[0]
            self.assertEqual(event['task_env_step'], 200000)
            self.assertEqual(event['critic_optimizer_step'], updates)
            self.assertEqual(event['target_update'], 'polyak')
            for name, value in target.items():
                self.assertTrue(torch.equal(value, algorithm._target_qf1.state_dict()[name]))
            online = copy.deepcopy(algorithm._qf1.state_dict())
            algorithm._update_targets()
            for name, value in target.items():
                expected = value * (1. - algorithm._tau) + online[name] * algorithm._tau
                self.assertTrue(torch.equal(expected, algorithm._target_qf1.state_dict()[name]))
            # Critic update count alone must not cause a clip at 210k env.
            algorithm.global_env_step = 1710000
            algorithm._critic_optimizer_steps = 200000
            algorithm._after_critic_update(None, 1)
            algorithm._after_critic_update(None, 0)
            self.assertEqual(len(algorithm._singular_clip_events), 1)

    def test_coincident_boundary_period_skips_outgoing_clip(self):
        algorithm = make_algo()
        algorithm._sampler = SimpleNamespace(_envs=[SimpleNamespace(cur_seq_idx=2)])
        algorithm.global_env_step = 400000
        algorithm._task_env_start_step = 200000
        algorithm._after_critic_update(None, 1)
        self.assertEqual(algorithm._singular_clip_events, [])
        algorithm.task_change(1)
        algorithm._after_critic_update(None, 2)
        self.assertEqual(len(algorithm._singular_clip_events), 1)
        self.assertEqual(algorithm._singular_clip_events[0]['trigger'], 'task_start')
        self.assertEqual(algorithm._singular_clip_events[0]['task_env_step'], 0)

    def test_real_sac_loop_exact_1_5m_including_warmup_for_three_tasks(self):
        # Exercise the actual train loop at the full formal clock, replacing
        # environment sampling and costly gradient math only (not scheduling).
        for collection_batch in (500, 1000):
            environment = SimpleNamespace(cur_seq_idx=0, reset=Mock())
            algorithm = make_algo(
                exact_task_budget=True, steps_per_epoch=10000 // collection_batch,
                sampler=SimpleNamespace(_envs=[environment]))
            algorithm.recent_trajectory = SimpleNamespace(append=lambda _: None,
                                                         clear=lambda: None)
            algorithm.save_results = Mock()
            evaluations = []
            def evaluate(_):
                evaluations.append((algorithm.global_env_step, algorithm.seq_idx,
                                    len(algorithm._singular_clip_events)))
                return [0.]
            algorithm._evaluate_policy = evaluate
            def update(position):
                algorithm._critic_optimizer_steps += 1
                algorithm._after_critic_update(None, position)
                return torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
            algorithm.train_once = update
            collected = [0]
            warmups = []
            def collect(_, position, requested):
                count = requested or collection_batch
                self.assertEqual(position, environment.cur_seq_idx)
                if algorithm.replay_buffer.n_transitions_stored == 0:
                    warmups.append((position, count))
                collected[0] += count
                environment.cur_seq_idx = collected[0] // 1500000
                return [dict(observations=np.zeros((count, 3), dtype='float32'),
                             next_observations=np.zeros((count, 3), dtype='float32'),
                             actions=np.zeros((count, 2), dtype='float32'),
                             rewards=np.zeros(count, dtype='float32'),
                             step_types=np.full(count, StepType.MID))]
            trainer = SimpleNamespace(
                _train_args=SimpleNamespace(batch_size=collection_batch),
                step_epochs=lambda: range(450), step_itr=0, obtain_samples=collect)
            with redirect_stdout(io.StringIO()):
                algorithm.train(trainer)
            self.assertEqual(collected[0], 4500000)
            self.assertEqual(warmups, [(0, 10000), (1, 10000), (2, 10000)])
            self.assertEqual([e[0] for e in evaluations], list(range(10000, 4500001, 10000)))
            self.assertEqual(evaluations[149], (1500000, 0, 0))
            self.assertEqual(evaluations[299], (3000000, 1, 8))
            events = algorithm._singular_clip_events
            self.assertEqual(len(events), 16)
            for position in (1, 2):
                self.assertEqual([e['task_env_step'] for e in events if e['task'] == position],
                                 [0, 200000, 400000, 600000, 800000, 1000000, 1200000, 1400000])
            self.assertEqual(algorithm.seq_idx, 2)  # no nonexistent final entry

    def test_evaluation_selects_current_and_seen_occurrence_heads(self):
        algorithm = make_algo(count=4)
        algorithm.seq_idx = 2
        algorithm.global_env_step = 3100000
        environment = SimpleNamespace(cur_seq_idx=2)
        algorithm._sampler = SimpleNamespace(_envs=[environment])
        module = 'garage.torch.algos.sac_singular_clip.'
        with patch(module + 'obtain_evaluation_episodes') as obtain, \
                patch(module + 'log_performance', return_value=[1.]) as log:
            algorithm._evaluate_policy(0)
            self.assertEqual([c[0][2] for c in obtain.call_args_list], [2])
            obtain.reset_mock()
            environment.cur_seq_idx = 3
            algorithm._evaluate_policy(1)
            self.assertEqual([c[0][2] for c in obtain.call_args_list], [0, 1, 2])
            self.assertEqual([c[1]['prefix'] for c in log.call_args_list],
                             ['test/2/task/', 'test/0/task/', 'test/1/task/', 'test/2/task/'])

    def test_heterogeneous_dmc_update_and_full_parameter_jacobian(self):
        specs = [EnvSpec(akro.Box(-np.inf, np.inf, shape=(size,)),
                         akro.Box(-1., 1., shape=(2,)), max_episode_length=10)
                 for size in (3, 5, 3)]
        algorithm = make_algo(spec=specs)
        for position, size in enumerate((3, 5, 3)):
            batch = samples(obs=size)
            losses = algorithm.optimize_policy(batch, seq_idx=position)
            self.assertTrue(all(torch.isfinite(loss) for loss in losses))
            jacobian = empirical_jacobian(algorithm._qf1, batch['observation'][:2],
                                          batch['action'][:2], position)
            self.assertEqual(jacobian.shape[0], 2)
            self.assertTrue(torch.isfinite(jacobian).all())

    def test_full_probe_collects_and_records_for_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            algorithm = make_algo(bellman_probe=True, bellman_probe_dir=directory,
                                  bellman_probe_size=8, log_name='clip_test')
            batch = {key: value.numpy() for key, value in samples().items()}
            algorithm._append_bellman_probe(batch, 0)
            algorithm.global_env_step = 100000
            metric = algorithm._run_bellman_probe('interval', 0)
            self.assertIsNotNone(metric)
            self.assertEqual(len(algorithm._bellman_probe_buffers[0]['observation']), 8)
            algorithm.task_change(0)
            event_path = Path(algorithm._bellman_probe_run_dir) / 'singular_clip_events.jsonl'
            event = json.loads(event_path.read_text().splitlines()[0])
            self.assertEqual(event['task'], 1)
            self.assertEqual(event['global_env_step'], 100000)

    def test_dmc_factory_accepts_formal_clip_and_diagnostics(self):
        from args import parse_args
        from garage.algo_factory import get_algo
        with tempfile.TemporaryDirectory() as directory:
            argv = ['main_garage.py', '--env_type', 'dm_control',
                    '--rl_method', 'sac', '--cl_method', 'finetuning',
                    '--sac_singular_clip', 'True', '--bellman_probe', 'True',
                    '--bellman_spectral_stats', 'True', '--wandb', 'False',
                    '--bellman_probe_dir', directory, '--proc_name', 'factory_test']
            with patch('sys.argv', argv):
                args = parse_args()
            self.assertEqual(args.singular_clip_start_task, 1)
            spec = EnvSpec(akro.Box(-np.inf, np.inf, shape=(5,)),
                           akro.Box(-1., 1., shape=(1,)), max_episode_length=1000)
            with patch('garage.algo_factory.LocalSampler', return_value=None):
                algorithm = get_algo(args, [spec] * 4, 4, [], [], (1000, 10),
                                     ['balance', 'swingup', 'balance', 'swingup'])
            self.assertIsInstance(algorithm, FinetuningSACSingularClip)
            self.assertIsInstance(algorithm._qf1_optimizer, torch.optim.Adam)
            self.assertEqual(algorithm._qf1._layers[0][0].out_features, 1024)
            heads = algorithm.policy._module._shared_mean_log_std_network._output_layers
            self.assertEqual(len(heads), 8)  # mean/std pair for every position
            args.sac_optimizer = 'muon'
            with self.assertRaises(ValueError):
                get_algo(args, [spec] * 4, 4, [], [], (1000, 10), ['task'] * 4)

    def test_formal_matrix_and_occurrences(self):
        jobs = assignments(Path('/repo'), Path('/artifacts'))
        self.assertEqual(len(jobs), 15)
        self.assertEqual(sum(j['task_count'] for j in jobs), 120)
        self.assertEqual(sum(j['total_train_env_steps'] for j in jobs), 180000000)
        self.assertEqual(sum(j['expected_clip_events'] for j in jobs), 840)
        self.assertEqual(len({j['run_id'] for j in jobs}), 15)
        expected = {'F1': 10, 'F2': 10, 'F3': 10, 'D-W6': 6, 'D-C4': 4}
        for job in jobs:
            self.assertEqual(job['task_count'], expected[job['sequence']])
            self.assertIn(job['seed'], (1, 2, 3))
            command = shlex.split(job['command'])
            for flag, value in (('--sac_optimizer', 'adam'),
                                ('--steps_per_task', '1500000'),
                                ('--singular_clip_interval', '200000'),
                                ('--singular_clip_start_task', '1'),
                                ('--wandb', 'True'), ('--bellman_spectral_stats', 'True'),
                                ('--train_task_count', str(job['task_count']))):
                self.assertEqual(command[command.index(flag) + 1], value)
            positions = job['task_positions']
            self.assertEqual([p['policy_head'] for p in positions], list(range(len(positions))))
            if job['sequence'] == 'D-W6':
                self.assertEqual([p['occurrence'] for p in positions], [1, 1, 1, 2, 2, 2])
            if job['sequence'] == 'D-C4':
                self.assertEqual([p['occurrence'] for p in positions], [1, 1, 2, 2])

    def test_real_dmc_revisits_train_evaluate_and_record(self):
        from dm_control import suite
        from garage.envs import normalize
        from garage.experiment import CLTaskSampler
        from garage.sampler import LocalSampler
        from garage.trainer import Trainer
        from scripts.generate_baseline_matrix import SEQUENCES

        for sequence in ('D-W6', 'D-C4'):
            spec = SEQUENCES[sequence]
            names = ['DMControl-{}-{}'.format(*suite.ALL_TASKS[i])
                     for i in spec['task_indices']]
            self.assertEqual(names, list(spec['tasks']))
        task_sampler = CLTaskSampler(None, 2000, 7, env_type='dm_control',
                                     wrapper=lambda env: normalize(env))
        training, evaluation = task_sampler.sample([3, 5, 3, 5])
        for training_env, eval_env in zip(training[0].envs_before_make, evaluation):
            self.assertIsNot(training_env, eval_env)
        with tempfile.TemporaryDirectory() as directory:
            algorithm = make_algo(
                spec=[env.spec for env in evaluation], count=4, eval_env=evaluation,
                task_names=list(SEQUENCES['D-C4']['tasks']),
                no_stats=False, exact_task_budget=True, steps_per_epoch=1,
                min_buffer_size=1000, singular_clip_interval=1000,
                num_evaluation_episodes=1, max_episode_length_eval=10,
                scalar_log_interval=1000, feature_stats_interval=1000,
                hessian_stats_interval=1000, bellman_probe=True,
                bellman_probe_size=8, bellman_probe_interval=1000,
                bellman_spectral_stats=True, bellman_spectral_anchor_size=4,
                bellman_spectral_task_steps=(1000, 2000),
                bellman_probe_dir=directory, log_name='dmc_clip_smoke')
            algorithm._sampler = LocalSampler(
                agents=algorithm.policy, envs=training,
                max_episode_length=1000, n_workers=1)
            algorithm.save_results = Mock()
            trainer = Trainer(None)
            trainer.setup(algorithm, evaluation[0])
            trainer.enable_logging = False
            trainer.save = Mock()
            result = trainer.train(n_epochs=8, batch_size=1000)
            self.assertTrue(np.isfinite(result))
            self.assertEqual(algorithm.global_env_step, 8000)
            self.assertEqual(algorithm.seq_idx, 3)
            # The final 2k point has no incoming task to deduplicate against,
            # so this deliberately shortened test also clips once at the end.
            self.assertEqual([e['task'] for e in algorithm._singular_clip_events],
                             [1, 1, 2, 2, 3, 3, 3])
            self.assertEqual(len(algorithm.results['Policy hessian rank']), 8)
            self.assertEqual(len(algorithm._bellman_probe_buffers), 4)
            metrics = Path(algorithm._bellman_probe_metrics_path).read_text().splitlines()
            self.assertGreaterEqual(len(metrics), 8)
            self.assertTrue(all(isinstance(json.loads(line), dict) for line in metrics))
            for env in evaluation:
                env.close()


if __name__ == '__main__':
    unittest.main()
