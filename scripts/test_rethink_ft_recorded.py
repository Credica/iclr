"""Protocol checks; run directly with the reset-distill Python environment."""
import copy
import json
from pathlib import Path
import pickle
import sys
import unittest

import akro
import numpy as np
import torch

from garage import EnvSpec
from garage.torch import as_torch_dict
from garage.torch.algos.finetuning import Finetuning_SAC
from rethink_ft_recorded import (PAIRS, build_algorithm,
                                 compute_plasticity_metrics, isolated_rng,
                                 parse_args, rng_state, restore_rng, stable_seed,
                                 state_hash, wandb_payload)


def equal_rng(first, second):
    assert first['python'] == second['python']
    assert first['numpy'][0] == second['numpy'][0]
    assert np.array_equal(first['numpy'][1], second['numpy'][1])
    assert first['numpy'][2:] == second['numpy'][2:]
    assert torch.equal(first['torch'], second['torch'])
    assert len(first['cuda']) == len(second['cuda'])
    assert all(torch.equal(a, b) for a, b in zip(first['cuda'], second['cuda']))


class ProtocolChecks(unittest.TestCase):
    @staticmethod
    def _samples(seed=17):
        generator = np.random.RandomState(seed)
        return as_torch_dict(dict(
            observation=generator.normal(size=(64, 39)).astype('float32'),
            next_observation=generator.normal(size=(64, 39)).astype('float32'),
            action=generator.uniform(-1, 1, size=(64, 4)).astype('float32'),
            reward=generator.uniform(0, 10, size=(64, 1)).astype('float32'),
            terminal=np.zeros((64, 1), dtype='float32')))

    def test_wandb_payload_keeps_all_numeric_metrics(self):
        row = {
            'global_env_step': 12000,
            'task_env_step': 12000,
            'train_task': 'reach-v2',
            'actor_loss': -3.5,
            'gradient_norms_last_update': [1.25, 2.5],
            'last_minibatch': {
                'q1': {'mean': 4.0, 'quantiles': [1., 2., 3., 4., 5.]},
            },
            'policy_feature_rank': 17,
            'training_rng_preserved': True,
            'missing': None,
        }
        assert wandb_payload('train', row) == {
            'clock/global_env_step': 12000,
            'clock/task_env_step': 12000,
            'train/actor_loss': -3.5,
            'train/gradient_norms_last_update/0': 1.25,
            'train/gradient_norms_last_update/1': 2.5,
            'train/last_minibatch/q1/mean': 4.0,
            'train/last_minibatch/q1/quantiles/0': 1.0,
            'train/last_minibatch/q1/quantiles/1': 2.0,
            'train/last_minibatch/q1/quantiles/2': 3.0,
            'train/last_minibatch/q1/quantiles/3': 4.0,
            'train/last_minibatch/q1/quantiles/4': 5.0,
            'train/policy_feature_rank': 17,
            'train/training_rng_preserved': True,
        }

    def test_plasticity_metrics_are_complete_and_side_effect_free(self):
        torch.set_num_threads(1)
        spec = EnvSpec(akro.Box(-np.inf, np.inf, shape=(5,)),
                       akro.Box(-1., 1., shape=(2,)), max_episode_length=10)
        algo = build_algorithm(spec, 9, 'cpu', hidden_sizes=(4, 4))
        generator = np.random.RandomState(23)
        samples = as_torch_dict(dict(
            observation=generator.normal(size=(8, 5)).astype('float32'),
            next_observation=generator.normal(size=(8, 5)).astype('float32'),
            action=generator.uniform(-1, 1, size=(8, 2)).astype('float32'),
            reward=generator.uniform(0, 1, size=(8, 1)).astype('float32'),
            terminal=np.zeros((8, 1), dtype='float32')))
        for _ in range(3):
            algo.capture = algo.capture_update = False
            algo.optimize_policy(samples, 0)
            algo._update_targets()
        previous = {name: copy.deepcopy(model.state_dict())
                    for name, model in zip(('policy', 'qf1', 'qf2'),
                                           (algo.policy, algo._qf1, algo._qf2))}
        before_rng = rng_state()
        before_hashes = [state_hash(model) for model in
                         (algo.policy, algo._qf1, algo._qf2)]
        metrics, current = compute_plasticity_metrics(
            algo, samples, 0, previous, normalization_count=8)
        equal_rng(before_rng, rng_state())
        assert before_hashes == [state_hash(model) for model in
                                 (algo.policy, algo._qf1, algo._qf2)]
        expected = {
            '{}_{}'.format(network, metric)
            for network in ('policy', 'qf1', 'qf2')
            for metric in ('zero_ratio', 'feature_rank', 'hessian_rank',
                           'weight_change')
        }
        assert expected <= set(metrics)
        assert all(np.isfinite(metrics[key]) for key in expected)
        assert all(metrics[name + '_weight_change'] == 0.
                   for name in ('policy', 'qf1', 'qf2'))
        assert set(current) == {'policy', 'qf1', 'qf2'}

    def test_rng_isolation_and_update_equivalence(self):
        torch.set_num_threads(1)
        spec = EnvSpec(akro.Box(-np.inf, np.inf, shape=(39,)),
                       akro.Box(-1., 1., shape=(4,)), max_episode_length=500)
        instrumented = build_algorithm(spec, 1, 'cpu')
        original = build_algorithm(spec, 1, 'cpu')
        original.__class__ = Finetuning_SAC
        samples = self._samples()
        for _ in range(3):
            before = rng_state()
            with isolated_rng(888):
                np.random.normal(size=100)
                torch.randn(200)
                _ = instrumented.policy(samples['observation'], 0)[0].mean
            equal_rng(before, rng_state())
            instrumented.capture = instrumented.capture_update = True
            losses = instrumented.optimize_policy(samples, 0)
            instrumented._update_targets()
            after = rng_state()
            restore_rng(before)
            reference_losses = original.optimize_policy(samples, 0)
            original._update_targets()
            equal_rng(after, rng_state())
            for first, second in zip(losses, reference_losses):
                assert torch.equal(first, second)
            for first, second in zip(instrumented.networks, original.networks):
                assert state_hash(first) == state_hash(second)
        old_q = [state_hash(instrumented._qf1), state_hash(instrumented._qf2)]
        old_actor = state_hash(instrumented.policy)
        old_adam = copy.deepcopy(instrumented._qf1_optimizer.state_dict())
        instrumented.task_change(0)
        assert old_q == [state_hash(instrumented._qf1), state_hash(instrumented._qf2)]
        assert old_actor == state_hash(instrumented.policy)
        assert state_hash(instrumented._qf1) == state_hash(instrumented._target_qf1)
        assert state_hash(instrumented._qf2) == state_hash(instrumented._target_qf2)
        assert float(instrumented._log_alpha.exp()) == 1.
        assert not instrumented._alpha_optimizer.state_dict()['state']
        for key, state in old_adam['state'].items():
            for field, value in state.items():
                assert torch.equal(
                    value,
                    instrumented._qf1_optimizer.state_dict()['state'][key][field])

    def test_formal_q_reset_clears_critic_adam(self):
        torch.set_num_threads(1)
        spec = EnvSpec(akro.Box(-np.inf, np.inf, shape=(39,)),
                       akro.Box(-1., 1., shape=(4,)), max_episode_length=500)
        algorithm = build_algorithm(spec, 3, 'cpu')
        algorithm._q_reset = True
        initial_q1 = copy.deepcopy(algorithm._random_qf1_state_dict)
        initial_q2 = copy.deepcopy(algorithm._random_qf2_state_dict)
        algorithm.optimize_policy(self._samples(23), 0)
        assert algorithm._qf1_optimizer.state_dict()['state']
        assert algorithm._qf2_optimizer.state_dict()['state']
        algorithm.task_change(0)
        assert not algorithm._qf1_optimizer.state_dict()['state']
        assert not algorithm._qf2_optimizer.state_dict()['state']
        for key, value in initial_q1.items():
            assert torch.equal(value, algorithm._qf1.state_dict()[key])
            assert torch.equal(value, algorithm._target_qf1.state_dict()[key])
        for key, value in initial_q2.items():
            assert torch.equal(value, algorithm._qf2.state_dict()[key])
            assert torch.equal(value, algorithm._target_qf2.state_dict()[key])

    def test_pairs(self):
        assert len(PAIRS) == 6
        for forward, reverse in [('P1', 'P2'), ('P3', 'P4'), ('P5', 'P6')]:
            assert PAIRS[forward] == PAIRS[reverse][::-1]
        assert stable_seed(1, 'reach-v2', 'train') != stable_seed(1, 'reach-v2', 'eval')

    def test_wandb_is_enabled_by_default_and_can_be_disabled(self):
        required = ['--pair', 'P1', '--seed', '1', '--output', 'unused', '--smoke']
        defaults = parse_args(required)
        assert defaults.wandb is True
        assert defaults.bellman_probe is True
        assert defaults.bellman_spectral_stats is True
        assert parse_args(required + ['--wandb', 'false']).wandb is False
        with self.assertRaises(AssertionError):
            parse_args(required + ['--bellman-probe', 'false'])


def audit_smoke(root):
    root = Path(root)
    status = json.loads((root / 'status.json').read_text())
    assert status['status'] == 'completed', status
    assert status['global_env_step'] == 3000
    assert status['global_critic_updates'] == 1000
    for task in (0, 1):
        warmup = np.load(str(root / 'anchors' / ('task_%d_warmup.npz' % task)))
        panels = np.load(str(root / 'anchors' / ('task_%d_panels.npz' % task)))
        assert len(warmup['observation']) == 1000
        assert not set(warmup['episode'][panels['panel_0']]) & set(warmup['episode'][panels['panel_1']])
        checkpoint = torch.load(str(root / 'checkpoints' / ('task_%d_env_0001000.pt' % task)), map_location='cpu')
        assert checkpoint['clocks']['task_critic_updates'] == 0
        boundary = torch.load(str(root / 'checkpoints' / ('task_%d_env_0001500_full.pt' % task)), map_location='cpu')
        assert boundary['full_boundary_state']
        assert boundary['replay'].n_transitions_stored == 1500
        # Native MetaWorld serialization includes MuJoCo model/state, cached
        # task variables, and environment RNG. Verify a step after round-trip.
        env = pickle.loads(boundary['environment'])
        env_copy = pickle.loads(pickle.dumps(env, protocol=4))
        state = rng_state()
        first = env.step(np.zeros(4))
        restore_rng(state)
        second = env_copy.step(np.zeros(4))
        np.testing.assert_allclose(first[0], second[0], rtol=0, atol=1e-10)
        assert abs(first[1]-second[1]) < 1e-10
        env.close()
        env_copy.close()
    window = root / 'target_windows' / 'B_env_0001000'
    assert json.loads((window / 'complete.json').read_text())['updates'] == 100
    vectors = np.load(str(window / 'critic_parameters.npy'), mmap_mode='r')
    assert vectors.shape[0] == 101 and np.isfinite(vectors).all()
    rows = torch.load(str(window / 'rows_through_0100.pt'), map_location='cpu')
    references = [row for row in rows if row['kind'] == 'reference']
    online = [row for row in rows if row['kind'] == 'online_minibatch']
    assert len(references) == 101 and len(online) == 100
    index = np.concatenate([np.load(str(root / 'anchors' / 'task_1_panels.npz'))[key]
                            for key in ('panel_0', 'panel_1')])
    rewards = np.load(str(root / 'anchors' / 'task_1_warmup.npz'))['reward'][index].reshape(-1)
    for row in references:
        expected = rewards + .99 * (np.minimum(row['target_q1'], row['target_q2']) - row['alpha'] * row['log_pi'])
        np.testing.assert_allclose(expected, row['target'], rtol=1e-6, atol=1e-6)
    episodes = [json.loads(line) for line in (root / 'eval_episodes.jsonl').read_text().splitlines()]
    assert all(row['eval_task_position'] <= row['train_task_position'] for row in episodes)
    assert all(row['length'] == 500 for row in episodes)
    assert all('success_any' in row and 'reset_seed' in row for row in episodes)
    training = [json.loads(line) for line in
                (root / 'train_metrics.jsonl').read_text().splitlines()]
    plasticity = [row for row in training if 'policy_feature_rank' in row]
    assert plasticity
    required_plasticity = {
        '{}_{}'.format(network, metric)
        for network in ('policy', 'qf1', 'qf2')
        for metric in ('zero_ratio', 'feature_rank', 'hessian_rank',
                       'weight_change')
    }
    assert all(required_plasticity <= set(row) for row in plasticity)
    assert all(row['plasticity_rng_preserved'] for row in plasticity)
    a_end = torch.load(str(root / 'checkpoints' / 'task_0_env_0001500_full.pt'), map_location='cpu')
    b_start = torch.load(str(root / 'checkpoints' / 'task_1_env_0000000_full.pt'), map_location='cpu')
    for name in ('policy', 'qf1', 'qf2'):
        assert all(torch.equal(value, b_start['models'][name][key])
                   for key, value in a_end['models'][name].items())
    assert b_start['replay'].n_transitions_stored == 0
    print('SMOKE_ARTIFACT_AUDIT_PASSED', root)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--audit-smoke':
        audit_smoke(sys.argv[2])
    else:
        unittest.main()
