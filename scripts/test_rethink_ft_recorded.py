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
from rethink_ft_recorded import (PAIRS, build_algorithm, isolated_rng, rng_state,
                                 restore_rng, stable_seed, state_hash)


def equal_rng(first, second):
    assert first['python'] == second['python']
    assert first['numpy'][0] == second['numpy'][0]
    assert np.array_equal(first['numpy'][1], second['numpy'][1])
    assert first['numpy'][2:] == second['numpy'][2:]
    assert torch.equal(first['torch'], second['torch'])
    assert len(first['cuda']) == len(second['cuda'])
    assert all(torch.equal(a, b) for a, b in zip(first['cuda'], second['cuda']))


class ProtocolChecks(unittest.TestCase):
    def test_rng_isolation_and_update_equivalence(self):
        torch.set_num_threads(1)
        spec = EnvSpec(akro.Box(-np.inf, np.inf, shape=(39,)),
                       akro.Box(-1., 1., shape=(4,)), max_episode_length=500)
        instrumented = build_algorithm(spec, 1, 'cpu')
        original = build_algorithm(spec, 1, 'cpu')
        original.__class__ = Finetuning_SAC
        generator = np.random.RandomState(17)
        samples = as_torch_dict(dict(
            observation=generator.normal(size=(64, 39)).astype('float32'),
            next_observation=generator.normal(size=(64, 39)).astype('float32'),
            action=generator.uniform(-1, 1, size=(64, 4)).astype('float32'),
            reward=generator.uniform(0, 10, size=(64, 1)).astype('float32'),
            terminal=np.zeros((64, 1), dtype='float32')))
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
                assert torch.equal(value, instrumented._qf1_optimizer.state_dict()['state'][key][field])

    def test_pairs(self):
        assert len(PAIRS) == 6
        for forward, reverse in [('P1', 'P2'), ('P3', 'P4'), ('P5', 'P6')]:
            assert PAIRS[forward] == PAIRS[reverse][::-1]
        assert stable_seed(1, 'reach-v2', 'train') != stable_seed(1, 'reach-v2', 'eval')


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
