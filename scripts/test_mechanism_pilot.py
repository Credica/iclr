import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from garage.torch.algos.branch_budget import remaining_branch_steps
from garage.torch.algos.mechanism_recorder import MechanismRecorder
from scripts.test_sac_singular_clip import make_algo, samples


class PilotChecks(unittest.TestCase):
    def setUp(self):
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def test_budget_includes_b_and_c(self):
        self.assertEqual(remaining_branch_steps({'seq_idx': 0}, 3, 1000000), 2000000)
        self.assertEqual(remaining_branch_steps({'seq_idx': 1}, 3, 1000000), 1000000)
        with self.assertRaises(ValueError):
            remaining_branch_steps({'seq_idx': 2}, 3, 1000000)

    def test_branch_qreset_clears_adam_clip_preserves_it(self):
        a = make_algo()
        a.optimize_policy(samples(), 0)
        checkpoint = dict(global_step=1000000, seq_idx=0)
        for key, model in (('policy', a.policy), ('qf1', a._qf1), ('qf2', a._qf2),
                           ('target_qf1', a._target_qf1), ('target_qf2', a._target_qf2)):
            checkpoint[key] = model.state_dict()
        for key, optimizer in (('policy_optimizer', a._policy_optimizer),
                               ('qf1_optimizer', a._qf1_optimizer), ('qf2_optimizer', a._qf2_optimizer)):
            checkpoint[key] = optimizer.state_dict()
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'a.pt')
            torch.save(checkpoint, path)
            for reset in (False, True):
                b = make_algo(branch_checkpoint=path, q_reset=reset,
                              sampler=SimpleNamespace(_envs=[SimpleNamespace()]),
                              singular_clip_max=8.)
                self.assertEqual(b.seq_idx, 1)
                self.assertEqual(b._sampler._envs[0].cur_seq_idx, 1)
                self.assertEqual(bool(b._qf1_optimizer.state), not reset)
                self.assertEqual(bool(b._qf2_optimizer.state), not reset)
                self.assertTrue(b._policy_optimizer.state)
                if not reset:
                    b._clip_at_task_start(1)
                    self.assertEqual(b._singular_clip_events[0]['task_env_step'], 0)
                    for x, y in zip(b._qf1.parameters(), b._target_qf1.parameters()):
                        self.assertTrue(torch.equal(x, y))

    def test_recording_does_not_change_updates_or_rng(self):
        torch.manual_seed(21)
        a = make_algo(min_buffer_size=8)
        torch.manual_seed(21)
        b = make_algo(min_buffer_size=8)
        data = {k: v.cpu().numpy() for k, v in samples().items()}
        for algorithm in (a, b):
            algorithm.seq_idx = 1
            algorithm.global_env_step = 10000
            algorithm._bellman_probe_buffers[1] = copy.deepcopy(data)
            algorithm.replay_buffer.add_path(copy.deepcopy(data))
        with tempfile.TemporaryDirectory() as folder:
            options = SimpleNamespace(exact_sac_task_budget=True, bellman_probe=True,
                                      branch_checkpoint=None, seed=1)
            recorder = MechanismRecorder(b, Path(folder) / 'record', options, window_updates=2)
            np.random.seed(34)
            torch.manual_seed(34)
            for _ in range(2):
                a.train_once(1)
                a.global_step += 1
            expected_rng = torch.get_rng_state()
            expected_np = np.random.get_state()
            np.random.seed(34)
            torch.manual_seed(34)
            for _ in range(2):
                b.train_once(1)
                b.global_step += 1
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            self.assertTrue(np.array_equal(expected_np[1], np.random.get_state()[1]))
            for x, y in zip(a.networks, b.networks):
                for p, q in zip(x.parameters(), y.parameters()):
                    self.assertTrue(torch.equal(p, q))
            self.assertTrue((recorder.root / 'task1_env10000_window' / 'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
