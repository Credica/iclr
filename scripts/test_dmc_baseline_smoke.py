"""Short heterogeneous-DMC interface checks for baseline algorithms."""
import unittest

import akro
import numpy as np
import torch

from garage import EnvSpec
from garage.replay_buffer import PathBuffer
from garage.torch import as_torch_dict
from garage.torch.algos import P_and_C_SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


def dmc_style_specs():
    return [
        EnvSpec(akro.Box(-np.inf, np.inf, shape=(3,)),
                akro.Box(-1., 1., shape=(2,)), max_episode_length=10),
        EnvSpec(akro.Box(-np.inf, np.inf, shape=(5,)),
                akro.Box(-1., 1., shape=(1,)), max_episode_length=10),
    ]


class DMCBaselineSmokeChecks(unittest.TestCase):

    def test_pandc_runs_update_on_second_heterogeneous_task(self):
        specs = dmc_style_specs()
        policy = TanhGaussianMLPPolicy(
            specs, n_tasks=2, hidden_sizes=(8, 8), adaptor=True,
            no_stats=True)
        qf1 = ContinuousMLPQFunction(
            specs, hidden_sizes=(8, 8), no_stats=True)
        qf2 = ContinuousMLPQFunction(
            specs, hidden_sizes=(8, 8), no_stats=True)
        algorithm = P_and_C_SAC(
            policy=policy, qf1=qf1, qf2=qf2, env_spec=specs,
            sampler=None,
            replay_buffer=PathBuffer(capacity_in_transitions=100),
            num_tasks=1, eval_env=[], gradient_steps_per_itr=1,
            seed=19, no_stats=True, use_wandb=False,
            bellman_probe=False, bellman_spectral_stats=False,
            q_reset=False, policy_reset=False, task_names=None,
            buffer_batch_size=4, fixed_alpha=0.01, multi_input=True,
            cl_reg_coef=1.0, compress_step=4, bc=False,
            reset_column=True, reset_adaptor=True)
        samples = as_torch_dict({
            'observation': np.random.randn(4, 5).astype('float32'),
            'next_observation': np.random.randn(4, 5).astype('float32'),
            'action': np.random.uniform(-1, 1, (4, 1)).astype('float32'),
            'reward': np.random.randn(4, 1).astype('float32'),
            'terminal': np.zeros((4, 1), dtype='float32'),
        })

        losses = algorithm.optimize_policy(samples, seq_idx=1)

        self.assertTrue(all(torch.isfinite(loss) for loss in losses))


if __name__ == '__main__':
    unittest.main()
