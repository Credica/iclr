"""Checks for single-task teacher loading into heterogeneous R&D policies."""
import unittest

import akro
import numpy as np
import torch

from garage import EnvSpec
from garage.torch.algos.rnd import RND_SAC
from garage.torch.policies import TanhGaussianMLPPolicy


class RNDTeacherMappingChecks(unittest.TestCase):

    def test_dmc_teacher_maps_input_slice_and_destination_head(self):
        specs = [
            EnvSpec(akro.Box(-np.inf, np.inf, shape=(3,)),
                    akro.Box(-1., 1., shape=(2,)), max_episode_length=10),
            EnvSpec(akro.Box(-np.inf, np.inf, shape=(5,)),
                    akro.Box(-1., 1., shape=(1,)), max_episode_length=10),
        ]
        teacher = TanhGaussianMLPPolicy(
            [specs[1]], n_tasks=1, hidden_sizes=(4, 4), no_stats=True)
        student = TanhGaussianMLPPolicy(
            specs, n_tasks=2, hidden_sizes=(4, 4), no_stats=True)
        with torch.no_grad():
            for parameter in teacher.parameters():
                parameter.fill_(0.375)

        algorithm = RND_SAC.__new__(RND_SAC)
        algorithm._copy_teacher_policy_state(
            student, teacher.state_dict(), seq_idx=1)

        state = student.state_dict()
        first = ('_module._shared_mean_log_std_network.'
                 '_layers.0.linear.weight')
        self.assertTrue(torch.all(state[first][:, 3:] == 0.375))
        self.assertTrue(torch.all(state[
            '_module._shared_mean_log_std_network.'
            '_output_layers.2.linear.weight'] == 0.375))
        self.assertTrue(torch.all(state[
            '_module._shared_mean_log_std_network.'
            '_output_layers.3.linear.weight'] == 0.375))

    def test_teacher_paths_match_single_task_log_names(self):
        algorithm = RND_SAC.__new__(RND_SAC)
        algorithm._teacher_root = __import__('pathlib').Path('/artifacts')
        algorithm._teacher_steps = 1500000
        algorithm._seed = 2
        self.assertEqual(
            '/artifacts/models/sac_models/'
            'policy_dm_control_sac_DMControl-walker-run_1500000_2.pt',
            str(algorithm._teacher_model_path('DMControl-walker-run')))


if __name__ == '__main__':
    unittest.main()
