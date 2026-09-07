#!/usr/bin/env python3
"""Unit checks for the SAC spectral-regularization baseline."""

import unittest

import akro
import numpy as np
import torch
from torch import nn

from garage import EnvSpec
from garage.replay_buffer import PathBuffer
from garage.torch import as_torch_dict
from garage.torch.algos import SpectralRegularizedSAC
from garage.torch.algos.spectral_regularization import (
    LayerSpectralRegularizer)
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


class SpectralRegularizerTest(unittest.TestCase):

    def test_k2_objective_matches_exact_svd(self):
        network = nn.Sequential(nn.Linear(2, 2))
        layer = network[0]
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[3., 0.], [0., 1.]]))
            layer.bias.copy_(torch.tensor([1., 2.]))

        regularizer = LayerSpectralRegularizer(power_iterations=50)
        loss, diagnostics = regularizer.loss(network, 'test')

        expected = (3. ** 2 - 1.) ** 2 + (1. ** 2 + 2. ** 2) ** 2
        self.assertAlmostEqual(loss.item(), expected, places=4)
        self.assertEqual(diagnostics['regularized_layers'], 1)

    def test_regularizer_is_differentiable_for_weight_and_bias(self):
        network = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
        regularizer = LayerSpectralRegularizer(power_iterations=2)
        loss, _ = regularizer.loss(network, 'gradient')
        loss.backward()

        for module in network.modules():
            if isinstance(module, nn.Linear):
                self.assertIsNotNone(module.weight.grad)
                self.assertTrue(torch.isfinite(module.weight.grad).all())
                self.assertIsNotNone(module.bias.grad)
                self.assertTrue(torch.isfinite(module.bias.grad).all())

    def test_state_round_trip_preserves_power_vectors(self):
        network = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        source = LayerSpectralRegularizer(power_iterations=1)
        source.loss(network, 'roundtrip')

        restored = LayerSpectralRegularizer(power_iterations=1)
        restored.load_state_dict(source.state_dict())
        self.assertEqual(
            list(source._left_vectors), list(restored._left_vectors))
        for key in source._left_vectors:
            self.assertTrue(torch.equal(
                source._left_vectors[key], restored._left_vectors[key]))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            LayerSpectralRegularizer(power_iterations=0)

    def test_power_initialization_does_not_consume_torch_rng(self):
        network = nn.Sequential(nn.Linear(3, 4))
        before = torch.get_rng_state().clone()
        LayerSpectralRegularizer().loss(network, 'rng')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_full_sac_update_regularizes_actor_and_twin_critics(self):
        torch.manual_seed(7)
        spec = EnvSpec(
            akro.Box(-np.inf, np.inf, shape=(3,)),
            akro.Box(-1., 1., shape=(2,)),
            max_episode_length=10)
        policy = TanhGaussianMLPPolicy(
            spec, n_tasks=2, hidden_sizes=(8, 8), no_stats=True)
        qf1 = ContinuousMLPQFunction(
            spec, hidden_sizes=(8, 8), no_stats=True)
        qf2 = ContinuousMLPQFunction(
            spec, hidden_sizes=(8, 8), no_stats=True)
        algorithm = SpectralRegularizedSAC(
            policy=policy, qf1=qf1, qf2=qf2, env_spec=spec,
            sampler=None,
            replay_buffer=PathBuffer(capacity_in_transitions=100),
            num_tasks=1, eval_env=[], gradient_steps_per_itr=1,
            seed=7, no_stats=True, use_wandb=False,
            bellman_probe=False, bellman_spectral_stats=False,
            q_reset=False, policy_reset=False, task_names=None,
            buffer_batch_size=8, actor_coef=1e-4, critic_coef=1e-4,
            power_iterations=1)
        samples = as_torch_dict({
            'observation': np.random.randn(8, 3).astype('float32'),
            'next_observation': np.random.randn(8, 3).astype('float32'),
            'action': np.random.uniform(-1, 1, (8, 2)).astype('float32'),
            'reward': np.random.randn(8, 1).astype('float32'),
            'terminal': np.zeros((8, 1), dtype='float32'),
        })

        actor_outputs = (
            algorithm.policy._module._shared_mean_log_std_network
            ._output_layers)
        inactive_before = [
            parameter.detach().clone()
            for layer in actor_outputs[2:4]
            for parameter in layer.parameters()
        ]
        losses = algorithm.optimize_policy(samples, seq_idx=0)

        self.assertEqual(len(losses), 3)
        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        stats = algorithm._spectral_regularization_last_stats
        self.assertGreater(stats['actor']['regularized_layers'], 0)
        self.assertGreater(stats['qf1']['regularized_layers'], 0)
        self.assertGreater(stats['qf2']['regularized_layers'], 0)
        inactive_after = [
            parameter.detach()
            for layer in actor_outputs[2:4]
            for parameter in layer.parameters()
        ]
        self.assertTrue(all(torch.equal(before, after) for before, after in
                            zip(inactive_before, inactive_after)))
        checkpoint = algorithm.spectral_regularization_checkpoint_state()
        self.assertEqual(checkpoint['actor_coefficient'], 1e-4)
        self.assertGreater(len(checkpoint['regularizer']['left_vectors']), 0)

    def test_heterogeneous_dmc_style_specs_support_a_sac_update(self):
        specs = [
            EnvSpec(akro.Box(-np.inf, np.inf, shape=(3,)),
                    akro.Box(-1., 1., shape=(2,)), max_episode_length=10),
            EnvSpec(akro.Box(-np.inf, np.inf, shape=(5,)),
                    akro.Box(-1., 1., shape=(1,)), max_episode_length=10),
        ]
        policy = TanhGaussianMLPPolicy(
            specs, n_tasks=2, hidden_sizes=(8, 8), no_stats=True)
        qf1 = ContinuousMLPQFunction(
            specs, hidden_sizes=(8, 8), no_stats=True)
        qf2 = ContinuousMLPQFunction(
            specs, hidden_sizes=(8, 8), no_stats=True)
        algorithm = SpectralRegularizedSAC(
            policy=policy, qf1=qf1, qf2=qf2, env_spec=specs,
            sampler=None,
            replay_buffer=PathBuffer(capacity_in_transitions=100),
            num_tasks=1, eval_env=[], gradient_steps_per_itr=1,
            seed=11, no_stats=True, use_wandb=False,
            bellman_probe=False, bellman_spectral_stats=False,
            q_reset=False, policy_reset=False, task_names=None,
            buffer_batch_size=4, actor_coef=1e-4, critic_coef=1e-4,
            power_iterations=1, fixed_alpha=0.01, multi_input=True)
        samples = as_torch_dict({
            'observation': np.random.randn(4, 5).astype('float32'),
            'next_observation': np.random.randn(4, 5).astype('float32'),
            'action': np.random.uniform(-1, 1, (4, 1)).astype('float32'),
            'reward': np.random.randn(4, 1).astype('float32'),
            'terminal': np.zeros((4, 1), dtype='float32'),
        })

        losses = algorithm.optimize_policy(samples, seq_idx=1)

        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        actor_keys = [
            key for key in algorithm._spectral_regularizer._left_vectors
            if key.startswith('actor.')]
        self.assertTrue(any('_output_layers.2.' in key for key in actor_keys))
        self.assertTrue(any('_output_layers.3.' in key for key in actor_keys))
        self.assertFalse(any('_output_layers.0.' in key for key in actor_keys))
        self.assertFalse(any('_output_layers.1.' in key for key in actor_keys))


if __name__ == '__main__':
    unittest.main()
