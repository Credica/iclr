"""Read-only Bellman spectral and target-fitting measurements for critics."""

import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_TASK_STEPS = (
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    1_500_000,
)


def empirical_jacobian(critic, observations, actions, task_idx):
    """Return the per-example full-parameter Jacobian of scalar Q values."""
    parameters = [parameter for parameter in critic.parameters()
                  if parameter.requires_grad]
    rows = []
    for row_idx in range(observations.shape[0]):
        value = critic(
            observations[row_idx:row_idx + 1],
            actions[row_idx:row_idx + 1],
            seq_idx=task_idx).reshape(())
        gradients = torch.autograd.grad(value, parameters)
        rows.append(torch.cat([gradient.reshape(-1)
                               for gradient in gradients]))
    return torch.stack(rows, dim=0)


def spectral_direction_stats(jacobian, directions, relative_ridge,
                             slow_fraction=0.30):
    """Compute empirical-NTK burden and energy in its slowest directions."""
    parameter_count = jacobian.shape[1]
    kernel = torch.matmul(jacobian, jacobian.t()) / parameter_count
    kernel = 0.5 * (kernel + kernel.t())
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    mean_eigenvalue = torch.trace(kernel) / kernel.shape[0]
    ridge = relative_ridge * mean_eigenvalue
    regularized_kernel = kernel + ridge * torch.eye(
        kernel.shape[0], dtype=kernel.dtype, device=kernel.device)

    inverse_directions = torch.linalg.solve(regularized_kernel, directions)
    burden = (mean_eigenvalue *
              (directions * inverse_directions).sum() /
              directions.square().sum())

    coefficients = torch.matmul(eigenvectors.t(), directions)
    energy_per_direction = coefficients.square().sum(dim=1)
    slow_count = int(math.ceil(slow_fraction * kernel.shape[0]))
    slow_energy = (energy_per_direction[:slow_count].sum() /
                   energy_per_direction.sum())
    return {
        'burden': burden,
        'slow_energy': slow_energy,
        'eigenvalues': eigenvalues,
        'energy_per_eigendirection': energy_per_direction,
        'mean_eigenvalue': mean_eigenvalue,
        'ridge': ridge,
        'slow_count': slow_count,
    }


def fit_fixed_target(critic, observations, actions, task_idx, target,
                     learning_rate, steps=200):
    """Fit a frozen scalar target using a cloned critic and fresh Adam."""
    fitted_critic = copy.deepcopy(critic)
    optimizer = torch.optim.Adam(
        fitted_critic.parameters(), lr=learning_rate)
    curve = []
    for fit_step in range(steps + 1):
        predictions = fitted_critic(
            observations, actions, seq_idx=task_idx).flatten()
        loss = F.mse_loss(predictions, target)
        if fit_step == 0:
            initial_loss = loss.detach()
        curve.append((loss.detach() / initial_loss).cpu())
        if fit_step < steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return torch.stack(curve)


def deterministic_soft_bellman_directions(
        anchor, task_idx, qf1, qf2, target_qf1, target_qf2, policy, alpha,
        discount, reward_scale, target_count):
    """Build the deterministic soft-Bellman direction bank for both critics."""
    observations = anchor['observation']
    actions = anchor['action']
    rewards = anchor['reward'].flatten()
    next_observations = anchor['next_observation']
    terminals = anchor['terminal'].flatten()

    with torch.no_grad():
        q1 = qf1(observations, actions, seq_idx=task_idx).flatten()
        q2 = qf2(observations, actions, seq_idx=task_idx).flatten()
        next_action_dist = policy(next_observations, task_idx)[0]
        base_dist = next_action_dist._normal.base_dist
        action_axis = torch.arange(
            1, base_dist.loc.shape[-1] + 1,
            device=base_dist.loc.device,
            dtype=base_dist.loc.dtype).unsqueeze(0)
        q1_directions = []
        q2_directions = []
        for target_idx in range(target_count):
            noise = torch.sin((target_idx + 1) * action_axis)
            pre_tanh = base_dist.loc + base_dist.scale * noise
            next_actions = torch.tanh(pre_tanh)
            next_log_pi = next_action_dist.log_prob(
                value=next_actions, pre_tanh_value=pre_tanh)
            next_q1 = target_qf1(
                next_observations, next_actions,
                seq_idx=task_idx).flatten()
            next_q2 = target_qf2(
                next_observations, next_actions,
                seq_idx=task_idx).flatten()
            target = rewards * reward_scale + (
                1. - terminals) * discount * (
                    torch.min(next_q1, next_q2) - alpha * next_log_pi)
            q1_directions.append(target - q1)
            q2_directions.append(target - q2)
    return (torch.stack(q1_directions, dim=1),
            torch.stack(q2_directions, dim=1))


class BellmanSpectralStats:
    """Compute full-Jacobian Bellman measurements without changing SAC."""

    def __init__(self, run_dir, anchor_size=64, target_count=8,
                 relative_ridge=1e-3, fit_learning_rate=3e-4,
                 reference_dir=None, task_steps=DEFAULT_TASK_STEPS, seed=0):
        self._anchor_size = anchor_size
        self._target_count = target_count
        self._relative_ridge = relative_ridge
        self._fit_learning_rate = fit_learning_rate
        self._reference_dir = reference_dir
        self._task_steps = tuple(task_steps)
        self._seed = seed
        self._artifact_dir = os.path.join(run_dir, 'spectral_stats')
        os.makedirs(self._artifact_dir, exist_ok=True)

        config = {
            'kernel': 'full_parameter_empirical_ntk_JJT_over_p',
            'anchor_size': self._anchor_size,
            'target_count': self._target_count,
            'relative_ridge': self._relative_ridge,
            'slow_fraction': 0.30,
            'fit_optimizer': 'fresh_adam',
            'fit_learning_rate': self._fit_learning_rate,
            'fit_steps': [50, 200],
            'bellman_fit_direction': 'mean_online_soft_bellman_direction',
            'direction_source': (
                'recomputed_from_current_policy_critics_and_target_critics_'
                'at_every_measurement'),
            'generic_control': 'fixed_gaussian_equal_l2_norm',
            'reference_dir': self._reference_dir,
            'reference_transition_banks': ['initial', 'coverage'],
            'reference_contains_frozen_directions': False,
            'task_steps': list(self._task_steps),
        }
        with open(os.path.join(
                self._artifact_dir, 'spectral_stats_config.json'),
                'w') as config_file:
            json.dump(config, config_file, indent=2)

    def should_run(self, event, task_step):
        """Return whether the requested event belongs to the fixed schedule."""
        return event in ('task_boundary', 'final') or task_step in self._task_steps

    @staticmethod
    def _task_file_name(task_name):
        name = task_name
        if name.endswith('-v2'):
            name = name[:-3]
        return name.replace('-', '_') + '.pt'

    @staticmethod
    def _as_tensor(value, device):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).float().to(device)
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    def _load_anchors(self, task_name, task_idx, probe, device):
        if self._reference_dir is not None:
            reference_path = os.path.join(
                self._reference_dir, self._task_file_name(task_name))
            reference = torch.load(reference_path, map_location=device)
            if reference.get('format_version') != 2:
                raise ValueError(
                    'Bellman reference must be a version-2 transition-only '
                    'reference: {}'.format(reference_path))
            anchors = {}
            for bank_name in ('initial', 'coverage'):
                bank = reference['banks'][bank_name]
                anchors[bank_name] = {
                    'observation': self._as_tensor(
                        bank['observation'], device),
                    'action': self._as_tensor(bank['action'], device),
                    'reward': self._as_tensor(bank['reward'], device),
                    'next_observation': self._as_tensor(
                        bank['next_observation'], device),
                    'terminal': self._as_tensor(bank['terminal'], device),
                    'anchor_id': bank['anchor_id'],
                }
            return anchors

        rows = min(self._anchor_size, len(probe['observation']))
        anchor = {
            key: self._as_tensor(value[:rows], device)
            for key, value in probe.items()
        }
        anchor['anchor_id'] = 'run_seed{}_task{}_first{}'.format(
            self._seed, task_idx, rows)
        anchor['direction_bank_id'] = (
            'online_deterministic_soft_bellman_{}').format(
                self._target_count)
        return {'online': anchor}

    def _online_bellman_directions(
            self, anchor, task_idx, qf1, qf2, target_qf1, target_qf2,
            policy, alpha, discount, reward_scale):
        return deterministic_soft_bellman_directions(
            anchor, task_idx, qf1, qf2, target_qf1, target_qf2, policy,
            alpha, discount, reward_scale, self._target_count)

    def _generic_direction(self, bellman_direction, task_idx, bank_name):
        generator = torch.Generator(device='cpu')
        bank_seed = sum(ord(character) for character in bank_name)
        generator.manual_seed(
            self._seed * 1009 + task_idx * 9173 + bank_seed + 1729)
        random_direction = torch.randn(
            bellman_direction.shape, generator=generator)
        random_direction = random_direction.to(
            device=bellman_direction.device,
            dtype=bellman_direction.dtype)
        return (random_direction * bellman_direction.norm() /
                random_direction.norm())

    def _critic_measurement(self, name, critic, observations, actions,
                            task_idx, directions, bank_name):
        jacobian = empirical_jacobian(
            critic, observations, actions, task_idx)
        spectral = spectral_direction_stats(
            jacobian, directions, self._relative_ridge)

        with torch.no_grad():
            current_values = critic(
                observations, actions, seq_idx=task_idx).flatten()
        bellman_direction = directions.mean(dim=1)
        bellman_target = current_values + bellman_direction
        generic_direction = self._generic_direction(
            bellman_direction, task_idx, bank_name)
        generic_target = current_values + generic_direction
        bellman_curve = fit_fixed_target(
            critic, observations, actions, task_idx, bellman_target,
            self._fit_learning_rate)
        generic_curve = fit_fixed_target(
            critic, observations, actions, task_idx, generic_target,
            self._fit_learning_rate)

        scalars = {
            'bellman_shape_burden_{}'.format(name):
                spectral['burden'].item(),
            'bellman_slowest30_energy_{}'.format(name):
                spectral['slow_energy'].item(),
            'bellman_fit_residual_50_{}'.format(name):
                bellman_curve[50].item(),
            'bellman_fit_residual_200_{}'.format(name):
                bellman_curve[200].item(),
            'generic_fit_residual_50_{}'.format(name):
                generic_curve[50].item(),
            'generic_fit_residual_200_{}'.format(name):
                generic_curve[200].item(),
        }
        artifact = {
            'jacobian_eigenvalues_{}'.format(name):
                spectral['eigenvalues'].cpu(),
            'bellman_energy_per_eigendirection_{}'.format(name):
                spectral['energy_per_eigendirection'].cpu(),
            'bellman_fitting_curve_{}'.format(name): bellman_curve,
            'generic_fitting_curve_{}'.format(name): generic_curve,
            'mean_jacobian_eigenvalue_{}'.format(name):
                spectral['mean_eigenvalue'].cpu(),
            'kernel_ridge_{}'.format(name): spectral['ridge'].cpu(),
            'slow_direction_count_{}'.format(name): spectral['slow_count'],
        }
        return scalars, artifact

    def run(self, event, global_step, task_step, task_idx, task_name, probe,
            qf1, qf2, target_qf1, target_qf2, policy, alpha, discount,
            reward_scale):
        """Run both critic probes, save curves/spectra, and return scalars."""
        device = next(qf1.parameters()).device
        anchors = self._load_anchors(
            task_name, task_idx, probe, device)
        scalars = {}
        artifact = {
            'event': event,
            'global_step': global_step,
            'task_step': task_step,
            'current_task': task_idx,
            'current_task_name': task_name,
            'direction_definition': (
                'online_current_checkpoint_soft_bellman'),
            'banks': {},
        }

        for bank_name, anchor in anchors.items():
            observations = anchor['observation'][:self._anchor_size]
            actions = anchor['action'][:self._anchor_size]
            q1_directions, q2_directions = self._online_bellman_directions(
                anchor, task_idx, qf1, qf2, target_qf1, target_qf2,
                policy, alpha, discount, reward_scale)
            q1_scalars, q1_artifact = self._critic_measurement(
                'q1', qf1, observations, actions, task_idx, q1_directions,
                bank_name)
            q2_scalars, q2_artifact = self._critic_measurement(
                'q2', qf2, observations, actions, task_idx, q2_directions,
                bank_name)

            bank_scalars = {}
            bank_scalars.update(q1_scalars)
            bank_scalars.update(q2_scalars)
            for metric_name in (
                    'bellman_shape_burden',
                    'bellman_slowest30_energy',
                    'bellman_fit_residual_50',
                    'bellman_fit_residual_200',
                    'generic_fit_residual_50',
                    'generic_fit_residual_200'):
                bank_scalars[metric_name + '_mean'] = 0.5 * (
                    bank_scalars[metric_name + '_q1'] +
                    bank_scalars[metric_name + '_q2'])

            scalars['anchor_id_' + bank_name] = anchor['anchor_id']
            scalars['direction_bank_id_' + bank_name] = (
                '{}_online_soft{}_globalstep{}'.format(
                    anchor['anchor_id'], self._target_count, global_step))
            for metric_name, value in bank_scalars.items():
                if metric_name.startswith('bellman_'):
                    output_name = 'bellman_{}_{}'.format(
                        bank_name, metric_name[len('bellman_'):])
                else:
                    output_name = 'generic_{}_{}'.format(
                        bank_name, metric_name[len('generic_'):])
                scalars[output_name] = value

            bank_artifact = {
                'anchor_id': anchor['anchor_id'],
                'direction_bank_id': scalars[
                    'direction_bank_id_' + bank_name],
                'observations': observations.detach().cpu(),
                'actions': actions.detach().cpu(),
                'rewards': anchor['reward'][:self._anchor_size].detach().cpu(),
                'next_observations': anchor[
                    'next_observation'][:self._anchor_size].detach().cpu(),
                'terminals': anchor[
                    'terminal'][:self._anchor_size].detach().cpu(),
                'bellman_directions_q1': q1_directions.detach().cpu(),
                'bellman_directions_q2': q2_directions.detach().cpu(),
            }
            bank_artifact.update(q1_artifact)
            bank_artifact.update(q2_artifact)
            artifact['banks'][bank_name] = bank_artifact

        artifact_name = 'task{}_step{}_{}.pt'.format(
            task_idx, task_step, event)
        torch.save(artifact, os.path.join(self._artifact_dir, artifact_name))
        return scalars
