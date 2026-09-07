"""PPO value-spectrum measurements on fixed transition references."""

import copy
import json
import os

import torch
import torch.nn.functional as F

from garage.torch.algos.bellman_spectral_stats import spectral_direction_stats


DEFAULT_PPO_TASK_STEPS = (
    15_000,
    60_000,
    105_000,
    510_000,
    1_005_000,
    1_500_000,
)


def spectral_ranks(eigenvalues):
    """Return entropy rank, stable rank, and top-eigenvalue mass."""
    eigenvalues = eigenvalues.clamp_min(0.)
    probabilities = eigenvalues / eigenvalues.sum()
    positive = probabilities > 0
    entropy_rank = torch.exp(-(
        probabilities[positive] * probabilities[positive].log()).sum())
    stable_rank = eigenvalues.sum().square() / eigenvalues.square().sum()
    top_mass = eigenvalues[-1] / eigenvalues.sum()
    return entropy_rank, stable_rank, top_mass


def demand_effective_rank(directions):
    """Return the entropy effective rank of a function-direction bank."""
    singular_values = torch.linalg.svdvals(directions)
    probabilities = singular_values.square()
    probabilities = probabilities / probabilities.sum()
    positive = probabilities > 0
    return torch.exp(-(
        probabilities[positive] * probabilities[positive].log()).sum())


def empirical_value_jacobian(value_function, observations, task_idx):
    """Return Jacobians of value means with respect to the mean network."""
    parameters = [
        parameter
        for parameter in value_function.module._mean_module.parameters()
        if parameter.requires_grad
    ]
    rows = []
    for row_idx in range(observations.shape[0]):
        value = value_function(
            observations[row_idx:row_idx + 1],
            seq_idx=task_idx).reshape(())
        gradients = torch.autograd.grad(value, parameters)
        rows.append(torch.cat([
            gradient.reshape(-1) for gradient in gradients
        ]))
    return torch.stack(rows, dim=0)


def empirical_policy_mean_jacobian(policy, observations, task_idx):
    """Return Jacobians of policy means with respect to the mean network."""
    parameters = [
        parameter
        for parameter in policy._module._mean_module.parameters()
        if parameter.requires_grad
    ]
    rows = []
    for row_idx in range(observations.shape[0]):
        mean = policy(
            observations[row_idx:row_idx + 1], task_idx)[0].mean.flatten()
        for action_idx in range(mean.shape[0]):
            gradients = torch.autograd.grad(
                mean[action_idx], parameters, retain_graph=True,
                allow_unused=True)
            rows.append(torch.cat([
                (torch.zeros_like(parameter) if gradient is None else gradient)
                .reshape(-1)
                for parameter, gradient in zip(parameters, gradients)
            ]))
    return torch.stack(rows, dim=0)


def fit_fixed_value_target(value_function, observations, task_idx, target,
                           learning_rate, steps=200):
    """Fit a frozen value target with a cloned mean network."""
    fitted_value = copy.deepcopy(value_function)
    optimizer = torch.optim.Adam(
        fitted_value.module._mean_module.parameters(), lr=learning_rate)
    curve = []
    for fit_step in range(steps + 1):
        predictions = fitted_value(
            observations, seq_idx=task_idx).flatten()
        loss = F.mse_loss(predictions, target)
        if fit_step == 0:
            initial_loss = loss.detach()
        curve.append((loss.detach() / initial_loss).cpu())
        if fit_step < steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return torch.stack(curve)


def fit_fixed_policy_target(policy, observations, task_idx, target,
                            learning_rate, steps=200):
    """Fit frozen successful actions with a cloned policy mean network."""
    mean_module = policy._module._mean_module
    cached_feature = mean_module._feature
    cached_features = mean_module._features
    mean_module._feature = None
    mean_module._features = None
    fitted_mean = copy.deepcopy(mean_module)
    mean_module._feature = cached_feature
    mean_module._features = cached_features
    optimizer = torch.optim.Adam(
        fitted_mean.parameters(), lr=learning_rate)
    curve = []
    for fit_step in range(steps + 1):
        predictions = fitted_mean(observations)[task_idx]
        loss = F.mse_loss(predictions, target)
        if fit_step == 0:
            initial_loss = loss.detach()
        curve.append((loss.detach() / initial_loss).cpu())
        if fit_step < steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return torch.stack(curve)


class PPOValueSpectralStats:
    """Measure the current PPO value critic on fixed task transitions."""

    def __init__(self, run_dir, reference_dir, reference_mode='load',
                 anchor_size=64, relative_ridge=1e-3,
                 fit_learning_rate=5e-4, discount=0.99,
                 task_steps=DEFAULT_PPO_TASK_STEPS, seed=0):
        self._reference_dir = reference_dir
        self._reference_mode = reference_mode
        self._anchor_size = anchor_size
        self._relative_ridge = relative_ridge
        self._fit_learning_rate = fit_learning_rate
        self._discount = discount
        self._task_steps = tuple(task_steps)
        self._seed = seed
        self._artifact_dir = os.path.join(run_dir, 'spectral_stats')
        self._metrics_path = os.path.join(run_dir, 'metrics.jsonl')
        os.makedirs(self._artifact_dir, exist_ok=True)

        config = {
            'algorithm': 'ppo',
            'critic': 'state_value_mean',
            'demand': (
                'reference_defined_fixed_targets_or_online_one_step_td'),
            'direction_source': 'reference_file_format',
            'fixed_target_direction_preprocessing': (
                'primary_mc_residual_plus_multistep_innovations_'
                'row_centered_and_column_unit_l2'),
            'kernel': 'full_value_mean_parameter_empirical_ntk_JJT_over_p',
            'anchor_size': self._anchor_size,
            'relative_ridge': self._relative_ridge,
            'discount': self._discount,
            'slow_fraction': 0.30,
            'fit_optimizer': 'fresh_adam_on_cloned_value_mean_network',
            'fit_learning_rate': self._fit_learning_rate,
            'fit_steps': [50, 200],
            'generic_control': 'fixed_gaussian_equal_l2_norm',
            'reference_dir': self._reference_dir,
            'reference_mode': self._reference_mode,
            'reference_transition_banks': ['initial', 'coverage'],
            'reference_contains_frozen_directions': True,
            'policy_demand': (
                'successful_teacher_action_minus_current_policy_mean'),
            'policy_kernel': (
                'full_policy_mean_parameter_empirical_ntk_JJT_over_p'),
            'policy_fit_target': 'successful_teacher_action',
            'task_steps': list(self._task_steps),
        }
        with open(os.path.join(
                self._artifact_dir, 'spectral_stats_config.json'),
                'w') as config_file:
            json.dump(config, config_file, indent=2)

    @staticmethod
    def _task_file_name(task_name):
        name = task_name[:-3] if task_name.endswith('-v2') else task_name
        return name.replace('-', '_') + '.pt'

    def _reference_path(self, task_name):
        return os.path.join(
            self._reference_dir, self._task_file_name(task_name))

    def should_run(self, event, task_step):
        return event in ('task_boundary', 'final') or task_step in self._task_steps

    def _load_anchors(self, task_name, device):
        reference_path = self._reference_path(task_name)
        reference = torch.load(reference_path, map_location=device)
        if reference.get('format_version') not in (2, 3):
            raise ValueError(
                'PPO spectral reference must be version 2 or 3: '
                'reference: {}'.format(reference_path))
        anchors = {}
        for bank_name in ('initial', 'coverage'):
            bank = reference['banks'][bank_name]
            anchors[bank_name] = {
                'observations': torch.as_tensor(
                    bank['observation'], dtype=torch.float32, device=device),
                'actions': torch.as_tensor(
                    bank['action'], dtype=torch.float32, device=device),
                'rewards': torch.as_tensor(
                    bank['reward'], dtype=torch.float32, device=device),
                'next_observations': torch.as_tensor(
                    bank['next_observation'], dtype=torch.float32,
                    device=device),
                'terminals': torch.as_tensor(
                    bank['terminal'], dtype=torch.float32, device=device),
                'anchor_id': bank['anchor_id'],
            }
            if reference['format_version'] == 3:
                anchors[bank_name]['bellman_targets'] = torch.as_tensor(
                    bank['bellman_targets'], dtype=torch.float32,
                    device=device)
                anchors[bank_name]['target_bank_id'] = bank[
                    'target_bank_id']
                anchors[bank_name]['target_definition'] = reference[
                    'target_definition']
        return anchors

    def _online_value_direction(self, value_function, anchor, task_idx):
        with torch.no_grad():
            current_values = value_function(
                anchor['observations'], seq_idx=task_idx).flatten()
            next_values = value_function(
                anchor['next_observations'], seq_idx=task_idx).flatten()
            direction = (
                anchor['rewards'].flatten() +
                self._discount * (1. - anchor['terminals'].flatten()) *
                next_values - current_values)
        return direction.unsqueeze(1)

    def _generic_direction(self, value_direction, task_idx, bank_name):
        generator = torch.Generator(device='cpu')
        bank_seed = sum(ord(character) for character in bank_name)
        generator.manual_seed(
            self._seed * 1013 + task_idx * 9181 + bank_seed + 1877)
        random_direction = torch.randn(
            value_direction.shape, generator=generator).to(
                device=value_direction.device,
                dtype=value_direction.dtype)
        return (random_direction * value_direction.norm() /
                random_direction.norm())

    def run(self, event, global_step, task_step, task_idx, task_name,
            value_function, policy=None):
        """Recompute current one-step TD demand on both fixed banks."""
        device = next(value_function.parameters()).device
        anchors = self._load_anchors(task_name, device)
        fixed_target_reference = 'bellman_targets' in anchors['initial']
        direction_definition = (
            anchors['initial']['target_definition']
            if fixed_target_reference else
            'online_current_checkpoint_one_step_value_td')
        metric = {
            'event': event,
            'global_step': global_step,
            'task_step': task_step,
            'current_task': task_idx,
            'current_task_name': task_name,
            'direction_definition': direction_definition,
        }
        artifact = dict(metric)
        artifact['banks'] = {}

        for bank_name, full_anchor in anchors.items():
            anchor = {
                key: value[:self._anchor_size]
                if torch.is_tensor(value) else value
                for key, value in full_anchor.items()
            }
            observations = anchor['observations']
            with torch.no_grad():
                current_values = value_function(
                    observations, seq_idx=task_idx).flatten()
            if fixed_target_reference:
                fixed_targets = anchor['bellman_targets']
                value_target = fixed_targets[:, 0]
                primary_direction = value_target - current_values
                target_innovations = (
                    fixed_targets[:, 1:] - fixed_targets[:, :1])
                directions = torch.cat([
                    primary_direction.unsqueeze(1), target_innovations
                ], dim=1)
                directions = directions - directions.mean(dim=0, keepdim=True)
                directions = directions / directions.norm(
                    dim=0, keepdim=True)
                direction_bank_id = anchor['target_bank_id']
            else:
                directions = self._online_value_direction(
                    value_function, anchor, task_idx)
                value_target = current_values + directions.flatten()
                primary_direction = directions.flatten()
                direction_bank_id = '{}_online_td_globalstep{}'.format(
                    anchor['anchor_id'], global_step)
            value_direction = primary_direction
            generic_direction = self._generic_direction(
                value_direction.unsqueeze(1), task_idx, bank_name).flatten()
            generic_target = current_values + generic_direction
            value_curve = fit_fixed_value_target(
                value_function, observations, task_idx, value_target,
                self._fit_learning_rate)
            generic_curve = fit_fixed_value_target(
                value_function, observations, task_idx, generic_target,
                self._fit_learning_rate)
            jacobian = empirical_value_jacobian(
                value_function, observations, task_idx)
            spectral = spectral_direction_stats(
                jacobian, directions, self._relative_ridge)
            entropy_rank, stable_rank, top_mass = spectral_ranks(
                spectral['eigenvalues'])
            direction_rank = demand_effective_rank(directions)

            if policy is not None:
                teacher_actions = anchor['actions']
                with torch.no_grad():
                    current_policy_mean = policy(
                        observations, task_idx)[0].mean
                policy_direction = teacher_actions - current_policy_mean
                policy_jacobian = empirical_policy_mean_jacobian(
                    policy, observations, task_idx)
                policy_spectral = spectral_direction_stats(
                    policy_jacobian,
                    policy_direction.flatten().unsqueeze(1),
                    self._relative_ridge)
                policy_entropy_rank, policy_stable_rank, policy_top_mass = (
                    spectral_ranks(policy_spectral['eigenvalues']))
                policy_curve = fit_fixed_policy_target(
                    policy, observations, task_idx, teacher_actions,
                    self._fit_learning_rate)

            prefix = 'ppo_value_{}_'.format(bank_name)
            metric['anchor_id_' + bank_name] = anchor['anchor_id']
            metric['direction_bank_id_' + bank_name] = direction_bank_id
            metric[prefix + 'shape_burden'] = spectral['burden'].item()
            metric[prefix + 'absolute_inverse_burden'] = (
                spectral['burden'] / spectral['mean_eigenvalue']).item()
            metric[prefix + 'slowest30_energy'] = spectral[
                'slow_energy'].item()
            metric[prefix + 'kernel_mean_eigenvalue'] = spectral[
                'mean_eigenvalue'].item()
            metric[prefix + 'kernel_entropy_rank'] = entropy_rank.item()
            metric[prefix + 'kernel_stable_rank'] = stable_rank.item()
            metric[prefix + 'kernel_top1_mass'] = top_mass.item()
            metric[prefix + 'demand_effective_rank'] = direction_rank.item()
            metric[prefix + 'fit_residual_50'] = value_curve[50].item()
            metric[prefix + 'fit_residual_200'] = value_curve[200].item()
            metric['ppo_generic_{}_fit_residual_50'.format(
                bank_name)] = generic_curve[50].item()
            metric['ppo_generic_{}_fit_residual_200'.format(
                bank_name)] = generic_curve[200].item()
            if policy is not None:
                policy_prefix = 'ppo_policy_{}_'.format(bank_name)
                metric[policy_prefix + 'shape_burden'] = policy_spectral[
                    'burden'].item()
                metric[policy_prefix + 'absolute_inverse_burden'] = (
                    policy_spectral['burden'] /
                    policy_spectral['mean_eigenvalue']).item()
                metric[policy_prefix + 'slowest30_energy'] = policy_spectral[
                    'slow_energy'].item()
                metric[policy_prefix + 'kernel_mean_eigenvalue'] = (
                    policy_spectral['mean_eigenvalue'].item())
                metric[policy_prefix + 'kernel_entropy_rank'] = (
                    policy_entropy_rank.item())
                metric[policy_prefix + 'kernel_stable_rank'] = (
                    policy_stable_rank.item())
                metric[policy_prefix + 'kernel_top1_mass'] = (
                    policy_top_mass.item())
                metric[policy_prefix + 'action_mse'] = F.mse_loss(
                    current_policy_mean, teacher_actions).item()
                metric[policy_prefix + 'fit_residual_50'] = (
                    policy_curve[50].item())
                metric[policy_prefix + 'fit_residual_200'] = (
                    policy_curve[200].item())

            artifact['banks'][bank_name] = {
                'anchor_id': anchor['anchor_id'],
                'direction_bank_id': metric[
                    'direction_bank_id_' + bank_name],
                'observations': observations.detach().cpu(),
                'actions': anchor['actions'].detach().cpu(),
                'rewards': anchor['rewards'].detach().cpu(),
                'next_observations': anchor[
                    'next_observations'].detach().cpu(),
                'terminals': anchor['terminals'].detach().cpu(),
                'value_directions': directions.detach().cpu(),
                'primary_value_direction': primary_direction.detach().cpu(),
                'fixed_bellman_targets': (
                    fixed_targets.detach().cpu()
                    if fixed_target_reference else None),
                'jacobian_eigenvalues': spectral['eigenvalues'].cpu(),
                'value_energy_per_eigendirection': spectral[
                    'energy_per_eigendirection'].cpu(),
                'value_fitting_curve': value_curve,
                'generic_fitting_curve': generic_curve,
                'mean_jacobian_eigenvalue': spectral[
                    'mean_eigenvalue'].cpu(),
                'kernel_ridge': spectral['ridge'].cpu(),
                'slow_direction_count': spectral['slow_count'],
            }
            if policy is not None:
                artifact['banks'][bank_name].update({
                    'teacher_actions': teacher_actions.detach().cpu(),
                    'current_policy_mean': current_policy_mean.detach().cpu(),
                    'policy_direction': policy_direction.detach().cpu(),
                    'policy_jacobian_eigenvalues': policy_spectral[
                        'eigenvalues'].cpu(),
                    'policy_energy_per_eigendirection': policy_spectral[
                        'energy_per_eigendirection'].cpu(),
                    'policy_fitting_curve': policy_curve,
                })

        artifact['value_function_state_dict'] = {
            name: value.detach().cpu()
            for name, value in value_function.state_dict().items()
        }
        if policy is not None:
            artifact['policy_state_dict'] = {
                name: value.detach().cpu()
                for name, value in policy.state_dict().items()
            }
        artifact_name = 'task{}_step{}_{}.pt'.format(
            task_idx, task_step, event)
        torch.save(artifact, os.path.join(self._artifact_dir, artifact_name))
        with open(self._metrics_path, 'a') as metrics_file:
            metrics_file.write(json.dumps(metric) + '\n')
        print('PPO_VALUE_SPECTRAL', json.dumps(metric), flush=True)
        return metric
