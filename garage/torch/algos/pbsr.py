"""Prospective Bellman Spectral Reserve for actor-critic value networks.

This module implements the inexpensive head-NTK variant.  Its probes are
constructed only from the current task minibatch and are detached before the
spectral objective is evaluated.  The regularizer therefore changes the
critic learning geometry without optimizing the Bellman targets themselves.
"""

import torch


def normalized_probe_directions(directions):
    """Center and independently unit-normalize Bellman probe columns."""
    centered = directions - directions.mean(dim=0, keepdim=True)
    norms = torch.linalg.norm(centered, dim=0, keepdim=True)
    return centered / norms.clamp_min(torch.finfo(centered.dtype).eps)


def pbsr_head_objective(features, directions, ridge=1e-3,
                        compute_diagnostics=False):
    """Return directional inverse burden for the exact linear-head NTK.

    The kernel is the exact NTK of the linear output-head weight block.
    Dividing it by its mean eigenvalue removes the global feature scale, while
    unit-normalizing each demand column makes all probes equally weighted.
    """
    if features.ndim != 2 or directions.ndim != 2:
        raise ValueError('PBSR features and directions must be matrices.')
    if features.shape[0] != directions.shape[0]:
        raise ValueError('PBSR features and directions need equal row counts.')

    kernel = torch.matmul(features, features.t()) / features.shape[1]
    kernel = 0.5 * (kernel + kernel.t())
    mean_eigenvalue = torch.trace(kernel) / kernel.shape[0]
    normalized_kernel = kernel / mean_eigenvalue

    probe_directions = normalized_probe_directions(directions.detach())
    regularized_kernel = normalized_kernel + ridge * torch.eye(
        kernel.shape[0], dtype=kernel.dtype, device=kernel.device)
    cholesky = torch.linalg.cholesky(regularized_kernel)
    solved = torch.cholesky_solve(probe_directions, cholesky)
    objective = (
        probe_directions * solved).sum() / probe_directions.shape[1]

    diagnostics = {
        'loss': float(objective.detach()),
        'kernel_mean_eigenvalue': float(mean_eigenvalue.detach()),
        'probe_columns': int(probe_directions.shape[1]),
        'anchor_rows': int(probe_directions.shape[0]),
    }
    if compute_diagnostics:
        with torch.no_grad():
            eigenvalues, eigenvectors = torch.linalg.eigh(normalized_kernel)
            slow_count = max(
                1, int((0.30 * kernel.shape[0]) + 0.999999))
            coefficients = torch.matmul(
                eigenvectors.t(), probe_directions)
            energy = coefficients.square().sum(dim=1)
            slow_energy = energy[:slow_count].sum() / energy.sum()
        diagnostics.update({
            'normalized_min_eigenvalue': float(eigenvalues[0]),
            'normalized_max_eigenvalue': float(eigenvalues[-1]),
            'slowest30_energy': float(slow_energy),
        })
    return objective, diagnostics


def gradient_ratio_coefficient(primary_loss, regularizer, parameters, ratio):
    """Scale the regularizer so its gradient norm is ``ratio`` times TD's."""
    parameters = tuple(parameter for parameter in parameters
                       if parameter.requires_grad)
    primary_gradients = torch.autograd.grad(
        primary_loss, parameters, retain_graph=True, allow_unused=True)
    regularizer_gradients = torch.autograd.grad(
        regularizer, parameters, retain_graph=True, allow_unused=True)

    primary_squared_norm = sum(
        gradient.square().sum() for gradient in primary_gradients
        if gradient is not None)
    regularizer_squared_norm = sum(
        gradient.square().sum() for gradient in regularizer_gradients
        if gradient is not None)
    primary_norm = torch.sqrt(primary_squared_norm)
    regularizer_norm = torch.sqrt(regularizer_squared_norm)
    coefficient = ratio * primary_norm / regularizer_norm
    diagnostics = {
        'td_grad_norm': float(primary_norm.detach()),
        'regularizer_grad_norm': float(regularizer_norm.detach()),
        'coefficient': float(coefficient.detach()),
        'weighted_gradient_ratio': float(
            (coefficient * regularizer_norm / primary_norm).detach()),
    }
    return coefficient.detach(), diagnostics


def deterministic_soft_bellman_probe_bank(
        samples, task_idx, q1_values, q2_values, target_qf1, target_qf2,
        policy, alpha, discount, reward_scale, target_count):
    """Construct a detached current-task soft-Bellman demand ensemble."""
    observations = samples['observation']
    rewards = samples['reward'].flatten()
    next_observations = samples['next_observation']
    terminals = samples['terminal'].flatten()

    with torch.no_grad():
        next_action_dist = policy(next_observations, task_idx)[0]
        base_dist = next_action_dist._normal.base_dist
        action_axis = torch.arange(
            1, base_dist.loc.shape[-1] + 1,
            dtype=base_dist.loc.dtype,
            device=base_dist.loc.device).unsqueeze(0)
        targets = []
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
            targets.append(
                rewards * reward_scale +
                (1. - terminals) * discount * (
                    torch.min(next_q1, next_q2) - alpha * next_log_pi))
        target_bank = torch.stack(targets, dim=1)
        q1_directions = target_bank - q1_values.detach().flatten().unsqueeze(1)
        q2_directions = target_bank - q2_values.detach().flatten().unsqueeze(1)
    return q1_directions, q2_directions


def _deterministic_unit_vectors(count, width, device, dtype, phase):
    """Create fixed reproducible probe vectors without learned parameters."""
    row_axis = torch.arange(
        1, count + 1, dtype=dtype, device=device).unsqueeze(1)
    column_axis = torch.arange(
        1, width + 1, dtype=dtype, device=device).unsqueeze(0)
    vectors = torch.sin(
        row_axis * column_axis * 0.017 + phase) + torch.cos(
            row_axis * column_axis * 0.031 + 0.5 * phase)
    return vectors / torch.linalg.norm(vectors, dim=1, keepdim=True)


def structured_counterfactual_bellman_probe_bank(
        samples, task_idx, target_qf1, target_qf2, policy, discount,
        target_count, seed):
    """Build Task-current, future-task-free Bellman-compatible probes.

    Each column applies a one-step Bellman operator to a frozen random value
    head.  Its action comes from a three-component mixture: actor mean, a
    deterministic noisy actor action, or a task-independent action.  A frozen
    random cumulant supplies reward variation.  The returned bank is detached;
    only the online critic kernel is optimized by PBSR.
    """
    observations = samples['observation']
    actions = samples['action']
    next_observations = samples['next_observation']
    terminals = samples['terminal'].flatten()

    with torch.no_grad():
        _ = target_qf1(observations, actions, seq_idx=task_idx)
        current_features_q1 = target_qf1._feature.detach().clone()
        _ = target_qf2(observations, actions, seq_idx=task_idx)
        current_features_q2 = target_qf2._feature.detach().clone()
        current_features = 0.5 * (
            current_features_q1 + current_features_q2)

        value_heads = _deterministic_unit_vectors(
            target_count, current_features.shape[1], observations.device,
            observations.dtype, phase=0.13 * (seed + 1))
        cumulant_inputs = torch.cat(
            (observations, actions, next_observations), dim=1)
        cumulant_heads = _deterministic_unit_vectors(
            target_count, cumulant_inputs.shape[1], observations.device,
            observations.dtype, phase=0.29 * (seed + 1))

        next_action_dist = policy(next_observations, task_idx)[0]
        base_dist = next_action_dist._normal.base_dist
        action_axis = torch.arange(
            1, actions.shape[1] + 1, dtype=actions.dtype,
            device=actions.device).unsqueeze(0)
        directions = []
        for probe_idx in range(target_count):
            policy_component = probe_idx % 3
            if policy_component == 0:
                next_actions = torch.tanh(base_dist.loc)
            elif policy_component == 1:
                noise = torch.sin((probe_idx + 1) * action_axis)
                next_actions = torch.tanh(
                    base_dist.loc + base_dist.scale * noise)
            else:
                next_actions = torch.sin(
                    (probe_idx + 1) * action_axis).expand_as(actions)

            _ = target_qf1(
                next_observations, next_actions, seq_idx=task_idx)
            next_features_q1 = target_qf1._feature.detach().clone()
            _ = target_qf2(
                next_observations, next_actions, seq_idx=task_idx)
            next_features_q2 = target_qf2._feature.detach().clone()
            next_features = 0.5 * (
                next_features_q1 + next_features_q2)

            value_head = value_heads[probe_idx]
            q_current = torch.matmul(current_features, value_head)
            q_next = torch.matmul(next_features, value_head)
            cumulant = torch.tanh(torch.matmul(
                cumulant_inputs, cumulant_heads[probe_idx]))
            directions.append(
                cumulant + (1. - terminals) * discount * q_next - q_current)
    return torch.stack(directions, dim=1)


def structured_value_bellman_probe_bank(
        observations, actions, rewards, next_observations, terminals,
        task_idx, value_function, discount, target_count, seed):
    """Build future-task-free Bellman probes for an on-policy value critic.

    PPO already supplies actions and successor states from its on-policy
    trajectories, so no SAC-style next-action mixture is needed.  Frozen
    random value heads and frozen random cumulants span different compatible
    Bellman demands on those transitions.
    """
    with torch.no_grad():
        _ = value_function(observations, seq_idx=task_idx)
        current_features = value_function._feature.detach().clone()
        _ = value_function(next_observations, seq_idx=task_idx)
        next_features = value_function._feature.detach().clone()

        value_heads = _deterministic_unit_vectors(
            target_count, current_features.shape[1], observations.device,
            observations.dtype, phase=0.17 * (seed + 1))
        cumulant_inputs = torch.cat(
            (observations, actions, next_observations), dim=1)
        cumulant_heads = _deterministic_unit_vectors(
            target_count, cumulant_inputs.shape[1], observations.device,
            observations.dtype, phase=0.37 * (seed + 1))

        directions = []
        nonterminal = 1. - terminals.flatten()
        for probe_idx in range(target_count):
            value_head = value_heads[probe_idx]
            value_current = torch.matmul(current_features, value_head)
            value_next = torch.matmul(next_features, value_head)
            cumulant = torch.tanh(torch.matmul(
                cumulant_inputs, cumulant_heads[probe_idx]))
            directions.append(
                cumulant + nonterminal * discount * value_next -
                value_current)
    return torch.stack(directions, dim=1)


class PBSRHead:
    """Compute and gradient-balance PBSR-head losses for twin SAC critics."""

    def __init__(self, anchor_size=64, target_count=8, ridge=1e-3,
                 gradient_ratio=0.1, seed=0):
        self.anchor_size = int(anchor_size)
        self.target_count = int(target_count)
        self.ridge = float(ridge)
        self.gradient_ratio = float(gradient_ratio)
        self.seed = int(seed)

    def direction_banks(
            self, samples, task_idx, q1_values, q2_values, target_qf1,
            target_qf2, policy, alpha, discount, reward_scale):
        rows = min(self.anchor_size, samples['observation'].shape[0])
        anchor = {key: value[:rows] for key, value in samples.items()}
        del q1_values, q2_values, alpha, reward_scale
        directions = structured_counterfactual_bellman_probe_bank(
            anchor, task_idx, target_qf1, target_qf2, policy, discount,
            self.target_count, self.seed)
        return directions, directions

    def critic_loss(self, critic, samples, task_idx, directions, td_loss,
                    compute_diagnostics=False):
        rows = directions.shape[0]
        _ = critic(
            samples['observation'][:rows], samples['action'][:rows],
            seq_idx=task_idx)
        features = critic._feature
        regularizer, diagnostics = pbsr_head_objective(
            features, directions, self.ridge, compute_diagnostics)
        coefficient, gradient_diagnostics = gradient_ratio_coefficient(
            td_loss, regularizer, critic.parameters(), self.gradient_ratio)
        diagnostics.update(gradient_diagnostics)
        total_loss = td_loss + coefficient * regularizer
        diagnostics.update({
            'td_loss': float(td_loss.detach()),
            'weighted_regularizer': float(
                (coefficient * regularizer).detach()),
            'total_loss': float(total_loss.detach()),
        })
        return total_loss, diagnostics


class PBSRValueHead:
    """Compute PBSR-head loss for an on-policy scalar value function."""

    def __init__(self, anchor_size=64, target_count=8, ridge=1e-3,
                 gradient_ratio=0.1, seed=0):
        self.anchor_size = int(anchor_size)
        self.target_count = int(target_count)
        self.ridge = float(ridge)
        self.gradient_ratio = float(gradient_ratio)
        self.seed = int(seed)

    def value_loss(self, value_function, observations, actions, rewards,
                   next_observations, terminals, task_idx, primary_loss,
                   discount, compute_diagnostics=False):
        rows = min(self.anchor_size, observations.shape[0])
        directions = structured_value_bellman_probe_bank(
            observations[:rows], actions[:rows], rewards[:rows],
            next_observations[:rows], terminals[:rows], task_idx,
            value_function, discount, self.target_count, self.seed)
        _ = value_function(observations[:rows], seq_idx=task_idx)
        features = value_function._feature
        regularizer, diagnostics = pbsr_head_objective(
            features, directions, self.ridge, compute_diagnostics)
        coefficient, gradient_diagnostics = gradient_ratio_coefficient(
            primary_loss, regularizer, value_function.parameters(),
            self.gradient_ratio)
        diagnostics.update(gradient_diagnostics)
        total_loss = primary_loss + coefficient * regularizer
        diagnostics.update({
            'value_loss': float(primary_loss.detach()),
            'weighted_regularizer': float(
                (coefficient * regularizer).detach()),
            'total_loss': float(total_loss.detach()),
        })
        return total_loss, diagnostics
