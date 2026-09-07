"""Demand-complete Prospective Bellman Spectral Reserve for PPO.

The original PPO port used a rank-limited value-feature objective whose
probe covariance was zero outside the sampled probe span.  Under a fixed
kernel trace, that objective is minimized by moving all spectral mass into
the probe span.  This module instead constructs network-independent,
multi-horizon Bellman reserve directions and analytically completes their
covariance for a prescribed minimum tangent bandwidth before evaluating the
inverse burden.

Both the value and policy shared backbones are trained.  The value demand
contains the actual Monte-Carlo and one-step TD residuals.  The policy demand
contains the exact coefficients of the active (unclipped) Gaussian PPO mean
head gradient.  No future-task transition, reward, or checkpoint is used.
"""

import math

import torch

from garage.torch.algos.pbsr import _deterministic_unit_vectors


def normalized_direction_columns(directions):
    """Center, remove zero columns, and unit-normalize function demands."""
    centered = directions - directions.mean(dim=0, keepdim=True)
    norms = torch.linalg.norm(centered, dim=0)
    tolerance = torch.finfo(centered.dtype).eps * math.sqrt(
        centered.shape[0])
    valid = norms > tolerance
    if not bool(valid.any()):
        raise ValueError('At least one nonconstant demand direction is needed.')
    return centered[:, valid] / norms[valid].unsqueeze(0)


def _standardize_columns(values):
    centered = values - values.mean(dim=0, keepdim=True)
    scale = torch.sqrt(centered.square().mean(dim=0, keepdim=True))
    return centered / scale.clamp_min(torch.finfo(values.dtype).eps)


def build_trajectory_reserve_bank(
        observations, actions, next_observations, terminals, path_ends,
        discount, probe_count, horizons, seed, task_idx):
    """Construct frozen raw-input multi-horizon Bellman reserve directions.

    A random Fourier value probe q_j(s) and cumulant c_j(s,a,s') are fixed by
    ``seed``.  For every transition and horizon h this computes

        sum_{k=0}^{h-1} gamma^k c_j(z_{t+k})
        + gamma^h q_j(s_{t+h}) - q_j(s_t),

    truncating at episode boundaries and bootstrapping from the recorded final
    successor for time-limit endings.  The construction is independent of the
    current value features, so a collapsing critic cannot collapse its probe.
    """
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError('PPO reserve inputs must be flattened matrices.')
    if observations.shape[0] != actions.shape[0]:
        raise ValueError('PPO reserve inputs need equal transition counts.')

    with torch.no_grad():
        observation_mean = observations.mean(dim=0, keepdim=True)
        observation_scale = torch.sqrt(
            (observations - observation_mean).square().mean(
                dim=0, keepdim=True)).clamp_min(
                    torch.finfo(observations.dtype).eps)
        observations = (
            observations - observation_mean) / observation_scale
        next_observations = (
            next_observations - observation_mean) / observation_scale
        transition_inputs = _standardize_columns(torch.cat(
            (observations, actions, next_observations), dim=1))
        phase = 0.19 * (seed + 1) + 0.41 * (task_idx + 1)
        value_heads = _deterministic_unit_vectors(
            probe_count, observations.shape[1], observations.device,
            observations.dtype, phase=phase)
        cumulant_heads = _deterministic_unit_vectors(
            probe_count, transition_inputs.shape[1], observations.device,
            observations.dtype, phase=phase + 0.73)

        q_current = torch.sin(
            torch.matmul(observations, value_heads.t()) + phase)
        q_next = torch.sin(
            torch.matmul(next_observations, value_heads.t()) + phase)
        cumulants = torch.tanh(
            torch.matmul(transition_inputs, cumulant_heads.t()) + phase)

        terminals = terminals.flatten().bool()
        path_ends = path_ends.flatten().bool()
        transition_count = observations.shape[0]
        starts = torch.arange(
            transition_count, device=observations.device)
        episode_ids = torch.cumsum(path_ends.long(), dim=0) - path_ends.long()
        reserve_columns = []

        for horizon in horizons:
            cumulative = torch.zeros_like(cumulants)
            targets = torch.zeros_like(cumulants)
            active = torch.ones(
                transition_count, dtype=torch.bool,
                device=observations.device)
            for offset in range(int(horizon)):
                raw_indices = starts + offset
                within_batch = raw_indices < transition_count
                indices = raw_indices.clamp_max(transition_count - 1)
                same_episode = episode_ids[indices] == episode_ids[starts]
                valid = active & within_batch & same_episode
                cumulative[valid] += (
                    discount ** offset) * cumulants[indices[valid]]

                stop = valid & (
                    terminals[indices] | path_ends[indices] |
                    (offset == int(horizon) - 1))
                targets[stop] = cumulative[stop]
                bootstrap = stop & ~terminals[indices]
                targets[bootstrap] += (
                    discount ** (offset + 1)) * q_next[indices[bootstrap]]
                active = valid & ~stop

            reserve_columns.append(targets - q_current)

        return torch.cat(reserve_columns, dim=1).detach()


def completed_demand_covariance(directions, minimum_bandwidth=0.5):
    """Complete demand covariance for a prescribed tangent bandwidth.

    For trace-normalized K, minimizing ``Tr[Omega K^-1]`` gives
    ``lambda_i(K) proportional to sqrt(lambda_i(Omega))``.  The complement
    variance is therefore solved analytically so that every unobserved
    centered direction has optimal kernel eigenvalue ``minimum_bandwidth^2``.
    This retains the observed demand anisotropy and introduces neither an
    arbitrary 0.5 covariance mixture nor a separate spectrum-only loss.
    """
    if not 0. < minimum_bandwidth < 1.:
        raise ValueError('minimum_bandwidth must lie strictly between 0 and 1.')
    normalized = normalized_direction_columns(directions.detach())
    sample_count = normalized.shape[0]
    empirical = torch.matmul(normalized, normalized.t())
    empirical = empirical / normalized.shape[1]
    empirical = 0.5 * (empirical + empirical.t())

    with torch.no_grad():
        eigenvalues, eigenvectors = torch.linalg.eigh(empirical)
        tolerance = (
            eigenvalues[-1] * max(empirical.shape) *
            torch.finfo(empirical.dtype).eps)
        positive = eigenvalues > tolerance
        if not bool(positive.any()):
            raise ValueError('Demand covariance has no positive eigenvalue.')
        positive_vectors = eigenvectors[:, positive]
        observed_projector = torch.matmul(
            positive_vectors, positive_vectors.t())
        identity = torch.eye(
            sample_count, dtype=empirical.dtype, device=empirical.device)
        constant = torch.ones(
            sample_count, 1, dtype=empirical.dtype,
            device=empirical.device) / math.sqrt(sample_count)
        centered_projector = identity - torch.matmul(constant, constant.t())
        complement_projector = centered_projector - observed_projector
        complement_projector = 0.5 * (
            complement_projector + complement_projector.t())
        centered_dimension = sample_count - 1
        complement_dimension = centered_dimension - int(positive.sum())
        target_kernel_floor = minimum_bandwidth ** 2
        if complement_dimension > 0:
            observed_sqrt_sum = torch.sqrt(eigenvalues[positive]).sum()
            denominator = (
                centered_dimension -
                complement_dimension * target_kernel_floor)
            completion_floor = (
                target_kernel_floor * observed_sqrt_sum /
                denominator).square()
        else:
            completion_floor = eigenvalues.new_zeros(())
        completed = (
            empirical + completion_floor * complement_projector)
        completed = 0.5 * (completed + completed.t())
        completed_trace = torch.trace(completed)
        covariance = completed / completed_trace
        complement_mass = (
            completion_floor * torch.trace(complement_projector) /
            completed_trace)

    diagnostics = {
        'demand_rank': int(positive.sum()),
        'demand_columns': int(normalized.shape[1]),
        'completion_floor': float(completion_floor),
        'completion_mass': float(complement_mass),
        'minimum_bandwidth': float(minimum_bandwidth),
        'target_kernel_eigenvalue_floor': float(minimum_bandwidth ** 2),
    }
    return normalized, covariance, diagnostics


def _mean_direction_burden(normalized_directions, cholesky):
    solved = torch.cholesky_solve(normalized_directions, cholesky)
    return (normalized_directions * solved).sum() / normalized_directions.shape[1]


def demand_complete_inverse_burden(
        tangent, current_directions, reserve_directions, ridge=1e-3,
        minimum_bandwidth=0.5, compute_diagnostics=False):
    """Evaluate inverse burden under a demand-completed covariance."""
    if tangent.ndim != 2:
        raise ValueError('The head tangent must be a matrix.')
    if tangent.shape[0] != current_directions.shape[0]:
        raise ValueError('Tangent and demand rows must match.')

    current = normalized_direction_columns(current_directions.detach())
    reserve = normalized_direction_columns(reserve_directions.detach())
    all_directions = torch.cat((current, reserve), dim=1)
    normalized, covariance, covariance_stats = (
        completed_demand_covariance(
            all_directions, minimum_bandwidth=minimum_bandwidth))

    kernel = torch.matmul(tangent, tangent.t())
    kernel = 0.5 * (kernel + kernel.t())
    mean_eigenvalue = torch.trace(kernel) / (kernel.shape[0] - 1)
    normalized_kernel = kernel / mean_eigenvalue
    regularized_kernel = normalized_kernel + ridge * torch.eye(
        kernel.shape[0], dtype=kernel.dtype, device=kernel.device)
    cholesky = torch.linalg.cholesky(regularized_kernel)
    solved_covariance = torch.cholesky_solve(covariance, cholesky)
    objective = torch.trace(solved_covariance)

    diagnostics = dict(covariance_stats)
    diagnostics.update({
        'loss': float(objective.detach()),
        'kernel_mean_eigenvalue': float(mean_eigenvalue.detach()),
        'anchor_rows': int(tangent.shape[0]),
        'tangent_columns': int(tangent.shape[1]),
        'current_columns': int(current.shape[1]),
        'reserve_columns': int(reserve.shape[1]),
        'current_burden': float(
            _mean_direction_burden(current, cholesky).detach()),
        'reserve_burden': float(
            _mean_direction_burden(reserve, cholesky).detach()),
    })

    if compute_diagnostics:
        with torch.no_grad():
            eigenvalues, eigenvectors = torch.linalg.eigh(normalized_kernel)
            positive_eigenvalues = eigenvalues.clamp_min(0.)
            probabilities = positive_eigenvalues / positive_eigenvalues.sum()
            nonzero_probabilities = probabilities[probabilities > 0]
            effective_rank = torch.exp(-(
                nonzero_probabilities * nonzero_probabilities.log()).sum())
            tolerance = (
                positive_eigenvalues[-1] * max(normalized_kernel.shape) *
                torch.finfo(normalized_kernel.dtype).eps)
            learned = eigenvalues > tolerance
            learned_vectors = eigenvectors[:, learned]
            projected = torch.matmul(
                learned_vectors,
                torch.matmul(learned_vectors.t(), normalized))
            nullspace_energy = (
                (normalized - projected).square().sum() /
                normalized.square().sum())
        diagnostics.update({
            'kernel_min_eigenvalue': float(eigenvalues[0]),
            'kernel_max_eigenvalue': float(eigenvalues[-1]),
            'kernel_top_share': float(
                positive_eigenvalues[-1] / positive_eigenvalues.sum()),
            'kernel_effective_rank': float(effective_rank),
            'kernel_numerical_rank': int(learned.sum()),
            'demand_nullspace_energy': float(nullspace_energy),
        })
    return objective, diagnostics


def backbone_gradient_ratio_coefficient(
        primary_loss, regularizer, parameters, ratio):
    """Match PBSR gradient norm to a fraction of the same backbone's loss."""
    parameters = tuple(parameter for parameter in parameters
                       if parameter.requires_grad)
    primary_gradients = torch.autograd.grad(
        primary_loss, parameters, retain_graph=True, allow_unused=True)
    regularizer_gradients = torch.autograd.grad(
        regularizer, parameters, retain_graph=True, allow_unused=True)
    primary_squared_norm = primary_loss.new_zeros(())
    regularizer_squared_norm = primary_loss.new_zeros(())
    gradient_dot = primary_loss.new_zeros(())
    for primary_gradient, regularizer_gradient in zip(
            primary_gradients, regularizer_gradients):
        if primary_gradient is not None:
            primary_squared_norm = (
                primary_squared_norm + primary_gradient.square().sum())
        if regularizer_gradient is not None:
            regularizer_squared_norm = (
                regularizer_squared_norm +
                regularizer_gradient.square().sum())
        if primary_gradient is not None and regularizer_gradient is not None:
            gradient_dot = gradient_dot + (
                primary_gradient * regularizer_gradient).sum()

    primary_norm = torch.sqrt(primary_squared_norm)
    regularizer_norm = torch.sqrt(regularizer_squared_norm)
    epsilon = torch.finfo(primary_loss.dtype).eps
    coefficient = ratio * primary_norm / (regularizer_norm + epsilon)
    cosine = gradient_dot / (
        primary_norm * regularizer_norm + epsilon)
    diagnostics = {
        'primary_backbone_grad_norm': float(primary_norm.detach()),
        'regularizer_backbone_grad_norm': float(regularizer_norm.detach()),
        'gradient_cosine': float(cosine.detach()),
        'coefficient': float(coefficient.detach()),
        'weighted_gradient_ratio': float((
            coefficient * regularizer_norm /
            (primary_norm + epsilon)).detach()),
    }
    return coefficient.detach(), diagnostics


def _value_backbone_parameters(value_function):
    return value_function.module._mean_module._layers.parameters()


def _policy_backbone_parameters(policy):
    return policy._module._mean_module._layers.parameters()


def gaussian_ppo_mean_head_directions(
        distribution, old_distribution, actions, advantages, clip_range):
    """Return exact active PPO mean-output gradient coefficients."""
    old_log_likelihood = old_distribution.log_prob(actions)
    log_likelihood = distribution.log_prob(actions)
    likelihood_ratio = (log_likelihood - old_log_likelihood).exp()
    active = (
        ((advantages >= 0.) &
         (likelihood_ratio <= 1. + clip_range)) |
        ((advantages < 0.) &
         (likelihood_ratio >= 1. - clip_range)))
    base_distribution = distribution.base_dist
    mean_score = (
        (actions - base_distribution.loc) /
        base_distribution.scale.square())
    ppo_coefficient = (
        advantages * likelihood_ratio * active.to(advantages.dtype))
    return (
        ppo_coefficient.unsqueeze(1) * mean_score,
        mean_score,
        active)


class PPOPBSRV2:
    """Demand-complete actor/value spectral reserve used on every task."""

    def __init__(self, anchor_size=64, probe_count=8, horizons=(1, 3, 5),
                 ridge=1e-3, gradient_ratio=0.1, minimum_bandwidth=0.5,
                 seed=0,
                 train_actor=True):
        self.anchor_size = int(anchor_size)
        self.probe_count = int(probe_count)
        self.horizons = tuple(int(horizon) for horizon in horizons)
        self.ridge = float(ridge)
        self.gradient_ratio = float(gradient_ratio)
        self.minimum_bandwidth = float(minimum_bandwidth)
        self.seed = int(seed)
        self.train_actor = bool(train_actor)

    def build_reserve_bank(
            self, observations, actions, next_observations, terminals,
            path_ends, discount, task_idx):
        return build_trajectory_reserve_bank(
            observations, actions, next_observations, terminals, path_ends,
            discount, self.probe_count, self.horizons, self.seed, task_idx)

    def value_loss(
            self, value_function, observations, returns, rewards,
            next_observations, terminals, reserve_directions, task_idx,
            primary_loss, discount, compute_diagnostics=False):
        rows = min(self.anchor_size, observations.shape[0])
        observations = observations[:rows]
        next_observations = next_observations[:rows]
        returns = returns[:rows]
        rewards = rewards[:rows]
        terminals = terminals[:rows]
        reserve_directions = reserve_directions[:rows]

        with torch.no_grad():
            values = value_function(
                observations, seq_idx=task_idx).flatten()
            next_values = value_function(
                next_observations, seq_idx=task_idx).flatten()
            monte_carlo = returns.flatten() - values
            one_step_td = (
                rewards.flatten() + discount *
                (1. - terminals.flatten()) * next_values - values)
            current_directions = torch.stack(
                (monte_carlo, one_step_td), dim=1)

        _ = value_function(observations, seq_idx=task_idx)
        features = value_function._feature
        tangent = features - features.mean(dim=0, keepdim=True)
        regularizer, diagnostics = demand_complete_inverse_burden(
            tangent, current_directions, reserve_directions, self.ridge,
            self.minimum_bandwidth, compute_diagnostics)
        coefficient, gradient_stats = backbone_gradient_ratio_coefficient(
            primary_loss, regularizer,
            _value_backbone_parameters(value_function), self.gradient_ratio)
        diagnostics.update(gradient_stats)
        total_loss = primary_loss + coefficient * regularizer
        diagnostics.update({
            'primary_loss': float(primary_loss.detach()),
            'weighted_regularizer': float(
                (coefficient * regularizer).detach()),
            'total_loss': float(total_loss.detach()),
            'geometry': 'value_mean_head',
        })
        return total_loss, diagnostics

    def policy_loss(
            self, policy, old_policy, observations, actions, advantages,
            reserve_directions, task_idx, primary_loss, clip_range,
            compute_diagnostics=False):
        rows = min(self.anchor_size, observations.shape[0])
        observations = observations[:rows]
        actions = actions[:rows]
        advantages = advantages[:rows]
        reserve_directions = reserve_directions[:rows]

        with torch.no_grad():
            old_distribution = old_policy(observations, task_idx)[0]
            distribution = policy(observations, task_idx)[0]
            current_directions, mean_score, active = (
                gaussian_ppo_mean_head_directions(
                    distribution, old_distribution, actions, advantages,
                    clip_range))

            action_heads = _deterministic_unit_vectors(
                reserve_directions.shape[1], actions.shape[1],
                actions.device, actions.dtype,
                phase=0.53 * (self.seed + 1) + 0.31 * (task_idx + 1))
            projected_score = torch.matmul(mean_score, action_heads.t())
            policy_reserve = reserve_directions * projected_score

        _ = policy(observations, task_idx)[0]
        features = policy._feature
        tangent = features - features.mean(dim=0, keepdim=True)
        regularizer, diagnostics = demand_complete_inverse_burden(
            tangent, current_directions, policy_reserve, self.ridge,
            self.minimum_bandwidth, compute_diagnostics)
        coefficient, gradient_stats = backbone_gradient_ratio_coefficient(
            primary_loss, regularizer,
            _policy_backbone_parameters(policy), self.gradient_ratio)
        diagnostics.update(gradient_stats)
        total_loss = primary_loss + coefficient * regularizer
        diagnostics.update({
            'primary_loss': float(primary_loss.detach()),
            'weighted_regularizer': float(
                (coefficient * regularizer).detach()),
            'total_loss': float(total_loss.detach()),
            'active_clip_fraction': float(active.float().mean()),
            'geometry': 'policy_mean_head',
        })
        return total_loss, diagnostics
