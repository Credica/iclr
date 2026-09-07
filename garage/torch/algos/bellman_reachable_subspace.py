"""Analysis utilities for Bellman-reachable residual subspaces.

This module is deliberately independent of a learned policy, critic, and
optimizer.  Given a fixed batch of one-step transitions, it constructs the
network-independent random-Fourier Bellman residuals

    d_j = c_j(s, a, s') + gamma (1 - terminal) q_j(s') - q_j(s),

where ``q_j`` and ``c_j`` are frozen random Fourier value probes and
cumulants.  The residual columns can then be centered, normalized, and
compressed with an SVD into an empirical Bellman-reachable basis.

For N anchors, centering removes the constant output mode, so the largest
possible reachable rank is N - 1.  The tangent margin implemented below is
the smallest eigenvalue of the full-Jacobian tangent kernel restricted to
that reachable basis after normalizing the mean eigenvalue of the centered
kernel to one.
"""

import math

import torch


def _population_moments(values):
    """Return population mean and scale for a matrix of design samples."""
    mean = values.mean(dim=0, keepdim=True)
    scale = torch.sqrt(
        (values - mean).square().mean(dim=0, keepdim=True))
    scale = scale.clamp_min(torch.finfo(values.dtype).eps)
    return mean, scale


def _transition_tensors(transitions):
    """Read and shape-check a transition mapping without retaining gradients."""
    required = ('observation', 'action', 'next_observation', 'terminal')
    missing = [key for key in required if key not in transitions]
    if missing:
        raise KeyError('Missing transition tensors: {}'.format(missing))

    observations = transitions['observation'].detach()
    actions = transitions['action'].detach()
    next_observations = transitions['next_observation'].detach()
    terminals = transitions['terminal'].detach().flatten()
    if observations.ndim != 2 or next_observations.ndim != 2:
        raise ValueError('Observations must be rank-two matrices.')
    if actions.ndim != 2:
        raise ValueError('Actions must be a rank-two matrix.')
    transition_count = observations.shape[0]
    if (next_observations.shape != observations.shape or
            actions.shape[0] != transition_count or
            terminals.shape[0] != transition_count):
        raise ValueError('All transition tensors must have equal row counts.')
    if not observations.is_floating_point():
        raise TypeError('Transition tensors must use a floating dtype.')

    actions = actions.to(
        device=observations.device, dtype=observations.dtype)
    next_observations = next_observations.to(
        device=observations.device, dtype=observations.dtype)
    terminals = terminals.to(
        device=observations.device, dtype=observations.dtype)
    return observations, actions, next_observations, terminals


def _cpu_random(shape, generator, dtype, device, uniform=False):
    """Draw on CPU so a seed defines the same probes on CPU and CUDA."""
    if uniform:
        values = torch.rand(shape, generator=generator, dtype=dtype)
    else:
        values = torch.randn(shape, generator=generator, dtype=dtype)
    return values.to(device=device)


def _random_fourier_parameters(input_dim, output_count, generator, bandwidth,
                               dtype, device):
    """Draw frozen RBF random Fourier frequencies and phases."""
    frequencies = _cpu_random(
        (input_dim, output_count), generator, dtype, device)
    frequencies = frequencies / bandwidth
    phases = 2. * math.pi * _cpu_random(
        (output_count,), generator, dtype, device,
        uniform=True)
    return frequencies, phases


def _evaluate_random_fourier_features(inputs, frequencies, phases):
    """Evaluate one already-fitted random Fourier coordinate system."""
    return math.sqrt(2.) * torch.cos(
        torch.matmul(inputs, frequencies) + phases)


def _frozen_cpu(tensor):
    """Copy a fitted tensor into a portable, gradient-free probe payload."""
    return tensor.detach().cpu().clone()


def fit_one_step_rff_bellman_probe(
        transitions, probe_count=64, discount=0.99, seed=0,
        value_bandwidth=1., cumulant_bandwidth=1.):
    """Fit and freeze an A-only RFF Bellman probe.

    Every population moment and Fourier parameter in the returned dictionary
    is determined solely by ``transitions``.  Evaluation on a later reveal
    batch therefore cannot change the coordinate system fitted on A.
    """
    if probe_count < 1:
        raise ValueError('probe_count must be positive.')
    if not 0. <= discount <= 1.:
        raise ValueError('discount must lie in [0, 1].')
    if value_bandwidth <= 0. or cumulant_bandwidth <= 0.:
        raise ValueError('RFF bandwidths must be positive.')

    with torch.no_grad():
        observations, actions, next_observations, terminals = (
            _transition_tensors(transitions))
        observation_design = torch.cat(
            (observations, next_observations), dim=0)
        observation_mean, observation_scale = _population_moments(
            observation_design)
        action_mean, action_scale = _population_moments(actions)
        current_inputs = (
            observations - observation_mean) / observation_scale
        next_inputs = (
            next_observations - observation_mean) / observation_scale
        action_inputs = (actions - action_mean) / action_scale
        raw_cumulant_inputs = torch.cat((
            current_inputs,
            action_inputs,
            next_inputs,
            terminals.unsqueeze(1),
        ), dim=1)
        cumulant_mean, cumulant_scale = _population_moments(
            raw_cumulant_inputs)

        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed))
        value_frequencies, value_phases = _random_fourier_parameters(
            observations.shape[1], probe_count, generator, value_bandwidth,
            observations.dtype, observations.device)
        cumulant_frequencies, cumulant_phases = (
            _random_fourier_parameters(
                raw_cumulant_inputs.shape[1], probe_count, generator,
                cumulant_bandwidth, observations.dtype, observations.device))

    return {
        'format_version': 1,
        'definition': 'a_only_one_step_rff_bellman_probe',
        'probe_count': int(probe_count),
        'observation_dim': int(observations.shape[1]),
        'action_dim': int(actions.shape[1]),
        'discount': float(discount),
        'seed': int(seed),
        'value_bandwidth': float(value_bandwidth),
        'cumulant_bandwidth': float(cumulant_bandwidth),
        'observation_mean': _frozen_cpu(observation_mean),
        'observation_scale': _frozen_cpu(observation_scale),
        'action_mean': _frozen_cpu(action_mean),
        'action_scale': _frozen_cpu(action_scale),
        'cumulant_mean': _frozen_cpu(cumulant_mean),
        'cumulant_scale': _frozen_cpu(cumulant_scale),
        'value_frequencies': _frozen_cpu(value_frequencies),
        'value_phases': _frozen_cpu(value_phases),
        'cumulant_frequencies': _frozen_cpu(cumulant_frequencies),
        'cumulant_phases': _frozen_cpu(cumulant_phases),
    }


def _probe_tensor(probe, key, reference):
    """Move one frozen probe tensor to the reveal batch without refitting it."""
    return probe[key].to(device=reference.device, dtype=reference.dtype)


def evaluate_one_step_rff_bellman_probe(
        transitions, probe, discount=None, return_components=False):
    """Evaluate a fitted A-only probe on B without recomputing any moments."""
    with torch.no_grad():
        observations, actions, next_observations, terminals = (
            _transition_tensors(transitions))
        if observations.shape[1] != probe['observation_dim']:
            raise ValueError('Reveal observation dimension differs from A.')
        if actions.shape[1] != probe['action_dim']:
            raise ValueError('Reveal action dimension differs from A.')
        if discount is None:
            discount = probe['discount']
        if not 0. <= discount <= 1.:
            raise ValueError('discount must lie in [0, 1].')

        observation_mean = _probe_tensor(
            probe, 'observation_mean', observations)
        observation_scale = _probe_tensor(
            probe, 'observation_scale', observations)
        action_mean = _probe_tensor(probe, 'action_mean', observations)
        action_scale = _probe_tensor(probe, 'action_scale', observations)
        current_inputs = (
            observations - observation_mean) / observation_scale
        next_inputs = (
            next_observations - observation_mean) / observation_scale
        action_inputs = (actions - action_mean) / action_scale
        raw_cumulant_inputs = torch.cat((
            current_inputs,
            action_inputs,
            next_inputs,
            terminals.unsqueeze(1),
        ), dim=1)
        cumulant_inputs = (
            raw_cumulant_inputs -
            _probe_tensor(probe, 'cumulant_mean', observations)) / (
                _probe_tensor(probe, 'cumulant_scale', observations))

        value_frequencies = _probe_tensor(
            probe, 'value_frequencies', observations)
        value_phases = _probe_tensor(probe, 'value_phases', observations)
        cumulant_frequencies = _probe_tensor(
            probe, 'cumulant_frequencies', observations)
        cumulant_phases = _probe_tensor(
            probe, 'cumulant_phases', observations)
        value_current = _evaluate_random_fourier_features(
            current_inputs, value_frequencies, value_phases)
        value_next = _evaluate_random_fourier_features(
            next_inputs, value_frequencies, value_phases)
        cumulants = _evaluate_random_fourier_features(
            cumulant_inputs, cumulant_frequencies, cumulant_phases)
        directions = (
            cumulants + discount * (1. - terminals).unsqueeze(1) * value_next -
            value_current).detach()

    if return_components:
        return directions, {
            'value_current': value_current.detach(),
            'value_next': value_next.detach(),
            'cumulants': cumulants.detach(),
            'nonterminal': (1. - terminals).detach(),
        }
    return directions


def one_step_rff_bellman_direction_bank(
        transitions, probe_count=64, discount=0.99, seed=0,
        value_bandwidth=1., cumulant_bandwidth=1.,
        return_components=False):
    """Construct a frozen one-step RFF Bellman residual bank.

    Args:
        transitions (Mapping[str, torch.Tensor]): Transition tensors with keys
            ``observation``, ``action``, ``next_observation``, and ``terminal``.
            Rewards are intentionally not used: the RFF cumulants describe a
            prospective family of reward signals without future-task data.
        probe_count (int): Number of paired value/cumulant probes.
        discount (float): One-step Bellman discount.
        seed (int): Local seed used only for the frozen Fourier probes.
        value_bandwidth (float): RBF bandwidth for value probes.
        cumulant_bandwidth (float): RBF bandwidth for cumulant probes.
        return_components (bool): Also return the value and cumulant matrices
            used to assemble the residuals.

    Returns:
        torch.Tensor: A detached ``[transition_count, probe_count]`` residual
            matrix.  If ``return_components`` is true, returns this tensor and
            a dictionary containing ``value_current``, ``value_next``, and
            ``cumulants``.
    """
    probe = fit_one_step_rff_bellman_probe(
        transitions, probe_count=probe_count, discount=discount, seed=seed,
        value_bandwidth=value_bandwidth,
        cumulant_bandwidth=cumulant_bandwidth)
    return evaluate_one_step_rff_bellman_probe(
        transitions, probe, return_components=return_components)


def center_and_normalize_directions(directions, tolerance=None):
    """Center residual columns, discard constants, and give each unit norm."""
    if directions.ndim != 2:
        raise ValueError('directions must be a rank-two matrix.')
    centered = directions - directions.mean(dim=0, keepdim=True)
    norms = torch.linalg.norm(centered, dim=0)
    if tolerance is None:
        scale = directions.detach().abs().max().clamp_min(1.)
        tolerance = (
            torch.finfo(directions.dtype).eps *
            math.sqrt(directions.shape[0]) * scale)
    valid = norms > tolerance
    if not bool(valid.any()):
        raise ValueError('No nonconstant Bellman direction remains.')
    return centered[:, valid] / norms[valid].unsqueeze(0)


def bellman_reachable_basis(directions, rank_rtol=None):
    """Return the left-SVD basis, singular values, and numerical rank.

    Columns are centered and normalized before the SVD.  With N transition
    rows, the returned rank is therefore at most N - 1.
    """
    normalized = center_and_normalize_directions(directions)
    left_vectors, singular_values, _ = torch.linalg.svd(
        normalized, full_matrices=False)
    if rank_rtol is None:
        rank_rtol = (
            max(normalized.shape) * torch.finfo(normalized.dtype).eps)
    if rank_rtol < 0.:
        raise ValueError('rank_rtol must be nonnegative.')
    threshold = singular_values[0] * rank_rtol
    rank = int((singular_values > threshold).sum().item())
    basis = left_vectors[:, :rank]
    return basis, singular_values, rank


def bellman_subspace_coverage(directions, basis):
    """Return the fraction of centered direction energy captured by ``basis``.

    ``basis`` is expected to have orthonormal columns, as returned by
    :func:`bellman_reachable_basis`.
    """
    if directions.ndim != 2 or basis.ndim != 2:
        raise ValueError('directions and basis must be rank-two matrices.')
    if directions.shape[0] != basis.shape[0]:
        raise ValueError('directions and basis must have equal row counts.')
    centered = directions - directions.mean(dim=0, keepdim=True)
    total_energy = centered.square().sum()
    if not bool(total_energy > 0.):
        raise ValueError('Coverage is undefined for constant directions.')
    coefficients = torch.matmul(basis.t(), centered)
    return coefficients.square().sum() / total_energy


def trace_normalized_reachable_tangent_spectrum(jacobian, basis):
    """Return tangent eigenvalues inside a Bellman-reachable basis.

    The empirical full-parameter tangent kernel is ``K = J_c J_c^T / p``,
    where ``J_c`` centers Jacobian rows.  It is divided by
    ``Tr(K) / (N - 1)`` so the mean eigenvalue in the centered anchor-output
    space is one.  The returned spectrum is that of ``U^T K_normalized U``.
    """
    if jacobian.ndim != 2 or basis.ndim != 2:
        raise ValueError('jacobian and basis must be rank-two matrices.')
    if jacobian.shape[0] != basis.shape[0]:
        raise ValueError('jacobian and basis must have equal row counts.')
    if jacobian.shape[0] < 2 or jacobian.shape[1] < 1:
        raise ValueError('jacobian must contain at least two rows and one column.')
    if basis.shape[1] < 1:
        raise ValueError('basis must contain at least one direction.')

    centered_jacobian = jacobian - jacobian.mean(dim=0, keepdim=True)
    kernel = torch.matmul(centered_jacobian, centered_jacobian.t())
    kernel = kernel / jacobian.shape[1]
    kernel = 0.5 * (kernel + kernel.t())
    centered_dimension = jacobian.shape[0] - 1
    mean_eigenvalue = torch.trace(kernel) / centered_dimension
    if not bool(mean_eigenvalue > 0.):
        raise ValueError('The centered Jacobian has zero tangent energy.')
    normalized_kernel = kernel / mean_eigenvalue
    reachable_kernel = torch.matmul(
        basis.t(), torch.matmul(normalized_kernel, basis))
    reachable_kernel = 0.5 * (reachable_kernel + reachable_kernel.t())
    return torch.linalg.eigvalsh(reachable_kernel).clamp_min(0.)


def trace_normalized_reachable_tangent_margin(jacobian, basis):
    """Return the slowest trace-normalized tangent eigenvalue reachable by U."""
    spectrum = trace_normalized_reachable_tangent_spectrum(jacobian, basis)
    return spectrum[0]
