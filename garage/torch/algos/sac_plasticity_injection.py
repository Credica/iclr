"""Bellman-burden width selection for SAC plasticity injection."""

import torch

from garage.torch.algos.bellman_spectral_stats import empirical_jacobian


def _module_jacobian(module, inputs):
    parameters = module.trainable_parameters()
    rows = []
    for row_idx in range(inputs.shape[0]):
        value = module(inputs[row_idx:row_idx + 1]).reshape(())
        gradients = torch.autograd.grad(value, parameters)
        rows.append(torch.cat([
            gradient.reshape(-1) for gradient in gradients
        ]))
    return torch.stack(rows, dim=0)


def _kernel(jacobian, parameter_count):
    kernel = torch.matmul(jacobian, jacobian.t()) / parameter_count
    return 0.5 * (kernel + kernel.t())


def inverse_bellman_burden(kernel, directions, ridge):
    """Return Tr[D D^T (K + rho I)^-1] / Tr[D D^T]."""
    identity = torch.eye(
        kernel.shape[0], dtype=kernel.dtype, device=kernel.device)
    inverse_directions = torch.linalg.solve(
        kernel + ridge * identity, directions)
    return ((directions * inverse_directions).sum() /
            directions.square().sum())


def normalized_inverse_bellman_burden(kernel, directions, relative_ridge):
    """Evaluate inverse burden on the unit-mean-eigenvalue kernel."""
    mean_eigenvalue = torch.trace(kernel) / kernel.shape[0]
    normalized_kernel = kernel / mean_eigenvalue
    burden = inverse_bellman_burden(
        normalized_kernel, directions,
        torch.as_tensor(
            relative_ridge, dtype=kernel.dtype, device=kernel.device))
    return burden, mean_eigenvalue


def select_plasticity_width(
        qf1, qf2, fresh_qf1, fresh_qf2, branch1, branch2,
        observations, actions, task_idx, q1_directions, q2_directions,
        candidate_widths, relative_ridge, fixed_width=None):
    """Choose the least width restoring both critic burdens to fresh level."""
    inputs = qf1.plasticity_injection_inputs(
        observations, actions, seq_idx=task_idx)
    critics = (qf1, qf2)
    fresh_critics = (fresh_qf1, fresh_qf2)
    branches = (branch1, branch2)
    directions = (q1_directions, q2_directions)

    base_kernels = []
    base_burdens = []
    fresh_burdens = []
    fresh_mean_eigenvalues = []
    parameter_counts = []
    for critic, fresh_critic, demand in zip(
            critics, fresh_critics, directions):
        parameter_count = sum(
            parameter.numel() for parameter in critic.parameters()
            if parameter.requires_grad)
        base_jacobian = empirical_jacobian(
            critic, observations, actions, task_idx)
        fresh_jacobian = empirical_jacobian(
            fresh_critic, observations, actions, task_idx)
        base_kernel = _kernel(base_jacobian, parameter_count)
        fresh_kernel = _kernel(fresh_jacobian, parameter_count)
        base_burden, _ = normalized_inverse_bellman_burden(
            base_kernel, demand, relative_ridge)
        fresh_burden, fresh_mean_eigenvalue = (
            normalized_inverse_bellman_burden(
                fresh_kernel, demand, relative_ridge))

        parameter_counts.append(parameter_count)
        base_kernels.append(base_kernel)
        base_burdens.append(base_burden)
        fresh_burdens.append(fresh_burden)
        fresh_mean_eigenvalues.append(fresh_mean_eigenvalue)

    widths = ([int(fixed_width)] if fixed_width is not None else
              sorted(int(width) for width in candidate_widths))
    rows = []
    for width in widths:
        critic_burdens = []
        burden_ratios = []
        kernel_traces = []
        mean_eigenvalues = []
        for branch, base_kernel, parameter_count, demand, fresh_burden \
                in zip(branches, base_kernels, parameter_counts, directions,
                       fresh_burdens):
            branch.set_active_width(width)
            branch_jacobian = _module_jacobian(branch, inputs)
            candidate_kernel = (
                base_kernel + _kernel(branch_jacobian, parameter_count))
            burden, mean_eigenvalue = normalized_inverse_bellman_burden(
                candidate_kernel, demand, relative_ridge)
            critic_burdens.append(float(burden.detach()))
            burden_ratios.append(float((burden / fresh_burden).detach()))
            kernel_traces.append(float(torch.trace(candidate_kernel).detach()))
            mean_eigenvalues.append(float(mean_eigenvalue.detach()))
        rows.append({
            'width': width,
            'qf1_burden': critic_burdens[0],
            'qf2_burden': critic_burdens[1],
            'qf1_to_fresh_ratio': burden_ratios[0],
            'qf2_to_fresh_ratio': burden_ratios[1],
            'worst_to_fresh_ratio': max(burden_ratios),
            'qf1_kernel_trace': kernel_traces[0],
            'qf2_kernel_trace': kernel_traces[1],
            'qf1_mean_eigenvalue': mean_eigenvalues[0],
            'qf2_mean_eigenvalue': mean_eigenvalues[1],
        })

    selected_width = widths[0]
    if fixed_width is None:
        selected_width = min(
            rows, key=lambda row: row['worst_to_fresh_ratio'])['width']
        for row in rows:
            if row['worst_to_fresh_ratio'] <= 1.:
                selected_width = row['width']
                break

    branch1.set_active_width(selected_width)
    branch2.set_active_width(selected_width)
    return {
        'selected_width': selected_width,
        'selection_rule': (
            'fixed_width' if fixed_width is not None else
            'minimum_width_reaching_fresh_else_minimum_worst_burden_ratio'),
        'base_qf1_burden': float(base_burdens[0].detach()),
        'base_qf2_burden': float(base_burdens[1].detach()),
        'fresh_qf1_burden': float(fresh_burdens[0].detach()),
        'fresh_qf2_burden': float(fresh_burdens[1].detach()),
        'fresh_qf1_mean_eigenvalue': float(
            fresh_mean_eigenvalues[0].detach()),
        'fresh_qf2_mean_eigenvalue': float(
            fresh_mean_eigenvalues[1].detach()),
        'candidates': rows,
    }
