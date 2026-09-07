"""Bellman-burden optimizer geometry for PPO value learning."""

import torch


def build_multistep_value_targets(rewards, next_values, terminals,
                                  path_ends, returns, horizons, discount):
    """Build bootstrapped h-step and Monte Carlo targets on one rollout."""
    rewards = rewards.flatten()
    next_values = next_values.flatten()
    terminals = terminals.flatten().bool()
    path_ends = path_ends.flatten().bool()
    one_step = rewards + discount * (~terminals).to(rewards.dtype) * next_values

    requested = sorted(set(int(horizon) for horizon in horizons))
    if not requested or requested[0] < 1:
        raise ValueError('PPO BOLT horizons must be positive integers.')

    targets = {}
    previous = one_step
    if 1 in requested:
        targets[1] = previous
    for horizon in range(2, requested[-1] + 1):
        current = one_step.clone()
        continues = (~path_ends[:-1]) & (~terminals[:-1])
        current[:-1][continues] = (
            rewards[:-1][continues] +
            discount * previous[1:][continues])
        previous = current
        if horizon in requested:
            targets[horizon] = current

    columns = [targets[horizon] for horizon in requested]
    columns.append(returns.flatten())
    return torch.stack(columns, dim=1).detach()


class PPOBellmanBurdenController:
    """Track real value demands and rotate Adam's active learning axes."""

    def __init__(self, value_function, optimizer, rank=4, rho=0.5,
                 ridge=0.1, calibration_size=32, update_interval=1000,
                 history_columns=24, active_tasks=None,
                 reset_first_moment=False):
        if rank <= 0 or rank % 2:
            raise ValueError('PPO BOLT rank must be a positive even integer.')
        if rho < 0.0 or rho >= 1.0:
            raise ValueError('PPO BOLT rho must lie in [0, 1).')
        self.value_function = value_function
        self.optimizer = optimizer
        self.rank = int(rank)
        self.rho = float(rho)
        self.ridge = float(ridge)
        self.calibration_size = int(calibration_size)
        self.update_interval = int(update_interval)
        self.history_columns = int(history_columns)
        self.active_tasks = (
            None if active_tasks is None
            else set(int(task) for task in active_tasks))
        self.reset_first_moment = bool(reset_first_moment)
        self._raw_pullbacks = None
        self._task_step = 0
        self._task_idx = 0
        self.last_stats = {}

    @property
    def active(self):
        return (self.active_tasks is None or
                self._task_idx in self.active_tasks)

    def start_task(self, task_idx):
        """Start a task with no inherited velocity or demand statistics."""
        self._task_idx = int(task_idx)
        self._task_step = 0
        self._raw_pullbacks = None
        self.last_stats = {}
        if self.reset_first_moment:
            self.optimizer.reset_first_moment()
        self.optimizer.clear_bolt_metric()

    def step_complete(self):
        self._task_step += 1

    @staticmethod
    def _flatten_gradients(outputs, parameters):
        rows = []
        for row_idx in range(outputs.shape[0]):
            gradients = torch.autograd.grad(
                outputs[row_idx], parameters,
                retain_graph=row_idx + 1 < outputs.shape[0],
                allow_unused=True)
            rows.append(torch.cat([
                (torch.zeros_like(parameter) if gradient is None else gradient)
                .reshape(-1)
                for parameter, gradient in zip(parameters, gradients)
            ]))
        return torch.stack(rows)

    @staticmethod
    def _burden(kernel, covariance, ridge):
        rows = kernel.shape[0]
        mean_eigenvalue = torch.trace(kernel) / rows
        normalized = kernel / mean_eigenvalue
        regularized = normalized + ridge * torch.eye(
            rows, dtype=kernel.dtype, device=kernel.device)
        return torch.trace(torch.linalg.solve(regularized, covariance))

    def _value_jacobian(self, observations, task_idx, parameters):
        values = self.value_function(
            observations, seq_idx=task_idx).flatten()
        jacobian = self._flatten_gradients(values, parameters).detach()
        return values.detach(), jacobian

    @staticmethod
    def _demand_statistics(target_bank, values):
        residuals = target_bank.detach() - values.unsqueeze(1)
        constant = residuals.mean(dim=0, keepdim=True).expand_as(residuals)
        shape = residuals - constant
        demands = torch.cat((constant, shape), dim=1)
        covariance = torch.matmul(demands, demands.t())
        covariance = covariance / torch.trace(covariance)
        return demands, covariance

    def maybe_refresh(self, observations, target_bank, task_idx):
        """Refresh the low-rank optimizer metric on a current minibatch."""
        self._task_idx = int(task_idx)
        if not self.active:
            self.optimizer.clear_bolt_metric()
            return None
        if self._task_step % self.update_interval:
            return None

        calibration_rows = min(
            self.calibration_size, observations.shape[0] // 2)
        heldout_rows = calibration_rows
        calibration_observations = observations[:calibration_rows]
        calibration_targets = target_bank[:calibration_rows]
        heldout_observations = observations[
            calibration_rows:calibration_rows + heldout_rows]
        heldout_targets = target_bank[
            calibration_rows:calibration_rows + heldout_rows]
        parameters = [parameter for parameter in
                      self.value_function.parameters()
                      if parameter.requires_grad]

        calibration_values, calibration_jacobian = self._value_jacobian(
            calibration_observations, task_idx, parameters)
        heldout_values, heldout_jacobian = self._value_jacobian(
            heldout_observations, task_idx, parameters)
        calibration_demands, calibration_covariance = (
            self._demand_statistics(
                calibration_targets, calibration_values))
        heldout_demands, heldout_covariance = self._demand_statistics(
            heldout_targets, heldout_values)

        raw_pullbacks = torch.matmul(
            calibration_jacobian.t(), calibration_demands)
        if self._raw_pullbacks is None:
            self._raw_pullbacks = raw_pullbacks
        else:
            self._raw_pullbacks = torch.cat(
                (self._raw_pullbacks, raw_pullbacks), dim=1)
            self._raw_pullbacks = self._raw_pullbacks[
                :, -self.history_columns:]

        preconditioner = self.optimizer.preview_preconditioner().detach()
        sqrt_preconditioner = preconditioner.sqrt()
        whitened_history = (
            sqrt_preconditioner.unsqueeze(1) * self._raw_pullbacks)
        left, singular_values, _ = torch.svd(whitened_history)
        tolerance = singular_values[0] * max(whitened_history.shape) * 1e-6
        history_rank = int((singular_values > tolerance).sum())
        if history_rank < self.rank:
            self.optimizer.clear_bolt_metric()
            self.last_stats = {
                'active': False,
                'task': int(task_idx),
                'task_value_step': int(self._task_step),
                'history_rank': history_rank,
                'required_rank': self.rank,
            }
            return self.last_stats

        basis = left[:, :self.rank].contiguous()
        calibration_whitened_jacobian = (
            calibration_jacobian * sqrt_preconditioner.unsqueeze(0))
        calibration_kernel = torch.matmul(
            calibration_whitened_jacobian,
            calibration_whitened_jacobian.t())

        work_kernel = calibration_kernel.double()
        work_covariance = calibration_covariance.double()
        mean_eigenvalue = torch.trace(work_kernel) / calibration_rows
        normalized_kernel = work_kernel / mean_eigenvalue
        identity = torch.eye(
            calibration_rows, dtype=work_kernel.dtype,
            device=work_kernel.device)
        inverse = torch.linalg.inv(
            normalized_kernel + self.ridge * identity)
        weighted_inverse = torch.matmul(
            torch.matmul(inverse, work_covariance), inverse)
        kernel_gradient = (
            -weighted_inverse +
            torch.trace(weighted_inverse @ normalized_kernel) /
            calibration_rows *
            identity) / mean_eigenvalue

        output_basis = torch.matmul(
            calibration_whitened_jacobian, basis).double()
        active_gradient = torch.matmul(
            torch.matmul(output_basis.t(), kernel_gradient), output_basis)
        active_gradient = 0.5 * (
            active_gradient + active_gradient.t())
        gradient_eigenvalues, rotation = torch.linalg.eigh(active_gradient)
        fast_count = self.rank // 2
        metric_eigenvalues = torch.cat((
            torch.full((fast_count,), 1.0 + self.rho,
                       dtype=rotation.dtype, device=rotation.device),
            torch.full((self.rank - fast_count,), 1.0 - self.rho,
                       dtype=rotation.dtype, device=rotation.device),
        ))
        active_metric = rotation @ torch.diag(metric_eigenvalues) @ rotation.t()

        active_delta = active_metric - torch.eye(
            self.rank, dtype=active_metric.dtype,
            device=active_metric.device)
        calibration_rotated_kernel = (
            work_kernel + output_basis @ active_delta @ output_basis.t())
        calibration_burden_before = self._burden(
            work_kernel, work_covariance, self.ridge)
        calibration_burden_after = self._burden(
            calibration_rotated_kernel, work_covariance, self.ridge)

        heldout_whitened_jacobian = (
            heldout_jacobian * sqrt_preconditioner.unsqueeze(0))
        heldout_kernel = torch.matmul(
            heldout_whitened_jacobian, heldout_whitened_jacobian.t()).double()
        heldout_output_basis = torch.matmul(
            heldout_whitened_jacobian, basis).double()
        heldout_rotated_kernel = (
            heldout_kernel +
            heldout_output_basis @ active_delta @ heldout_output_basis.t())
        heldout_burden_before = self._burden(
            heldout_kernel, heldout_covariance.double(), self.ridge)
        heldout_burden_after = self._burden(
            heldout_rotated_kernel, heldout_covariance.double(), self.ridge)

        self.optimizer.set_bolt_metric(
            basis, active_metric.to(dtype=basis.dtype))
        demand_singular_values = torch.linalg.svdvals(
            calibration_demands.double())
        demand_tolerance = (
            demand_singular_values[0] *
            max(calibration_demands.shape) * 1e-6)
        demand_rank = int(
            (demand_singular_values > demand_tolerance).sum())
        self.last_stats = {
            'active': True,
            'task': int(task_idx),
            'task_value_step': int(self._task_step),
            'calibration_rows': int(calibration_rows),
            'heldout_rows': int(heldout_rows),
            'demand_columns': int(calibration_demands.shape[1]),
            'demand_rank': demand_rank,
            'history_rank': history_rank,
            'active_rank': self.rank,
            'rho': self.rho,
            'ridge': self.ridge,
            'calibration_burden_before': float(
                calibration_burden_before),
            'calibration_burden_after': float(
                calibration_burden_after),
            'calibration_burden_ratio': float(
                calibration_burden_after / calibration_burden_before),
            'heldout_burden_before': float(heldout_burden_before),
            'heldout_burden_after': float(heldout_burden_after),
            'heldout_burden_ratio': float(
                heldout_burden_after / heldout_burden_before),
            'kernel_mean_eigenvalue': float(mean_eigenvalue),
            'gradient_eigenvalue_min': float(gradient_eigenvalues[0]),
            'gradient_eigenvalue_max': float(gradient_eigenvalues[-1]),
        }
        return self.last_stats
