"""Adam with a low-rank Bellman-conditioned metric on value updates."""

import math

import torch
from torch.optim import Optimizer


class BellmanBurdenAdam(Optimizer):
    """Adam whose final preconditioned direction can be rotated in low rank.

    The first- and second-moment states are updated from the unmodified
    gradient.  The inherited Adam second moment supplies the diagonal metric.
    By default, the low-rank metric is applied only to the current gradient
    innovation; historical first-moment velocity follows vanilla Adam.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, norm_match=True,
                 update_mode='innovation'):
        if lr <= 0.0:
            raise ValueError('Learning rate must be positive.')
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError('Adam betas must lie in [0, 1).')
        if eps <= 0.0:
            raise ValueError('Adam epsilon must be positive.')
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        if update_mode not in ('innovation', 'full_moment'):
            raise ValueError(
                'BOLT update_mode must be innovation or full_moment.')
        self._bolt_basis = None
        self._bolt_metric = None
        self._norm_match = bool(norm_match)
        self._update_mode = update_mode
        self.last_step_stats = {}

    def _ordered_parameters(self):
        return [parameter for group in self.param_groups
                for parameter in group['params']]

    @staticmethod
    def _initialize_state(state, parameter):
        if state:
            return
        state['step'] = 0
        state['first_moment_step'] = 0
        state['exp_avg'] = torch.zeros_like(parameter)
        state['exp_avg_sq'] = torch.zeros_like(parameter)

    def reset_first_moment(self):
        """Discard optimization velocity while retaining Adam scale state."""
        for parameter in self._ordered_parameters():
            state = self.state[parameter]
            if state:
                state['exp_avg'].zero_()
                state['first_moment_step'] = 0

    def clear_bolt_metric(self):
        """Use the unmodified Adam metric."""
        self._bolt_basis = None
        self._bolt_metric = None

    def set_bolt_metric(self, basis, active_metric):
        """Install an orthonormal basis and its SPD active metric."""
        self._bolt_basis = basis.detach()
        self._bolt_metric = active_metric.detach()

    @torch.no_grad()
    def preview_preconditioner(self):
        """Return the diagonal Adam preconditioner used by the next step."""
        pieces = []
        for group in self.param_groups:
            beta2 = group['betas'][1]
            eps = group['eps']
            weight_decay = group['weight_decay']
            for parameter in group['params']:
                state = self.state[parameter]
                self._initialize_state(state, parameter)
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                elif weight_decay != 0.0:
                    gradient = gradient.add(parameter, alpha=weight_decay)
                next_second = state['exp_avg_sq'].mul(beta2).addcmul(
                    gradient, gradient, value=1.0 - beta2)
                next_step = state['step'] + 1
                correction = 1.0 - beta2 ** next_step
                denominator = next_second.sqrt().div_(
                    math.sqrt(correction)).add_(eps)
                pieces.append(denominator.reciprocal().reshape(-1))
        return torch.cat(pieces)

    @torch.no_grad()
    def step(self, closure=None):
        """Take one Adam step and apply the current BOLT metric."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        records = []
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError(
                        'BellmanBurdenAdam does not support sparse gradients.')
                if weight_decay != 0.0:
                    gradient = gradient.add(parameter, alpha=weight_decay)

                state = self.state[parameter]
                self._initialize_state(state, parameter)
                state['step'] += 1
                state['first_moment_step'] += 1
                first_correction = 1.0 - beta1 ** state['first_moment_step']
                historical = beta1 * state['exp_avg'] / first_correction
                innovation = (
                    (1.0 - beta1) * gradient / first_correction)
                state['exp_avg'].mul_(beta1).add_(
                    gradient, alpha=1.0 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2)

                second_correction = 1.0 - beta2 ** state['step']
                first = state['exp_avg'] / first_correction
                denominator = state['exp_avg_sq'].sqrt().div(
                    math.sqrt(second_correction)).add(eps)
                preconditioner = denominator.reciprocal()
                records.append((parameter, first, historical, innovation,
                                preconditioner, group['lr']))

        if not records:
            return loss

        base_direction = torch.cat([
            (preconditioner * first).reshape(-1)
            for _, first, _, _, preconditioner, _ in records
        ])
        direction = base_direction
        if self._bolt_basis is not None:
            sqrt_preconditioner = torch.cat([
                preconditioner.sqrt().reshape(-1)
                for _, _, _, _, preconditioner, _ in records
            ])
            if self._update_mode == 'innovation':
                historical = torch.cat([
                    value.reshape(-1) for _, _, value, _, _, _ in records
                ])
                metric_input = torch.cat([
                    value.reshape(-1) for _, _, _, value, _, _ in records
                ])
            else:
                historical = None
                metric_input = torch.cat([
                    value.reshape(-1) for _, value, _, _, _, _ in records
                ])

            whitened = sqrt_preconditioner * metric_input
            coordinates = torch.matmul(self._bolt_basis.t(), whitened)
            rotated = torch.matmul(self._bolt_metric, coordinates)
            correction = torch.matmul(
                self._bolt_basis, rotated - coordinates)
            rotated_direction = (
                sqrt_preconditioner * (whitened + correction))
            if self._norm_match:
                base_metric_direction = preconditioner * metric_input
                rotated_direction = rotated_direction * (
                    base_metric_direction.norm() /
                    rotated_direction.norm())
            if self._update_mode == 'innovation':
                direction = preconditioner * historical + rotated_direction
            else:
                direction = rotated_direction

        offset = 0
        for parameter, _, _, _, _, learning_rate in records:
            size = parameter.numel()
            update = direction[offset:offset + size].view_as(parameter)
            parameter.add_(update, alpha=-learning_rate)
            offset += size

        self.last_step_stats = {
            'bolt_active': self._bolt_basis is not None,
            'update_mode': self._update_mode,
            'base_direction_norm': float(base_direction.norm()),
            'applied_direction_norm': float(direction.norm()),
        }
        return loss
