"""Spectral regularization utilities for continual SAC.

The objective follows Lewandowski et al., "Learning Continually by
Spectral Regularization" (ICLR 2025).  For exponent k=2, every affine layer
contributes

    (sigma_max(W) ** 2 - 1) ** 2 + ||b||_2 ** 4.

The leading singular value is estimated with stateful power iteration.  The
power vectors are detached from autograd; the resulting Rayleigh quotient is
still differentiable with respect to the weight matrix.
"""

from collections import OrderedDict
import hashlib

import torch
from torch import nn
from torch.nn import functional as F


class LayerSpectralRegularizer:
    """Compute the k=2 layerwise spectral regularization objective."""

    def __init__(self, power_iterations=1, eps=1e-12):
        if power_iterations < 1:
            raise ValueError('power_iterations must be positive.')
        if eps <= 0:
            raise ValueError('eps must be positive.')
        self.power_iterations = int(power_iterations)
        self.eps = float(eps)
        self._left_vectors = OrderedDict()

    def _initial_left_vector(self, weight, key):
        # Use a local, stable seed so initialization neither consumes the
        # experiment RNG nor risks the systematic cancellation of all-ones.
        seed = int.from_bytes(
            hashlib.sha256(key.encode('utf-8')).digest()[:8], 'little')
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
        vector = torch.randn(weight.shape[0], generator=generator)
        vector = vector.to(dtype=weight.dtype, device=weight.device)
        return F.normalize(vector, dim=0, eps=self.eps)

    def _top_singular_value(self, weight, key):
        matrix = weight.reshape(weight.shape[0], -1)
        left = self._left_vectors.get(key)
        if (left is None or left.shape[0] != matrix.shape[0] or
                left.dtype != matrix.dtype or left.device != matrix.device):
            left = self._initial_left_vector(matrix, key)

        with torch.no_grad():
            for _ in range(self.power_iterations):
                right = F.normalize(
                    torch.mv(matrix.detach().t(), left),
                    dim=0, eps=self.eps)
                left = F.normalize(
                    torch.mv(matrix.detach(), right),
                    dim=0, eps=self.eps)
            self._left_vectors[key] = left.detach().clone()

        # Detach the updated power vectors while retaining a differentiable
        # Rayleigh quotient with respect to the weight matrix.
        return torch.dot(left.detach(), torch.mv(matrix, right.detach()))

    def loss(self, network, namespace, compute_diagnostics=True,
             module_filter=None):
        """Return the summed regularizer and lightweight diagnostics."""
        layer_losses = []
        singular_values = []
        weight_penalties = []
        bias_penalties = []

        for name, module in network.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if module_filter is not None and not module_filter(name, module):
                continue
            key = '{}.{}'.format(namespace, name)
            sigma = self._top_singular_value(module.weight, key)
            weight_penalty = (sigma.square() - 1.).square()
            if module.bias is None:
                bias_penalty = weight_penalty.new_zeros(())
            else:
                # For k=2, (||b||_2 ** k) ** 2 = ||b||_2 ** 4.
                bias_penalty = module.bias.square().sum().square()
            layer_losses.append(weight_penalty + bias_penalty)
            if compute_diagnostics:
                singular_values.append(sigma.detach())
                weight_penalties.append(weight_penalty.detach())
                bias_penalties.append(bias_penalty.detach())

        if not layer_losses:
            raise ValueError(
                'Spectral regularization requires at least one nn.Linear layer.')

        objective = torch.stack(layer_losses).sum()
        if not compute_diagnostics:
            return objective, {}
        singular_values = torch.stack(singular_values)
        diagnostics = {
            'loss': float(objective.detach()),
            'weight_penalty': float(torch.stack(weight_penalties).sum()),
            'bias_penalty': float(torch.stack(bias_penalties).sum()),
            'mean_top_singular_value': float(singular_values.mean()),
            'max_top_singular_value': float(singular_values.max()),
            'min_top_singular_value': float(singular_values.min()),
            'regularized_layers': int(singular_values.numel()),
        }
        return objective, diagnostics

    def state_dict(self):
        """Return power-iteration state for exact checkpoint continuation."""
        return {
            'power_iterations': self.power_iterations,
            'eps': self.eps,
            'left_vectors': {
                key: value.detach().cpu().clone()
                for key, value in self._left_vectors.items()
            },
        }

    def load_state_dict(self, state):
        """Restore power-iteration state; vectors move lazily to each device."""
        if int(state['power_iterations']) != self.power_iterations:
            raise ValueError('Checkpoint power_iterations does not match.')
        self._left_vectors = OrderedDict(
            (key, value.detach().clone())
            for key, value in state['left_vectors'].items())
