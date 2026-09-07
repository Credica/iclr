"""Critic-only regularizers for the PPO method comparison."""

import torch

from garage.torch.algos.ppo_pbsr_v2 import (
    backbone_gradient_ratio_coefficient,
)


def actual_demand_inverse_burden(head_tangent, residual, ridge=1e-3):
    """Measure the adaptation burden of the current value residual.

    ``head_tangent`` contains the current value features and the output-bias
    column.  The residual is detached because it defines the demand being
    fitted rather than an additional value-error gradient.
    """
    demand = residual.flatten().detach()
    kernel = torch.matmul(head_tangent, head_tangent.t())
    regularized_kernel = kernel + ridge * torch.eye(
        kernel.shape[0], dtype=kernel.dtype, device=kernel.device)
    cholesky = torch.linalg.cholesky(regularized_kernel)
    solved = torch.cholesky_solve(demand[:, None], cholesky).flatten()
    objective = ridge * torch.dot(demand, solved) / demand.shape[0]
    diagnostics = {
        'loss': float(objective.detach()),
        'residual_mean': float(demand.mean()),
        'residual_mean_square': float(demand.square().mean()),
        'kernel_trace': float(torch.trace(kernel).detach()),
        'kernel_mean_eigenvalue': float(
            (torch.trace(kernel) / kernel.shape[0]).detach()),
        'anchor_rows': int(head_tangent.shape[0]),
        'tangent_columns': int(head_tangent.shape[1]),
    }
    return objective, diagnostics


def layer_spectral_regularizer(layer_parameters):
    """Evaluate the k=2 spectral regularizer on linear layers."""
    top_singular_values = torch.stack([
        torch.linalg.svdvals(weight)[0]
        for weight, _ in layer_parameters
    ])
    biases = tuple(bias for _, bias in layer_parameters)
    weight_penalty = (
        top_singular_values.square() - 1.).square().sum()
    bias_penalty = torch.stack([
        bias.square().sum() for bias in biases
    ]).sum()
    objective = weight_penalty + bias_penalty
    diagnostics = {
        'loss': float(objective.detach()),
        'weight_penalty': float(weight_penalty.detach()),
        'bias_penalty': float(bias_penalty.detach()),
        'mean_top_singular_value': float(
            top_singular_values.mean().detach()),
        'max_top_singular_value': float(
            top_singular_values.max().detach()),
        'min_top_singular_value': float(
            top_singular_values.min().detach()),
        'regularized_layers': int(top_singular_values.shape[0]),
    }
    return objective, diagnostics


def _value_backbone_parameters(value_function):
    return value_function.module._mean_module._layers.parameters()


def _value_mean_parameters(value_function):
    return value_function.module._mean_module.parameters()


def _value_linear_layer_parameters(value_function):
    mean_module = value_function.module._mean_module
    layers = tuple(mean_module._layers) + tuple(mean_module._output_layers)
    return tuple((layer._modules['linear'].weight,
                  layer._modules['linear'].bias) for layer in layers)


class PPOActualDemandBurden:
    """Regularize value geometry for the observed return residual."""

    def __init__(self, anchor_size=64, ridge=1e-3, gradient_ratio=0.1):
        self.anchor_size = int(anchor_size)
        self.ridge = float(ridge)
        self.gradient_ratio = float(gradient_ratio)
        self.train_actor = False

    def value_loss(self, value_function, observations, returns, task_idx,
                   primary_loss, compute_diagnostics=False):
        del compute_diagnostics
        rows = min(self.anchor_size, observations.shape[0])
        observations = observations[:rows]
        returns = returns[:rows].flatten()

        values = value_function(observations, seq_idx=task_idx).flatten()
        features = value_function._feature
        bias_tangent = torch.ones(
            rows, 1, dtype=features.dtype, device=features.device)
        head_tangent = torch.cat((features, bias_tangent), dim=1)
        regularizer, diagnostics = actual_demand_inverse_burden(
            head_tangent, returns - values, self.ridge)
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
            'geometry': 'value_linear_head_actual_return_demand',
        })
        return total_loss, diagnostics


class PPOValueLayerSpectralRegularizer:
    """Apply standard layer spectral regularization to the value network."""

    def __init__(self, gradient_ratio=0.1):
        self.gradient_ratio = float(gradient_ratio)
        self.train_actor = False

    def value_loss(self, value_function, primary_loss,
                   compute_diagnostics=False):
        del compute_diagnostics
        regularizer, diagnostics = layer_spectral_regularizer(
            _value_linear_layer_parameters(value_function))
        coefficient, gradient_stats = backbone_gradient_ratio_coefficient(
            primary_loss, regularizer,
            _value_mean_parameters(value_function), self.gradient_ratio)
        diagnostics.update(gradient_stats)
        total_loss = primary_loss + coefficient * regularizer
        diagnostics.update({
            'primary_loss': float(primary_loss.detach()),
            'weighted_regularizer': float(
                (coefficient * regularizer).detach()),
            'total_loss': float(total_loss.detach()),
            'geometry': 'value_all_linear_layer_spectral_norm_and_bias',
        })
        return total_loss, diagnostics
