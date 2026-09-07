"""Spectrum-aware interpolation between PPO GAE and Monte Carlo advantage."""

import torch


def exact_tanh_mlp_ntk(mean_module, observations):
    """Return the exact empirical NTK of a scalar tanh MLP."""
    activations = [observations]
    for layer in mean_module._layers:
        activations.append(torch.tanh(layer.linear(activations[-1])))

    output_layer = mean_module._output_layers[0].linear
    if output_layer.out_features != 1:
        raise ValueError('The PPO value mean network must have scalar output')

    kernel = torch.matmul(activations[-1], activations[-1].t()) + 1.
    delta = output_layer.weight.expand(observations.shape[0], -1)
    delta = delta * (1. - activations[-1].square())

    for layer_idx in range(len(mean_module._layers) - 1, -1, -1):
        previous = activations[layer_idx]
        kernel = kernel + (
            torch.matmul(delta, delta.t()) *
            (torch.matmul(previous, previous.t()) + 1.))
        if layer_idx > 0:
            weight = mean_module._layers[layer_idx].linear.weight
            delta = torch.matmul(delta, weight)
            delta = delta * (1. - activations[layer_idx].square())

    parameter_count = sum(
        parameter.numel() for parameter in mean_module.parameters())
    return kernel / parameter_count


def _spectral_mc_interpolation(kernel, gae, monte_carlo, relative_ridge):
    kernel = 0.5 * (kernel + kernel.t())
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    eigenvalues = eigenvalues.clamp_min(0.)
    mean_eigenvalue = eigenvalues.mean()
    ridge = relative_ridge * mean_eigenvalue
    weights = ridge / (eigenvalues + ridge)
    difference = monte_carlo - gae
    coefficients = torch.matmul(eigenvectors.t(), difference)
    correction = torch.matmul(eigenvectors, weights * coefficients)

    probabilities = eigenvalues / eigenvalues.sum()
    positive = probabilities > 0
    entropy_rank = torch.exp(-(
        probabilities[positive] * probabilities[positive].log()).sum())
    return gae + correction, {
        'mc_weight': weights.mean(),
        'kernel_entropy_rank': entropy_rank,
        'kernel_top1_mass': eigenvalues[-1] / eigenvalues.sum(),
        'correction_norm': correction.norm(),
        'difference_norm': difference.norm(),
    }


def spectral_trust_advantage(value_function, padded_observations, lengths,
                             gae_advantages, monte_carlo_advantages,
                             seq_idx, relative_ridge=0.1, block_size=128):
    """Use MC advantage only in slow value-critic tangent directions."""
    mean_module = value_function.module._mean_module
    corrected_episodes = []
    diagnostics = []

    for episode_idx, length in enumerate(lengths):
        episode_gae = gae_advantages[episode_idx, :length]
        episode_mc = monte_carlo_advantages[episode_idx, :length]
        episode_observations = padded_observations[episode_idx, :length]
        episode_corrected = []
        for start in range(0, int(length), block_size):
            end = min(start + block_size, int(length))
            observations = episode_observations[start:end]
            if value_function._multi_input:
                observations = value_function._zero_pad_per_task[
                    seq_idx](observations)
            with torch.no_grad():
                kernel = exact_tanh_mlp_ntk(mean_module, observations)
                corrected, stats = _spectral_mc_interpolation(
                    kernel, episode_gae[start:end], episode_mc[start:end],
                    relative_ridge)
            episode_corrected.append(corrected)
            diagnostics.append(stats)
        corrected_episodes.append(torch.cat(episode_corrected))

    corrected = torch.cat(corrected_episodes)
    correction_norm = torch.stack([
        item['correction_norm'] for item in diagnostics]).square().sum().sqrt()
    difference_norm = torch.stack([
        item['difference_norm'] for item in diagnostics]).square().sum().sqrt()
    return corrected, {
        'mc_weight': torch.stack([
            item['mc_weight'] for item in diagnostics]).mean().item(),
        'kernel_entropy_rank': torch.stack([
            item['kernel_entropy_rank'] for item in diagnostics]).mean().item(),
        'kernel_top1_mass': torch.stack([
            item['kernel_top1_mass'] for item in diagnostics]).mean().item(),
        'correction_fraction': (
            correction_norm / difference_norm.clamp_min(1e-12)).item(),
        'blocks': len(diagnostics),
    }
