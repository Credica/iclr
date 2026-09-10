#!/usr/bin/env python3
"""CPU-only spectrum/demand audit of the archived single-seed ABC run.

No environment calls, online updates, plotting, or modification of source runs.
The eight sin-noise targets match the legacy probe, not independent MC samples.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_rethink_dynamic import NAMES, params_from_state, forward, digest, write_json
from analyze_rethink_subspace import geometry, demand_family, demand_score

TASKS = ('sweep-into-v2', 'push-wall-v2', 'window-close-v2')
FIELDS = ('observation', 'action', 'reward', 'next_observation', 'terminal')
POLICY = '_module._shared_mean_log_std_network.'


def torch_critic(state, observations, actions):
    x = torch.cat((observations, actions), dim=-1)
    for prefix in ('_layers.0.linear', '_layers.1.linear'):
        x = torch.relu(torch.nn.functional.linear(x, state[prefix + '.weight'], state[prefix + '.bias']))
    return torch.nn.functional.linear(x, state[NAMES[4]], state[NAMES[5]]).squeeze(-1)


def policy_parameters(state, observations, task):
    x = observations
    for prefix in ('_layers.0.linear', '_layers.1.linear'):
        x = torch.relu(torch.nn.functional.linear(x, state[POLICY + prefix + '.weight'], state[POLICY + prefix + '.bias']))
    outputs = []
    for head in (2 * task, 2 * task + 1):
        prefix = POLICY + '_output_layers.%d.linear' % head
        outputs.append(torch.nn.functional.linear(x, state[prefix + '.weight'], state[prefix + '.bias']))
    return outputs[0], outputs[1].clamp(float(state['_module.min_std_param']), float(state['_module.max_std_param']))


def legacy_targets(saved, bank, task):
    """Original float32 sin-noise + tanh log-Jacobian epsilon=1e-6 probe."""
    obs = torch.as_tensor(bank['next_observation'], dtype=torch.float32)
    with torch.no_grad():
        loc, log_std = policy_parameters(saved['policy'], obs, task)
        std = log_std.exp()
        axis = torch.arange(1, loc.shape[1] + 1, dtype=torch.float32)[None, :]
        alpha = saved['log_alpha'].exp().reshape(-1)
        alpha = alpha[task] if len(alpha) > 1 else alpha[0]
        rewards = torch.as_tensor(bank['reward'], dtype=torch.float32).flatten()
        bootstrap = .99 * (1 - torch.as_tensor(bank['terminal'], dtype=torch.float32).flatten())
        targets = []
        for index in range(8):
            pre = loc + std * torch.sin((index + 1) * axis)
            action = torch.tanh(pre)
            # Do not substitute the numerically different softplus log-Jacobian.
            logp = torch.distributions.Normal(loc, std).log_prob(pre).sum(-1)
            logp -= torch.log((1 - action.square()).clamp(0, 1) + 1e-6).sum(-1)
            q1 = torch_critic(saved['target_qf1'], obs, action)
            q2 = torch_critic(saved['target_qf2'], obs, action)
            targets.append(rewards + bootstrap * (torch.minimum(q1, q2) - alpha * logp))
        return torch.stack(targets).numpy().astype(np.float64)


def entry_state(saved):
    """Boundary code keeps actor/online critics, resets alpha and syncs targets."""
    result = dict(saved)
    result['target_qf1'], result['target_qf2'] = saved['qf1'], saved['qf2']
    result['log_alpha'] = torch.zeros_like(saved['log_alpha'])
    return result


def weight_metrics(weight):
    singular = np.linalg.svd(weight, compute_uv=False)
    squared = singular ** 2
    probability = squared / squared.sum()
    positive = probability > 0
    return singular, dict(sigma_max=float(singular[0]), sigma_min=float(singular[-1]),
        frobenius_norm=float(np.sqrt(squared.sum())),
        stable_rank=float(squared.sum() / squared[0]),
        entropy_rank=float(np.exp(-np.sum(probability[positive] * np.log(probability[positive])))),
        top1_energy=float(probability[0]),
        condition=float(singular[0] / singular[-1]) if singular[-1] > 0 else None)


def feature_rank(features, centered=False):
    if centered: features = features - features.mean(axis=0)
    values = np.linalg.svd(features, compute_uv=False) ** 2
    total = float(values.sum())
    return int(np.searchsorted(np.cumsum(values), .99 * total) + 1) if total > 1e-30 else 0


def flatten_score(score):
    result = {k: v for k, v in score.items() if not isinstance(v, dict)}
    for k, value in score.items():
        if isinstance(value, dict):
            result.update({k + '_' + name: item for name, item in value.items()})
    return result


def save_csv(path, rows):
    keys = sorted(set(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def analyze(root, output, log_path):
    torch.set_num_threads(1)
    root, output, log_path = root.resolve(), output.resolve(), log_path.resolve()
    assert root != output and root not in output.parents and output not in root.parents
    assert not (output / 'protocol.json').exists(), 'Use a new output directory'
    output.mkdir(parents=True, exist_ok=True)
    run = root / 'sac_transfer_abc_s1_1m'
    checkpoint_dir = run / 'checkpoints'
    paths = sorted(checkpoint_dir.glob('interval_task*_step*.pt'), key=lambda p: int(p.stem.split('step')[1]))
    assert len(paths) == 30
    manifest = json.loads((root / 'source_manifest.json').read_text())
    for name, expected in manifest.items():
        assert digest(root / 'source_snapshot' / name) == expected, name
    config = json.loads((root / 'run_config.json').read_text())
    assert config['task_names'] == list(TASKS) and config['gradient_updates_per_task'] == 1000000
    assert config['seed'] == 1 and config['evaluation_episodes'] == 20
    src = root / 'source_snapshot'
    tracked = paths + [root / 'run_config.json', root / 'source_manifest.json',
        run / 'metrics.jsonl', log_path] + [src / name for name in manifest]
    boundary_paths = [checkpoint_dir / ('task_boundary_task%d_step%d.pt' % (t, (t + 1) * 1000000)) for t in (0, 1)]
    final_path = checkpoint_dir / 'final_task2_step3000000.pt'
    tracked += boundary_paths + [final_path]
    sources = [dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=digest(p)) for p in tracked]
    hashes = {name: digest(Path(__file__).parent / name) for name in
              ('analyze_abc_spectrum.py', 'analyze_rethink_dynamic.py', 'analyze_rethink_subspace.py', 'analyze_rethink_dynamic_geometry.py')}
    write_json(output / 'protocol.json', dict(source=str(root), code_sha256=hashes, seed=1,
        checkpoints=30, clocks='100k critic optimizer updates, NOT environment steps',
        geometry='full uncentered JJ^T/n on fixed original 128-row per-task probe banks; float64 including biases',
        demand='eight deterministic sin-noise soft Bellman residual vectors; original multi-head actor and log_prob epsilon',
        primary_fixed_demand='post-boundary entry demand on saved B/C inputs; alpha=1 and targets=online; retrospective reconstruction',
        trajectory_demand='80 columns from ten checkpoints; descriptive endogenous family, not a full Bellman operator',
        ridges=[.0001, .001, .01], primary_ridge=.001,
        slow='trace-normalized GD eta=.45/n, squared residual retention >=.5 at 1000 updates, not actual Adam modes',
        raw='raw burden = shape burden/mean eigenvalue, with per-kernel relative ridge',
        limits=['single seed', 'no saved initialization', 'no exact-step windows', 'only first 128 probe transitions/task',
                'different B/C distributions and policy heads', 'no matched fresh/reset or Clip branch in this run'],
        no_online_training=True))
    saved = [torch.load(str(p), map_location='cpu') for p in paths]
    final = torch.load(str(final_path), map_location='cpu')
    for name in ('qf1', 'qf2', 'policy', 'target_qf1', 'target_qf2'):
        assert all(torch.equal(v, final[name][k]) for k, v in saved[-1][name].items())
    banks = final['probe_buffers']
    assert set(banks) == {0, 1, 2}
    xs, bank_quality = {}, []
    bank_arrays = {}
    for task in range(3):
        b = banks[task]
        assert set(b) == set(FIELDS)
        assert len(b['observation']) == 128
        assert all(np.isfinite(v).all() for v in b.values())
        xs[task] = np.concatenate((b['observation'], b['action']), axis=1).astype(np.float64)
        unique = int(len(np.unique(xs[task], axis=0)))
        assert unique == 128
        bank_quality.append(dict(task=task, rows=128, unique_inputs=unique,
            terminal_count=int(np.sum(b['terminal'])), reward_min=float(np.min(b['reward'])), reward_max=float(np.max(b['reward'])),
            episode_ids_saved=False, origin='first collected probe transitions; no held-out guarantee'))
        for key in FIELDS: bank_arrays['task%d_%s' % (task, key)] = b[key]
    np.savez_compressed(str(output / 'fixed_inputs.npz'), **bank_arrays)
    write_json(output / 'input_quality.json', bank_quality)
    histories, weight_rows, demands, geometries, arrays, current_targets = [], [], {}, {}, {}, {}
    precision = []
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    for index, snapshot in enumerate(saved):
        step, task = (index + 1) * 100000, index // 10
        assert snapshot['global_step'] == step and snapshot['seq_idx'] == task
        assert snapshot['critic_optimizer_steps'] == (index % 10 + 1) * 100000
        for flag in ('muon', 'singular_clip', 'dsr_v2', 'plasticity_injections', 'demand_aligned_reserve'):
            assert not snapshot.get(flag), flag
        for t, bank in snapshot['probe_buffers'].items():
            for key in FIELDS: np.testing.assert_array_equal(bank[key], banks[t][key])
        target = legacy_targets(snapshot, banks[task], task)
        current_targets[step] = target
        record = next(r for r in metrics if r['event'] == 'interval' and r['global_step'] == step)
        for qi in (1, 2):
            params = params_from_state(snapshot['qf%d' % qi])
            assert [p.shape for p in params] == [(256, 43), (256,), (256, 256), (256,), (1, 256), (1,)]
            for layer, wi in enumerate((0, 2, 4), 1):
                singular, desc = weight_metrics(params[wi])
                weight_rows.append(dict(step=step, task=task, critic=qi, layer=layer, **desc))
                arrays['step%d_q%d_W%d_singular' % (step, qi, layer)] = singular
            for panel in range(3):
                geom = geometry(params, xs[panel])
                geometries[step, panel, qi] = geom
                _, cache = forward(params, xs[panel])
                row = dict(step=step, trained_task=task, input_task=panel, critic=qi,
                    input_available_during_training=panel <= task, **geom['summary'])
                row['kernel_top1_trace_fraction'] = float(geom['eigenvalues'][-1] / geom['eigenvalues'].sum())
                row['feature_rank99_raw'] = feature_rank(cache[2])
                row['feature_rank99_centered'] = feature_rank(cache[2], True)
                row['q_rms'] = float(np.sqrt(np.mean(geom['q'] ** 2)))
                histories.append(row)
                for key in ('q', 'eigenvalues', 'eigenvectors'):
                    arrays['step%d_task%d_q%d_%s' % (step, panel, qi, key)] = geom[key]
            q = geometries[step, task, qi]['q']
            d = target - q[None, :]
            demands[step, qi] = d
            arrays['step%d_q%d_demand' % (step, qi)] = d
            arrays['step%d_targets' % step] = target
            with torch.no_grad():
                q32 = torch_critic(snapshot['qf%d' % qi], torch.as_tensor(banks[task]['observation'], dtype=torch.float32),
                                   torch.as_tensor(banks[task]['action'], dtype=torch.float32)).numpy().astype(np.float64)
            reconstruction = float(np.linalg.norm(q - q32) / max(np.linalg.norm(d) / math.sqrt(8), 1e-30))
            check = dict(step=step, critic=qi, q_error_over_residual=reconstruction)
            assert reconstruction < .005, check
            if qi == 1:
                recorded_td = record['task_td_abs'][str(task)]
                reconstructed_td = float(np.abs(target - q32).mean())
                td_error = abs(reconstructed_td - recorded_td) / max(recorded_td, 1e-30)
                check.update(recorded_td=recorded_td, reconstructed_td=reconstructed_td, td_relative_error=td_error)
                assert td_error < .02, check
            precision.append(check)
        print('CHECKPOINT', step, 'task', task, flush=True)
    entries = {}
    for task, boundary_path in zip((1, 2), boundary_paths):
        boundary = torch.load(str(boundary_path), map_location='cpu')
        previous = saved[task * 10 - 1]
        for name in ('qf1', 'qf2', 'policy'):
            assert all(torch.equal(v, previous[name][k]) for k, v in boundary[name].items())
        virtual = entry_state(boundary)
        target = legacy_targets(virtual, banks[task], task)
        arrays['task%d_entry_targets' % task] = target
        for qi in (1, 2):
            q = geometries[task * 1000000, task, qi]['q']
            entries[task, qi] = target - q[None, :]
            arrays['task%d_q%d_entry_demand' % (task, qi)] = entries[task, qi]
    score_rows = []
    for task in range(3):
        steps = [(task * 10 + i + 1) * 100000 for i in range(10)]
        for qi in (1, 2):
            trajectory = np.concatenate([demands[step, qi] for step in steps])
            families = {'first_checkpoint': demands[steps[0], qi], 'trajectory': trajectory,
                'trajectory_centered': trajectory - trajectory.mean(axis=0)}
            if task in (1, 2):
                families['entry'] = entries[task, qi]
                families['entry_centered'] = entries[task, qi] - entries[task, qi].mean(axis=0)
            for step in steps: families['current_%d' % step] = demands[step, qi]
            for family, d in families.items():
                c, space, desc = demand_family(d)
                arrays['task%d_q%d_%s_second_moment' % (task, qi, family)] = c
                arrays['task%d_q%d_%s_basis95' % (task, qi, family)] = space
                model_steps = steps if family.startswith('current_') else [s['global_step'] for s in saved]
                if family.startswith('current_'): model_steps = [int(family.split('_')[1])]
                eta = min(3e-4, .45 / max(geometries[s, task, qi]['eigenvalues'][-1] for s in model_steps))
                for step in model_steps:
                    score = demand_score(c, space, geometries[step, task, qi], eta)
                    score_rows.append(dict(input_task=task, critic=qi, family=family, kernel_step=step,
                        demand_rank95=desc['rank95'], demand_energy=desc['energy'],
                        mean_vector_energy_fraction=desc.get('mean_vector_energy_fraction'), **flatten_score(score)))
    with log_path.open('rb') as handle: log = pickle.load(handle)
    evaluations = []
    for index in range(30):
        for task in range(3):
            prefix = 'test/%d/%s/' % (task, TASKS[task])
            assert len(log[prefix + 'SuccessRate']) == 30
            evaluations.append(dict(step=(index + 1) * 100000, trained_task=index // 10,
                evaluated_task=task, success=float(log[prefix + 'SuccessRate'][index]),
                mean_return=float(log[prefix + 'AverageReturn'][index]),
                episodes=int(log[prefix + 'NumEpisodes'][index])))
    save_csv(output / 'weight_spectra.csv', weight_rows)
    save_csv(output / 'learning_spectra.csv', histories)
    save_csv(output / 'demand_spectra.csv', score_rows)
    save_csv(output / 'evaluation.csv', evaluations)
    np.savez_compressed(str(output / 'spectral_arrays.npz'), **arrays)
    for record in sources:
        st = Path(record['path']).stat()
        assert (st.st_size, st.st_mtime_ns) == (record['bytes'], record['mtime_ns'])
    write_json(output / 'inputs.json', sources)
    write_json(output / 'precision_checks.json', precision)
    write_json(output / 'complete.json', dict(checkpoints=30, critics=2, fixed_input_panels=3,
        full_jacobian_kernels=len(histories), weight_spectra=len(weight_rows), source_files=len(sources),
        code_sha256=hashes, source_size_mtime_unchanged=True,
        max_q_error_over_residual=max(r['q_error_over_residual'] for r in precision),
        max_legacy_td_relative_error=max(r.get('td_relative_error', 0) for r in precision)))
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output, args.log)
