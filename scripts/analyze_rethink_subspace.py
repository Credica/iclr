#!/usr/bin/env python3
"""Offline R3-A/B/C on fixed B inputs; never starts an environment or training.

Primary input: all P1--P6, seeds 1--3, 10k/100k/500k saved windows.
The empirical residual family is not the full Bellman operator's subspace.
"""
import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

# Full boundary snapshots contain garage space/sampler metadata. Make their
# package importable when this file is launched directly from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_rethink_dynamic import (NAMES, params_from_state, forward, ntk,
    gradient, jvp, load_window, recorded_params, digest, write_json, relative_rmse)
from analyze_rethink_dynamic_geometry import spectral_summary
from summarize_rethink_dynamic import evaluations, success_auc

STARTS = (10000, 100000, 500000)
PREFIX = 100
RIDGES = (1e-4, 1e-3, 1e-2)
ENERGY = .95


def safe_ratio(a, b):
    return float(a / b) if b > 1e-30 else None


def eigen(matrix):
    values, basis = np.linalg.eigh((matrix + matrix.T) / 2)
    assert values[0] >= -1e-9 * max(1., values[-1]), values[0]
    return np.maximum(values, 0.), basis


def covariance(rows):
    """Rows are separate demand vectors. Do NOT average before their outer products."""
    rows = np.asarray(rows, dtype=np.float64)
    assert rows.ndim == 2 and len(rows) > 0
    return rows.T @ rows / len(rows)


def demand_family(rows):
    matrix = covariance(rows)
    vals, vectors = eigen(matrix)
    total = float(vals.sum())
    if total <= 1e-30:
        return matrix, vectors[:, :0], dict(energy=total, rank95=0)
    rank = int(np.searchsorted(np.cumsum(vals[::-1]), ENERGY * total) + 1)
    positive = vals > 0
    prob = vals[positive] / total
    summary = dict(energy=total, rank95=rank,
        entropy_rank=float(np.exp(-np.sum(prob * np.log(prob)))),
        leading_fraction=float(vals[-1] / total),
        mean_vector_energy_fraction=float(np.square(np.mean(rows, axis=0)).sum() / total))
    return matrix, vectors[:, -rank:], summary


def geometry(params, x):
    q, cache = forward(params, x)
    values, basis = eigen(ntk(params, cache))
    return dict(q=q, eigenvalues=values, eigenvectors=basis,
                summary=spectral_summary(values))


def shape_response(values, horizon=1000):
    """Same trace n, same eta=.45/n for EVERY kernel; not the SAC Adam lr."""
    mean = float(values.mean())
    normalized = values / mean
    eta = .45 / len(values)
    factor = 1 - 2 * eta * normalized
    assert factor.min() >= -1e-12 and factor.max() <= 1 + 1e-12
    retention = np.maximum(factor, 0.) ** (2 * horizon)
    return normalized, eta, retention, retention >= .5


def demand_score(matrix, subspace, geom, common_raw_eta):
    vals, basis = geom['eigenvalues'], geom['eigenvectors']
    projected = basis.T @ matrix @ basis
    energy = np.maximum(np.diag(projected), 0.)
    total = float(energy.sum())
    assert np.isclose(total, np.trace(matrix), rtol=1e-9, atol=1e-9)
    if total <= 1e-30:
        return dict(total_energy=total, valid=False)
    normalized, eta, retention, slow = shape_response(vals)
    fast_count = int(np.searchsorted(np.cumsum(vals[::-1]), .9 * vals.sum()) + 1)
    prob = energy / total
    out = dict(valid=True, total_energy=total, shape_eta=eta,
        shape_remaining1000=float(prob @ retention),
        slow_energy_fraction=float(prob[slow].sum()), slow_dimension=int(slow.sum()),
        kernel_fast90_dimension=fast_count, fast90_demand_energy=float(prob[-fast_count:].sum()),
        raw_common_eta=float(common_raw_eta),
        raw_remaining1000=float(prob @ ((1 - 2 * common_raw_eta * vals) ** 2000)))
    assert np.min(1 - 2 * common_raw_eta * vals) >= -1e-10
    for ridge in RIDGES:
        contributions = prob / (normalized + ridge)
        burden = float(contributions.sum())
        key = 'ridge_%g' % ridge
        out[key] = dict(shape_burden=burden, raw_burden=burden / float(vals.mean()),
            slow_burden_fraction=float(contributions[slow].sum() / burden))
    if subspace.shape[1]:
        overlap = basis[:, -fast_count:].T @ subspace
        squared_cos = np.linalg.svd(overlap, compute_uv=False) ** 2
        # Pad missing principal cosines when the demand space has larger dimension.
        padded = np.r_[squared_cos, np.zeros(subspace.shape[1] - len(squared_cos))]
        out.update(subspace_fast90_overlap=float(padded.mean()),
                   subspace_fast90_min_cos2=float(padded.min()),
                   uniform_subspace_shape_burden=float(np.sum(
                       np.square(basis.T @ subspace) / (normalized[:, None] + 1e-3)) / subspace.shape[1]))
    return out


def signed_accounting(d, parts):
    """parts: [target, -GD_anchor, -sampling, -Adam, -nonlinearity]."""
    delta = parts.sum(axis=0)
    return np.sum(parts * (2 * d + delta)[None, :, :], axis=(1, 2)) / d.shape[1]


def tracking_summary(d, components, basis, slow, left=PREFIX):
    """Fixed basis selected before the evaluation suffix; targets remain moving."""
    coefficients = d @ basis
    projected = np.asarray([c @ basis for c in components])
    delta = np.diff(coefficients, axis=0)
    identity = np.max(np.abs(projected.sum(axis=0) - delta))
    scale = max(1., float(np.max(np.abs(projected))))
    assert identity < 1e-8 * scale, identity
    result = dict(projection_identity_max_abs=float(identity), groups={})
    for label, mask in (('slow', slow), ('fast', ~slow), ('all', np.ones(len(slow), dtype=bool))):
        if not mask.any():
            result['groups'][label] = dict(dimension=0)
            continue
        r = coefficients[left:, mask]
        parts = projected[:, left:, mask]
        # Normalization remains by full n, not the size of the selected subspace.
        n = coefficients.shape[1]
        energy = np.square(r).sum(axis=1) / n
        signed = signed_accounting(r[:-1], parts) * r.shape[1] / n
        closure = float(abs(signed.sum() - (energy[-1] - energy[0])))
        assert closure < 1e-7 * max(1., float(np.abs(signed).sum()))
        update = -parts[1:].sum(axis=0)
        demand = parts[0]
        row = dict(dimension=int(mask.sum()), initial_mse=float(energy[0]), final_mse=float(energy[-1]),
            auc=float(np.trapz(energy) / (len(energy) - 1)),
            final_over_initial=safe_ratio(energy[-1], energy[0]),
            auc_over_initial=safe_ratio(np.trapz(energy) / (len(energy) - 1), energy[0]),
            correction_gain=safe_ratio(np.sum(r[:-1] * update), np.square(r[:-1]).sum()),
            target_rms=float(np.sqrt(np.square(demand).sum() / demand.size)),
            update_rms=float(np.sqrt(np.square(update).sum() / update.size)),
            signed_delta_mse=dict(zip(('target', 'anchor_gd', 'sampling', 'adam', 'nonlinear'), map(float, signed))),
            signed_energy_identity_abs=closure)
        result['groups'][label] = row
    return result, coefficients, projected


def read_snapshot(path, registry):
    obj = torch.load(str(path), map_location='cpu')
    models = obj.get('models', obj)
    params = [params_from_state(models[name]) for name in ('qf1', 'qf2')]
    stat = path.stat()
    registry[str(path.resolve())] = dict(path=str(path.resolve()), bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns, sha256=digest(path))
    return params


def actual_trajectory(data, qi):
    """Reconstruct every Q and actual local-J update, plus exact relative bookkeeping."""
    x, y = data['x'], data['y']
    n = len(x)
    qvalues = np.empty((1001, n))
    parts = np.empty((5, 1000, n))
    lr = data['first']['optimizers'][('qf1', 'qf2')[qi]]['param_groups'][0]['lr']
    audit = 0.
    for t in range(1000):
        p, pn = recorded_params(data, qi, t), recorded_params(data, qi, t + 1)
        q, cache = forward(p, x)
        qn, _ = forward(pn, x)
        qvalues[t], qvalues[t + 1] = q, qn
        audit = max(audit, relative_rmse(q, data['saved_q'][qi, t]))
        lin = jvp(p, [b - a for a, b in zip(p, pn)], cache)
        anchor = jvp(p, [-lr * g for g in gradient(p, x, y[t], cache)], cache)
        sample = jvp(p, [-lr * g for g in gradient(p, data['xs'][t], data['ys'][t])], cache)
        parts[:, t] = np.stack((y[t + 1] - y[t], -anchor, -(sample - anchor),
                               -(lin - sample), -((qn - q) - lin)))
    audit = max(audit, relative_rmse(qvalues[-1], data['saved_q'][qi, -1]))
    assert audit < 2e-5, audit
    return qvalues, parts, audit


def run_pair(job):
    root, out, pair, seed, code_hash = job
    torch.set_num_threads(1)
    began = time.time()
    root, out = Path(root), Path(out)
    tag = 'P%d_s%d' % (pair, seed)
    folder = out / tag
    folder.mkdir(exist_ok=True)
    if (folder / 'complete.json').exists():
        assert json.loads((folder / 'complete.json').read_text())['code_hash'] == code_hash
        return dict(tag=tag, status='cached')
    run = root / ('P%d_ft_s%d' % (pair, seed))
    fresh = root / ('P%d_ft_s%d' % (pair + 1 if pair % 2 else pair - 1, seed))
    registry = {}
    initial = read_snapshot(run / 'checkpoints/initial_models.pt', registry)
    fresh_initial = read_snapshot(fresh / 'checkpoints/initial_models.pt', registry)
    for p, f in zip(initial, fresh_initial):
        assert all(np.array_equal(a, b) for a, b in zip(p, f))
    snapshots = {'initial': initial}
    for t in (10000, 50000, 100000, 500000, 1000000, 1500000):
        suffix = '_full' if t == 1500000 else ''
        snapshots['A_%07d' % t] = read_snapshot(run / ('checkpoints/task_0_env_%07d%s.pt' % (t, suffix)), registry)
    for t in (10000, 50000, 100000, 500000):
        snapshots['B_%07d' % t] = read_snapshot(run / ('checkpoints/task_1_env_%07d.pt' % t), registry)
        snapshots['fresh_%07d' % t] = read_snapshot(fresh / ('checkpoints/task_0_env_%07d.pt' % t), registry)
    for a, b in zip(snapshots['A_1500000'], snapshots['B_0010000']):
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
    windows = []
    history_rows, history_arrays, histories = [], {}, None
    fixed_x = None
    for start in STARTS:
        data = load_window(root, pair, seed, start)
        for rec in data['inputs']: registry[rec['path']] = rec
        if fixed_x is None:
            fixed_x = data['x'].copy()
            # Check paired B/fresh warm-up transitions on the SAME indices.
            manifest = json.loads((data['window'] / 'manifest.json').read_text())
            ids = np.asarray(manifest['anchor_indices'])
            for bank_name in ('observation', 'action', 'reward', 'next_observation', 'terminal'):
                with np.load(str(run / 'anchors/task_1_warmup.npz')) as b, np.load(str(fresh / 'anchors/task_0_warmup.npz')) as f:
                    assert np.array_equal(b[bank_name][ids], f[bank_name][ids]), bank_name
            fresh_bank = fresh / 'anchors/task_0_warmup.npz'
            st = fresh_bank.stat()
            registry[str(fresh_bank.resolve())] = dict(path=str(fresh_bank.resolve()), bytes=st.st_size,
                mtime_ns=st.st_mtime_ns, sha256=digest(fresh_bank))
            histories = [{label: geometry(params[qi], fixed_x) for label, params in snapshots.items()} for qi in (0, 1)]
            for qi in (0, 1):
                for label, geom in histories[qi].items():
                    history_rows.append(dict(pair=pair, seed=seed, critic=qi + 1, stage=label, **geom['summary']))
                    for key in ('q', 'eigenvalues', 'eigenvectors'):
                        history_arrays['q%d_%s_%s' % (qi + 1, label, key)] = geom[key]
            np.savez_compressed(str(folder / 'history_spectra.npz'), **history_arrays)
            write_json(folder / 'history.json', history_rows)
        else:
            assert np.array_equal(fixed_x, data['x']), 'B inputs changed across windows'
        destination = folder / ('B%07d' % start)
        destination.mkdir(exist_ok=True)
        rows, arrays = [], {}
        for qi in (0, 1):
            q, parts, qaudit = actual_trajectory(data, qi)
            residuals = data['y'] - q
            current = geometry(recorded_params(data, qi, PREFIX), fixed_x)
            endpoint = geometry(recorded_params(data, qi, 1000), fixed_x)
            comparisons = dict(initial=histories[qi]['initial'], A_end=histories[qi]['A_1500000'],
                ft_start=histories[qi]['B_%07d' % start],
                fresh_matched=histories[qi]['fresh_%07d' % start], ft_prefix=current)
            # Cross the SAME D with all saved B-stage kernels, retrospectively.
            for other_start in STARTS:
                comparisons['B_%07d' % other_start] = histories[qi]['B_%07d' % other_start]
            raw_eta = min(3e-4, .45 / max(float(g['eigenvalues'][-1]) for g in comparisons.values()))
            families = dict(prefix_residual=residuals[:PREFIX + 1], all_residual=residuals,
                prefix_centered=residuals[:PREFIX + 1] - residuals[:PREFIX + 1].mean(axis=0),
                all_centered=residuals - residuals.mean(axis=0),
                target_increments=np.diff(data['y'], axis=0))
            family_results = {}
            for name, demands in families.items():
                matrix, subspace, desc = demand_family(demands)
                scores = {label: demand_score(matrix, subspace, g, raw_eta) for label, g in comparisons.items()}
                family_results[name] = dict(structure=desc, geometries=scores)
                arrays['q%d_%s_covariance' % (qi + 1, name)] = matrix
                arrays['q%d_%s_basis95' % (qi + 1, name)] = subspace
            _, _, _, slow = shape_response(current['eigenvalues'])
            tracking, coeff, projected = tracking_summary(residuals, parts, current['eigenvectors'], slow)
            # A family selected with the prefix only, measured on the suffix.
            _, subspace, _ = demand_family(families['prefix_residual'])
            prefix_energy = np.square(residuals[:PREFIX + 1]).sum()
            suffix_energy = np.square(residuals[PREFIX + 1:]).sum()
            subspace_transfer = dict(
                prefix_energy_coverage=safe_ratio(np.square(residuals[:PREFIX + 1] @ subspace).sum(), prefix_energy),
                suffix_energy_coverage=safe_ratio(np.square(residuals[PREFIX + 1:] @ subspace).sum(), suffix_energy))
            ev0, v0 = current['eigenvalues'], current['eigenvectors']
            ev1, v1 = endpoint['eigenvalues'], endpoint['eigenvectors']
            # Fixed slow projector retention in end-kernel and rotation diagnostic.
            kn0 = (v0 * (ev0 / ev0.mean())) @ v0.T
            kn1 = (v1 * (ev1 / ev1.mean())) @ v1.T
            _, _, _, slow1 = shape_response(ev1)
            stability = dict(normalized_kernel_relative_change=float(np.linalg.norm(kn1 - kn0) / np.linalg.norm(kn0)),
                slow_projector_overlap=safe_ratio(np.square(v0[:, slow].T @ v1[:, slow1]).sum(), slow.sum()))
            deltaq = np.diff(q, axis=0)
            nonlinear_ratio = math.sqrt(np.square(parts[4]).sum() / max(np.square(deltaq).sum(), 1e-30))
            row = dict(pair=pair, seed=seed, start=start, critic=qi + 1, families=family_results,
                tracking=tracking, prefix_subspace_transfer=subspace_transfer, stability=stability,
                reconstruction_relative_rmse_max=qaudit, nonlinear_over_actual_change_rms=float(nonlinear_ratio))
            rows.append(row)
            arrays['q%d_residuals' % (qi + 1)] = residuals
            arrays['q%d_targets' % (qi + 1)] = data['y']
            arrays['q%d_outputs' % (qi + 1)] = q
            arrays['q%d_prefix_kernel_eigenvalues' % (qi + 1)] = ev0
            arrays['q%d_prefix_kernel_basis' % (qi + 1)] = v0
            arrays['q%d_slow_mask' % (qi + 1)] = slow
            arrays['q%d_residual_mode_coefficients' % (qi + 1)] = coeff
            arrays['q%d_signed_update_mode_components' % (qi + 1)] = projected
            print('CRITIC', tag, start, qi + 1, 'seconds', round(time.time() - began, 1), flush=True)
        np.savez_compressed(str(destination / 'subspace_arrays.npz'), **arrays)
        write_json(destination / 'results.json', rows)
        write_json(destination / 'clocks.json', data['clocks'])
        windows.extend(rows)
        del data, arrays, parts, projected
    for rec in registry.values():
        st = Path(rec['path']).stat()
        assert (st.st_size, st.st_mtime_ns) == (rec['bytes'], rec['mtime_ns']), rec['path']
    write_json(folder / 'inputs.json', list(registry.values()))
    write_json(folder / 'complete.json', dict(code_hash=code_hash, windows=len(STARTS), critics=2,
        source_files=len(registry), seconds=time.time() - began))
    return dict(tag=tag, status='complete', seconds=round(time.time() - began, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, nargs='+', default=list(range(1, 7)))
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    assert root != out and root not in out.parents and out not in root.parents
    assert 1 <= args.workers <= 2
    out.mkdir(parents=True, exist_ok=True)
    hashes = {name: digest(Path(__file__).parent / name) for name in
        ('analyze_rethink_subspace.py', 'analyze_rethink_dynamic.py',
         'analyze_rethink_dynamic_geometry.py', 'summarize_rethink_dynamic.py')}
    protocol = dict(source_root=str(root), code_hashes=hashes, starts=STARTS, prefix_updates=PREFIX,
        pairs=list(range(1, 7)), seeds=[1, 2, 3], updates=1000, anchors=256,
        kernel='full parameter uncentered JJT/n, float64; biases included',
        demand='D_emp has one recorded y_t-Q_t column per state; not column-averaged or a full fixed Bellman operator',
        families=['prefix_residual', 'all_residual', 'prefix_centered', 'all_centered', 'target_increments'],
        subspace='top singular directions capturing 95% demand energy; uncentered primary, centered sensitivity',
        ridges=RIDGES, primary_ridge=.001,
        shape='K/mean_eigenvalue; fixed eta .45/n, squared-error retention after 1000 calibrated GD updates',
        raw='same demand, common stable eta across comparison geometries per window and critic; reported separately',
        cross_geometry='each D also evaluated on all three B-stage kernels; future-stage kernels are retrospective controls, not available-at-entry predictions',
        slow='shape-normalized squared-error retention >=.5 after 1000; threshold is not an Adam speed claim',
        timing='prefix states 0..100 select demand; kernel at 100 defines modes; suffix updates 100..999 tests tracking',
        signed_components=['target', '-anchor_GD', '-sampling_difference', '-actual_minus_minibatch_SGD', '-nonlinear'],
        accounting='exact symmetric cross-term allocation, not causal percentages; sampling includes replay/panel and stochastic-target differences',
        endpoint='online evaluation snapshots frozen before this analysis; success-AUC 0..500k, not final 1.5M result',
        scope='no environment calls, actor updates, online training, Clip replay or git push',
        inference='paired three-seed descriptive results; critics/windows/time points are not independent seeds')
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == json.loads(json.dumps(protocol)), 'Protocol changed'
    else:
        write_json(protocol_path, protocol)
        frozen = {}
        for p in range(1, 7):
            for seed in (1, 2, 3):
                run = root / ('P%d_ft_s%d' % (p, seed))
                ev = evaluations(run)
                frozen[run.name] = [r for r in ev if r['task_env_step'] <= 500000]
                assert success_auc(ev, 1, 0, 500000) >= 0
        write_json(out / 'evaluation_snapshot.json', frozen)
    jobs = [(str(root), str(out), p, s, hashes) for p in args.pairs for s in args.seeds]
    print('PAIRS', len(jobs), 'WORKERS', args.workers, 'OUTPUT', out, flush=True)
    if args.workers == 1:
        for job in jobs: print('JOB', json.dumps(run_pair(job)), flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_pair, job) for job in jobs]
            for f in concurrent.futures.as_completed(futures): print('JOB', json.dumps(f.result()), flush=True)
    print('COMPLETE', len(jobs), flush=True)


if __name__ == '__main__':
    main()
