#!/usr/bin/env python3
"""Independent real-checkpoint Jacobian and entry-demand inverse-solve audit."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch

from analyze_rethink_dynamic import digest, write_json
from analyze_abc_spectrum import torch_critic


def audit(out):
    torch.set_num_threads(1)
    protocol = json.loads((out / 'protocol.json').read_text())
    for name, expected in protocol['code_sha256'].items():
        assert digest(Path(__file__).parent / name) == expected, name
    sources = json.loads((out / 'inputs.json').read_text())
    for source in sources:
        path = Path(source['path']); st = path.stat()
        assert (st.st_size, st.st_mtime_ns) == (source['bytes'], source['mtime_ns'])
        assert digest(path) == source['sha256'], str(path)
    arrays = np.load(str(out / 'spectral_arrays.npz'))
    banks = np.load(str(out / 'fixed_inputs.npz'))
    with (out / 'demand_spectra.csv').open() as f: rows = list(csv.DictReader(f))
    records = []
    ids = np.linspace(0, 127, 8, dtype=int)
    for step in (1000000, 2000000, 3000000):
        task = step // 1000000 - 1
        path = Path(protocol['source']) / 'sac_transfer_abc_s1_1m/checkpoints' / ('interval_task%d_step%d.pt' % (task, step))
        saved = torch.load(str(path), map_location='cpu')
        for panel in (1, 2):
            obs = torch.from_numpy(banks['task%d_observation' % panel][ids]).double()
            action = torch.from_numpy(banks['task%d_action' % panel][ids]).double()
            for qi in (1, 2):
                params = {k: v.double().requires_grad_() for k, v in saved['qf%d' % qi].items()}
                q = torch_critic(params, obs, action)
                jac = []
                for value in q:
                    grads = torch.autograd.grad(value, tuple(params.values()), retain_graph=True)
                    jac.append(torch.cat([g.flatten() for g in grads]))
                jac = torch.stack(jac).detach().numpy()
                expected = jac @ jac.T / 128
                basis = arrays['step%d_task%d_q%d_eigenvectors' % (step, panel, qi)]
                values = arrays['step%d_task%d_q%d_eigenvalues' % (step, panel, qi)]
                actual = (basis[ids] * values) @ basis[ids].T
                err = float(np.linalg.norm(expected - actual) / np.linalg.norm(expected))
                assert err < 1e-10, err
                records.append(dict(step=step, panel=panel, critic=qi, relative_jacobian_kernel_error=err))
    errors = []
    for task in (1, 2):
        for qi in (1, 2):
            d = arrays['task%d_q%d_entry_demand' % (task, qi)]
            target = arrays['task%d_entry_targets' % task]
            q = arrays['step%d_task%d_q%d_q' % (task * 1000000, task, qi)]
            np.testing.assert_array_equal(d, target - q[None, :])
            covariance = d.T @ d / len(d)
            np.testing.assert_allclose(covariance, arrays['task%d_q%d_entry_second_moment' % (task, qi)], rtol=1e-12, atol=1e-10)
            for step in range(100000, 3000001, 100000):
                v = arrays['step%d_task%d_q%d_eigenvectors' % (step, task, qi)]
                lam = arrays['step%d_task%d_q%d_eigenvalues' % (step, task, qi)]
                kn = (v * (lam / lam.mean())) @ v.T
                expected = np.trace(np.linalg.solve(kn + .001 * np.eye(128), covariance)) / np.trace(covariance)
                row = next(r for r in rows if r['family'] == 'entry' and int(r['input_task']) == task and int(r['critic']) == qi and int(r['kernel_step']) == step)
                value = float(row['ridge_0.001_shape_burden'])
                err = float(abs(expected - value) / max(abs(value), 1e-30))
                assert err < 1e-7, err
                errors.append(err)
    result = dict(real_checkpoint_autograd_checks=len(records), max_real_kernel_relative_error=max(r['relative_jacobian_kernel_error'] for r in records),
        entry_inverse_direct_solve_checks=len(errors), max_entry_inverse_relative_error=max(errors),
        source_sha256_verified=len(sources), original_sources_unchanged=True, records=records,
        audit_script_sha256=digest(__file__))
    write_json(out / 'independent_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); audit(args.output.resolve())
