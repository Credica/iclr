#!/usr/bin/env python3
"""Final independent bookkeeping/precision audit of the R3 offline artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_rethink_dynamic import NAMES, digest, write_json


def relative(a, b):
    return float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-30))


def audit(out):
    torch.set_num_threads(1)
    protocol=json.loads((out/'protocol.json').read_text())
    source_root=Path(protocol['source_root'])
    folders=sorted(p.parent for p in out.glob('P*_s*/complete.json'))
    assert len(folders)==18
    rows=[]
    for folder in folders:
        p, s = folder.name.split('_')
        run=source_root/(p+'_ft_'+s)
        history=np.load(str(folder/'history_spectra.npz'))
        first_ids=first_noise=None
        for start in protocol['starts']:
            source=run/'target_windows'/('B_env_%07d'%start)
            manifest=json.loads((source/'manifest.json').read_text())
            ids=np.asarray(manifest['anchor_indices'])
            noise=np.load(str(source/'fixed_next_action_epsilon.npy'))
            if first_ids is None: first_ids, first_noise=ids, noise
            else:
                np.testing.assert_array_equal(ids,first_ids)
                np.testing.assert_array_equal(noise,first_noise)
            refs=[]
            for chunk in sorted(source.glob('rows_through_*.pt')):
                refs.extend(r for r in torch.load(str(chunk),map_location='cpu') if r['kind']=='reference')
            assert [r['index'] for r in refs]==list(range(1001))
            ys=np.asarray([r['target'] for r in refs],dtype=np.float64)
            clocks=json.loads((folder/('B%07d'%start)/'clocks.json').read_text())
            assert [c['task_env_step'] for c in clocks]==[r['task_env_step'] for r in refs]
            if start<500000:
                assert max(c['task_env_step'] for c in clocks)<(50000 if start==10000 else 150000)
            snap=torch.load(str(run/('checkpoints/task_1_env_%07d.pt'%start)),map_location='cpu')
            initial=torch.load(str(source/'start.pt'),map_location='cpu')
            values=np.load(str(folder/('B%07d'%start)/'subspace_arrays.npz'))
            for qi,name in enumerate(('qf1','qf2'),1):
                assert all(torch.equal(snap['models'][name][k],initial['models'][name][k]) for k in NAMES)
                q=values['q%d_outputs'%qi]
                saved=np.asarray([r['q%d'%qi] for r in refs],dtype=np.float64)
                d=values['q%d_residuals'%qi]
                basis=values['q%d_prefix_kernel_basis'%qi]
                coeff=values['q%d_residual_mode_coefficients'%qi]
                parts=values['q%d_signed_update_mode_components'%qi]
                slow=values['q%d_slow_mask'%qi]
                np.testing.assert_array_equal(values['q%d_targets'%qi],ys)
                np.testing.assert_array_equal(d,ys-q)
                np.testing.assert_allclose(q[0],history['q%d_B_%07d_q'%(qi,start)],rtol=1e-12,atol=1e-10)
                np.testing.assert_allclose(coeff,d@basis,rtol=1e-11,atol=1e-8)
                c=d[:101].T@d[:101]/101
                np.testing.assert_allclose(c,values['q%d_prefix_residual_covariance'%qi],rtol=1e-11,atol=1e-8)
                err=q-saved
                residual_saved=ys-saved
                per_state=np.linalg.norm(err,axis=1)/np.maximum(np.linalg.norm(residual_saved,axis=1),1e-30)
                delta=np.diff(coeff,axis=0)
                cancellation=np.max(np.abs(parts.sum(axis=0)-delta))
                row=dict(run=run.name,start=start,critic=qi,
                    Q_reconstruction_relative_RMS=relative(err,saved),
                    Q_reconstruction_over_recorded_residual_RMS=relative(err,residual_saved),
                    max_state_Q_error_over_residual=float(per_state.max()),
                    slow_projection_Q_error_over_residual=relative(err@basis[:,slow],residual_saved@basis[:,slow]),
                    mode_update_identity_max_abs=float(cancellation),
                    mode_update_identity_relative=relative(parts.sum(axis=0)-delta,delta))
                assert row['Q_reconstruction_over_recorded_residual_RMS']<.01,row
                rows.append(row)
            values.close()
        history.close()
        print('AUDITED',folder.name,flush=True)
    result=dict(windows=54,critic_trajectories=len(rows),
        same_anchor_indices_and_noise_across_windows=True,
        B_snapshot_equals_window_start_parameters=True,
        target_and_residual_matrices_reconstructed=True,
        prefix_subspace_and_projected_update_checks=True,
        future_intervals_start_after_entire_early_window=True,
        max_reconstruction_over_recorded_residual_RMS=max(r['Q_reconstruction_over_recorded_residual_RMS'] for r in rows),
        max_state_Q_error_over_residual=max(r['max_state_Q_error_over_residual'] for r in rows),
        max_slow_projection_error_over_residual=max(r['slow_projection_Q_error_over_residual'] for r in rows),
        audit_script_sha256=digest(__file__),rows=rows)
    write_json(out/'PRECISION_AND_PAIRING_AUDIT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.output.resolve())
