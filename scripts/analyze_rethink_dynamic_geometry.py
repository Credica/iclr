#!/usr/bin/env python3
"""Frozen-kernel forced-response audit on the same recorded moving targets.

This is a calibrated anchor-GD surrogate, not a second SAC replay or an
Adam convergence claim. The nonlinear, actual-minibatch results are separate.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_rethink_dynamic import (params_from_state,forward,ntk,clipped,digest,write_json)


def eigensystem(params,x):
    q,cache=forward(params,x)
    ev,basis=np.linalg.eigh(ntk(params,cache))
    assert ev[0]>=-1e-9*max(1.,ev[-1])
    return q,np.maximum(ev,0.),basis


def forced_response(eigenvalues,basis,d0,drift,jump,eta):
    """d[t+1]=(I-2 eta K)d[t]+u[t]; jump is propagated homogeneously."""
    factor=1-2*eta*eigenvalues
    assert factor.min()>=-1e-12 and factor.max()<=1+1e-12
    residual=basis.T@d0
    propagated_jump=basis.T@jump
    increments=drift@basis
    rows=[]
    for t in range(len(drift)+1):
        matched=np.mean(residual**2)
        same_target=np.mean((residual-propagated_jump)**2)
        cross=2*np.mean(residual*propagated_jump)
        cost=np.mean(propagated_jump**2)
        assert abs((matched-same_target)-(cross-cost))<1e-9*max(1.,matched,same_target)
        rows.append((matched,same_target,cross,cost))
        if t<len(drift):
            residual=factor*residual+increments[t]
            propagated_jump*=factor
    return np.asarray(rows)


def spectral_summary(ev):
    total=float(ev.sum())
    positive=ev>0
    prob=ev[positive]/max(total,1e-30)
    return dict(lambda_min=float(ev[0]),lambda_max=float(ev[-1]),trace=total,
                entropy_effective_rank=float(np.exp(-np.sum(prob*np.log(prob)))),
                participation_rank=float(total**2/max(np.square(ev).sum(),1e-30)),
                trace_rank99=int(np.searchsorted(np.cumsum(ev[::-1]),.99*total)+1))


def analyze(folder,root,script_hash):
    destination=folder/'geometry.json'
    if destination.exists():
        existing=json.loads(destination.read_text())
        if existing.get('script_sha256')==script_hash:return 'cached'
        raise RuntimeError('Geometry analysis script mismatch: '+str(destination))
    metadata=json.loads((folder/'summary.json').read_text())
    for source in json.loads((folder/'inputs.json').read_text()):
        stat=Path(source['path']).stat()
        assert (stat.st_size,stat.st_mtime_ns)==(source['bytes'],source['mtime_ns']),source['path']
    run=Path(metadata['source_run'])
    assert root==run.parent
    window=run/'target_windows'/('B_env_%07d'%metadata['start_env_step'])
    first=torch.load(str(window/'start.pt'),map_location='cpu')
    manifest=json.loads((window/'manifest.json').read_text())
    indices=np.asarray(manifest['anchor_indices'])
    with np.load(str(run/'anchors/task_1_warmup.npz')) as bank:
        x=np.concatenate((bank['observation'][indices],bank['action'][indices]),axis=1).astype(np.float64)
        reward=bank['reward'][indices].ravel().astype(np.float64)
        bootstrap=.99*(1-bank['terminal'][indices].ravel().astype(np.float64))
        next_observation=bank['next_observation'][indices].astype(np.float64)
    refs=[]
    for chunk in sorted(window.glob('rows_through_*.pt')):
        refs.extend(r for r in torch.load(str(chunk),map_location='cpu') if r['kind']=='reference')
    assert [r['index'] for r in refs]==list(range(1001))
    y=np.asarray([r['target'] for r in refs],dtype=np.float64)
    drift=np.diff(y,axis=0)
    alpha=np.asarray([float(r['alpha']) for r in refs],dtype=np.float64)
    target_q=np.minimum(np.asarray([r['target_q1'] for r in refs],dtype=np.float64),
                        np.asarray([r['target_q2'] for r in refs],dtype=np.float64))
    alpha_logpi=alpha[:,None]*np.asarray([r['log_pi'] for r in refs],dtype=np.float64)
    reconstructed=reward+bootstrap*(target_q-alpha_logpi)
    dq=bootstrap*np.diff(target_q,axis=0)
    de=-bootstrap*np.diff(alpha_logpi,axis=0)
    denominator=max(float(np.square(drift).sum()),1e-30)
    bellman_source=dict(alpha_start=float(alpha[0]),alpha_end=float(alpha[-1]),
        alpha_min=float(alpha.min()),alpha_max=float(alpha.max()),
        reward_rms=float(np.sqrt(np.mean(reward**2))),target_rms=float(np.sqrt(np.mean(y**2))),
        target_formula_relative_rmse=float(np.sqrt(np.mean((reconstructed-y)**2)/max(1.,np.mean(y**2)))),
        drift_formula_relative_rmse=float(np.sqrt(np.square(dq+de-drift).sum()/denominator)),
        target_critic_drift_rms_ratio=float(np.sqrt(np.square(dq).sum()/denominator)),
        entropy_drift_rms_ratio=float(np.sqrt(np.square(de).sum()/denominator)),
        target_critic_signed_energy_share=float((dq*drift).sum()/denominator),
        entropy_signed_energy_share=float((de*drift).sum()/denominator))
    assert bellman_source['target_formula_relative_rmse']<2e-5
    summaries=[];arrays={}
    originals=[params_from_state(first['models'][name]) for name in ('qf1','qf2')]
    projections=[clipped(params) for params in originals]
    next_x=np.concatenate((next_observation,np.asarray(refs[0]['next_action'],dtype=np.float64)),axis=1)
    q0=np.asarray([forward(params,x)[0] for params in originals])
    qc0=np.asarray([forward(params,x)[0] for params in projections])
    ft_sync=reward+bootstrap*(np.minimum(*[forward(params,next_x)[0] for params in originals])-alpha_logpi[0])
    clip_sync=reward+bootstrap*(np.minimum(*[forward(params,next_x)[0] for params in projections])-alpha_logpi[0])
    q_jump=qc0-q0;target_jump=clip_sync-y[0]
    cross=float(np.mean(q_jump*target_jump))
    entry_sync=dict(
        definition='instantaneous conditional diagnostic: same FT actor/action noise/alpha; synchronize targets to each online twin critic',
        not_online_result=True,
        ft_recorded_target_mse=float(np.mean((y[0]-q0)**2)),
        ft_hard_sync_target_mse=float(np.mean((ft_sync-q0)**2)),
        clip_common_ft_target_mse=float(np.mean((y[0]-qc0)**2)),
        clip_hard_sync_target_mse=float(np.mean((clip_sync-qc0)**2)),
        q_jump_rms=float(np.sqrt(np.mean(q_jump**2))),
        target_jump_rms=float(np.sqrt(np.mean(target_jump**2))),
        residual_jump_rms=float(np.sqrt(np.mean((target_jump-q_jump)**2))),
        q_target_jump_cosine=cross/np.sqrt(max(float(np.mean(q_jump**2)*np.mean(target_jump**2)),1e-30)))
    for qi,name in enumerate(('qf1','qf2'),1):
        params=originals[qi-1];projected=projections[qi-1]
        q,ev,basis=eigensystem(params,x)
        qc,evc,basisc=eigensystem(projected,x)
        d0=y[0]-q;jump=qc-q
        assert np.isclose(np.mean(d0*d0),metadata['decomposition'][qi-1]['initial_mse'],rtol=1e-9)
        matrices=[]
        for index in (0,2,4):
            before=np.linalg.svd(params[index],compute_uv=False)
            after=np.linalg.svd(projected[index],compute_uv=False)
            matrices.append(dict(weight_index=index,min_before=float(before.min()),max_before=float(before.max()),
                min_after=float(after.min()),max_after=float(after.max()),
                fraction_raised=float(np.mean(before<.25)),fraction_lowered=float(np.mean(before>4)),
                relative_frobenius_change=float(np.linalg.norm(projected[index]-params[index])/np.linalg.norm(params[index]))))
        rec=dict(critic=name,ft_kernel=spectral_summary(ev),clip_kernel=spectral_summary(evc),layers=matrices,
                 initial_residual_mse=float(np.mean(d0*d0)),q_jump_mse=float(np.mean(jump*jump)),conditions={})
        for condition in ('common_stable_lr','separately_stable_lr'):
            common=min(3e-4,.45/max(ev[-1],evc[-1],1e-30))
            eta0=common if condition=='common_stable_lr' else min(3e-4,.45/max(ev[-1],1e-30))
            etac=common if condition=='common_stable_lr' else min(3e-4,.45/max(evc[-1],1e-30))
            original=forced_response(ev,basis,d0,drift,np.zeros_like(jump),eta0)
            intervention=forced_response(evc,basisc,d0,drift,jump,etac)
            # Columns are FT error, Clip matched-correction error, Clip same-target
            # error, geometric gain, Q-jump alignment benefit, Q-jump cost.
            curve=np.column_stack((original[:,0],intervention[:,0],intervention[:,1],
                original[:,0]-intervention[:,0],intervention[:,2],intervention[:,3]))
            closure=np.max(np.abs((curve[:,0]-curve[:,2])-(curve[:,3]+curve[:,4]-curve[:,5])))
            assert closure<1e-8*max(1.,float(np.abs(curve).max()))
            area=np.trapz(curve,axis=0)/1000
            rec['conditions'][condition]=dict(ft_eta=float(eta0),clip_eta=float(etac),
                ft_auc=float(area[0]),clip_correction_auc=float(area[1]),clip_target_auc=float(area[2]),
                geometry_gain_auc=float(area[3]),jump_alignment_gain_auc=float(area[4]),jump_cost_auc=float(area[5]),
                identity_max_abs=float(closure),ft_final_mse=float(curve[-1,0]),
                clip_correction_final_mse=float(curve[-1,1]),clip_target_final_mse=float(curve[-1,2]))
            arrays['q%d_%s'%(qi,condition)]=curve
        for label,vals in (('ft_eigenvalues',ev),('clip_eigenvalues',evc)):
            arrays['q%d_%s'%(qi,label)]=vals
        summaries.append(rec)
    np.savez_compressed(str(folder/'geometry_curves.npz'),**arrays)
    write_json(destination,dict(script_sha256=script_hash,critics=summaries,bellman_source=bellman_source,
        entry_sync=entry_sync,
        pair=metadata['pair'],seed=metadata['seed'],start_env_step=metadata['start_env_step']))
    return 'complete'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    torch.set_num_threads(1)
    protocol=json.loads((out/'protocol.json').read_text())
    folders=sorted(p.parent for p in out.glob('P*_s*_B*/complete.json'))
    if not args.allow_partial:assert len(folders)==36
    script_hash=digest(__file__)
    write_json(out/'geometry_protocol.json',dict(script_sha256=script_hash,
        equation='d[t+1]=(I-2*eta*K_start)d[t]+(y[t+1]-y[t])',updates=1000,
        kernel='same 256-anchor full-parameter uncentered JJT/n, frozen at window start',
        clip='one all-Linear weight projection [0.25,4], no bias or Adam change',
        controls=['common_stable_lr','separately_stable_lr'],
        eta='min(3e-4,0.45/lambda_max); common uses the maximum across both models',
        caveat='frozen-kernel full-anchor GD surrogate; not actual minibatch Adam, not a theorem about nonlinear SAC',
        identity='E_FT-E_Clip_same_target = geometry_gain + propagated_jump_alignment - propagated_jump_cost',
        entry_sync='additional instantaneous diagnostic only; regenerate one entry target with synchronized FT/Clip critics, same recorded FT actor/alpha/noise; no online rollout or dynamic performance claim',
        curve_columns=['ft_mse','clip_correction_mse','clip_target_mse','geometry_gain','jump_alignment_gain','jump_cost']))
    for folder in folders:
        began=time.time();status=analyze(folder,Path(protocol['source_root']),script_hash)
        print(folder.name,status,'seconds',round(time.time()-began,2),flush=True)
    print('COMPLETE_GEOMETRY',len(folders),flush=True)


if __name__=='__main__':main()
