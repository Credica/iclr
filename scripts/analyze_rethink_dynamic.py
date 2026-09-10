#!/usr/bin/env python3
"""Read-only dynamic-window analysis and CPU replay of recorded SAC critics.

No environment, W&B, actor training, or online target generation is involved.
Outputs must be outside the source run tree. See protocol.json for definitions.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

PREFIXES = ('_layers.0.linear', '_layers.1.linear', '_output_layers.0.linear')
NAMES = tuple(p + '.' + k for p in PREFIXES for k in ('weight', 'bias'))
BRANCHES = ('ft_carried', 'clip_carried_target', 'clip_carried_correction',
            'ft_fresh', 'clip_fresh_target', 'clip_fresh_correction')
COMPONENTS = ('target', 'anchor_gd', 'sampling', 'adam', 'nonlinear')
VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def params_from_state(state):
    assert set(state) == set(NAMES), list(state)
    return [state[name].detach().cpu().numpy().astype(np.float64) for name in NAMES]


def forward(params, x):
    w1, b1, w2, b2, w3, b3 = params
    h1 = np.maximum(x @ w1.T + b1, 0.)
    h2 = np.maximum(h1 @ w2.T + b2, 0.)
    q = (h2 @ w3.T + b3).ravel()
    return q, (x, h1, h2)


def gradient(params, x, target, cache=None):
    if cache is None:
        q, cache = forward(params, x)
    else:
        q = (cache[2] @ params[4].T + params[5]).ravel()
    _, h1, h2 = cache
    dq = (2. / len(x)) * (q - target)
    d2 = (dq[:, None] @ params[4]) * (h2 > 0)
    d1 = (d2 @ params[2]) * (h1 > 0)
    return [d1.T @ x, d1.sum(0), d2.T @ h1, d2.sum(0),
            dq[None, :] @ h2, np.asarray([dq.sum()])]


def jvp(params, delta, cache):
    x, h1, h2 = cache
    d1 = (x @ delta[0].T + delta[1]) * (h1 > 0)
    d2 = (d1 @ params[2].T + h1 @ delta[2].T + delta[3]) * (h2 > 0)
    return (d2 @ params[4].T + h2 @ delta[4].T + delta[5]).ravel()


def ntk(params, cache):
    """Exact uncentered full-parameter J J^T / n, including all biases."""
    x, h1, h2 = cache
    d2 = (h2 > 0) * params[4]
    d1 = (d2 @ params[2]) * (h1 > 0)
    gram = h2 @ h2.T + 1.
    gram += (d2 @ d2.T) * (h1 @ h1.T + 1.)
    gram += (d1 @ d1.T) * (x @ x.T + 1.)
    return (gram + gram.T) / (2. * len(x))


def clipped(params):
    result = [p.copy() for p in params]
    for index in (0, 2, 4):
        u, s, vh = np.linalg.svd(result[index], full_matrices=False)
        result[index] = (u * np.clip(s, .25, 4.)) @ vh
        check = np.linalg.svd(result[index], compute_uv=False)
        assert check.min() >= .25 - 1e-9 and check.max() <= 4. + 1e-9
    return result


class Adam:
    """NumPy Adam; saved moments are copied, never mutated in the input tree."""
    def __init__(self, params, saved=None):
        self.lr, self.beta1, self.beta2, self.eps = 3e-4, .9, .999, 1e-8
        self.step = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        if saved is not None:
            assert len(saved['param_groups']) == 1
            group = saved['param_groups'][0]
            assert group['weight_decay'] == 0 and not group['amsgrad']
            assert not group.get('maximize', False)
            self.lr, self.eps = group['lr'], group['eps']
            self.beta1, self.beta2 = group['betas']
            states = [saved['state'][i] for i in group['params']]
            assert len(states) == len(params)
            steps = [int(s['step']) for s in states]
            assert len(set(steps)) == 1
            self.step = steps[0]
            self.m = [s['exp_avg'].cpu().numpy().astype(np.float64) for s in states]
            self.v = [s['exp_avg_sq'].cpu().numpy().astype(np.float64) for s in states]
            assert all(p.shape == m.shape == v.shape for p, m, v in zip(params, self.m, self.v))

    def update(self, grads):
        self.step += 1
        result = []
        c1, c2 = 1. - self.beta1 ** self.step, 1. - self.beta2 ** self.step
        for i, g in enumerate(grads):
            self.m[i] *= self.beta1
            self.m[i] += (1. - self.beta1) * g
            self.v[i] *= self.beta2
            self.v[i] += (1. - self.beta2) * g * g
            result.append(-self.lr / c1 * self.m[i] / (np.sqrt(self.v[i] / c2) + self.eps))
        return result


def squared(x):
    return float(np.mean(np.square(x)))


def relative_rmse(actual, reference):
    return math.sqrt(squared(actual - reference) / max(1., squared(reference)))


def residual_accounting(d, u, anchor_gd, sample_gd, linear_actual, actual_change):
    """Signed, reference-relative accounting, not causal percentages."""
    sampling = sample_gd - anchor_gd
    adam = linear_actual - sample_gd
    nonlinear = actual_change - linear_actual
    parts = np.stack([u, -anchor_gd, -sampling, -adam, -nonlinear])
    dd = u - actual_change
    # Symmetric allocation of cross terms: sum_i <v_i, 2d + delta_d>.
    contributions = np.mean(parts * (2. * d + dd)[None, :], axis=1)
    closure = np.max(np.abs(parts.sum(0) - dd))
    energy_error = abs(contributions.sum() - (squared(d + dd) - squared(d)))
    return parts, contributions, float(closure), float(energy_error)


def spectral_block(params, x, y, q, t, stride):
    _, cache = forward(params, x)
    matrix = ntk(params, cache)
    ev, basis = np.linalg.eigh(matrix)
    assert ev[0] >= -1e-9 * max(1., ev[-1]), ev[0]
    ev = np.maximum(ev, 0.)
    eta = min(3e-4, .45 / max(ev[-1], 1e-30))
    retention = np.power(1. - 2. * eta * ev, 2000)
    mask = retention >= .5
    nslow = int(math.ceil(.3 * len(ev)))
    end = min(t + stride, len(y) - 1)
    dcoeff = basis.T @ (y[t] - q)
    changes = np.diff(y[t:end+1], axis=0)
    energy = np.square(changes @ basis).sum(0) if len(changes) else np.zeros_like(ev)
    first_energy = np.square(changes[0] @ basis) if len(changes) else np.zeros_like(ev)
    return dict(index=t, block_end=end, eta=eta, eigenvalues=ev, eigenvectors=basis,
                residual_energy=dcoeff*dcoeff, drift_energy=energy,
                first_drift_energy=first_energy, retention1000=retention,
                slow_mask=mask, bottom_count=nslow)


def load_window(root, pair, seed, start):
    run = Path(root) / ('P%d_ft_s%d' % (pair, seed))
    window = run / 'target_windows' / ('B_env_%07d' % start)
    complete = json.loads((window / 'complete.json').read_text())
    assert complete['updates'] == 1000
    manifest = json.loads((window / 'manifest.json').read_text())
    assert manifest['updates'] == 1000 and manifest['nominal_start_env_step'] == start
    first = torch.load(str(window / 'start.pt'), map_location='cpu')
    last = torch.load(str(window / 'end.pt'), map_location='cpu')
    refs, batches = [], []
    chunks = sorted(window.glob('rows_through_*.pt'))
    assert len(chunks) == 10
    for path in chunks:
        for row in torch.load(str(path), map_location='cpu'):
            (refs if row['kind'] == 'reference' else batches).append(row)
    assert [r['index'] for r in refs] == list(range(1001))
    assert [r['index'] for r in batches] == list(range(1000))
    assert all(refs[t+1]['global_critic_updates'] == refs[t]['global_critic_updates'] + 1
               for t in range(1000))
    assert first['clocks']['task_env_step'] == start
    assert refs[0]['task_env_step'] == start
    indices = np.asarray(manifest['anchor_indices'])
    assert len(indices) == 256 and len(set(indices)) == 256
    with np.load(str(run / 'anchors/task_1_warmup.npz')) as bank:
        assert not set(bank['episode'][indices[:128]]) & set(bank['episode'][indices[128:]])
        x = np.concatenate([bank['observation'][indices], bank['action'][indices]], 1).astype(np.float64)
    y = np.stack([r['target'] for r in refs]).astype(np.float64)
    saved_q = np.stack([np.stack([r[name] for r in refs]) for name in ('q1', 'q2')]).astype(np.float64)
    xs = [np.concatenate([b['samples']['observation'], b['samples']['action']], 1).astype(np.float64)
          for b in batches]
    ys = [np.asarray(b['predictions']['target'], dtype=np.float64).ravel() for b in batches]
    batch_q = [np.stack([np.asarray(b['predictions'][q], dtype=np.float64).ravel()
                         for b in batches]) for q in ('q1', 'q2')]
    weights = np.load(str(window / 'critic_parameters.npy'), mmap_mode='r')
    shapes = [first['models']['qf1'][name].shape for name in NAMES]
    sizes = [int(np.prod(shape)) for shape in shapes]
    width = sum(sizes)
    assert weights.shape == (1001, 2 * width) and weights.dtype == np.float32
    for qi, name in enumerate(('qf1', 'qf2')):
        assert first['optimizer_parameter_names'][name] == list(NAMES)
        for index, ckpt in ((0, first), (1000, last)):
            flat = np.concatenate([v.ravel() for v in params_from_state(ckpt['models'][name])])
            assert np.array_equal(flat, weights[index, qi*width:(qi+1)*width])
    paths = [window / f for f in ('start.pt', 'end.pt', 'manifest.json', 'complete.json',
                                 'critic_parameters.npy', 'fixed_next_action_epsilon.npy')]
    paths += chunks + [run / 'anchors/task_1_warmup.npz']
    inputs = [dict(path=str(p.resolve()), bytes=p.stat().st_size,
                   mtime_ns=p.stat().st_mtime_ns, sha256=digest(p)) for p in paths]
    return dict(run=run, window=window, first=first, last=last, x=x, y=y, saved_q=saved_q,
                xs=xs, ys=ys, batch_q=batch_q, weights=weights, width=width,
                shapes=shapes, sizes=sizes, inputs=inputs,
                clocks=[{k:r[k] for k in ('global_env_step','task_env_step',
                                        'global_critic_updates','task_critic_updates')} for r in refs])


def recorded_params(data, qi, t):
    flat = data['weights'][t, qi*data['width']:(qi+1)*data['width']].astype(np.float64)
    result, offset = [], 0
    for shape, size in zip(data['shapes'], data['sizes']):
        result.append(flat[offset:offset+size].reshape(shape))
        offset += size
    return result


def analyze_critic(data, qi, stride):
    x, y = data['x'], data['y']
    qname = ('qf1', 'qf2')[qi]
    actual_adam = Adam(recorded_params(data, qi, 0), data['first']['optimizers'][qname])
    metrics = []
    spectra = []
    q, cache = forward(recorded_params(data, qi, 0), x)
    for t in range(1000):
        params = recorded_params(data, qi, t)
        following = recorded_params(data, qi, t+1)
        q, cache = forward(params, x)
        next_q, _ = forward(following, x)
        d = y[t] - q
        u = y[t+1] - y[t]
        delta = [b-a for a,b in zip(params,following)]
        linear = jvp(params, delta, cache)
        ga = gradient(params, x, y[t], cache)
        qb, batchcache = forward(params, data['xs'][t])
        gb = gradient(params, data['xs'][t], data['ys'][t], batchcache)
        qaudit = relative_rmse(q, data['saved_q'][qi,t])
        baudit = relative_rmse(qb, data['batch_q'][qi][t])
        assert max(qaudit,baudit) < 2e-5, (qname,t,qaudit,baudit)
        anchor_gd = jvp(params, [-actual_adam.lr*g for g in ga], cache)
        sample_gd = jvp(params, [-actual_adam.lr*g for g in gb], cache)
        predicted_delta = actual_adam.update(gb)
        delta_error = sum(np.square(a-b).sum() for a,b in zip(delta,predicted_delta))
        delta_norm = sum(np.square(a).sum() for a in delta)
        parts, contrib, closure, energy_error = residual_accounting(
            d,u,anchor_gd,sample_gd,linear,next_q-q)
        scale = max(1., np.max(np.abs(parts)), np.max(np.abs(d)))
        assert closure < 1e-8*scale
        assert energy_error < 1e-7*max(1.,np.abs(contrib).sum(),squared(d))
        row = dict(index=t, residual_mse=squared(d), next_residual_mse=squared(y[t+1]-next_q),
                   target_drift_mse=squared(u), actual_q_change_mse=squared(next_q-q),
                   linear_change_mse=squared(linear), nonlinear_mse=squared(parts[4]),
                   reconstruction_relative_rmse=qaudit, batch_reconstruction_relative_rmse=baudit,
                   adam_step_relative_l2=math.sqrt(delta_error/max(delta_norm,1e-30)),
                   identity_max_abs=closure, energy_identity_abs=energy_error,
                   target_update_cosine=float(np.dot(u,next_q-q)/max(np.linalg.norm(u)*np.linalg.norm(next_q-q),1e-30)),
                   squared_residual_drop_old_target=squared(d)-squared(y[t]-next_q))
        for i,name in enumerate(COMPONENTS):
            row[name+'_mse'] = squared(parts[i])
            row[name+'_energy_contribution'] = float(contrib[i])
        metrics.append(row)
        if t % stride == 0:
            spectra.append(spectral_block(params,x,y,q,t,stride))
    params = recorded_params(data,qi,1000)
    final_q,_ = forward(params,x)
    spectra.append(spectral_block(params,x,y,final_q,1000,stride))
    end_adam = Adam(params,data['last']['optimizers'][qname])
    moment_errors = {}
    for attr in ('m','v'):
        left,right = getattr(actual_adam,attr),getattr(end_adam,attr)
        moment_errors[attr+'_relative_l2'] = math.sqrt(
            sum(np.square(a-b).sum() for a,b in zip(left,right))/
            max(1e-30,sum(np.square(b).sum() for b in right)))
    assert actual_adam.step == end_adam.step
    q0 = forward(recorded_params(data,qi,0),x)[0]
    total_u = sum(float(s['drift_energy'].sum()) for s in spectra)
    total_bottom = sum(float(s['drift_energy'][:s['bottom_count']].sum()) for s in spectra)
    total_slow = sum(float(s['drift_energy'][s['slow_mask']].sum()) for s in spectra)
    total_retention = sum(float(np.dot(s['drift_energy'],s['retention1000'])) for s in spectra)
    null_bottom = spectra[0]['bottom_count']/len(x)
    null_slow = sum(float(s['drift_energy'].sum())*float(s['slow_mask'].mean()) for s in spectra)/max(total_u,1e-30)
    initial_mse, final_mse = squared(y[0]-q0), squared(y[-1]-final_q)
    summary = dict(critic=qname,initial_mse=initial_mse,final_mse=final_mse,
                   final_over_initial=final_mse/max(initial_mse,1e-30),
                   target_net_drift_over_initial_residual=math.sqrt(squared(y[-1]-y[0])/max(initial_mse,1e-30)),
                   drift_bottom30_energy_fraction=total_bottom/max(total_u,1e-30),
                   isotropic_bottom30_fraction=null_bottom,
                   drift_budget_slow_energy_fraction=total_slow/max(total_u,1e-30),
                   isotropic_budget_slow_fraction=null_slow,
                   drift_predicted_residual1000=total_retention/max(total_u,1e-30),
                   nonlinear_over_qchange_rms=math.sqrt(sum(r['nonlinear_mse'] for r in metrics)/max(1e-30,sum(r['actual_q_change_mse'] for r in metrics))),
                   residual_drop_old_target_mean=float(np.mean([r['squared_residual_drop_old_target'] for r in metrics])),
                   reconstruction_relative_rmse_max=max(r['reconstruction_relative_rmse'] for r in metrics),
                   adam_step_relative_l2_mean=float(np.mean([r['adam_step_relative_l2'] for r in metrics])),
                   adam_step_relative_l2_max=max(r['adam_step_relative_l2'] for r in metrics),
                   moment_audit=moment_errors,
                   signed_energy_contributions={name:sum(r[name+'_energy_contribution'] for r in metrics) for name in COMPONENTS},
                   component_rms_over_qchange={name:math.sqrt(sum(r[name+'_mse'] for r in metrics)/max(1e-30,sum(r['actual_q_change_mse'] for r in metrics))) for name in COMPONENTS})
    energy_sum=sum(summary['signed_energy_contributions'].values())
    assert abs(energy_sum-(final_mse-initial_mse))<1e-6*max(1.,initial_mse,abs(energy_sum))
    return summary,metrics,spectra


def replay_critic(data, qi):
    """All original 64-example minibatches/targets; panels are evaluation only."""
    x,y=data['x'],data['y']
    qname=('qf1','qf2')[qi]
    parent=recorded_params(data,qi,0)
    clip=clipped(parent)
    original_q=forward(parent,x)[0]
    jump=forward(clip,x)[0]-original_q
    batch_jumps=[forward(clip,xb)[0]-forward(parent,xb)[0] for xb in data['xs']]
    results,curves={},{}
    for name in BRANCHES:
        params=[p.copy() for p in (clip if name.startswith('clip') else parent)]
        optimizer=Adam(params,data['first']['optimizers'][qname] if 'carried' in name else None)
        corrected=name.endswith('correction')
        offset=jump if corrected else np.zeros_like(jump)
        curve=[]
        for t in range(1001):
            q,_=forward(params,x)
            residual=y[t]+offset-q
            curve.append([squared(residual[:128]),squared(residual[128:]),
                          squared(q-data['saved_q'][qi,t])])
            assert np.isfinite(curve[-1]).all(), (name,t)
            if t<1000:
                target=data['ys'][t]+batch_jumps[t] if corrected else data['ys'][t]
                grads=gradient(params,data['xs'][t],target)
                delta=optimizer.update(grads)
                for p,dp in zip(params,delta):p+=dp
        curve=np.asarray(curve)
        curves[name]=curve
        denom=np.mean(np.square(y[0]-original_q).reshape(2,128),axis=1)
        results[name]=dict(initial_mse=curve[0,:2].tolist(),final_mse=curve[-1,:2].tolist(),
            normalized_auc=(np.trapz(curve[:,:2],axis=0)/1000/np.maximum(denom,1e-30)).tolist(),
            final_over_ft_initial=(curve[-1,:2]/np.maximum(denom,1e-30)).tolist(),
            final_over_own_initial=(curve[-1,:2]/np.maximum(curve[0,:2],1e-30)).tolist(),
            actual_ft_trajectory_rmse=math.sqrt(float(curve[:,2].mean())),
            actual_ft_trajectory_relative_rmse=math.sqrt(float(curve[:,2].mean())/max(1.,squared(data['saved_q'][qi]))))
        if corrected:
            assert np.allclose(curve[0,:2],denom,rtol=1e-9,atol=1e-9)
    audit=results['ft_carried']['actual_ft_trajectory_relative_rmse']
    return dict(critic=qname,branches=results,q_jump_rms=math.sqrt(squared(jump)),
                q_jump_over_ft_initial_residual=math.sqrt(squared(jump)/max(squared(y[0]-original_q),1e-30)),
                original_replay_relative_rmse=audit),curves


def run_job(job):
    root,out,pair,seed,start,stride,script_hash=job
    torch.set_num_threads(1)
    tag='P%d_s%d_B%07d'%(pair,seed,start)
    folder=Path(out)/tag
    folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'complete.json'
    if receipt.exists():
        existing=json.loads(receipt.read_text())
        if existing['script_sha256']==script_hash:
            return dict(tag=tag,status='cached',seconds=0)
        raise RuntimeError('Existing analysis has a different script hash: '+str(folder))
    began=time.time()
    data=load_window(root,pair,seed,start)
    write_json(folder/'inputs.json',data['inputs'])
    summaries=[];step_arrays={};spectral_arrays={};replays=[];replay_arrays={}
    for qi in (0,1):
        summary,metrics,spectra=analyze_critic(data,qi,stride)
        summaries.append(summary)
        for key in metrics[0]:step_arrays['q%d_%s'%(qi+1,key)]=np.asarray([r[key] for r in metrics])
        for key in spectra[0]:spectral_arrays['q%d_%s'%(qi+1,key)]=np.asarray([r[key] for r in spectra])
        print('DECOMPOSED',tag,'q%d'%(qi+1),'seconds',round(time.time()-began,1),flush=True)
        replay,curves=replay_critic(data,qi)
        replays.append(replay)
        for name,curve in curves.items():replay_arrays['q%d_%s'%(qi+1,name)]=curve
        print('REPLAYED',tag,'q%d'%(qi+1),'FT relative RMSE',replay['original_replay_relative_rmse'],
              'seconds',round(time.time()-began,1),flush=True)
    np.savez_compressed(str(folder/'step_metrics.npz'),**step_arrays)
    np.savez_compressed(str(folder/'spectra.npz'),**spectral_arrays)
    np.savez_compressed(str(folder/'replay_curves.npz'),**replay_arrays)
    write_json(folder/'clocks.json',data['clocks'])
    write_json(folder/'summary.json',dict(pair=pair,seed=seed,start_env_step=start,
        source_run=str(data['run']),updates=1000,anchors=256,decomposition=summaries,replay=replays))
    # Read-only input audit catches unexpected changes while the job was running.
    for rec in data['inputs']:
        stat=Path(rec['path']).stat()
        assert (stat.st_size,stat.st_mtime_ns)==(rec['bytes'],rec['mtime_ns']),rec['path']
    write_json(receipt,dict(status='complete',script_sha256=script_hash,seconds=time.time()-began,
        input_files=len(data['inputs']),outputs=['summary.json','step_metrics.npz','spectra.npz','replay_curves.npz']))
    return dict(tag=tag,status='complete',seconds=time.time()-began)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='E1 runs directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pairs',type=int,nargs='+',default=list(range(1,7)))
    parser.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
    parser.add_argument('--starts',type=int,nargs='+',default=[10000,100000])
    parser.add_argument('--spectral-stride',type=int,default=100)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    assert args.spectral_stride>0 and 1000%args.spectral_stride==0
    assert 1<=args.workers<=2
    root,out=args.root.resolve(),args.output.resolve()
    assert root!=out and root not in out.parents and out not in root.parents
    out.mkdir(parents=True,exist_ok=True)
    sh=digest(__file__)
    protocol=dict(version=VERSION,source_root=str(root),script_sha256=sh,precision='float64 CPU',
        target_source='fixed recorded FT actor/target critics/alpha; not regenerated by replay branches',
        parameter_scope='both online critics, all three Linear weights and biases',
        clip='one [0.25,4] weight-only SVD projection at window start; no online intervention',
        clocks='1000 critic updates, not 1000 independently evaluated environment points',
        branches=BRANCHES,replay_data='same 1000 original 64-example minibatches and exact stored targets',
        panels='two disjoint-episode 128-example warmup panels; both are probe evaluation panels, not guaranteed unseen in replay',
        matched_correction='target_t(x)+Q_clip_start(x)-Q_ft_start(x); constant initial-function offset',
        optimizer='Adam lr=3e-4, original carried moments versus fresh moments',
        kernel='exact analytic uncentered full-parameter JJT/n, biases included',
        spectral_stride=args.spectral_stride,
        projection='current K at block start; project next block target increments onto this frozen basis',
        spectral_budget='H=1000, GD MSE operator I-2*eta*K; eta=min(3e-4,0.45/lambda_max)',
        slow_definitions='bottom ceil(0.30*n) and modes retaining >=0.5 squared error after H; isotropic references reported',
        decomposition='delta_d = u - anchor_GD - sampling - Adam - nonlinear',
        sampling='actual minibatch/stochastic-target gradient minus full 256-anchor fixed-noise gradient; not pure sampling variance',
        adam='actual parameter displacement minus same-minibatch SGD reference, mapped through current J',
        signed_energy='C_i=<v_i,2*d+delta_d>/n; exact symmetric cross-term allocation, not causal percentages',
        inference='three seeds; directions/windows/critics/checkpoints are not extra independent seeds')
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text())==json.loads(json.dumps(protocol)), 'Protocol mismatch'
    else:write_json(out/'protocol.json',protocol)
    jobs=[(str(root),str(out),p,s,t,args.spectral_stride,sh)
          for p in args.pairs for s in args.seeds for t in args.starts]
    print('JOBS',len(jobs),'WORKERS',args.workers,'OUTPUT',out,flush=True)
    results=[]
    if args.workers==1:
        for job in jobs:
            result=run_job(job);results.append(result);print('JOB',json.dumps(result),flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_job,job):job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                result=future.result();results.append(result);print('JOB',json.dumps(result),flush=True)
    stamp=time.strftime('%Y%m%d_%H%M%S')
    write_json(out/('invocation_'+stamp+'.json'),dict(jobs=results,pairs=args.pairs,
            seeds=args.seeds,starts=args.starts,workers=args.workers,
            root=str(root),output=str(out)))
    print('COMPLETE',len(results),flush=True)


if __name__=='__main__':
    main()
