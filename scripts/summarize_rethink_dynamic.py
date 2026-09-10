#!/usr/bin/env python3
"""Aggregate completed dynamic-window jobs without editing the original runs."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr


def evaluations(run):
    with (run/'eval_summary.jsonl').open() as handle:
        # Live runs may be appending a later evaluation. Only consume complete lines.
        return [json.loads(line) for line in handle if line.strip() and line.endswith('\n')]


def success_auc(records,position,left,right):
    points={int(r['task_env_step']):float(r['success_rate']) for r in records
            if r['train_task_position']==r['eval_task_position']==position
            and left<=r['task_env_step']<=right}
    expected=list(range(left,right+1,50000))
    assert sorted(points)==expected
    return float(np.trapz([points[t] for t in expected],expected)/(right-left))


def mean_sd(values):
    a=np.asarray(values,dtype=float)
    return float(a.mean()),float(a.std(ddof=1)) if len(a)>1 else None


def csv_write(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def format_ms(values,scale=1):
    m,s=mean_sd(np.asarray(values)*scale)
    return '%.3g ± %.3g'%(m,s) if s is not None else '%.3g (n=1)'%m


def pooled_auc(critic_branches,branch):
    """Pool raw MSE before normalization; panels/critics have equal size."""
    values=[]
    for branches in critic_branches:
        initial=np.asarray(branches['ft_carried']['initial_mse'])
        values.extend(np.asarray(branches[branch]['normalized_auc'])*initial)
    return float(np.mean(values))


def initial_jump_accounting(ft_mse,clip_mse,jump_mse):
    """||d-j||²-||d||² = ||j||²-2<d,j>, in mean-square units."""
    cross=clip_mse-ft_mse-jump_mse
    cosine=-cross/(2*np.sqrt(max(ft_mse*jump_mse,1e-30)))
    return dict(jump_cost=jump_mse,cross_cost=cross,cosine=float(cosine))


def matched_block_direction_score(energy,score,block_weights):
    """Normalize direction within each block, then apply shared time weights."""
    energy=np.asarray(energy);score=np.asarray(score);block_weights=np.asarray(block_weights)
    assert energy.shape==score.shape and block_weights.shape==energy.shape[:1]
    total=energy.sum(1)
    assert np.all((total>0)|(block_weights==0))
    return float(np.dot(block_weights,(energy*score).sum(1)/np.maximum(total,1e-30))/
                 max(block_weights.sum(),1e-30))


def summarize(folder):
    result=json.loads((folder/'summary.json').read_text())
    ds=result['decomposition']
    p,seed,start=result['pair'],result['seed'],result['start_env_step']
    run=Path(result['source_run'])
    counterpart=p+1 if p%2 else p-1
    ev=evaluations(run)
    fresh=evaluations(run.parent/('P%d_ft_s%d'%(counterpart,seed)))
    row=dict(pair=p,seed=seed,start_env_step=start,
        delta_success_auc_0_100k=success_auc(ev,1,0,100000)-success_auc(fresh,0,0,100000),
        delta_success_auc_0_300k=success_auc(ev,1,0,300000)-success_auc(fresh,0,0,300000),
        delta_success_auc_100k_300k=success_auc(ev,1,100000,300000)-success_auc(fresh,0,100000,300000),
        initial_td_mse=np.mean([d['initial_mse'] for d in ds]),
        final_td_mse=np.mean([d['final_mse'] for d in ds]),
        drift_bottom30=np.mean([d['drift_bottom30_energy_fraction'] for d in ds]),
        isotropic_bottom30=ds[0]['isotropic_bottom30_fraction'],
        drift_budget_slow=np.mean([d['drift_budget_slow_energy_fraction'] for d in ds]),
        isotropic_budget_slow=np.mean([d['isotropic_budget_slow_fraction'] for d in ds]),
        drift_retention1000=np.mean([d['drift_predicted_residual1000'] for d in ds]),
        reconstruction_relative_rmse_max=max(d['reconstruction_relative_rmse_max'] for d in ds),
        adam_step_relative_l2_mean=np.mean([d['adam_step_relative_l2_mean'] for d in ds]),
        adam_moment_relative_l2_max=max(v for d in ds for v in d['moment_audit'].values()),
        ft_replay_relative_rmse=max(d['original_replay_relative_rmse'] for d in result['replay']))
    row['final_over_initial_td_mse']=row['final_td_mse']/row['initial_td_mse']
    parts=('target','anchor_gd','sampling','adam','nonlinear')
    with np.load(str(folder/'step_metrics.npz')) as steps:
        qenergy=sum(float(steps['q%d_actual_q_change_mse'%q].sum()) for q in (1,2))
        for name in parts:
            energy=sum(float(steps['q%d_%s_mse'%(q,name)].sum()) for q in (1,2))
            row[name+'_rms_over_qchange']=np.sqrt(energy/max(qenergy,1e-30))
            row[name+'_signed_delta_mse_over_initial']=np.mean([
                d['signed_energy_contributions'][name] for d in ds])/row['initial_td_mse']
        row['target_qchange_cosine']=np.mean([steps['q%d_target_update_cosine'%q].mean() for q in (1,2)])
        row['identity_abs_max']=max(float(steps['q%d_identity_max_abs'%q].max()) for q in (1,2))
        row['energy_identity_abs_max']=max(float(steps['q%d_energy_identity_abs'%q].max()) for q in (1,2))
        row['one_step_nonlinear_relative_max']=max(float(np.sqrt(
            steps['q%d_nonlinear_mse'%q]/np.maximum(steps['q%d_actual_q_change_mse'%q],1e-30)).max()) for q in (1,2))
        original_areas=[];original_mean_errors=[]
        for q in (1,2):
            error=np.r_[steps['q%d_residual_mse'%q],steps['q%d_next_residual_mse'%q][-1]]
            original_areas.append(float(np.trapz(error)/1000))
            original_mean_errors.append(float(error.mean()))
        row['original_online_td_mse_auc']=float(np.mean(original_areas))
    row['linear_update_signed_delta_mse_over_initial']=sum(
        row[k+'_signed_delta_mse_over_initial'] for k in ('anchor_gd','sampling','adam'))
    row['actual_update_signed_delta_mse_over_initial']=(row['linear_update_signed_delta_mse_over_initial']
        +row['nonlinear_signed_delta_mse_over_initial'])
    assert np.isclose(row['target_signed_delta_mse_over_initial']+
        row['actual_update_signed_delta_mse_over_initial'],row['final_over_initial_td_mse']-1,rtol=1e-6,atol=1e-6)
    with np.load(str(folder/'spectra.npz')) as spectra:
        slow_first=[];slow_last=[];null_first=[]
        residual_bottom_num=residual_total=0.
        residual_slow_num=residual_null_num=0.
        raw_sgd_multipliers=[];stable_steps=[]
        current_drift_bottom_num=current_drift_energy=0.
        matched_residual_slow=[];matched_residual_retention=[];matched_residual_bottom=[]
        for q in (1,2):
            d=spectra['q%d_residual_energy'%q]
            masks=spectra['q%d_slow_mask'%q]
            slow_first.append(d[0,masks[0]].sum()/max(d[0].sum(),1e-30))
            slow_last.append(d[-1,masks[-1]].sum()/max(d[-1].sum(),1e-30))
            null_first.append(masks[0].mean())
            bottom=int(spectra['q%d_bottom_count'%q][0])
            residual_total+=d.sum()
            residual_bottom_num+=d[:,:bottom].sum()
            residual_slow_num+=(d*masks).sum()
            residual_null_num+=(d.sum(1)*masks.mean(1)).sum()
            eigenvalues=spectra['q%d_eigenvalues'%q]
            raw_sgd_multipliers.extend(2*3e-4*eigenvalues[:,-1])
            stable_steps.extend(spectra['q%d_eta'%q])
            # Compare new increments and the current residual in the SAME K_t,
            # using the SAME block weights. The final endpoint has no future block.
            incoming=spectra['q%d_drift_energy'%q]
            weights=incoming.sum(1)
            retention=spectra['q%d_retention1000'%q]
            matched_residual_slow.append(matched_block_direction_score(d,masks,weights))
            matched_residual_retention.append(matched_block_direction_score(d,retention,weights))
            bottom_mask=np.broadcast_to(np.arange(d.shape[1])<bottom,d.shape)
            matched_residual_bottom.append(matched_block_direction_score(d,bottom_mask,weights))
            energy=spectra['q%d_first_drift_energy'%q]
            bottom=int(spectra['q%d_bottom_count'%q][0])
            current_drift_bottom_num+=energy[:,:bottom].sum()
            current_drift_energy+=energy.sum()
        row['residual_initial_budget_slow']=np.mean(slow_first)
        row['residual_final_budget_slow']=np.mean(slow_last)
        row['current_snapshot_drift_bottom30']=current_drift_bottom_num/max(current_drift_energy,1e-30)
        row['snapshot_residual_bottom30']=residual_bottom_num/max(residual_total,1e-30)
        row['snapshot_residual_budget_slow']=residual_slow_num/max(residual_total,1e-30)
        row['snapshot_residual_budget_slow_null']=residual_null_num/max(residual_total,1e-30)
        row['raw_sgd_2lr_lambda_max']=max(raw_sgd_multipliers)
        row['stable_spectral_lr_min']=min(stable_steps)
        row['stable_spectral_lr_max']=max(stable_steps)
        row['current_residual_slow_matched_to_drift_blocks']=np.mean(matched_residual_slow)
        row['current_residual_retention1000_matched_to_drift_blocks']=np.mean(matched_residual_retention)
        row['current_residual_bottom30_matched_to_drift_blocks']=np.mean(matched_residual_bottom)
        row['incoming_minus_current_slow_fraction']=(row['drift_budget_slow']-
            row['current_residual_slow_matched_to_drift_blocks'])
        row['incoming_minus_current_retention1000']=(row['drift_retention1000']-
            row['current_residual_retention1000_matched_to_drift_blocks'])
    # Primary replay statistic pools both equally sized panels and both critics.
    # Do not average separately normalized panels: their initial errors can differ greatly.
    branches=result['replay'][0]['branches']
    for branch in branches:
        final=[];initial=[]
        for replay in result['replay']:
            b=replay['branches'][branch]
            final.extend(b['final_mse']);initial.extend(b['initial_mse'])
        row[branch+'_mse_auc']=pooled_auc([r['branches'] for r in result['replay']],branch)
        row[branch+'_final_mse']=float(np.mean(final))
        row[branch+'_initial_mse']=float(np.mean(initial))
        row[branch+'_normalized_auc']=row[branch+'_mse_auc']/row['initial_td_mse']
    for optimizer in ('carried','fresh'):
        base=row['ft_'+optimizer+'_mse_auc']
        for condition in ('target','correction'):
            name='clip_'+optimizer+'_'+condition
            row[name+'_auc_ratio_to_ft']=row[name+'_mse_auc']/base
            row[name+'_final_ratio_to_ft']=row[name+'_final_mse']/row['ft_'+optimizer+'_final_mse']
    with np.load(str(folder/'replay_curves.npz')) as curves:
        for branch in branches:
            row[branch+'_mse_at200']=float(np.mean([curves['q%d_%s'%(q,branch)][200,:2] for q in (1,2)]))
            row[branch+'_normalized_mse_at200']=row[branch+'_mse_at200']/row['initial_td_mse']
        for optimizer in ('carried','fresh'):
            for condition in ('target','correction'):
                name='clip_'+optimizer+'_'+condition
                row[name+'_ratio_to_ft_at200']=row[name+'_mse_at200']/row['ft_'+optimizer+'_mse_at200']
    row['ft_replay_auc_ratio_to_original']=row['ft_carried_mse_auc']/row['original_online_td_mse_auc']
    row['ft_replay_error_over_original_residual']=np.sqrt(
        np.mean([r['branches']['ft_carried']['actual_ft_trajectory_rmse']**2 for r in result['replay']])/
        np.mean(original_mean_errors))
    row['ft_replay_audit_pass']=int(abs(row['ft_replay_auc_ratio_to_original']-1)<.01
                                  and row['ft_replay_error_over_original_residual']<.05)
    row['q_jump_rms']=np.sqrt(np.mean([r['q_jump_rms']**2 for r in result['replay']]))
    row['q_jump_over_initial_residual']=row['q_jump_rms']/np.sqrt(row['initial_td_mse'])
    jump=initial_jump_accounting(row['initial_td_mse'],row['clip_carried_target_initial_mse'],row['q_jump_rms']**2)
    row['initial_q_jump_residual_cosine']=jump['cosine']
    row['initial_q_jump_cost_over_residual']=jump['jump_cost']/row['initial_td_mse']
    row['initial_q_jump_cross_cost_over_residual']=jump['cross_cost']/row['initial_td_mse']
    row['clip_initial_mse_ratio_to_ft']=row['clip_carried_target_initial_mse']/row['initial_td_mse']
    row['ft_fresh_auc_ratio_to_carried']=row['ft_fresh_mse_auc']/row['ft_carried_mse_auc']
    row['clip_fresh_target_auc_ratio_to_carried']=row['clip_fresh_target_mse_auc']/row['clip_carried_target_mse_auc']
    row['clip_fresh_correction_auc_ratio_to_carried']=row['clip_fresh_correction_mse_auc']/row['clip_carried_correction_mse_auc']
    assert abs(jump['cosine'])<=1+1e-7
    geometry_path=folder/'geometry.json'
    if geometry_path.exists():
        geometry_record=json.loads(geometry_path.read_text())
        geometry=geometry_record['critics']
        for key,value in geometry_record['bellman_source'].items():row['bellman_'+key]=value
        sync=geometry_record['entry_sync']
        for key,value in sync.items():
            if isinstance(value,(int,float)):row['entry_sync_'+key]=value
        row['entry_sync_clip_ratio_to_recorded_ft']=sync['clip_hard_sync_target_mse']/sync['ft_recorded_target_mse']
        row['entry_sync_clip_ratio_to_synced_ft']=sync['clip_hard_sync_target_mse']/sync['ft_hard_sync_target_mse']
        for label in ('ft','clip'):
            for key in ('lambda_max','trace','entropy_effective_rank','participation_rank','trace_rank99'):
                row[label+'_kernel_'+key]=np.mean([g[label+'_kernel'][key] for g in geometry])
        for condition in ('common_stable_lr','separately_stable_lr'):
            group=[g['conditions'][condition] for g in geometry]
            base=np.mean([g['ft_auc'] for g in group])
            for key in group[0]:
                row['surrogate_'+condition+'_'+key]=np.mean([g[key] for g in group])
            for variant in ('correction','target'):
                row['surrogate_'+condition+'_clip_'+variant+'_auc_ratio_to_ft']=np.mean(
                    [g['clip_'+variant+'_auc'] for g in group])/base
    assert all(np.isfinite(v) for v in row.values())
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    folders=sorted(p.parent for p in out.glob('P*_s*_B*/complete.json'))
    geometry_count=sum((folder/'geometry.json').exists() for folder in folders)
    assert geometry_count in (0,len(folders)), 'Complete geometry for all selected windows before aggregation'
    rows=[summarize(p) for p in folders]
    expected={(p,s,t) for p in range(1,7) for s in (1,2,3) for t in (10000,100000)}
    actual={(r['pair'],r['seed'],r['start_env_step']) for r in rows}
    assert actual<=expected and len(actual)==len(rows)
    if not args.allow_partial:assert actual==expected,sorted(expected-actual)
    assert rows,'No complete windows'
    prefix='PARTIAL_' if args.allow_partial else ''
    csv_write(out/(prefix+'window_metrics.csv'),rows)
    groups=[]
    for p in range(1,7):
        for t in (10000,100000):
            group=[r for r in rows if r['pair']==p and r['start_env_step']==t]
            if not group:continue
            rec=dict(pair=p,start_env_step=t,seeds=[r['seed'] for r in group])
            for key in rows[0]:
                if key not in ('pair','seed','start_env_step'):
                    m,s=mean_sd([r[key] for r in group]);rec[key]=dict(mean=m,sample_sd=s)
            groups.append(rec)
    correlations=[]
    # Direction/seed observations are related; these are descriptive, without p-values.
    for start in (10000,100000):
        group=[r for r in rows if r['start_env_step']==start]
        if len(group)<3:continue
        target='delta_success_auc_0_300k' if start==10000 else 'delta_success_auc_100k_300k'
        for feature in ('drift_bottom30','drift_budget_slow','drift_retention1000',
                        'residual_initial_budget_slow','nonlinear_rms_over_qchange',
                        'clip_carried_target_auc_ratio_to_ft','clip_carried_correction_auc_ratio_to_ft'):
            fx=[r[feature] for r in group];fy=[r[target] for r in group]
            rho=float(spearmanr(fx,fy).correlation) if len(set(fx))>1 and len(set(fy))>1 else float('nan')
            correlations.append(dict(start=start,feature=feature,outcome=target,n=len(group),
                                     spearman=rho if np.isfinite(rho) else None))
    evidence=dict(completed=len(rows),expected=36,missing=sorted(expected-actual),groups=groups,
        replay_audit=dict(criteria='MSE-AUC ratio within 1% and Q replay RMSE / original residual RMS below 5%',
                          failed=[dict(pair=r['pair'],seed=r['seed'],start=r['start_env_step'])
                                  for r in rows if not r['ft_replay_audit_pass']]),
        descriptive_correlations=correlations,
        provenance='protocol.json and each window inputs.json include definitions and SHA-256',
        online_scope='local P1-P6 FT versus matched fresh only; no matched formal online Clip/Reset branches supplied')
    with (out/(prefix+'aggregate.json')).open('w') as handle:json.dump(evidence,handle,indent=2,allow_nan=False)
    md=['# P1–P6 动态窗口分析'+('（未完成快照）' if args.allow_partial else ''),'',
        '更新时间：'+time.strftime('%Y-%m-%d %H:%M:%S')+'。',
        '','## 范围与核验','',
        '- 完成 %d/36 个窗口；每个 1000 次实际更新，包含双 critic。'%len(rows),
        '- 只使用本地 P1–P6。原训练、actor、target 网络和环境没有被改动。',
        '- FT 重放相对原 Q 轨迹的相对 RMSE：中位数 %.3g，最大 %.3g。'%(
            np.median([r['ft_replay_relative_rmse'] for r in rows]),max(r['ft_replay_relative_rmse'] for r in rows)),
        '- 逐步模型重建最大相对 RMSE %.3g；Adam moments 重建最大相对 L2 误差 %.3g。'%(
            max(r['reconstruction_relative_rmse_max'] for r in rows),max(r['adam_moment_relative_l2_max'] for r in rows)),
        '- FT 重放相对原轨迹的动态 MSE-AUC 比值范围 %.6g–%.6g；Q 重放误差 / 原 residual RMS 最大 %.3g。'%(
            min(r['ft_replay_auc_ratio_to_original'] for r in rows),max(r['ft_replay_auc_ratio_to_original'] for r in rows),
            max(r['ft_replay_error_over_original_residual'] for r in rows)),
        '- 重放核验：%d/%d 窗口满足 MSE-AUC 偏差 <1%% 且 Q 重放误差 < 原 residual RMS 的5%%；未通过者必须标注，不能当作可靠配对重放。'%(
            sum(r['ft_replay_audit_pass'] for r in rows),len(rows)),
        '- 表格按同方向、同窗口先汇总双 critic，再跨 seeds 计算均值±样本 SD；窗口/critic/更新不是额外独立 seeds。',
        '', '## 1. 新增 target 修正的方向','',
        'K=JJᵀ/n，包含全部参数和 bias，不中心化。每 100 次更新重算一次当前 K，投影随后 100 次 target 增量；另存恰在采样点的一步投影作口径检查。',
        '每个anchor使用一个跨时间固定的Gaussian next-action噪声；这是被记录的SAC target轨迹，不是精确的策略期望，更不是已知Q*。输入固定在B任务warm-up panel，不代表所有后续在线访问状态。',
        '“最慢30%”的各向同性能量参照为 30.08%。预算慢模态是稳定 GD 校准下 H=1000 后仍保留至少一半平方误差的模态；它不是对 Adam 的无条件学习速度判断。',
        '', '| 方向 | B窗口起点 | 新增需求落入最慢30%（%） | 预算慢模态能量（%） | 同维度各向同性参照（%） | 残差末/初 |',
        '|---|---:|---:|---:|---:|---:|']
    for g in groups:
        group=[r for r in rows if r['pair']==g['pair'] and r['start_env_step']==g['start_env_step']]
        md.append('| P%d | %dk | %s | %s | %s | %s |'%(g['pair'],g['start_env_step']//1000,
            format_ms([r['drift_bottom30'] for r in group],100),
            format_ms([r['drift_budget_slow'] for r in group],100),
            format_ms([r['isotropic_budget_slow'] for r in group],100),
            format_ms([r['final_over_initial_td_mse'] for r in group])))
    md+=['','新增需求的能量分布和剩余误差的分布分别列出，避免把二者混为一谈。下表的误差占比按11个谱快照、双critic的原始能量合并；两者均不是完整状态空间上的价值误差。',
         '', '| 方向 | B窗口起点 | 新增需求最慢30%：采样点即时口径（%） | 剩余误差最慢30%（%） | 剩余误差预算慢模态（%） |',
         '|---|---:|---:|---:|---:|']
    for g in groups:
        group=[r for r in rows if r['pair']==g['pair'] and r['start_env_step']==g['start_env_step']]
        md.append('| P%d | %dk | %s | %s | %s |'%(g['pair'],g['start_env_step']//1000,*[
            format_ms([r[k] for r in group],100) for k in ('current_snapshot_drift_bottom30',
                'snapshot_residual_bottom30','snapshot_residual_budget_slow')]))
    md+=['','### 新增需求是否比当前残差更难','',
         '在同一个K_t、同一个稳定GD步长下，直接比较当前残差d_t与随后100次新增target修正u的方向。两者使用相同的block权重（该block的新增修正总能量），避免时点加权不同造成假差异。',
         'R1000是校准固定核GD预测的归一化剩余平方误差，越大表示该方向在此代理模型下越难。它衡量方向难度，不包含需求绝对幅度，也不是实际Adam的1000步误差预测。',
         '', '| 方向 | B窗口起点 | 当前残差R1000（%） | 新增修正R1000（%） | 新增−当前预算慢模态能量（百分点） |',
         '|---|---:|---:|---:|---:|']
    for g in groups:
        group=[r for r in rows if r['pair']==g['pair'] and r['start_env_step']==g['start_env_step']]
        md.append('| P%d | %dk | %s | %s | %s |'%(g['pair'],g['start_env_step']//1000,*[
            format_ms([r[k] for r in group],100) for k in (
                'current_residual_retention1000_matched_to_drift_blocks','drift_retention1000',
                'incoming_minus_current_slow_fraction')]))
    md+=['','## 2. 实际更新分解','',
        'd=y−Q；Δd=u−g−s−a−r。g 为同 lr、固定256输入和固定噪声目标上的 GD 参考；s 为真实 minibatch/随机目标梯度与参考梯度的差；a 为实际 Adam 位移相对真实 minibatch SGD 的差；r=ΔQ−JΔθ 是非线性余项。',
        '采样项同时包含状态/动作 minibatch 与目标动作采样，不能细称纯 minibatch 方差。g、s、a 可能强烈抵消；不把它们的范数归一化成因果百分比。CSV 保存对 ΔMSE 的精确、有符号、对称交叉项分摊。',
        '非线性RMS比是逐步重算J的单步余项尺度，不是总误差的因果百分比，也不表示整个1000步过程具有固定K。',
        '', '| 方向 | B窗口起点 | 非线性RMS / 实际ΔQ RMS（%） | target漂移RMS / ΔQ RMS（%） |',
        '|---|---:|---:|---:|']
    for g in groups:
        group=[r for r in rows if r['pair']==g['pair'] and r['start_env_step']==g['start_env_step']]
        md.append('| P%d | %dk | %s | %s |'%(g['pair'],g['start_env_step']//1000,
            format_ms([r['nonlinear_rms_over_qchange'] for r in group],100),
            format_ms([r['target_rms_over_qchange'] for r in group],100)))
    md+=['','### 目标变化与实际更新分别做了什么','',
         '先看不涉及普通GD参考的三项，全部除以窗口初始MSE：正数增加误差，负数降低误差，三项之和等于“末/初MSE−1”。这是精确的对称交叉项分摊，不是独立因果效应。',
         '', '| 方向 | B窗口起点 | target变化 | 实际线性更新 JΔθ | 非线性余项 |',
         '|---|---:|---:|---:|---:|']
    for g in groups:
        group=[r for r in rows if r['pair']==g['pair'] and r['start_env_step']==g['start_env_step']]
        md.append('| P%d | %dk | %s | %s | %s |'%(g['pair'],g['start_env_step']//1000,*[
            format_ms([r[k+'_signed_delta_mse_over_initial'] for r in group])
            for k in ('target','linear_update','nonlinear')]))
    md+=['','### Adam／采样的参考相对账本','',
         '下表除以窗口初始 MSE。五列之和严格等于“末/初 MSE−1”；正数增加误差，负数降低误差。数值不是百分比或可相加的独立因果效应，尤其 GD/采样/Adam 可以出现很大的抵消。',
         '', '| 方向 | B窗口起点 | target漂移 | 固定panel GD | 采样差异 | Adam差异 | 非线性 |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        group=[r for r in rows if r['pair']==g['pair'] and r['start_env_step']==g['start_env_step']]
        md.append('| P%d | %dk | %s | %s | %s | %s | %s |'%(g['pair'],g['start_env_step']//1000,*[
            format_ms([r[k+'_signed_delta_mse_over_initial'] for r in group])
            for k in ('target','anchor_gd','sampling','adam','nonlinear')]))
    md+=['','## 3. 共同动态流的 Clip 对照','',
        '每个分支读取相同的1000个原始64样本 minibatch 和对应保存 targets。Clip 只在窗口入口对权重做一次[0.25,4]投影；不重新生成它自己的 targets。B=100k 的入口干预只是离线诊断，不是正式每200k协议的一次在线事件。',
        '同 target 比较包含 Q-jump；同 correction 使用 y_t(x)+Q_clip,0(x)−Q_ft,0(x)，消除初始函数差异。分别用 carried 与 fresh Adam，均为 lr=3e-4。',
        '同correction是离线诊断构造，不是额外的SAC算法：它匹配初始函数误差，仍允许后续非线性和Adam动力学不同，因此不单独证明只有初始K造成差异。',
        '两个128状态 panel 来自不同 warmup episodes，但可能出现在真实 replay minibatch 中，因此称固定诊断 panel，不宣称独立 held-out 验证集。',
        '下表为 Clip / FT 的1000更新动态 MSE-AUC 比值：小于1表示 Clip 误差更低。先在全部256输入及双 critic 上等权合并原始 MSE，再算比值，避免不同panel的初始误差量级影响归一化。',
        '', '| 方向 | B窗口起点 | carried：同target | carried：同correction | fresh：同target | fresh：同correction |',
        '|---|---:|---:|---:|---:|---:|']
    for g in groups:
        group=[r for r in rows if r['pair']==g['pair'] and r['start_env_step']==g['start_env_step']]
        md.append('| P%d | %dk | %s | %s | %s | %s |'%(g['pair'],g['start_env_step']//1000,*[
            format_ms([r['clip_'+o+'_'+c+'_auc_ratio_to_ft'] for r in group])
            for o,c in [('carried','target'),('carried','correction'),('fresh','target'),('fresh','correction')]]))
    md+=['','### Q 跳变与 Adam 状态对照','',
         'Q跳变/初始修正是 RMS 比；裁剪入口误差比直接测量函数扰动的即时净代价。fresh/carried 的比值小于1表示在这个离线目标流上清空该分支的 Adam 状态降低了误差面积；不等同于正式在线 reset 方法的收益。',
         '', '| 方向 | B窗口起点 | Q跳变/初始修正 | Clip入口MSE / FT入口MSE | FT：fresh/carried AUC | Clip同target：fresh/carried AUC |',
         '|---|---:|---:|---:|---:|---:|']
    for g in groups:
        group=[r for r in rows if r['pair']==g['pair'] and r['start_env_step']==g['start_env_step']]
        md.append('| P%d | %dk | %s | %s | %s | %s |'%(g['pair'],g['start_env_step']//1000,*[
            format_ms([r[k] for r in group]) for k in ('q_jump_over_initial_residual',
                'clip_initial_mse_ratio_to_ft','ft_fresh_auc_ratio_to_carried','clip_fresh_target_auc_ratio_to_carried')]))
    if geometry_count:
        md+=['','### 固定核、真实动态需求下的几何/函数代价分解','',
             '额外计算 d[t+1]=(I−2ηK₀)d[t]+u[t]，使用原始1000步 u[t]。这是固定核、完整anchor GD的校准模型，不能冒充实际小批次 Adam 重放。',
             '相同稳定步长使用 η=min(3e-4,0.45/max(λmax,FT,λmax,Clip))；分别校准使用各自 λmax，后者同时改变步长，不称纯几何对照。',
             '这里K有效秩采用 exp(−Σpᵢlog pᵢ)，pᵢ=λᵢ/Σλ；它不是训练日志里的feature rank。其他谱指标和原始特征值同样保存，不能把权重奇异值更均匀直接等同于学习核更好。',
             '动态误差恒等式也经数值核验：FT误差−Clip同target误差 = 同correction改善 + 传播后的Q跳变对齐项 − 传播后的Q跳变平方代价。详细逐步值见各 geometry_curves.npz。',
             '', '| 方向 | B窗口起点 | K有效秩 FT→Clip | 相同步长：同correction误差比 | 分别校准：同correction误差比 |',
             '|---|---:|---:|---:|---:|']
        for g in groups:
            group=[r for r in rows if r['pair']==g['pair'] and r['start_env_step']==g['start_env_step']]
            md.append('| P%d | %dk | %s → %s | %s | %s |'%(g['pair'],g['start_env_step']//1000,
                format_ms([r['ft_kernel_entropy_effective_rank'] for r in group]),
                format_ms([r['clip_kernel_entropy_effective_rank'] for r in group]),*[
                format_ms([r['surrogate_'+c+'_clip_correction_auc_ratio_to_ft'] for r in group])
                for c in ('common_stable_lr','separately_stable_lr')]))
        md+=['','### 入口 target 同步的即时核查','',
             '正式 Clip 在任务入口会同步 target critic，而共同外部target重放有意移除了这条反馈。因此额外核查：在B=10k、尚未做B优化的同源状态，用同一个已保存actor/action噪声/alpha，把target同步到裁剪后的双Q，再计算一次TD误差。也对FT做对称同步参照。',
             '这里只生成一个入口target，没有生成新的1000步目标流或在线收益；误差变小也可能来自目标自身收缩，不能据此证明更准确的价值估计或更高策略收益。',
             '', '| 方向 | Clip共同FT target入口误差比 | Clip同步target / 原记录FT误差 | Clip同步target / 对称同步FT误差 |',
             '|---|---:|---:|---:|']
        for p in range(1,7):
            group=[r for r in rows if r['pair']==p and r['start_env_step']==10000]
            if group:md.append('| P%d | %s | %s | %s |'%(p,*[
                format_ms([r[k] for r in group]) for k in ('clip_initial_mse_ratio_to_ft',
                    'entry_sync_clip_ratio_to_recorded_ft','entry_sync_clip_ratio_to_synced_ft')]))
    md+=['','## 4. 与本地在线结果对应','',
        '只有 E1 FT 与匹配 fresh 的正式在线记录；不存在可配对的正式在线 Clip/Reset 收益。下表不是把离线拟合改善冒充在线策略收益。300k 是此前共同已完成进度下的阶段性窗口，不是完整1.5M终点。',
        '', '| 方向 | 前100k FT−fresh success-AUC（百分点） | 前300k | 100k–300k |',
        '|---|---:|---:|---:|']
    for p in range(1,7):
        byseed={r['seed']:r for r in rows if r['pair']==p};group=list(byseed.values())
        if group:md.append('| P%d | %s | %s | %s |'%(p,*[format_ms([r[k] for r in group],100) for k in
            ['delta_success_auc_0_100k','delta_success_auc_0_300k','delta_success_auc_100k_300k']]))
    md+=['','## 数据与复算','',
        '- 完整定义：[protocol.json](protocol.json)。',
        '- 全部窗口级指标：[%swindow_metrics.csv](%swindow_metrics.csv)。'%(prefix,prefix),
        '- 三seed汇总及描述性相关：[%saggregate.json](%saggregate.json)。'%(prefix,prefix),
        '- 每个窗口目录保存 input SHA-256、逐步分解、完整谱/投影、所有重放曲线和完成凭据。',
        '- 主分析用全部1000次更新；CSV另存第200次更新的误差，以便与大纲原定200步探针对照，不挑选更有利的时间点。',
        '- 相关系数只作描述；不把18个方向/seed组合或36个窗口当作独立训练seeds做显著性宣称。',
        '- 这是固定 FT 外部目标流上的条件化比较；真实 Clip 会改变 target 同步、actor、探索和后续数据。本轮不能直接判定其在线胜负。','']
    (out/(prefix+'DYNAMIC_ANALYSIS.md')).write_text('\n'.join(md))
    print(json.dumps(dict(completed=len(rows),missing=len(expected-actual),
        ft_replay_rmse_max=max(r['ft_replay_relative_rmse'] for r in rows),
        carried_same_target_clip_wins=sum(r['clip_carried_target_auc_ratio_to_ft']<1 for r in rows),
        carried_same_correction_clip_wins=sum(r['clip_carried_correction_auc_ratio_to_ft']<1 for r in rows)),indent=2))


if __name__=='__main__':main()
