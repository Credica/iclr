#!/usr/bin/env python3
"""Aggregate the frozen R3-A/B/C protocol, retaining all directions and seeds."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analyze_rethink_dynamic import digest, write_json
from summarize_rethink_dynamic import success_auc, format_ms

TASKS = {1:'sweep-into → push-wall', 2:'push-wall → sweep-into',
    3:'button-press → button-press-wall', 4:'button-press-wall → button-press',
    5:'reach → window-close', 6:'window-close → reach'}


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def ratio(a, b):
    return float(a / b) if a is not None and b is not None and b > 1e-30 else None


def write_csv(path, rows):
    names = sorted(set(k for r in rows for k in r))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader(); writer.writerows(rows)


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(set(x)) < 2 or len(set(y)) < 2: return None
    return float(spearmanr(x, y).correlation)


def pool_window(records):
    first = records[0]
    row = dict(pair=first['pair'], seed=first['seed'], start=first['start'])
    for name in first['families']:
        fs = [r['families'][name] for r in records]
        for metric in ('rank95', 'energy', 'mean_vector_energy_fraction'):
            row[name + '_' + metric] = mean(f['structure'].get(metric) for f in fs)
        for model in fs[0]['geometries']:
            scores = [f['geometries'][model] for f in fs]
            for metric in ('shape_remaining1000', 'raw_remaining1000', 'slow_energy_fraction',
                           'subspace_fast90_overlap', 'uniform_subspace_shape_burden', 'fast90_demand_energy'):
                row[name + '_' + model + '_' + metric] = mean(s.get(metric) for s in scores)
            for ridge in ('0.0001', '0.001', '0.01'):
                for metric in ('shape_burden', 'raw_burden', 'slow_burden_fraction'):
                    row[name + '_' + model + '_r' + ridge + '_' + metric] = mean(
                        s.get('ridge_' + ridge, {}).get(metric) for s in scores)
        for ref in ('initial', 'fresh_matched', 'A_end'):
            for metric in ('shape_burden', 'raw_burden'):
                key = name + '_ft_start_r0.001_' + metric
                row[name + '_ft_over_' + ref + '_' + metric] = ratio(row[key], row[name + '_' + ref + '_r0.001_' + metric])
    for group in ('slow', 'fast', 'all'):
        gs = [r['tracking']['groups'][group] for r in records]
        for metric in ('initial_mse', 'final_mse', 'auc', 'correction_gain'):
            row[group + '_' + metric] = mean(g.get(metric) for g in gs)
        row[group + '_final_over_initial'] = ratio(row[group + '_final_mse'], row[group + '_initial_mse'])
        row[group + '_auc_over_initial'] = ratio(row[group + '_auc'], row[group + '_initial_mse'])
        for component in ('target', 'anchor_gd', 'sampling', 'adam', 'nonlinear'):
            row[group + '_' + component + '_delta_mse'] = mean(g.get('signed_delta_mse', {}).get(component) for g in gs)
    row['slow_initial_energy_fraction'] = ratio(row['slow_initial_mse'], row['all_initial_mse'])
    row['slow_final_energy_fraction'] = ratio(row['slow_final_mse'], row['all_final_mse'])
    row['slow_vs_fast_retention_ratio'] = ratio(row['slow_final_over_initial'], row['fast_final_over_initial'])
    row['slow_vs_fast_gain_ratio'] = ratio(row['slow_correction_gain'], row['fast_correction_gain'])
    row['prefix_subspace_suffix_coverage'] = mean(r['prefix_subspace_transfer']['suffix_energy_coverage'] for r in records)
    row['kernel_shape_relative_change'] = mean(r['stability']['normalized_kernel_relative_change'] for r in records)
    row['nonlinear_relative_rms'] = mean(r['nonlinear_over_actual_change_rms'] for r in records)
    row['reconstruction_error_max'] = max(r['reconstruction_relative_rmse_max'] for r in records)
    return row


def report(out, partial=False):
    protocol = json.loads((out / 'protocol.json').read_text())
    ev = json.loads((out / 'evaluation_snapshot.json').read_text())
    folders = sorted(p.parent for p in out.glob('P*_s*/complete.json'))
    if not partial: assert len(folders) == 18, len(folders)
    rows, histories, sources = [], [], {}
    audit_error = 0.
    for folder in folders:
        receipt = json.loads((folder / 'complete.json').read_text())
        assert receipt['code_hash'] == protocol['code_hashes']
        h = json.loads((folder / 'history.json').read_text())
        for stage in sorted(set(r['stage'] for r in h)):
            rs = [r for r in h if r['stage'] == stage]
            histories.append(dict(pair=rs[0]['pair'], seed=rs[0]['seed'], stage=stage,
                **{k:mean(r[k] for r in rs) for k in ('trace','entropy_effective_rank','participation_rank','lambda_min','lambda_max')}))
        for start in protocol['starts']:
            records = json.loads((folder / ('B%07d' % start) / 'results.json').read_text())
            assert len(records) == 2
            row = pool_window(records)
            p, s = row['pair'], row['seed']
            stage_history = next(r for r in histories if r['pair'] == p and r['seed'] == s and r['stage'] == 'B_%07d' % start)
            row['ft_kernel_entropy_rank'] = stage_history['entropy_effective_rank']
            row['ft_kernel_trace'] = stage_history['trace']
            for family in records[0]['families']:
                for metric in ('shape_burden','raw_burden'):
                    a=row[family+'_ft_start_r0.001_'+metric]
                    b=row[family+'_B_%07d_r0.001_'%start+metric]
                    assert np.isclose(a,b,rtol=1e-10,atol=1e-12)
            other = p + 1 if p % 2 else p - 1
            ft, fresh = ev['P%d_ft_s%d' % (p,s)], ev['P%d_ft_s%d' % (other,s)]
            row['delta_auc_0_500k'] = success_auc(ft,1,0,500000) - success_auc(fresh,0,0,500000)
            left = 50000 if start == 10000 else 150000 if start == 100000 else None
            row['future_delta_auc'] = None if left is None else success_auc(ft,1,left,500000) - success_auc(fresh,0,left,500000)
            row['future_interval_start'] = left
            rows.append(row)
            for r in records:
                for g in r['tracking']['groups'].values():
                    audit_error = max(audit_error, g.get('signed_energy_identity_abs', 0.))
        for source in json.loads((folder / 'inputs.json').read_text()):
            path=Path(source['path']); st=path.stat()
            assert (st.st_size,st.st_mtime_ns) == (source['bytes'],source['mtime_ns']), str(path)
            if str(path) in sources: assert sources[str(path)] == source
            sources[str(path)] = source
    write_csv(out / 'window_metrics.csv', rows)
    write_csv(out / 'history_metrics.csv', histories)
    correlations = []
    metrics = ['prefix_residual_ft_over_initial_shape_burden',
        'prefix_residual_ft_over_fresh_matched_shape_burden',
        'prefix_residual_ft_start_shape_remaining1000',
        'prefix_residual_ft_start_r0.001_shape_burden',
        'prefix_centered_ft_start_r0.001_shape_burden',
        'slow_vs_fast_retention_ratio','slow_correction_gain',
        'ft_kernel_entropy_rank','ft_kernel_trace',
        'prefix_residual_ft_start_r0.001_raw_burden']
    for start in protocol['starts']:
        rs = [r for r in rows if r['start'] == start]
        for metric in metrics:
            for outcome in ('delta_auc_0_500k','future_delta_auc'):
                valid = [r for r in rs if r.get(metric) is not None and r.get(outcome) is not None]
                if not valid: continue
                x = np.array([r[metric] for r in valid]); y = np.array([r[outcome] for r in valid])
                xc, yc = x.copy(), y.copy()
                for p in set(r['pair'] for r in valid):
                    mask = np.array([r['pair'] == p for r in valid])
                    xc[mask] -= x[mask].mean(); yc[mask] -= y[mask].mean()
                correlations.append(dict(start=start,metric=metric,outcome=outcome,n_runs=len(valid),
                    rho=correlation(x,y), within_direction_centered_rho=correlation(xc,yc)))
    write_json(out / 'correlations.json', correlations)
    audit = dict(complete=not partial, runs=len(folders), windows=len(rows), critics=2,
        reconstructed_critic_trajectories=2*len(rows), reconstructed_updates=2000*len(rows),
        unique_source_files=len(sources), source_sizes_and_mtimes_unchanged=True,
        reconstruction_relative_rmse_max=max(r['reconstruction_error_max'] for r in rows),
        signed_energy_identity_abs_max=audit_error,
        analysis_code_hashes=protocol['code_hashes'],summary_code_sha256=digest(__file__),
        generated_results_git_ignored=True)
    write_json(out / 'audit.json', audit)
    if partial:
        print(json.dumps(audit), flush=True)
        return
    lines=['# R3-A/B/C：学习谱、经验 Bellman 子空间与方向滞后', '',
        '本报告使用全部 P1–P6 × seeds 1/2/3，在固定 B 输入上完成三个离线检验。',
        '54 个窗口位于 B=10k/100k/500k，每个包含 1000 次实际更新；双 critic 不是额外 seeds。',
        '源数据只读，未启动环境、在线训练、Clip 新分支或 Git push。结果是经验局部检验，不是完整 Bellman 算子或 SAC 全局因果定理。','',
        '## 1. R3-A：同 B 输入的学习谱历史','',
        'K=JJᵀ/n，所有 critic 参数及 bias，未中心化。下表为双 critic 先平均、三个 seeds 均值±样本 SD。trace 比用逐 seed 的双 critic 汇总后计算。','',
        '| 方向 | 初始化 K 有效秩 | A 出口 K 有效秩 | B=100k K 有效秩 | B=500k K 有效秩 | A 出口/初始化 trace |',
        '|---|---:|---:|---:|---:|---:|']
    for p in range(1,7):
        stages={stage:sorted([r for r in histories if r['pair']==p and r['stage']==stage],key=lambda r:r['seed'])
                for stage in ('initial','A_1500000','B_0100000','B_0500000')}
        vals=[format_ms([r['entropy_effective_rank'] for r in stages[stage]]) for stage in stages]
        traces=[a['trace']/b['trace'] for a,b in zip(stages['A_1500000'],stages['initial'])]
        lines.append('| P%d | %s | %s |'%(p,' | '.join(vals),format_ms(traces)))
    lines += ['', '完整 A/B/fresh 各阶段见 history_metrics.csv；不能将整体谱集中直接写成任务条件负迁移。', '',
        '## 2. R3-B：保留多列的经验需求与方向负担','',
        '主对象为窗口前 101 个状态的 D_emp=[y_t−Q_t]，没有先求列均值。下表比较同一 D 在不同模型核上的响应。',
        '95% 主子空间只用于结构描述；加权负担与谱响应使用完整未中心化二阶矩 D Dᵀ/m，不丢弃小能量慢方向（存档键 covariance 指此二阶矩）。',
        '逆谱负担 P̃=∑p_i/(λ_i/mean(λ)+ε)，ε=1e−3；比值 >1 表示该定义下 FT 负担更高。',
        '“慢方向”按 trace 归一化核、固定 η=.45/256、1000 步后保留至少一半平方误差定义，不是在线 Adam 的慢模态保证。',
        '原始尺度的逆谱负担、共同稳定步长的预测、ε=1e−4/1e−2 和中心化对照均保存在 CSV/逐 critic JSON。','',
        '| 方向 | 窗口 | D rank95 | FT/初始化 P̃ | FT/同进度 fresh P̃ | 慢方向需求能量 % | 慢方向逆谱负担份额 % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in range(1,7):
        for t in protocol['starts']:
            rs=[r for r in rows if r['pair']==p and r['start']==t]
            fields=[('prefix_residual_rank95',1),('prefix_residual_ft_over_initial_shape_burden',1),
                ('prefix_residual_ft_over_fresh_matched_shape_burden',1),
                ('prefix_residual_ft_start_slow_energy_fraction',100),
                ('prefix_residual_ft_start_r0.001_slow_burden_fraction',100)]
            lines.append('| P%d | %dk | %s |'%(p,t//1000,' | '.join(format_ms([r[k] for r in rs],scale) for k,scale in fields)))
    lines += ['', '### 去均值敏感性与原始尺度','',
        '中心化会删除均值需求，不能替代主结果；raw burden 则保留整体 Jacobian 尺度。两者不一致时应解释，不择优报告。','',
        '| 方向 | 窗口 | 未中心化均值能量份额 % | 中心化 D rank95 | 中心化 FT/fresh P̃ | 原始尺度 FT/fresh 逆谱负担 |',
        '|---|---:|---:|---:|---:|---:|']
    for p in range(1,7):
        for t in protocol['starts']:
            rs=[r for r in rows if r['pair']==p and r['start']==t]
            fields=[('prefix_residual_mean_vector_energy_fraction',100),('prefix_centered_rank95',1),
                ('prefix_centered_ft_over_fresh_matched_shape_burden',1),('prefix_residual_ft_over_fresh_matched_raw_burden',1)]
            lines.append('| P%d | %dk | %s |'%(p,t//1000,' | '.join(format_ms([r[k] for r in rs],scale) for k,scale in fields)))
    lines += ['', '### 交叉需求/核：固定其中一个对象再比较','',
        '同行固定同一个前缀 D，横向只换 B 阶段核；同列固定 K，纵向比较不同阶段需求。',
        '全部为 ε=1e−3 的形状负担。这是回溯条件化比较；未来时刻的 K/D 不能作为任务入口已知信息。','',
        '| 方向 | D 的窗口 | K_10k | K_100k | K_500k |', '|---|---:|---:|---:|---:|']
    for p in range(1,7):
        for t in protocol['starts']:
            rs=[r for r in rows if r['pair']==p and r['start']==t]
            keys=['prefix_residual_B_%07d_r0.001_shape_burden'%kt for kt in protocol['starts']]
            lines.append('| P%d | %dk | %s |'%(p,t//1000,' | '.join(format_ms([r[k] for r in rs]) for k in keys)))
    lines += ['', '## 3. R3-C：前缀确定方向，后缀检验实际修正','',
        '前 100 次更新确定需求，第 100 次更新处固定核方向，随后 900 次更新检验。',
        '“残差滞留比”=后缀末/初残差能量；不是冻结目标实验。比值需要结合目标注入和实际更新的有符号账本解释。',
        '实际修正系数=∑〈d,ΔQ〉/∑||d||²，按各组投影后计算；不是 SGD 学习率，也不是因果占比。','',
        '| 方向 | 窗口 | 慢方向残差滞留比 | 快方向残差滞留比 | 慢方向实际修正系数 | 快方向实际修正系数 |',
        '|---|---:|---:|---:|---:|---:|']
    for p in range(1,7):
        for t in protocol['starts']:
            rs=[r for r in rows if r['pair']==p and r['start']==t]
            fields=('slow_final_over_initial','fast_final_over_initial','slow_correction_gain','fast_correction_gain')
            lines.append('| P%d | %dk | %s |'%(p,t//1000,' | '.join(format_ms([r[k] for r in rs]) for k in fields)))
    lines += ['', '### 方向稳定性与分解','',
        '| 方向 | 窗口 | 前缀需求子空间覆盖后缀残差 % | 后缀起止 K 形状相对变化 | 全窗口单步非线性/实际 ΔQ RMS % |',
        '|---|---:|---:|---:|---:|']
    for p in range(1,7):
        for t in protocol['starts']:
            rs=[r for r in rows if r['pair']==p and r['start']==t]
            fields=[('prefix_subspace_suffix_coverage',100),('kernel_shape_relative_change',1),('nonlinear_relative_rms',100)]
            lines.append('| P%d | %dk | %s |'%(p,t//1000,' | '.join(format_ms([r[k] for r in rs],scale) for k,scale in fields)))
    lines += ['', '每个窗口保存逐模态 Δd=u−GD_anchor−sampling−Adam_difference−nonlinear 的完整有符号分量；',
        'CSV 提供 slow/fast/all 的累积误差变化。Adam 项相对同实际 minibatch SGD 定义；sampling 同时包含 replay/panel 与 target-action 随机性差异。',
        '非线性比例覆盖全部 1000 次更新，跟踪和误差账本取后 900 次更新。较小单步非线性不等于整段核不变；起止 K 差异也不是中途最大变化。原始 SGD、采样和 Adam 项可能很大且抵消，不将其转为因果百分比。','',
        '## 4. 与后续在线负迁移的对应','',
        '在线结果冻结在所有 runs 已具备的 500k；0–500k 是阶段性获取，不是 1.5M 终点。',
        '10k 指标对应 50k–500k，100k 指标对应 150k–500k；500k 指标没有冻结范围内的后续区间，仅做同期关联。','',
        '| 方向 | 任务 | FT−fresh success-AUC 0–500k（百分点） |', '|---|---|---:|']
    for p in range(1,7):
        rs=[r for r in rows if r['pair']==p and r['start']==10000]
        lines.append('| P%d | %s | %s |'%(p,TASKS[p],format_ms([r['delta_auc_0_500k'] for r in rs],100)))
    lines += ['', '描述性相关：18 个任务方向/seed 组合，另给组内去均值敏感性；不做独立同分布或显著性/因果宣称。','',
        '| 窗口 | 指标 | 后续区间 rho | 方向内去均值 rho |', '|---|---|---:|---:|']
    for c in correlations:
        if c['outcome']=='future_delta_auc':
            lines.append('| %dk | %s | %s | %s |'%(c['start']//1000,c['metric'],
                'NA' if c['rho'] is None else '%.3f'%c['rho'],
                'NA' if c['within_direction_centered_rho'] is None else '%.3f'%c['within_direction_centered_rho']))
    lines += ['', '## 5. 复查与证据边界','',
        '- protocol.json：预先固定对象、范围、前缀/后缀、尺度、ridge 与代码校验。',
        '- history_metrics.csv / window_metrics.csv：全部 seeds 的汇总；均值之外保留各 run。',
        '- P*_s*/history_spectra.npz：完整历史核特征值、特征向量与输出。',
        '- P*_s*/B*/subspace_arrays.npz：需求矩阵可由 targets−outputs 精确恢复，另存 covariance、主子空间、逐模态残差和更新分量。',
        '- inputs.json / audit.json：原始数据 SHA256、大小与 mtime 核验；运行中的后续评估不覆盖冻结快照。',
        '- 需求来自 FT 轨迹及固定 warm-up 输入；不等于全域 Q*、完整固定 Bellman 算子或对其他策略分布的证明。',
        '- 实际 residual 是内生对象，快方向先被拟合也会使剩余 residual 显得更慢；不能仅凭残差谱反推失配导致控制失败。',
        '- 没有同源在线 Clip/Reset 干预结果；本报告不证明 Clip 的在线收益机制。',
        '- 三项实测检验完成，不意味着三条假说均得到支持。具体解释见 CONCLUSIONS.md。', '']
    (out/'SUBSPACE_ANALYSIS.md').write_text('\n'.join(lines))
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();report(args.output.resolve(),args.allow_partial)
