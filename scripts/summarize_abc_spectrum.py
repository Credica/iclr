#!/usr/bin/env python3
"""Make complete ABC spectrum tables; never selects checkpoints by outcome."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    with path.open() as f: return list(csv.DictReader(f))


def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def fmt(x):
    return '%.5g' % x


def summarize(out):
    complete = json.loads((out / 'complete.json').read_text())
    assert complete['full_jacobian_kernels'] == 180
    h, w, d, ev = [read(out / file) for file in ('learning_spectra.csv', 'weight_spectra.csv', 'demand_spectra.csv', 'evaluation.csv')]
    def hs(s, t): return [r for r in h if int(r['step']) == s and int(r['input_task']) == t]
    def ds(s, t, family): return [r for r in d if int(r['kernel_step']) == s and int(r['input_task']) == t and r['family'] == family]
    lines = ['# ABC：权重谱、固定输入学习谱与 Bellman 需求', '',
        '本表是单 seed 的完整离线结果；数字先按双 critic 平均，不是多 seed 均值，不能据此计算跨 seed 置信度。',
        'ABC=sweep-into → push-wall → window-close，每任务 1M 次 optimizer updates；不是 1M 环境步。',
        '横轴为累计 critic updates。每任务额外 warm-up 交互与评估不在该横轴中；每 100k updates 评估 20 episodes。',
        '全部 30 checkpoint、三组固定 128 输入、双 critic；无初始化 checkpoint、无逐步动态窗口。','',
        '## 1. 完整学习谱历史','',
        'full K=JJᵀ/128，未中心化，含所有 critic 参数和 bias。B/C 输入固定为各自最早保存的 probe bank；未来任务输入上的早期模型结果仅是回溯测量。','',
        '| 累计 updates | B 输入 K 有效秩 | B top1 trace % | B trace | C 输入 K 有效秩 | C top1 trace % | C trace |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for s in range(100000, 3000001, 100000):
        values = []
        for t in (1, 2):
            rows = hs(s, t)
            values += [mean(rows, 'entropy_effective_rank'), 100 * mean(rows, 'kernel_top1_trace_fraction'), mean(rows, 'trace')]
        lines.append('| %s | %s |' % (fmt(s), ' | '.join(map(fmt, values))))
    lines += ['', 'A 输入上的同定义完整结果见 learning_spectra.csv（没有因主表聚焦 B/C 而删除 A）。','',
        '## 2. 完整权重谱历史','',
        'stable rank=||W||F²/σmax²；输出头为 1×256，仅有一个奇异值，不用其 rank=1 宣称谱坍缩。完整奇异值数组已保存。','',
        '| 累计 updates | W1 σmax | W1 σmin | W1 stable rank | W2 σmax | W2 σmin | W2 stable rank | 输出头范数 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in range(100000, 3000001, 100000):
        values = []
        for layer in (1, 2):
            rows = [r for r in w if int(r['step']) == s and int(r['layer']) == layer]
            values += [mean(rows, k) for k in ('sigma_max', 'sigma_min', 'stable_rank')]
        values.append(mean([r for r in w if int(r['step']) == s and int(r['layer']) == 3], 'sigma_max'))
        lines.append('| %s | %s |' % (fmt(s), ' | '.join(map(fmt, values))))
    lines += ['', '## 3. 固定入口需求，只更换学习核','',
        '分别用 A 出口/B 出口模型重建 B/C 入口：保留 actor 与 online critic，按原边界代码 alpha=1、target=online。',
        '输入来自保存的该任务 probe bank；不是当时在线已知的未来输入。每组 D 有 8 个原探针 sin-noise target−Q 向量，不是先求平均，也不是完整 Bellman 算子的子空间。',
        '以下每个任务的 D 固定不变；95% 主子空间只描述结构，负担保留全部方向。',
        'P̃=Σpᵢ/(λᵢ/mean(λ)+1e−3)，raw=P̃/mean(λ)，不是固定绝对 ridge 或实际 Adam 收敛率。',
        '慢方向按同一归一化 GD η=.45/128、1000 次更新后平方误差保留≥.5 定义。','',
        '| 核时点 | B 入口 D 负担 | B 原始负担 | B 慢方向需求 % | C 入口 D 负担 | C 原始负担 | C 慢方向需求 % |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for s in range(100000, 3000001, 100000):
        values = []
        for t in (1, 2):
            rows = ds(s, t, 'entry')
            values += [mean(rows, 'ridge_0.001_shape_burden'), mean(rows, 'ridge_0.001_raw_burden'), 100 * mean(rows, 'slow_energy_fraction')]
        lines.append('| %s | %s |' % (fmt(s), ' | '.join(map(fmt, values))))
    lines += ['', '### 各自入口：双 critic 与敏感性均保留','',
        '| 任务 | critic | ε=1e−4 负担 | ε=1e−3 | ε=1e−2 | 去列均值 ε=1e−3 | 入口 D rank95 | 均值需求能量 % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in (1, 2):
        for q in (1, 2):
            row = next(r for r in ds(t * 1000000, t, 'entry') if int(r['critic']) == q)
            centered = next(r for r in ds(t * 1000000, t, 'entry_centered') if int(r['critic']) == q)
            vals = [float(row['ridge_' + ridge + '_shape_burden']) for ridge in ('0.0001', '0.001', '0.01')]
            vals += [float(centered['ridge_0.001_shape_burden']), float(row['demand_rank95']), 100 * float(row['mean_vector_energy_fraction'])]
            lines.append('| %s | %d | %s |' % ('ABC'[t], q, ' | '.join(map(fmt, vals))))
    lines += ['', '## 4. 每个 checkpoint 的当前需求及实际评价','',
        '这里 D 随时间变化，不能把该表当成只改 K 的实验。表内 MSE=||D||F²/(8×128)。',
        '含所有 30 个当前任务评估点；不是只展示失败的 B 或成功的 C。','',
        '| 任务 | task-local updates | success % | 当前 D MSE | P̃ | raw | 慢方向能量 % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for t in range(3):
        for local in range(100000, 1000001, 100000):
            s = t * 1000000 + local
            rows = ds(s, t, 'current_%d' % s)
            e = next(r for r in ev if int(r['step']) == s and int(r['evaluated_task']) == t)
            values = [100 * float(e['success']), mean(rows, 'demand_energy') / 128,
                mean(rows, 'ridge_0.001_shape_burden'), mean(rows, 'ridge_0.001_raw_burden'), 100 * mean(rows, 'slow_energy_fraction')]
            lines.append('| %s | %s | %s |' % ('ABC'[t], fmt(local), ' | '.join(map(fmt, values))))
    lines += ['', '## 5. 多时刻需求矩阵：每任务 80 列','',
        '拼接该任务十个 checkpoint 的八列残差，等列权重、能量加权，不先平均。由已发生的轨迹构造，仅用于回溯解释。','',
        '| 任务 | 核阶段 | 多时刻 D rank95 | 未中心化负担 | 去列均值负担 |',
        '|---|---:|---:|---:|---:|']
    for t in (1, 2):
        for s in (t * 1000000, t * 1000000 + 100000, t * 1000000 + 500000, (t + 1) * 1000000):
            rows = ds(s, t, 'trajectory'); centered = ds(s, t, 'trajectory_centered')
            vals = [mean(rows, 'demand_rank95'), mean(rows, 'ridge_0.001_shape_burden'), mean(centered, 'ridge_0.001_shape_burden')]
            lines.append('| %s | %s | %s |' % ('ABC'[t], fmt(s), ' | '.join(map(fmt, vals))))
    lines += ['', '## 6. 可解释范围','',
        '- 权重谱、full-J 学习谱、特征谱不是同一个对象。',
        '- B/C 是不同任务、不同输入 bank、不同 actor head；其差异支持任务条件性描述，不是随机化控制的几何因果效应。',
        '- 入口需求重建不是已保存的入口逐步轨迹；固定需求的核响应是代理，不是实际在线 Adam 拟合实验。',
        '- 原始八个 sin 向量确定性且跨输入共享噪声，不是八个独立 seeds 或无偏 Monte Carlo 估计。',
        '- 128 条输入是最早收集的连续 probe 样本，没有 episode ID、跨实例覆盖或未进入 replay 的保证。',
        '- ABC 只有 seed 1，没有本组同源 fresh/reset/Clip 对照，也没有初始化模型或 1000-update 参数窗口。',
        '- 旧版训练预算、actor-head/边界和评估协议不同，不能并为当前 1.5M 环境步 P1–P6 的同协议统计。',
        '- 不用序列中的相邻 checkpoint 充当独立 seeds；不把这里的经验子空间称为完整 Bellman 算子或 Q* 修正。',
        '- 具体结果解释见 ABC_SPECTRAL_CONCLUSIONS.md；原始数组、CSV 与逐 critic 结果全部保留。','']
    (out / 'ABC_SPECTRAL_TABLES.md').write_text('\n'.join(lines))
    print('WROTE', out / 'ABC_SPECTRAL_TABLES.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); summarize(args.output.resolve())
