"""DSR v2：分阶段需求估计与可验证的新增谱动力学。

目标仅为减轻新任务的负向迁移。原 critic 保留 Adam；新分支在对齐后
固定特征，只用无动量 SGD 更新输出头。新增核严格为 lr * Phi Phi^T，
原 critic 的 Adam 核仍只是冻结二阶矩的局部代理，不能视为闭环保证。
"""

import copy
import math

import torch
from torch import nn


class ReserveBranch(nn.Module):
    """可对动作求导的固定特征分支；固定参数不等于切断输入梯度。"""

    def __init__(self, input_dim, hidden_dim, feature_dim, head_lr=1.0):
        super().__init__()
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim), nn.Tanh())
        self.output_weight = nn.Parameter(torch.zeros(feature_dim, 1))
        self.register_buffer('feature_scale', torch.ones(()))
        # 更新规则随 state_dict 保存，恢复后不会隐式换成 Adam。
        self.register_buffer('head_lr', torch.tensor(float(head_lr)))

    def features(self, inputs):
        return self.feature_scale * self.feature_network(inputs)

    def forward(self, inputs):
        return self.features(inputs) @ self.output_weight

    def freeze_features(self):
        for parameter in self.feature_network.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None


def reserve_branches(critic):
    return [branch for branch in getattr(critic, '_plasticity_injection_branches', ())
            if isinstance(branch, ReserveBranch)]


def install_branch(critic, target_critic, branch):
    """零函数安装；输出头由单独的 SGD 更新，不加入原 Adam 参数组。"""
    branch.freeze_features()
    critic.append_plasticity_injection(branch)
    target_critic.append_plasticity_injection(copy.deepcopy(branch))


def zero_head_gradients(critic):
    # actor backward 也会产生 Q 参数梯度，下一次 critic 更新前必须清除。
    for branch in reserve_branches(critic):
        branch.output_weight.grad = None


def step_heads(critic):
    """对同一个 SAC TD loss 做无动量 SGD，无二阶矩重缩放。"""
    with torch.no_grad():
        for branch in reserve_branches(critic):
            if branch.output_weight.grad is not None:
                branch.output_weight.add_(
                    branch.output_weight.grad, alpha=-float(branch.head_lr))


def restore_branches(critic, target_critic, state):
    """先按检查点中的张量形状恢复分支拓扑，再由 SAC 加载所有参数。"""
    prefix = '_plasticity_injection_branches.'
    indices = sorted(int(key[len(prefix):].split('.')[0]) for key in state
                     if key.startswith(prefix) and key.endswith('.head_lr'))
    reference = next(critic.parameters())
    for index in indices:
        if index != len(critic._plasticity_injection_branches):
            raise ValueError('DSR v2 检查点的分支顺序与当前 critic 不一致')
        base = prefix + str(index) + '.'
        hidden, input_dim = state[base + 'feature_network.0.weight'].shape
        feature_dim = state[base + 'output_weight'].shape[0]
        branch = ReserveBranch(input_dim, hidden, feature_dim,
                               float(state[base + 'head_lr'])).to(reference)
        install_branch(critic, target_critic, branch)


def effective_kernel(critic, optimizer, inputs, task):
    """原参数用冻结 Adam 代理；已安装的固定特征头使用精确 SGD 度量。"""
    parameters = [p for p in critic.parameters() if p.requires_grad]
    groups = {id(p): group for group in optimizer.param_groups
              for p in group['params']}
    head_rates = {id(b.output_weight): float(b.head_lr)
                  for b in reserve_branches(critic)}
    metric = []
    for parameter in parameters:
        if id(parameter) in head_rates:
            value = torch.full_like(parameter, head_rates[id(parameter)])
        else:
            group = groups[id(parameter)]
            state = optimizer.state.get(parameter, {})
            step = float(state.get('step', 0))
            if step == 0:
                # 正常运行在预热后触发；该退化值支持空状态的单元验证。
                value = torch.full_like(parameter, group['lr'])
            else:
                moment = state['exp_avg_sq'] / (1 - group['betas'][1] ** step)
                value = group['lr'] / (moment.sqrt() + group['eps'])
        metric.append(value.reshape(-1).detach().double())
    metric = torch.cat(metric)
    rows = []
    action_dim = critic._action_dim
    for row in inputs:
        prediction = critic(row[None, :-action_dim], row[None, -action_dim:],
                            seq_idx=task).reshape(())
        gradients = torch.autograd.grad(prediction, parameters, allow_unused=True)
        # 辅助预测头可能不参与 Q 输出，其 Jacobian 列应为零，不能让阶段检查崩溃。
        rows.append(torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for p, g in zip(parameters, gradients)]).double())
    jacobian = torch.stack(rows)
    kernel = (jacobian * metric) @ jacobian.t()
    return ((kernel + kernel.t()) * 0.5).detach()


def inverse_burden(kernel, demands, ridge):
    eye = torch.eye(len(kernel), dtype=kernel.dtype, device=kernel.device)
    solved = torch.linalg.solve(kernel + ridge * eye, demands)
    return (demands * solved).sum() / demands.shape[1]


def soft_demands(samples, task, critics, targets, policy, alpha,
                 discount, reward_scale, count):
    """每次阶段检查重新采样当前策略动作，不复用边界的固定正弦探针。"""
    with torch.no_grad():
        predictions = [q(samples['observation'], samples['action'],
                         seq_idx=task).flatten() for q in critics]
        distribution = policy(samples['next_observation'], task)[0]
        columns = [[], []]
        for _ in range(count):
            pre_tanh, action = distribution.rsample_with_pre_tanh_value()
            log_pi = distribution.log_prob(value=action, pre_tanh_value=pre_tanh)
            next_q = [q(samples['next_observation'], action, seq_idx=task).flatten()
                      for q in targets]
            target = reward_scale * samples['reward'].flatten() + discount * (
                1 - samples['terminal'].flatten()) * (
                    torch.minimum(*next_q) - alpha * log_pi.flatten())
            for index in range(2):
                columns[index].append(target - predictions[index])
        return [torch.stack(column, dim=1).double() for column in columns]


class SACDSRV2:
    """有限阶段的需求刷新、独立留出检查和零函数谱扩展。"""

    def __init__(self, task_steps=(100000, 500000, 800000), rows=64,
                 hidden_dim=256, feature_dim=64, targets=8,
                 alignment_steps=200, alignment_lr=1e-3,
                 trace_ratio=1.0, ridge=1e-3, min_gain=0.01,
                 num_tasks=3):
        if (not task_steps or tuple(sorted(set(task_steps))) != tuple(task_steps)
                or min(task_steps) <= 0):
            raise ValueError('DSR v2 阶段步数必须为严格递增的正整数')
        if min(rows, hidden_dim, feature_dim, targets, num_tasks) < 1:
            raise ValueError('DSR v2 的样本数、网络维度和任务数必须为正')
        if (alignment_steps < 0 or not all(math.isfinite(x) and x > 0
                for x in (alignment_lr, trace_ratio, ridge))
                or not 0 <= min_gain < 1):
            raise ValueError('DSR v2 对齐参数不合法')
        self.task_steps = tuple(task_steps)
        self.rows = rows
        self.hidden_dim, self.feature_dim = hidden_dim, feature_dim
        self.targets, self.alignment_steps = targets, alignment_steps
        self.alignment_lr, self.trace_ratio = alignment_lr, trace_ratio
        self.ridge, self.min_gain = ridge, min_gain
        self.num_tasks = num_tasks
        self.records = []

    def state_dict(self):
        """保存完整配置，阶段索引与稳定性预算必须使用同一套定义。"""
        config = {key: value for key, value in vars(self).items() if key != 'records'}
        return {'version': 2, 'config': copy.deepcopy(config),
                'records': copy.deepcopy(self.records)}

    def load_state_dict(self, state):
        if state.get('version') != 2 or state.get('config') != self.state_dict()['config']:
            raise ValueError('DSR v2 恢复配置与检查点不一致，请使用原阶段和预算参数')
        self.records = copy.deepcopy(state['records'])

    def due_stage(self, task, task_step):
        if task == 0:
            return None
        due = [i for i, step in enumerate(self.task_steps) if step <= task_step]
        if not due:
            return None
        # 中途恢复时只检查最新阶段，避免在连续几个更新中补装过时分支。
        stage = due[-1]
        if any(r['task'] == task and r['stage'] >= stage for r in self.records):
            return None
        return stage

    @staticmethod
    def _scaled_kernel(branch, inputs, trace_budget, norm_budget):
        features = branch.feature_network(inputs).double()
        raw_kernel = features @ features.t()
        lr = float(branch.head_lr)
        # Tanh 特征绝对值不超过 1。限制所有新增头的总 lr*||phi||²，
        # 使固定目标、固定原 critic 时的 mean-MSE 头更新保持稳定。
        scale_square = torch.clamp(trace_budget / torch.trace(raw_kernel).clamp_min(1e-20),
                                   max=norm_budget / features.shape[1]) / lr
        return lr * scale_square * raw_kernel, scale_square

    def prepare_one(self, critic, target, optimizer, inputs, demands,
                    heldout_inputs, heldout_demands, task, branch_norm_budget):
        kernel = effective_kernel(critic, optimizer, inputs, task)
        heldout_kernel = effective_kernel(critic, optimizer, heldout_inputs, task)
        demands, heldout_demands = demands.double(), heldout_demands.double()
        ridge = (self.ridge * torch.trace(kernel) / len(kernel)).clamp_min(1e-12)
        action_dim = critic._action_dim
        features_x = critic.plasticity_injection_inputs(
            inputs[:, :-action_dim], inputs[:, -action_dim:], seq_idx=task).detach()
        heldout_x = critic.plasticity_injection_inputs(
            heldout_inputs[:, :-action_dim], heldout_inputs[:, -action_dim:], seq_idx=task).detach()
        branch = ReserveBranch(features_x.shape[1], self.hidden_dim, self.feature_dim).to(inputs)
        random_branch = copy.deepcopy(branch)
        trace_budget = self.trace_ratio * torch.trace(kernel) / len(self.task_steps)
        align_optimizer = torch.optim.Adam(branch.feature_network.parameters(), lr=self.alignment_lr)
        for _ in range(self.alignment_steps):
            extra, _ = self._scaled_kernel(branch, features_x, trace_budget, branch_norm_budget)
            objective = inverse_burden(kernel + extra, demands, ridge)
            if not torch.isfinite(objective):
                return {'accepted': False, 'reason': 'nonfinite_alignment'}
            align_optimizer.zero_grad()
            objective.backward()
            align_optimizer.step()
        with torch.no_grad():
            extra, scale_square = self._scaled_kernel(branch, features_x, trace_budget, branch_norm_budget)
            branch.feature_scale.copy_(scale_square.sqrt())
            heldout_phi = branch.features(heldout_x).double()
            heldout_extra = float(branch.head_lr) * heldout_phi @ heldout_phi.t()
            random_phi = random_branch.feature_network(heldout_x).double()
            random_extra = random_phi @ random_phi.t()
            # 留出集上匹配 trace，使接受标准比较方向而不是特征幅度。
            random_extra *= torch.trace(heldout_extra) / torch.trace(random_extra).clamp_min(1e-20)
            base = inverse_burden(heldout_kernel, heldout_demands, ridge)
            aligned = inverse_burden(heldout_kernel + heldout_extra, heldout_demands, ridge)
            random = inverse_burden(heldout_kernel + random_extra, heldout_demands, ridge)
            gain = (random - aligned) / random.clamp_min(1e-20)
            result = {
                'accepted': bool(torch.isfinite(gain) and gain > self.min_gain and aligned < base),
                'heldout_gain_over_random': float(gain),
                'heldout_base_burden': float(base),
                'heldout_aligned_burden': float(aligned),
                'heldout_random_burden': float(random),
                'train_base_burden': float(inverse_burden(kernel, demands, ridge)),
                'train_aligned_burden': float(inverse_burden(kernel + extra, demands, ridge)),
                'kernel_trace': float(torch.trace(extra)),
                'head_norm_bound': float(branch.head_lr * branch.feature_scale.square() * self.feature_dim),
            }
        if result['accepted']:
            install_branch(critic, target, branch)
        return result

    def prepare(self, samples, heldout, task, stage, task_step, critics,
                targets, optimizers, policy, alpha, discount, reward_scale):
        # 同一阶段内固定 target；下一阶段重新从新任务 replay 与当前网络估计。
        demands = soft_demands(samples, task, critics, targets, policy, alpha,
                               discount, reward_scale, self.targets)
        heldout_demands = soft_demands(heldout, task, critics, targets, policy, alpha,
                                       discount, reward_scale, self.targets)
        inputs = torch.cat((samples['observation'], samples['action']), dim=1)
        heldout_inputs = torch.cat((heldout['observation'], heldout['action']), dim=1)
        record = {'version': 2, 'task': int(task), 'stage': int(stage),
                  'task_step': int(task_step), 'scheduled_step': self.task_steps[stage]}
        for index in range(2):
            # 全序列所有可能安装的分支共享稳定性预算，不随阶段指数扩容。
            norm_budget = 0.5 / (max(self.num_tasks - 1, 1) * len(self.task_steps))
            record['qf' + str(index + 1)] = self.prepare_one(
                critics[index], targets[index], optimizers[index], inputs, demands[index],
                heldout_inputs, heldout_demands[index], task, norm_budget)
        self.records.append(record)
        return record
