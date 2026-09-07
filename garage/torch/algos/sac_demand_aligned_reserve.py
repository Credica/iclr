"""SAC 的需求对齐谱储备方法。

该文件只实现新任务加速，不处理旧任务遗忘。方法在 critic 上增加一个零输出
特征块，并在激活前用当前任务的 soft Bellman demand 调整该特征块的学习谱。
"""

import copy

import torch
from torch import nn

from garage.torch.algos.bellman_spectral_stats import (
    deterministic_soft_bellman_directions,
)


class DemandAlignedReserveBranch(nn.Module):
    """带零初始化输出头的可微状态动作特征块。"""

    def __init__(self, input_dim, hidden_dim, feature_dim):
        super().__init__()

        # 特征网络直接接收状态动作，因此激活后能够学习动作方向。
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh(),
        )

        # 输出头为零时，新增分支对所有输入的函数值和动作梯度都为零。
        self.output_weight = nn.Parameter(torch.zeros(feature_dim, 1))

        # 对齐结束后用该尺度把新增 kernel 调整到指定的谱预算。
        self.register_buffer('feature_scale', torch.ones(()))

    def raw_features(self, inputs):
        """返回尚未按谱预算缩放的特征。"""
        return self.feature_network(inputs)

    def features(self, inputs):
        """返回最终参与 critic 学习的特征。"""
        return self.feature_scale * self.raw_features(inputs)

    def forward(self, inputs):
        """计算零头残差；激活前该输出严格为零。"""
        return torch.matmul(self.features(inputs), self.output_weight)

    def alignment_parameters(self):
        """谱对齐阶段只更新特征网络，不更新零输出头。"""
        return self.feature_network.parameters()


def critic_jacobian(critic, observations, actions, task_idx):
    """计算一批状态动作上 critic 对全部可训练参数的经验 Jacobian。"""
    parameters = tuple(
        parameter for parameter in critic.parameters()
        if parameter.requires_grad)
    rows = []

    for row_idx in range(observations.shape[0]):
        value = critic(
            observations[row_idx:row_idx + 1],
            actions[row_idx:row_idx + 1],
            seq_idx=task_idx).reshape(())
        gradients = torch.autograd.grad(value, parameters)
        rows.append(torch.cat([
            gradient.reshape(-1) for gradient in gradients
        ]))

    return torch.stack(rows, dim=0)


def adam_effective_metric(optimizer, critic):
    """读取当前 Adam 二阶矩，构造冻结时刻的对角参数度量。"""
    parameters = tuple(
        parameter for parameter in critic.parameters()
        if parameter.requires_grad)
    parameter_groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group['params']
    }
    diagonal = []

    for parameter in parameters:
        group = parameter_groups[id(parameter)]
        state = optimizer.state[parameter]
        step = float(state['step'])
        beta2 = group['betas'][1]
        second_moment = state['exp_avg_sq'] / (1. - beta2 ** step)
        metric = group['lr'] / (second_moment.sqrt() + group['eps'])
        diagonal.append(metric.reshape(-1))

    return torch.cat(diagonal)


def critic_effective_kernel(
        critic, optimizer, observations, actions, task_idx):
    """计算考虑当前 Adam 二阶矩后的 inherited effective kernel。"""
    jacobian = critic_jacobian(
        critic, observations, actions, task_idx)
    metric = adam_effective_metric(optimizer, critic)
    weighted_jacobian = jacobian * metric.sqrt().unsqueeze(0)
    kernel = torch.matmul(weighted_jacobian, weighted_jacobian.t())
    return 0.5 * (kernel + kernel.t())


def soft_bellman_demands(
        samples, task_idx, qf1, qf2, target_qf1, target_qf2,
        policy, alpha, discount, reward_scale, target_count):
    """从 SAC 原生 replay 样本构造 twin critics 的 soft Bellman demand。"""
    q1_demands, q2_demands = deterministic_soft_bellman_directions(
        samples,
        task_idx,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        policy,
        alpha,
        discount,
        reward_scale,
        target_count,
    )
    return q1_demands.detach(), q2_demands.detach()


def inverse_burden(kernel, demands, ridge):
    """计算 Tr[D D^T (K + rho I)^-1]。"""
    identity = torch.eye(
        kernel.shape[0], dtype=kernel.dtype, device=kernel.device)
    solved = torch.linalg.solve(kernel + ridge * identity, demands)
    return (demands * solved).sum() / demands.shape[1]


def need_operator(kernel, demands, ridge):
    """计算决定最优新增方向的 A^-1 Omega A^-1。"""
    identity = torch.eye(
        kernel.shape[0], dtype=kernel.dtype, device=kernel.device)
    inverse_demands = torch.linalg.solve(
        kernel + ridge * identity, demands)
    return (
        torch.matmul(inverse_demands, inverse_demands.t())
        / demands.shape[1])


def reserve_kernel_with_budget(raw_features, kernel_trace_budget):
    """将 reserve kernel 的 trace 固定到给定函数空间预算。"""
    raw_kernel = torch.matmul(raw_features, raw_features.t())
    scale_square = kernel_trace_budget / torch.trace(raw_kernel)
    reserve_kernel = scale_square * raw_kernel
    return reserve_kernel, scale_square


def align_reserve(
        branch, inputs, base_kernel, demands, ridge,
        kernel_trace_budget, alignment_steps, alignment_lr):
    """在零输出状态下最小化加入 reserve 后的 Bellman burden。"""
    optimizer = torch.optim.Adam(
        branch.alignment_parameters(), lr=alignment_lr)

    for _ in range(alignment_steps):
        raw_features = branch.raw_features(inputs)
        reserve_kernel, _ = reserve_kernel_with_budget(
            raw_features, kernel_trace_budget)
        objective = inverse_burden(
            base_kernel + reserve_kernel, demands, ridge)

        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    # 将对齐时使用的 trace 预算固化到真正参与 critic forward 的特征上。
    with torch.no_grad():
        raw_features = branch.raw_features(inputs)
        reserve_kernel, scale_square = reserve_kernel_with_budget(
            raw_features, kernel_trace_budget)
        branch.feature_scale.copy_(scale_square.sqrt())
        aligned_burden = inverse_burden(
            base_kernel + reserve_kernel, demands, ridge)

    return aligned_burden, reserve_kernel


def install_reserve(critic, target_critic, critic_optimizer, branch):
    """把已对齐的零输出 reserve 同时加入 online 和 target critic。"""
    reference_parameter = critic_optimizer.param_groups[0]['params'][0]
    reference_step = critic_optimizer.state[reference_parameter]['step']

    critic.append_plasticity_injection(branch)
    target_critic.append_plasticity_injection(copy.deepcopy(branch))
    critic_optimizer.add_param_group({
        'params': branch.parameters(),
    })

    # 对齐目标中的 reserve kernel 使用单位参数度量，因此这里把新增参数的
    # Adam 二阶矩初始化为对应的单位预条件尺度，使理论 kernel 与实际更新一致。
    group = critic_optimizer.param_groups[-1]
    beta2 = group['betas'][1]
    step = float(reference_step)
    second_moment = (group['lr'] - group['eps']) ** 2
    stored_second_moment = second_moment * (1. - beta2 ** step)

    for parameter in branch.parameters():
        state = critic_optimizer.state[parameter]
        state['step'] = reference_step.clone()
        state['exp_avg'] = torch.zeros_like(parameter)
        state['exp_avg_sq'] = torch.full_like(
            parameter, stored_second_moment)


class SACDemandAlignedReserve:
    """负责 twin-critic reserve 对齐、复用判断和安装。"""

    def __init__(
            self,
            capacity_price,
            hidden_dim=256,
            feature_dim=64,
            target_count=8,
            relative_ridge=1e-3,
            trace_ratio=1.0,
            alignment_steps=200,
            alignment_lr=1e-3):
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.target_count = target_count
        self.relative_ridge = relative_ridge
        self.trace_ratio = trace_ratio
        self.capacity_price = capacity_price
        self.alignment_steps = alignment_steps
        self.alignment_lr = alignment_lr

    def _prepare_one(
            self, critic, target_critic, critic_optimizer,
            observations, actions, task_idx, demands):
        """为一个 critic 判断复用或安装新的 demand-aligned reserve。"""
        inputs = critic.plasticity_injection_inputs(
            observations, actions, seq_idx=task_idx).detach()
        base_kernel = critic_effective_kernel(
            critic, critic_optimizer, observations, actions, task_idx)
        base_kernel = base_kernel.detach()

        ridge = (
            self.relative_ridge
            * torch.trace(base_kernel)
            / base_kernel.shape[0])
        base_burden = inverse_burden(
            base_kernel, demands, ridge).detach()

        # KKT 条件给出是否可以直接复用已有学习谱。
        spectral_need = need_operator(
            base_kernel, demands, ridge)
        maximum_need = torch.linalg.eigvalsh(spectral_need)[-1].detach()
        reuse = bool(maximum_need <= self.capacity_price)

        if reuse:
            return {
                'reused': True,
                'base_burden': float(base_burden),
                'aligned_burden': float(base_burden),
                'maximum_need': float(maximum_need),
                'kernel_trace_budget': 0.0,
            }

        branch = DemandAlignedReserveBranch(
            critic._obs_dim + critic._action_dim,
            self.hidden_dim,
            self.feature_dim,
        ).to(observations.device)

        kernel_trace_budget = (
            self.trace_ratio * torch.trace(base_kernel))
        aligned_burden, reserve_kernel = align_reserve(
            branch,
            inputs,
            base_kernel,
            demands,
            ridge,
            kernel_trace_budget,
            self.alignment_steps,
            self.alignment_lr,
        )

        install_reserve(
            critic, target_critic, critic_optimizer, branch)

        return {
            'reused': False,
            'base_burden': float(base_burden),
            'aligned_burden': float(aligned_burden),
            'maximum_need': float(maximum_need),
            'kernel_trace_budget': float(torch.trace(reserve_kernel)),
        }

    def prepare(
            self,
            samples,
            task_idx,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            qf1_optimizer,
            qf2_optimizer,
            policy,
            alpha,
            discount,
            reward_scale):
        """根据一个新任务 replay batch 准备 twin critics。"""
        q1_demands, q2_demands = soft_bellman_demands(
            samples,
            task_idx,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            policy,
            alpha,
            discount,
            reward_scale,
            self.target_count,
        )

        observations = samples['observation']
        actions = samples['action']

        q1_result = self._prepare_one(
            qf1,
            target_qf1,
            qf1_optimizer,
            observations,
            actions,
            task_idx,
            q1_demands,
        )
        q2_result = self._prepare_one(
            qf2,
            target_qf2,
            qf2_optimizer,
            observations,
            actions,
            task_idx,
            q2_demands,
        )

        return {
            'qf1': q1_result,
            'qf2': q2_result,
        }
