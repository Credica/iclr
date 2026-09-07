"""SAC 的 Bellman 需求驱动几何正则：固定容量、完整 critic Jacobian。

本文件实现当前参数上的辅助正则，不包含虚拟 TD 更新或元梯度展开。
对 n 条当前任务转移，先平均多个下一动作的软 Bellman 目标，再定义
（d、E、C 均停止梯度）：

    d = stopgrad(mean(y_1, ..., y_m) - Q), E = ||d||² / n
    K = J Jᵀ / n
    C = (1 - xi) d dᵀ / n + xi E I / n
    R = tr[C (K + rho I)^(-1)] + scale_penalty * E * tr(K) / n
    weight = min(beta, max_grad_ratio * ||g_TD|| / ||g_R||)
    loss = 原 SAC 的 TD loss + stopgrad(weight) * R

不再用当前 trace 归一化：缩小无关强方向不能降低绝对需求逆核项。
尺度成本单独记录；它是限制核膨胀的有偏正则，不是无副作用的稳定器。
rho 现在具有原始核的绝对尺度。梯度上限只约束输入 Adam 的辅助梯度，
不等同于约束 Adam 的实际参数步长。仍不保证恢复零空间、修正错误的
bootstrap 或改善在线控制回报。均匀先验不使用旧任务数据。

使用方式（sac_kwargs 与现有 SAC 构造参数一致）：

    algo = SACBellmanGeometry(
        **sac_kwargs,
        geometry_coefficient=0.01,
        geometry_isotropic_fraction=0.1,
        geometry_ridge=0.01,
        geometry_scale_penalty=0.01,
        geometry_max_grad_ratio=0.1,
        geometry_target_samples=8,
        geometry_probe_size=32,
        geometry_first_task=1)

这些默认值是实现起点，尚未经任务实验调优。默认从第二个任务开始，每次
critic 更新都计算正则；first_task=0 可从首任务开始，coefficient=0 完全旁路。
原 SAC 使用 mean squared error 而不是 half-MSE，本文件保留其损失约定。
MetaWorld 顺序训练可通过 --bellman_geometry True 启用；导入不启动训练。
"""

import json
import math
from contextlib import contextmanager
from numbers import Integral

import torch

from garage.torch.algos.sac import SAC
from garage.torch.algos.finetuning import Finetuning_SAC


def _validate_penalty_options(isotropic_fraction, ridge):
    """允许 xi 的两个端点用于消融，主方法使用 0 < xi < 1。"""
    if (not math.isfinite(isotropic_fraction) or
            not 0.0 <= isotropic_fraction <= 1.0):
        raise ValueError('isotropic_fraction 必须是 [0, 1] 内的有限数')
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError('ridge 必须是正的有限数')


def _scalar_predictions(predictions):
    """只接受每条转移一个 Q 值，避免无意展开多输出头。"""
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    if predictions.ndim != 1 or predictions.numel() == 0:
        raise ValueError('Q 预测必须为非空的 [n] 或 [n, 1] 张量')
    if not predictions.is_floating_point():
        raise TypeError('Q 预测必须为浮点张量')
    if not bool(torch.isfinite(predictions).all()):
        raise FloatingPointError('Q 预测出现非有限值')
    return predictions


@contextmanager
def _preserve_forward_cache(module):
    """递归隔离额外 forward 的特征图和统计，不复制历史列表。"""
    names = ('_feature', '_features', '_stats')
    # policy 的统计实际由内部 MLP 写入，再逐层暴露到外面；只恢复顶层
    # 属性无法撤销内部列表追加，必须在 forward 前隔离每个子模块。
    saved_modules = [
        (child, {name: getattr(child, name) for name in names
                 if hasattr(child, name)})
        for child in module.modules()]
    try:
        for child, saved in saved_modules:
            if isinstance(saved.get('_stats'), dict):
                child._stats = {
                    key: ([] if isinstance(value, list) else
                          {} if isinstance(value, dict) else value)
                    for key, value in saved['_stats'].items()}
        yield
    finally:
        for child, saved in saved_modules:
            for name in names:
                if name in saved:
                    setattr(child, name, saved[name])
                elif hasattr(child, name):
                    delattr(child, name)


def full_critic_jacobian(predictions, parameters):
    """返回可继续求导的完整 J；不把参数的 .grad 当作 Jacobian。

    parameters 应传入 critic.parameters()，包含全部可训练层。未参与当前
    Q 输出的辅助头记为零列，冻结参数不计入可训练切空间。保留图以支持
    正则对 J 的反传，以及稍后对原 TD 损失的正常 backward。
    """
    predictions = _scalar_predictions(predictions)
    parameters = tuple(p for p in parameters if p.requires_grad)
    if not parameters:
        raise ValueError('critic 至少需要一个可训练参数')
    rows = []
    for value in predictions:
        if value.requires_grad:
            gradients = torch.autograd.grad(
                value, parameters, create_graph=True, retain_graph=True,
                allow_unused=True)
        else:
            gradients = (None,) * len(parameters)
        rows.append(torch.cat([
            (gradient if gradient is not None else torch.zeros_like(parameter))
            .reshape(-1)
            for parameter, gradient in zip(parameters, gradients)]))
    jacobian = torch.stack(rows).double()
    # 线性模型的 J 可以是常量。零连接使正则独立 backward 仍合法，且不会
    # 伪造几何梯度；真实的非零导数仍完全来自上面的 create_graph=True。
    return jacobian + predictions[0].double() * 0.0


def bellman_geometry_penalty(jacobian, demand, isotropic_fraction=0.1,
                            ridge=0.01, scale_penalty=0.01):
    """计算 R 及不持有计算图的标量诊断；所有求解使用 float64。

    demand 是单个当前 Bellman 残差向量，不把随机动作采样列自动解释为
    已识别的完整需求子空间。内部强制 detach，防止通过修改标签降低正则。
    """
    # _validate_penalty_options(isotropic_fraction, ridge)
    # if not math.isfinite(scale_penalty) or scale_penalty < 0:
    #     raise ValueError('scale_penalty 必须是非负有限数')
    # if (jacobian.ndim != 2 or min(jacobian.shape) == 0 or
    #         not jacobian.is_floating_point()):
    #     raise ValueError('jacobian 必须是非空浮点矩阵 [n, 参数数]')
    # 需求停止梯度是方法定义的一部分，不能随数值检查一起禁用。
    demand = _scalar_predictions(demand.detach())
    # if len(demand) != len(jacobian):
    #     raise ValueError('需求与 Jacobian 的样本行数不一致')
    # if not bool(torch.isfinite(jacobian).all()):
    #     raise FloatingPointError('critic Jacobian 出现非有限值')

    jacobian = jacobian.double()
    demand = demand.to(device=jacobian.device, dtype=torch.float64)
    count = len(jacobian)
    eye = torch.eye(count, dtype=torch.float64, device=jacobian.device)
    kernel = (jacobian @ jacobian.t()) / count
    kernel = (kernel + kernel.t()) * 0.5
    mean_eigenvalue = kernel.trace() / count
    if not bool(torch.isfinite(mean_eigenvalue)):
        raise FloatingPointError('critic 核的尺度溢出')
    zero_kernel = bool(mean_eigenvalue.detach() == 0)
    energy = demand.square().mean()
    covariance = ((1.0 - isotropic_fraction) *
                  torch.outer(demand, demand) / count +
                  isotropic_fraction * energy * eye / count)
    if not bool(torch.isfinite(covariance).all()):
        raise FloatingPointError('Bellman 需求协方差溢出')
    # 直接使用绝对核；否则只压低强方向就可能改善归一化指标。
    # ridge 不通过当前 trace 缩放，也不把冻结 Adam 二阶矩当作真实核。
    regularized_kernel = kernel + ridge * eye
    # 求解 A X = C 后取 tr(X)，不显式求逆，也不静默调整 ridge。
    cholesky = torch.linalg.cholesky(regularized_kernel)
    solved = torch.cholesky_solve(covariance, cholesky)
    inverse_burden = solved.trace()
    scale_cost = scale_penalty * energy * mean_eigenvalue
    penalty = inverse_burden + scale_cost
    if not bool(torch.isfinite(penalty)):
        raise FloatingPointError('Bellman 几何正则出现非有限值')

    with torch.no_grad():
        solved_demand = torch.cholesky_solve(demand[:, None], cholesky.detach())
        demand_burden = (demand[:, None] * solved_demand).sum() / count
        response = ((demand @ kernel.detach() @ demand) /
                    demand.square().sum()) if bool(energy > 0) else energy
    stats = {
        'penalty': float(penalty.detach()),
        'inverse_burden': float(inverse_burden.detach()),
        'demand_burden': float(demand_burden),
        'scale_cost': float(scale_cost.detach()),
        'demand_response': float(response),
        'demand_energy': float(energy),
        'kernel_trace': float(kernel.trace().detach()),
        'kernel_mean_eigenvalue': float(mean_eigenvalue.detach()),
        'rows': count,
        'parameters': jacobian.shape[1],
        'zero_kernel': zero_kernel,
        'zero_demand': bool(energy == 0),
    }
    return penalty, stats


def bound_geometry_coefficient(base_loss, penalty, parameters, coefficient,
                               max_grad_ratio=0.1):
    """只缩小辅助系数，保证其梯度范数不超过 TD 梯度的给定比例。

    不写入 parameter.grad；梯度与系数均停止梯度，保留原图供联合 backward。
    TD 或几何梯度为零时返回零，不利用 epsilon 把零梯度放大成更新。
    max_grad_ratio < 1 对普通 SGD 保留 TD 的一阶下降方向；实际 Adam 下
    仅保证输入梯度的比例，不能保证参数步长或 TD 损失必然下降。
    """
    # if not math.isfinite(coefficient) or coefficient < 0:
    #     raise ValueError('coefficient 必须是非负有限数')
    # if not math.isfinite(max_grad_ratio) or not 0 <= max_grad_ratio < 1:
    #     raise ValueError('max_grad_ratio 必须为 [0, 1) 内的有限数')
    # critic.parameters() 是一次性生成器；两次求梯度必须复用同一参数表。
    parameters = tuple(p for p in parameters if p.requires_grad)

    def gradient_norm(loss):
        if not parameters or not loss.requires_grad:
            return loss.new_zeros((), dtype=torch.float64)
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True)
        squared = loss.new_zeros((), dtype=torch.float64)
        for gradient in gradients:
            if gradient is not None:
                squared = squared + gradient.detach().double().square().sum()
        return squared.sqrt()

    td_norm = gradient_norm(base_loss)
    regularizer_norm = gradient_norm(penalty)
    if not bool(torch.isfinite(td_norm) & torch.isfinite(regularizer_norm)):
        raise FloatingPointError('TD 或几何梯度范数出现非有限值')
    weight = penalty.new_tensor(coefficient).detach()
    if bool(td_norm > 0) and bool(regularizer_norm > 0):
        weight = torch.minimum(weight, max_grad_ratio * td_norm / regularizer_norm)
        ratio = weight * regularizer_norm / td_norm
    else:
        weight = weight * 0.0
        ratio = weight
    return weight.detach(), {
        'effective_coefficient': float(weight),
        'td_grad_norm': float(td_norm),
        'geometry_grad_norm': float(regularizer_norm),
        'weighted_gradient_ratio': float(ratio),
    }


def averaged_soft_bellman_targets(samples_data, seq_idx, policy, target_qf1,
                                  target_qf2, alpha, discount, reward_scale,
                                  target_samples=8, seed=0, step=0):
    """先平均独立下一动作的目标，再由调用方构造残差外积。

    返回目标均值与均值的样本方差估计；仅降低下一动作 Monte Carlo 方差，
    不能消除转移噪声或 inherited bootstrap 偏差。均值与方差均无计算图。
    fork_rng 恢复 CPU 及涉及的 CUDA 随机流，不改变原 SAC 的策略采样序列。
    """
    if not isinstance(target_samples, Integral) or target_samples < 2:
        raise ValueError('target_samples 至少为 2，才能估计均值方差')
    if not isinstance(seed, Integral) or not isinstance(step, Integral) or step < 0:
        raise ValueError('seed 必须为整数，step 必须为非负整数')
    next_obs = samples_data['next_observation'].detach()
    rewards = _scalar_predictions(samples_data['reward'].detach())
    terminals = _scalar_predictions(samples_data['terminal'].detach())
    count = len(next_obs)
    if next_obs.ndim != 2 or len(rewards) != count or len(terminals) != count:
        raise ValueError('下一状态、奖励和终止标记的批次形状不匹配')
    devices = set()
    for module in (policy, target_qf1, target_qf2):
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.is_cuda:
                devices.add(tensor.device.index)
    if next_obs.is_cuda:
        devices.add(next_obs.device.index)
    local_seed = (int(seed) + 1000033 * int(step)) % (2 ** 63 - 1)

    with torch.no_grad(), torch.random.fork_rng(devices=sorted(devices)), \
            _preserve_forward_cache(policy), _preserve_forward_cache(target_qf1), \
            _preserve_forward_cache(target_qf2):
        # torch.manual_seed 会同时修改所有 CUDA 设备；只设本次已保存的流。
        torch.random.default_generator.manual_seed(local_seed)
        for device in sorted(devices):
            with torch.cuda.device(device):
                torch.cuda.manual_seed(local_seed)
        distribution = policy(next_obs, seq_idx)[0]
        entropy_scale = torch.as_tensor(alpha, device=next_obs.device,
                                        dtype=next_obs.dtype).detach()
        if entropy_scale.numel() == 1:
            entropy_scale = entropy_scale.reshape(())
        else:
            entropy_scale = _scalar_predictions(entropy_scale)
            if len(entropy_scale) != count:
                raise ValueError('逐样本 alpha 长度必须与目标批次一致')
        target_bank = []
        for _ in range(target_samples):
            pre_tanh, action = distribution.rsample_with_pre_tanh_value()
            log_pi = _scalar_predictions(distribution.log_prob(
                value=action, pre_tanh_value=pre_tanh))
            next_q1 = _scalar_predictions(target_qf1(next_obs, action, seq_idx=seq_idx))
            next_q2 = _scalar_predictions(target_qf2(next_obs, action, seq_idx=seq_idx))
            target = reward_scale * rewards + discount * (1.0 - terminals) * (
                torch.minimum(next_q1, next_q2) - entropy_scale * log_pi)
            target_bank.append(_scalar_predictions(target).double())
        values = torch.stack(target_bank)
        mean_target = values.mean(dim=0)
        variance_of_mean = values.var(dim=0, unbiased=True) / target_samples
    return mean_target.detach(), variance_of_mean.detach()


class SACBellmanGeometryRegularizer:
    """为已有 critic loss 添加正则，不持有网络、优化器或跨步计算图。"""

    def __init__(self, coefficient=0.01, isotropic_fraction=0.1, ridge=0.01,
                 probe_size=32, seed=0, scale_penalty=0.01, max_grad_ratio=0.1):
        _validate_penalty_options(isotropic_fraction, ridge)
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError('coefficient 必须是非负有限数')
        if not isinstance(probe_size, Integral) or probe_size < 2:
            raise ValueError('probe_size 必须为至少 2 的整数')
        if not math.isfinite(scale_penalty) or scale_penalty < 0:
            raise ValueError('scale_penalty 必须是非负有限数')
        if not math.isfinite(max_grad_ratio) or not 0 <= max_grad_ratio < 1:
            raise ValueError('max_grad_ratio 必须为 [0, 1) 内的有限数')
        if not isinstance(seed, Integral):
            raise ValueError('seed 必须为整数')
        self.coefficient = float(coefficient)
        self.isotropic_fraction = float(isotropic_fraction)
        self.ridge = float(ridge)
        self.scale_penalty = float(scale_penalty)
        self.max_grad_ratio = float(max_grad_ratio)
        self.probe_size = int(probe_size)
        self.seed = int(seed)

    def probe_indices(self, batch_size, step=0):
        """无放回子采样；不消耗全局 RNG，也没有需额外恢复的 RNG 状态。"""
        if not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError('batch_size 必须为正整数')
        if not isinstance(step, Integral) or step < 0:
            raise ValueError('step 必须为非负整数')
        if self.probe_size >= batch_size:
            return torch.arange(batch_size)
        generator = torch.Generator(device='cpu')
        generator.manual_seed((self.seed + 1000003 * step) % (2 ** 63 - 1))
        return torch.randperm(batch_size, generator=generator)[:self.probe_size]

    @staticmethod
    def _subset_task(seq_idx, indices, batch_size):
        """混合任务批次的任务索引也必须随转移一起取子集。"""
        if seq_idx is None or isinstance(seq_idx, Integral):
            return None if seq_idx is None else int(seq_idx)
        if isinstance(seq_idx, torch.Tensor):
            if seq_idx.ndim == 0:
                return int(seq_idx)
            if len(seq_idx) != batch_size:
                raise ValueError('逐样本任务索引与批次长度不一致')
            return seq_idx.detach().index_select(0, indices.to(seq_idx.device))
        if len(seq_idx) != batch_size:
            raise ValueError('逐样本任务索引与批次长度不一致')
        return [seq_idx[index] for index in indices.tolist()]

    def critic_loss(self, critic, samples_data, seq_idx, targets, base_loss,
                    step=0):
        """用调用方提供的均值 targets，在独立输入图上计算辅助项。

        base_loss 可直接传现有 SAC 的 TD loss。辅助输入和目标都 detach，
        确保正则只更新当前 critic；正常 actor 更新仍使用原 SAC 路径。
        """
        if self.coefficient == 0 or self.max_grad_ratio == 0:
            return base_loss, {'enabled': False, 'weighted_penalty': 0.0}
        observations = samples_data['observation']
        actions = samples_data['action']
        batch_size = len(observations)
        targets = _scalar_predictions(targets.detach())
        if (len(actions) != batch_size or len(targets) != batch_size or
                observations.ndim != 2 or actions.ndim != 2):
            raise ValueError('observation、action 和 targets 的批次形状不匹配')
        indices = self.probe_indices(batch_size, step)
        obs = observations.detach().index_select(0, indices.to(observations.device))
        act = actions.detach().index_select(0, indices.to(actions.device))
        target = targets.index_select(0, indices.to(targets.device))
        task = self._subset_task(seq_idx, indices, batch_size)

        # 辅助 forward 不能覆盖正常 TD forward 的特征，也不能向原有的
        # dormant/zero-ratio 统计列表追加额外观测。临时统计使用空容器，
        # 避免每次复制随训练增长的历史列表；最后恢复所有原对象。
        with _preserve_forward_cache(critic):
            predictions = _scalar_predictions(critic(obs, act, seq_idx=task))
            # 均值目标使用 double，避免在计算残差前再次舍入到网络精度。
            demand = (target.to(device=predictions.device, dtype=torch.float64) -
                      predictions.double()).detach()
            jacobian = full_critic_jacobian(predictions, critic.parameters())
            penalty, stats = bellman_geometry_penalty(
                jacobian, demand, self.isotropic_fraction, self.ridge,
                self.scale_penalty)
        # 在完整 TD batch 的梯度上设上限，双 critic 各自计算；不能拿损失
        # 数值的比例替代参数梯度的比例，也不能让系数本身参与反传。
        weight, gradient_stats = bound_geometry_coefficient(
            base_loss, penalty, critic.parameters(), self.coefficient,
            self.max_grad_ratio)
        weighted_penalty = weight * penalty
        stats.update(gradient_stats)
        stats.update({
            'enabled': True,
            'td_loss': float(base_loss.detach()),
            'weighted_penalty': float(weighted_penalty.detach()),
        })
        return base_loss + weighted_penalty, stats


class SACBellmanGeometry(SAC):
    """独立 SAC 子类：复用原采样、TD 目标、actor、温度及 target 更新。

    不修改原 SAC 文件或现有实验入口。辅助抽样只依赖 seed、任务和已完成
    的 critic 更新次数；继续训练时需传入相同的 geometry_* 构造参数。
    完整训练快照会保存这些属性；原有 probe checkpoint 不是完整算法配置。
    """

    def __init__(self, *args, geometry_coefficient=0.01,
                 geometry_isotropic_fraction=0.1, geometry_ridge=0.01,
                 geometry_probe_size=32, geometry_first_task=1,
                 geometry_scale_penalty=0.01, geometry_max_grad_ratio=0.1,
                 geometry_target_samples=8, **kwargs):
        self.geometry_regularizer = SACBellmanGeometryRegularizer(
            coefficient=geometry_coefficient,
            isotropic_fraction=geometry_isotropic_fraction,
            ridge=geometry_ridge, probe_size=geometry_probe_size,
            scale_penalty=geometry_scale_penalty,
            max_grad_ratio=geometry_max_grad_ratio,
            seed=kwargs.get('seed', 0))
        if (not isinstance(geometry_target_samples, Integral) or
                geometry_target_samples < 2):
            raise ValueError('geometry_target_samples 必须为至少 2 的整数')
        if (not isinstance(geometry_first_task, Integral) or
                geometry_first_task < 0):
            raise ValueError('geometry_first_task 必须为非负任务序号')
        if geometry_coefficient > 0:
            conflicts = [key for key in ('pbsr', 'demand_aligned_reserve', 'dsr_v2')
                         if kwargs.get(key, False)]
            if kwargs.get('plasticity_injection_mode', 'none') != 'none':
                conflicts.append('plasticity_injection_mode')
            if conflicts:
                raise ValueError('本版本请单独启用，不能混用：' + ', '.join(conflicts))
        self.geometry_first_task = int(geometry_first_task)
        self.geometry_target_samples = int(geometry_target_samples)
        self.geometry_last_stats = {}
        super().__init__(*args, **kwargs)
        if geometry_coefficient > 0 and any(
                len(getattr(critic, '_plasticity_injection_branches', ()))
                for critic in (self._qf1, self._qf2)):
            raise ValueError('固定容量几何版本不能从含扩容分支的 critic 开始')

    def _critic_objective(self, samples_data, seq_idx, return_predictions=False):
        # 原 SAC 的训练入口请求 predictions；常规 Hessian 诊断不请求。
        # 诊断保留普通 TD，避免对含完整 Jacobian 的正则再求高阶 Hessian。
        if (not return_predictions or self.geometry_regularizer.coefficient == 0
                or self.geometry_regularizer.max_grad_ratio == 0
                or int(seq_idx) < self.geometry_first_task):
            return super()._critic_objective(
                samples_data, seq_idx, return_predictions=return_predictions)
        loss1, loss2, pred1, pred2, targets = super()._critic_objective(
            samples_data, seq_idx, return_predictions=True)
        # 双 critic 使用相同辅助行；任务切换后行序列可复现且彼此区分。
        step = int(self._critic_optimizer_steps) + 10000019 * int(seq_idx)
        # 原 TD loss 继续使用父类的一次采样目标；辅助项单独估计目标均值。
        # 只对 probe 行做额外动作采样，两个 critic 共用均值，避免重复开销。
        indices = self.geometry_regularizer.probe_indices(
            len(samples_data['observation']), step)
        probe_data = {
            key: samples_data[key].detach().index_select(
                0, indices.to(samples_data[key].device))
            for key in ('observation', 'action', 'next_observation',
                        'reward', 'terminal')}
        alpha = self._get_log_alpha(samples_data).exp().detach()
        if alpha.numel() > 1:
            alpha = alpha.index_select(0, indices.to(alpha.device))
        geometry_targets, target_variance = averaged_soft_bellman_targets(
            probe_data, seq_idx, self.policy, self._target_qf1, self._target_qf2,
            alpha, self._discount, self._reward_scale,
            target_samples=self.geometry_target_samples,
            seed=self.geometry_regularizer.seed, step=step)
        # probe_data 的行数不超过 probe_size，因此 helper 内不会再次打乱行。
        loss1, stats1 = self.geometry_regularizer.critic_loss(
            self._qf1, probe_data, seq_idx, geometry_targets, loss1, step=step)
        loss2, stats2 = self.geometry_regularizer.critic_loss(
            self._qf2, probe_data, seq_idx, geometry_targets, loss2, step=step)
        # 只保存 Python 标量；不能让跨步日志持有二阶计算图。
        self.geometry_last_stats = {
            'task': int(seq_idx),
            'critic_optimizer_step': int(self._critic_optimizer_steps),
            'coefficient': self.geometry_regularizer.coefficient,
            'isotropic_fraction': self.geometry_regularizer.isotropic_fraction,
            'ridge': self.geometry_regularizer.ridge,
            'scale_penalty': self.geometry_regularizer.scale_penalty,
            'max_grad_ratio': self.geometry_regularizer.max_grad_ratio,
            'target_samples': self.geometry_target_samples,
            'target_variance_of_mean': float(target_variance.mean()),
            'qf1': stats1,
            'qf2': stats2,
        }
        # 原训练日志中的 Q loss 包含辅助项；另记原始 TD 和加权正则，避免
        # 把联合目标的变化误读成 Bellman 拟合误差的变化。
        if self._critic_optimizer_steps < 3 or self._critic_optimizer_steps % 1000 == 0:
            print('BELLMAN_GEOMETRY', json.dumps(self.geometry_last_stats), flush=True)
        return loss1, loss2, pred1, pred2, targets


class FinetuningSACBellmanGeometry(SACBellmanGeometry, Finetuning_SAC):
    """保留正式训练入口的 Finetuning/MTSAC 行为，叠加几何 critic 目标。

    协作式 super 按 Geometry → Finetuning → MTSAC → SAC 初始化。
    这样任务评估、日志和温度参数继续使用原有实现，普通 SAC helper 的
    独立测试也仍然适用；不复制原训练循环以免两条路径逐渐偏离。
    """
