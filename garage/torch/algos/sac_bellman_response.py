"""SAC 的 Bellman 更新响应修正（独立版本，无逆核及核收缩项）。

在真实 TD 更新后，以当前 Adam 状态做一次可微虚拟更新。校准集产生
更新，互不重叠的检查集评价 R = mean((d - (Q_virtual-Q))²)/mean(d²)。
两个集合的需求均停止梯度；前后两个 Q 都保留梯度。辅助参数位移独立于
Adam，范数至多为刚完成的真实 TD 位移的 response_step_ratio 倍。

虚拟梯度的数值等于当前均值 Bellman 目标的 MSE 梯度；对这个梯度求导时
固定需求，只微分网络响应，避免通过改变需求本身降低辅助目标。一次虚拟
更新不写回参数、优化器状态、随机流或网络特征缓存。检查集不是独立轨迹；
MC 目标均值也不能消除继承 bootstrap 偏差。本方法不保证在线回报提升。
"""

from collections import OrderedDict
import json
import math
from numbers import Integral
import os

import numpy as np
import torch
from torch.nn.utils.stateless import functional_call

from garage.torch import as_torch_dict
from garage.torch.algos.finetuning import Finetuning_SAC
from garage.torch.algos.sac_bellman_geometry import (
    _preserve_forward_cache, _scalar_predictions, averaged_soft_bellman_targets)


class _FiniteSqrt(torch.autograd.Function):
    """前向为精确 sqrt；零二阶矩处采用零次梯度，避免 0 * inf 产生 NaN。"""

    @staticmethod
    def forward(ctx, value):
        root = value.sqrt()
        ctx.save_for_backward(root)
        return root

    @staticmethod
    def backward(ctx, gradient):
        root, = ctx.saved_tensors
        safe = torch.where(root > 0, root, torch.ones_like(root))
        return torch.where(root > 0, gradient / (2 * safe),
                           torch.zeros_like(gradient))


def virtual_adam_parameters(named_parameters, gradients, optimizer):
    """计算下一次 Adam 参数值；保留当前梯度的图，历史状态只读并 detach。"""
    if type(optimizer) is not torch.optim.Adam:
        raise TypeError('响应版本要求标准 torch.optim.Adam')
    named_parameters = OrderedDict(named_parameters)
    gradients = tuple(gradients)
    if len(gradients) != len(named_parameters):
        raise ValueError('虚拟梯度与参数长度不匹配')
    groups = {id(p): group for group in optimizer.param_groups
              for p in group['params']}
    result = OrderedDict()
    for (name, parameter), gradient in zip(named_parameters.items(), gradients):
        if gradient is None:
            result[name] = parameter
            continue
        group = groups[id(parameter)]
        if group.get('maximize', False) or group.get('weight_decay', 0) != 0:
            raise ValueError('当前响应协议仅支持无 weight decay 的最小化 Adam')
        beta1, beta2 = group['betas']
        # get 不向 defaultdict 插入空状态，启动前的虚拟计算也没有副作用。
        state = optimizer.state.get(parameter, {})
        moment = state.get('exp_avg', torch.zeros_like(parameter)).detach()
        variance = state.get('exp_avg_sq', torch.zeros_like(parameter)).detach()
        step = float(state.get('step', 0)) + 1
        next_moment = moment * beta1 + gradient * (1 - beta1)
        next_variance = variance * beta2 + gradient.square() * (1 - beta2)
        if group.get('amsgrad', False):
            maximum = state.get('max_exp_avg_sq', torch.zeros_like(parameter))
            next_variance = torch.maximum(maximum.detach(), next_variance)
        denominator = (_FiniteSqrt.apply(next_variance) /
                       math.sqrt(1 - beta2 ** step) + group['eps'])
        result[name] = parameter - (group['lr'] / (1 - beta1 ** step)) * (
            next_moment / denominator)
    return result


def bellman_response_loss(critic, optimizer, train_data, query_data,
                          train_targets, query_targets, seq_idx,
                          fixed_demands=None, create_graph=True, collect_stats=True):
    """评价跨批次的一步函数增量；fixed_demands 用于同需求数值复核。"""
    named = OrderedDict((name, p) for name, p in critic.named_parameters()
                        if p.requires_grad)
    if not named:
        raise ValueError('critic 没有可训练参数')
    with _preserve_forward_cache(critic):
        train_q = _scalar_predictions(critic(
            train_data['observation'].detach(), train_data['action'].detach(),
            seq_idx=seq_idx))
        query_q = _scalar_predictions(critic(
            query_data['observation'].detach(), query_data['action'].detach(),
            seq_idx=seq_idx))
        if fixed_demands is None:
            train_d = (train_targets.detach().reshape(-1).double() -
                       train_q.double()).detach()
            query_d = (query_targets.detach().reshape(-1).double() -
                       query_q.double()).detach()
        else:
            train_d, query_d = (d.detach().double() for d in fixed_demands)
        if len(train_d) != len(train_q) or len(query_d) != len(query_q):
            raise ValueError('需求与样本长度不匹配')
        energy = query_d.square().mean()
        # 线性代理只用于得到 -2 J^T d/n，数值与 MSE 梯度完全一致。
        # 必须固定 d；直接对普通 MSE 二阶求导会把需求变化混进响应塑形。
        surrogate = -2 * (train_q.double() * train_d).mean()
        gradients = torch.autograd.grad(
            surrogate, tuple(named.values()), create_graph=create_graph,
            retain_graph=True, allow_unused=True)
        virtual = virtual_adam_parameters(named, gradients, optimizer)
        # 克隆 buffer，防止未来加入可变 buffer 的网络污染真实模型。
        virtual.update((name, value.detach().clone())
                       for name, value in critic.named_buffers())
        virtual_q = _scalar_predictions(functional_call(
            critic, virtual,
            (query_data['observation'].detach(), query_data['action'].detach()),
            {'seq_idx': seq_idx}))
        # query_q 不能 detach：差值的梯度应消去仅平移当前预测的途径。
        response = virtual_q.double() - query_q.double()
        loss = (query_d - response).square().mean() / (energy + 1e-12)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Bellman 更新响应出现非有限值')
        if bool(energy <= 1e-12):
            loss = loss * 0.0
        # 日志只每千步使用一次；其余步不做这些 reduction 和 GPU→CPU 取值。
        # 有限值检查及影响算法分支的判断仍按原频率执行。
        stats = ({
            'response_loss': float(loss.detach()),
            'demand_energy': float(energy),
            'response_energy': float(response.detach().square().mean()),
            'response_demand_inner_product': float(
                (response.detach() * query_d).mean()),
            'train_rows': len(train_d), 'query_rows': len(query_d),
        } if collect_stats else {})
    return loss, stats, (train_d, query_d)


def _norm(values):
    return torch.stack([v.detach().double().square().sum()
                        for v in values]).sum().sqrt()


def apply_bounded_response_step(parameters, gradients, td_delta, ratio,
                                collect_stats=True):
    """直接施加辅助位移，并按浮点舍入后的实际位移检查范数上限。"""
    parameters = tuple(parameters)
    gradients = tuple(torch.zeros_like(p) if g is None else g.detach()
                      for p, g in zip(parameters, gradients))
    td_norm, grad_norm = _norm(td_delta), _norm(gradients)
    if not bool(torch.isfinite(td_norm) & torch.isfinite(grad_norm)):
        raise FloatingPointError('TD 位移或辅助梯度出现非有限值')
    stats = ({'td_delta_norm': float(td_norm),
             'response_gradient_norm': float(grad_norm),
             'aux_delta_norm': 0.0, 'aux_to_td_delta_ratio': 0.0,
             'applied': False} if collect_stats else {})
    if ratio == 0 or bool(td_norm == 0) or bool(grad_norm <= 1e-12):
        stats['skip_reason'] = 'zero_or_tiny_update'
        return stats
    cap = ratio * td_norm
    scale = cap / grad_norm
    with torch.no_grad():
        old = [p.detach().clone() for p in parameters]
        # 浮点加法可能让实际位移略超目标；收缩后重新检查，不改 Adam 状态。
        for _ in range(4):
            candidates = [p - scale.to(p) * g for p, g in zip(old, gradients)]
            actual_norm = _norm([new - p for new, p in zip(candidates, old)])
            if bool(actual_norm <= cap):
                for p, candidate in zip(parameters, candidates):
                    p.copy_(candidate)
                if collect_stats:
                    stats.update(aux_delta_norm=float(actual_norm),
                                 aux_to_td_delta_ratio=float(actual_norm / td_norm),
                                 applied=bool(actual_norm > 0))
                return stats
            scale = scale * (0.99 * cap / actual_norm)
    stats['skip_reason'] = 'rounding_exceeds_budget'
    return stats


def response_branch_steps(checkpoint, task_count, steps_per_task, task_step=0):
    """从指定检查点计算剩余序列预算，包含恢复点之后的全部任务。"""
    start_task = int(checkpoint['seq_idx']) + (0 if task_step > 0 else 1)
    if not 0 <= start_task < task_count or not 0 <= task_step < steps_per_task:
        raise ValueError('恢复任务序号或任务内进度超出训练预算')
    return (task_count - start_task) * steps_per_task - task_step


class FinetuningSACBellmanResponse(Finetuning_SAC):
    """保留正式 SAC 流程，只在双 critic 更新后、actor 更新前修正响应。"""

    def __init__(self, *args, response_step_ratio=0.05, response_rows=64,
                 response_target_samples=8, response_update_interval=1,
                 response_first_task=1, **kwargs):
        if not math.isfinite(response_step_ratio) or not 0 <= response_step_ratio < 1:
            raise ValueError('response_step_ratio 必须在 [0, 1) 内')
        for name, value, minimum in (
                ('rows', response_rows, 2), ('target_samples', response_target_samples, 2),
                ('update_interval', response_update_interval, 1),
                ('first_task', response_first_task, 0)):
            if not isinstance(value, Integral) or value < minimum:
                raise ValueError('无效响应配置：' + name)
        if any(kwargs.get(key, False) for key in
               ('pbsr', 'demand_aligned_reserve', 'dsr_v2', 'q_reset',
                'policy_reset', 'infer', 'wasserstein', 'ReDo')):
            raise ValueError('响应版本必须单独启用')
        if kwargs.get('plasticity_injection_mode', 'none') != 'none':
            raise ValueError('响应版本不混用扩容分支')
        self.response_config = {
            'version': 1, 'step_ratio': float(response_step_ratio),
            'rows': int(response_rows), 'target_samples': int(response_target_samples),
            'update_interval': int(response_update_interval),
            'first_task': int(response_first_task), 'seed': int(kwargs.get('seed', 0))}
        self._bellman_response_enabled = True
        self._response_td_start = None
        self.response_last_stats = {}
        super().__init__(*args, **kwargs)
        if any(len(getattr(q, '_plasticity_injection_branches', ()))
               for q in (self._qf1, self._qf2)):
            raise ValueError('响应版本不能恢复已扩容的 critic')
        print('BELLMAN_RESPONSE_CONFIG', json.dumps(self.response_config), flush=True)

    def optimize_policy(self, samples_data, seq_idx):
        config = self.response_config
        due = (config['step_ratio'] > 0 and seq_idx >= config['first_task'] and
               self._critic_optimizer_steps % config['update_interval'] == 0)
        if due:
            self._response_td_start = [
                [p.detach().clone() for p in q.parameters()]
                for q in (self._qf1, self._qf2)]
        try:
            return super().optimize_policy(samples_data, seq_idx)
        finally:
            self._response_td_start = None

    def _response_batches(self, seq_idx):
        """从当前 replay 均匀无放回采样；不改变原 SAC 的 numpy 随机流。"""
        rows = self.response_config['rows']
        available = self.replay_buffer.n_transitions_stored
        if available < 2 * rows:
            raise ValueError('当前 replay 不足以构造互不重叠的响应批次')
        seed = (self.response_config['seed'] + 1000003 * self._critic_optimizer_steps +
                10000019 * int(seq_idx)) % (2 ** 32 - 1)
        generator = np.random.RandomState(seed)
        # 避免 choice(replace=False) 每一步分配百万长度的完整排列。
        chosen, seen = [], set()
        while len(chosen) < 2 * rows:
            for index in generator.randint(0, available, size=2 * rows):
                if int(index) not in seen:
                    chosen.append(int(index))
                    seen.add(int(index))
                    if len(chosen) == 2 * rows:
                        break
        data = as_torch_dict(self.replay_buffer.sample_transitions(
            2 * rows, idx=np.asarray(chosen)))
        return data, rows

    def _after_critic_update(self, samples_data, seq_idx):
        if self._response_td_start is None:
            return
        config = self.response_config
        diagnostic = self._critic_optimizer_steps <= 3 or self._critic_optimizer_steps % 1000 == 0
        data, rows = self._response_batches(seq_idx)
        alpha = self._get_log_alpha(data).exp().detach()
        targets, variance = averaged_soft_bellman_targets(
            data, seq_idx, self.policy, self._target_qf1, self._target_qf2,
            alpha, self._discount, self._reward_scale,
            target_samples=config['target_samples'], seed=config['seed'],
            step=self._critic_optimizer_steps + 10000019 * int(seq_idx))
        train = {key: value[:rows] for key, value in data.items()}
        query = {key: value[rows:] for key, value in data.items()}
        record = {'global_step': int(self.global_step + 1), 'task': int(seq_idx),
                  'critic_optimizer_step': int(self._critic_optimizer_steps)}
        if diagnostic:
            record['target_variance_of_mean'] = float(variance.mean())
        for index, (name, critic, optimizer) in enumerate((
                ('qf1', self._qf1, self._qf1_optimizer),
                ('qf2', self._qf2, self._qf2_optimizer))):
            parameters = tuple(critic.parameters())
            td_delta = [p.detach() - old for p, old in
                        zip(parameters, self._response_td_start[index])]
            loss, stats, demands = bellman_response_loss(
                critic, optimizer, train, query, targets[:rows], targets[rows:], seq_idx,
                collect_stats=diagnostic)
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            stats.update(apply_bounded_response_step(
                parameters, gradients, td_delta, config['step_ratio'],
                collect_stats=diagnostic))
            if diagnostic:
                # 固定同一需求及同一 Adam 状态，只复核辅助位移造成的响应变化。
                after, _, _ = bellman_response_loss(
                    critic, optimizer, train, query, targets[:rows], targets[rows:],
                    seq_idx, fixed_demands=demands, create_graph=False)
                stats['response_loss_after_aux'] = float(after.detach())
                record[name] = stats
        if diagnostic:
            # 此属性明确保存最近一次已记录的诊断，不保留每步的 GPU 张量。
            self.response_last_stats = record
            print('BELLMAN_RESPONSE', json.dumps(record), flush=True)
            if self._bellman_probe:
                with open(os.path.join(self._bellman_probe_run_dir,
                                       'response_updates.jsonl'), 'a') as stream:
                    stream.write(json.dumps(record) + '\n')

    def task_change(self, seq_idx):
        super().task_change(seq_idx)
        # 新任务训练奖励窗口只记录当前任务，避免混入上一个任务的高奖励。
        self.episode_rewards.clear()

    def response_checkpoint_state(self):
        return {'config': self.response_config, 'last_stats': self.response_last_stats}
