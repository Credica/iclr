"""适配 PyTorch 1.13 的单卡 Muon；非隐藏层参数继续使用标准 Adam。

Muon 核心参考 https://github.com/KellerJordan/Muon/blob/master/muon.py 。
Newton–Schulz 核心版权 (c) 2024 Keller Jordan，MIT 许可见同目录 MUON_LICENSE。
采用 PyTorch Muon 的 match_rms_adamw 缩放：0.2 * sqrt(max(rows, cols))。
这只是学习率尺度约定，不保证每一步与 Adam 的真实位移相等。
"""

import copy
import math

import torch


def orthogonalize(update, steps=5):
    """用五次 Newton–Schulz 迭代近似正交化，不对网络权重做投影。"""
    if update.ndim != 2:
        raise ValueError('Muon 仅接受二维隐藏层矩阵')
    # CUDA 使用原实现的 BF16；CPU 使用 FP32，兼容旧版 PyTorch。
    x = update.to(dtype=torch.bfloat16 if update.is_cuda else torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.t()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.t()
        # BF16 下乘法顺序会改变舍入；保持与作者参考实现相同的先缩放后乘积。
        polynomial = -4.7750 * gram + (2.0315 * gram) @ gram
        x = 3.4445 * x + polynomial @ x
    return x.t() if transposed else x


class MuonWithAdam(torch.optim.Optimizer):
    """一个优化器保存两类参数状态，便于 SAC 保存和恢复 checkpoint。"""

    def __init__(self, named_parameters, lr=3e-4, muon_lr=3e-4,
                 momentum=.95, ns_steps=5):
        if not all(math.isfinite(x) and x > 0 for x in (lr, muon_lr)):
            raise ValueError('Adam/Muon 学习率必须为有限正数')
        if not 0 <= momentum < 1 or ns_steps < 1:
            raise ValueError('无效的 Muon 动量或迭代次数')
        named = list(named_parameters)
        self._original_named_parameters = named
        muon, adam = [], []
        for name, parameter in named:
            # Garage 的隐藏层位于 _layers；输出头位于 _output_layers。
            # 明确按层名选择，避免误将 actor 的 mean/log_std 输出矩阵交给 Muon。
            hidden = '_layers' in name.split('.') and parameter.ndim == 2
            (muon if hidden else adam).append((name, parameter))
        if not muon:
            raise ValueError('没有识别到 Muon 隐藏层矩阵，拒绝静默退回 Adam')
        groups = []
        for use_muon, entries in ((True, muon), (False, adam)):
            if entries:
                groups.append(dict(params=[p for _, p in entries],
                                   param_names=[n for n, _ in entries],
                                   use_muon=use_muon,
                                   lr=muon_lr if use_muon else lr))
        super().__init__(groups, dict(momentum=momentum, ns_steps=ns_steps,
                                     betas=(.9, .999), eps=1e-8,
                                     weight_decay=0.,
                                     adjust_lr_fn='match_rms_adamw'))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                gradient = parameter.grad
                # 未使用的任务头没有梯度，遵循 PyTorch Adam 的跳过语义。
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError('当前 SAC Muon 不支持稀疏梯度')
                state = self.state[parameter]
                if group['use_muon']:
                    if not state:
                        state['momentum_buffer'] = torch.zeros_like(parameter)
                        state['step'] = 0
                    state['step'] += 1
                    beta = group['momentum']
                    buffer = state['momentum_buffer']
                    buffer.lerp_(gradient, 1 - beta)
                    # 非原地 lerp：保留 .grad，避免影响后续诊断或调用者。
                    update = gradient.lerp(buffer, beta)
                    update = orthogonalize(update, group['ns_steps'])
                    scale = .2 * math.sqrt(max(parameter.shape))
                    parameter.add_(update.to(parameter.dtype),
                                   alpha=-group['lr'] * scale)
                else:
                    if not state:
                        state.update(step=0, exp_avg=torch.zeros_like(parameter),
                                     exp_avg_sq=torch.zeros_like(parameter))
                    state['step'] += 1
                    beta1, beta2 = group['betas']
                    state['exp_avg'].mul_(beta1).add_(gradient, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(
                        gradient, gradient, value=1 - beta2)
                    step = int(state['step'])
                    denominator = (state['exp_avg_sq'].sqrt() /
                                   math.sqrt(1 - beta2 ** step)).add_(group['eps'])
                    parameter.addcdiv_(state['exp_avg'], denominator,
                                       value=-group['lr'] / (1 - beta1 ** step))
        return loss

    def load_state_dict(self, saved):
        """Muon 原样恢复；Adam→Muon 时仅重建隐藏矩阵的动量状态。"""
        if all('use_muon' in group for group in saved['param_groups']):
            current_names = [g['param_names'] for g in self.param_groups]
            if [g.get('param_names') for g in saved['param_groups']] != current_names:
                raise ValueError('Muon checkpoint 的参数分组不匹配')
            return super().load_state_dict(saved)
        # 旧 SAC 的 Adam 只有一组，顺序与 model.named_parameters() 一致。
        if len(saved['param_groups']) != 1:
            raise ValueError('只支持普通 SAC 单参数组 Adam checkpoint 的转换')
        original = saved['param_groups'][0]
        if original.get('weight_decay', 0) or original.get('amsgrad', False):
            raise ValueError('不支持带 weight decay 或 AMSGrad 的旧 checkpoint')
        if len(original['params']) != len(self._original_named_parameters):
            raise ValueError('Adam checkpoint 的参数数量不匹配')
        old_by_name = dict(zip((n for n, _ in self._original_named_parameters),
                               original['params']))
        mapped = self.state_dict()
        mapped['state'] = {}
        for new_group, live_group in zip(mapped['param_groups'], self.param_groups):
            if new_group['use_muon']:
                continue
            for key in ('lr', 'betas', 'eps'):
                new_group[key] = original[key]
            for name, new_id, parameter in zip(new_group['param_names'],
                                                new_group['params'], live_group['params']):
                old = saved['state'].get(old_by_name[name])
                if old:
                    if old['exp_avg'].shape != parameter.shape:
                        raise ValueError('Adam checkpoint 参数形状不匹配：' + name)
                    mapped['state'][new_id] = copy.deepcopy(old)
        super().load_state_dict(mapped)

    def description(self):
        """记录实际分组及超参数，不在训练的每一步打印。"""
        return [dict(method='muon' if g['use_muon'] else 'adam',
                     lr=g['lr'], parameters=list(g['param_names']),
                     shapes=[list(p.shape) for p in g['params']],
                     momentum=g['momentum'] if g['use_muon'] else None,
                     ns_steps=g['ns_steps'] if g['use_muon'] else None,
                     adjust_lr_fn=g['adjust_lr_fn'] if g['use_muon'] else None,
                     weight_decay=g['weight_decay']) for g in self.param_groups]
