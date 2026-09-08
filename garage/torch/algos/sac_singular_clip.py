"""普通 SAC 的双边权重谱裁剪：只修改两个在线 critic 的 Linear 权重。

方法来源：https://arxiv.org/abs/2608.18319 。这里是 critic-only 适配，
保留原始 Adam、网络结构、actor 更新和 target Polyak 更新。
"""

import json
import math
from numbers import Integral
import os

import torch

from garage import log_performance, obtain_evaluation_episodes
from garage.torch.algos.finetuning import Finetuning_SAC


@torch.no_grad()
def clip_weight_singular_values_(weight, lower=.25, upper=4.):
    """保留 SVD 的左右基底，将奇异值裁剪后写回原参数对象。"""
    if (not math.isfinite(lower) or not math.isfinite(upper) or
            not 0 < lower <= upper):
        raise ValueError('奇异值范围必须满足 0 < lower <= upper 且有限')
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError('谱裁剪要求有限的二维权重矩阵')
    # 裁剪很稀疏；使用 FP64 SVD 减少小奇异值和重构的数值误差。
    original = weight.detach().to(dtype=torch.float64)
    u, singular, vh = torch.linalg.svd(original, full_matrices=False)
    clipped = singular.clamp(lower, upper)
    below = int((singular < lower).sum())
    above = int((singular > upper).sum())
    if below or above:
        candidate = ((u * clipped.unsqueeze(0)) @ vh).to(weight.dtype)
        if not torch.isfinite(candidate).all():
            raise FloatingPointError('谱裁剪重构出现非有限参数')
        displacement = float((candidate.double() - original).norm())
        weight.copy_(candidate)
    else:
        # 已在区间内时保持逐位不变，避免无意义的 SVD 重构漂移。
        displacement = 0.
    return dict(shape=list(weight.shape), singular_min_before=float(singular.min()),
                singular_max_before=float(singular.max()),
                singular_min_projected=float(clipped.min()),
                singular_max_projected=float(clipped.max()),
                lifted=below, lowered=above, delta_norm=displacement)


class FinetuningSACSingularClip(Finetuning_SAC):
    """在真实 critic Adam 更新后、actor 更新前，周期性直接裁剪权重。"""

    def __init__(self, singular_clip_min=.25, singular_clip_max=4.,
                 singular_clip_interval=200000, singular_clip_start_task=1,
                 **kwargs):
        if (not math.isfinite(singular_clip_min) or
                not math.isfinite(singular_clip_max) or
                not 0 < singular_clip_min <= singular_clip_max):
            raise ValueError('无效的奇异值上下界')
        if not isinstance(singular_clip_interval, Integral) or singular_clip_interval < 1:
            raise ValueError('谱裁剪间隔必须为正整数')
        if (not isinstance(singular_clip_start_task, Integral) or
                singular_clip_start_task < 0):
            raise ValueError('谱裁剪起始任务必须为非负整数')
        self._singular_clip_enabled = True
        self._singular_clip_config = dict(lower=singular_clip_min, upper=singular_clip_max,
                                          interval=singular_clip_interval,
                                          start_task=int(singular_clip_start_task),
                                          scope='online_critics_all_linear_weights',
                                          optimizer='Adam', optimizer_state='preserve',
                                          target_update='entry_hard_sync_periodic_polyak',
                                          schedule='task_local_environment_steps')
        self._singular_clip_events = []
        self._singular_clip_event_keys = set()
        super().__init__(**kwargs)
        print('SINGULAR_CLIP_CONFIG', json.dumps(self._singular_clip_config), flush=True)

    def train(self, trainer):
        # Also support an explicit start_task=0 ablation; the formal default
        # is 1, so A receives no intervention at all.
        self._clip_at_task_start(int(self.seq_idx))
        return super().train(trainer)

    def _after_critic_update(self, samples, seq_idx):
        # The final update of a collection is immediately followed by the
        # ordinary Polyak update and diagnostics/evaluation. UTD does not
        # change this clock, and warm-up is included in the environment count.
        step = int(self.global_env_step - self._task_env_start_step)
        config = self._singular_clip_config
        if (int(seq_idx) < config['start_task'] or step == 0 or
                step % config['interval'] or
                not getattr(self, '_is_last_collection_update', True)):
            return
        if self._sampler is not None:
            next_task = self._sampler._envs[0].cur_seq_idx
            if (next_task != seq_idx and self._task_names and
                    next_task < len(self._task_names)):
                # A coincident outgoing interval/incoming boundary is one
                # entry event (after boundary evaluation), not two clips.
                return
        self._clip_critics(int(seq_idx), int(self.global_step + 1),
                           int(self._critic_optimizer_steps), 'interval')

    def _clip_critics(self, task, global_step, critic_optimizer_step, trigger):
        task_env_step = int(self.global_env_step - self._task_env_start_step)
        event_key = (int(task), task_env_step)
        if event_key in self._singular_clip_event_keys:
            return False
        config = self._singular_clip_config
        records = {}
        for label, model in (('qf1', self._qf1), ('qf2', self._qf2)):
            records[label] = {}
            for name, layer in model.named_modules():
                if isinstance(layer, torch.nn.Linear):
                    # 包括标量 Q 输出头的权重；偏置、Adam 状态和参数身份不变。
                    records[label][name + '.weight'] = clip_weight_singular_values_(
                        layer.weight, config['lower'], config['upper'])
        event = dict(global_step=int(global_step), task=int(task),
                     global_env_step=int(self.global_env_step),
                     task_env_step=task_env_step,
                     critic_optimizer_step=int(critic_optimizer_step),
                     target_update=('hard_sync' if trigger == 'task_start'
                                    else 'polyak'),
                     trigger=trigger, critics=records)
        self._singular_clip_event_keys.add(event_key)
        self._singular_clip_events.append(event)
        print('SINGULAR_CLIP', json.dumps(event), flush=True)
        if self._bellman_probe:
            path = os.path.join(self._bellman_probe_run_dir, 'singular_clip_events.jsonl')
            with open(path, 'a') as stream:
                stream.write(json.dumps(event) + '\n')
        return True

    def singular_clip_checkpoint_state(self):
        return dict(config=self._singular_clip_config, events=self._singular_clip_events)

    def _clip_at_task_start(self, task):
        if (task < self._singular_clip_config['start_task'] or
                (self._task_names and task >= len(self._task_names))):
            return
        if self._clip_critics(task, int(self.global_step), 0, 'task_start'):
            self._target_qf1.load_state_dict(self._qf1.state_dict())
            self._target_qf2.load_state_dict(self._qf2.state_dict())

    def task_change(self, seq_idx):
        result = super().task_change(seq_idx)
        self._clip_at_task_start(int(seq_idx) + 1)
        self.episode_rewards.clear()
        return result

    def _evaluate_policy(self, epoch):
        """Current task periodically; all seen occurrences at task exit."""
        at_boundary = self._sampler._envs[0].cur_seq_idx != self.seq_idx
        positions = range(self.seq_idx + 1) if at_boundary else [self.seq_idx]
        current_returns = None
        for position in positions:
            self.on_test_start(position)
            try:
                episodes = obtain_evaluation_episodes(
                    self.policy, self._eval_env[position], position,
                    self._max_episode_length_eval,
                    num_eps=self._num_evaluation_episodes,
                    deterministic=self._use_deterministic_evaluation)
            finally:
                self.on_test_end(position)
            # Position, not a unique task name, selects the actor head. DMC
            # revisit curves must not be merged into the first occurrence.
            name = self._task_names[position]
            prefix = 'test/{}/{}/'.format(position, name)
            returns = log_performance(
                epoch, episodes, self._discount, self.results,
                prefix=prefix, use_wandb=self._use_wandb)
            self.results.setdefault(prefix + 'Global env step', []).append(
                self.global_env_step)
            if position == self.seq_idx:
                current_returns = returns
        return current_returns
