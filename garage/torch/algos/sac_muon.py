"""普通顺序 SAC 的 Muon 版本：保留原始损失、网络和更新顺序。"""

import json

from garage.torch.algos.finetuning import Finetuning_SAC
from garage.torch.optimizers.muon import MuonWithAdam


def muon_branch_steps(checkpoint, task_count, steps_per_task, task_step=0):
    """续训预算包含检查点后面的全部任务，B 结束后继续训练 C。"""
    start = int(checkpoint['seq_idx']) + (0 if task_step > 0 else 1)
    if not 0 <= start < task_count or not 0 <= task_step < steps_per_task:
        raise ValueError('Muon 续训任务或任务内步数超出范围')
    return (task_count - start) * steps_per_task - task_step


class FinetuningSACMuon(Finetuning_SAC):
    def __init__(self, muon_lr=3e-4, muon_momentum=.95, muon_ns_steps=5, **kwargs):
        self._muon_enabled = True
        self._muon_options = dict(muon_lr=muon_lr, momentum=muon_momentum,
                                  ns_steps=muon_ns_steps)
        super().__init__(**kwargs)
        # 普通 SAC 对照只保存模型和成功率，不计算依赖 Adam 度量的谱探针。
        self._bellman_probe_buffers = {}
        print('SAC_MUON_CONFIG', json.dumps(self.muon_checkpoint_state()), flush=True)

    def _make_network_optimizer(self, model, lr):
        return MuonWithAdam(model.named_parameters(), lr=lr, **self._muon_options)

    def muon_checkpoint_state(self):
        return dict(options=self._muon_options,
                    policy=self._policy_optimizer.description(),
                    qf1=self._qf1_optimizer.description(),
                    qf2=self._qf2_optimizer.description(),
                    alpha_optimizer='Adam',
                    adam_checkpoint_conversion='reset_hidden_momentum_keep_adam_heads',
                    auxiliary_losses=False)

    def _append_bellman_probe(self, path, seq_idx):
        """复用 checkpoint 开关，但不额外收集谱分析样本。"""

    def _run_bellman_probe(self, event, task_idx):
        """Muon 实验只使用原始 SAC 训练；不计算 Bellman 辅助目标或探针。"""

    def task_change(self, seq_idx):
        super().task_change(seq_idx)
        # 清除跨任务奖励滑动统计，避免 B 初期显示 A 的奖励。
        self.episode_rewards.clear()
