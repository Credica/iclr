# 实验执行清单：Bellman Demand and Learning Spectra

本文档只记录需要实现、验证和运行的实验，是仓库中的唯一执行清单。论文结构、理论主张和图表叙事以 `PAPER_OUTLINE_20260907.md` 为准；理论推导以 `MOTIVATION_METHOD_THEORY_20260907.md` 为准。本文档中的预算和表格均为计划，不是已经获得的实验结果。

## 1. 固定实验口径

| 项目 | 固定决定 |
|---|---|
| 正式随机种子 | 1、2、3 |
| 单任务训练预算 | 1,500,000 个实际训练环境交互步，包含 10,000 步 warm-up |
| 任务内评估 | 每 50,000 环境步评估当前任务 50 episodes，另含任务入口与出口 |
| 任务边界评估 | 每个任务结束时评估全部已见任务，各 50 episodes；不评估未来任务 |
| Meta-World 网络 | 2×256 ReLU SAC；actor 按任务位置分配 head，critic 共享 |
| DMC 网络 | 2×1024 SAC；重复访问同名任务时分配新的 occurrence head |
| 统计单位 | seed；报告 3 seeds 的均值、样本标准差、全部 seed 曲线和配对差 |
| 训练数据与评估数据 | Meta-World 每个 task/seed 固定 50 个训练实例和独立的 50 个评估实例；DMC 固定 50 个评估 reset seeds |
| 结果边界 | episodes、tasks、checkpoints 和 transfer directions 都不能作为额外独立 seeds |

所有方法必须共享同一任务顺序、预算、实例 bank、评估协议和 head 语义。训练、评估、离线分析和方法专有计算分别记账，不声称不同方法的计算成本相同。

## 2. 执行顺序与硬门槛

实验按 E0→E5 执行。某一阶段的正确性门槛未通过时，不启动后续大规模训练。

| 阶段 | 内容 | 在线训练规模 | 当前状态 | 进入下一阶段的条件 |
|---|---|---:|---|---|
| E0 | 统一训练、评估、记录和恢复管线 | 仅 smoke runs | FT 记录版已完成首轮验证；其他方法待接入 | 时钟、RNG、边界事件、恢复和数据 schema 全部通过测试 |
| E1 | P1–P6 的 FT 首批运行 | 18 个双任务 runs | 2026-09-07 已启动 | 18 个 runs 完整结束，字段审计无缺失，结果只用于质检和后续同源分支 |
| E2 | 完整 6×6 rethink 矩阵：FT、Reset、Clip | 18 个源预训练 + 324 个目标分支 | 待实现/运行 | 三方法同源、共同 warm-up、边界行为和完整矩阵核验通过 |
| E3 | 64 状态 MRP 与真实 critic 离线机制实验 | 不增加正式在线 RL runs | 待实现 | 理论量、共同 correction、共同 target 和实际更新误差可重算 |
| E4 | 五条主序列 × 七方法 | 105 个序列配置 | 待实现/运行 | 七方法均满足相同协议，DMC 与重复任务路径通过 smoke tests |
| E5 | 汇总、配对统计和图表数据审计 | 不增加正式在线 RL runs | 待实现 | Tables 1–3 与 Figs. 1–4 的每个数均可追溯到 run、seed、checkpoint 和角色 |

E1 正在运行的进程使用启动时冻结的源码快照；仓库后续清理不会改变这些进程的代码。E1 只提供 FT 证据，不能提前支持 Reset 或 Clip 的因果结论。

## 3. E0：正式运行前的实现与正确性检查

### 3.1 统一环境时钟

- 每个任务恰好记录 1,500,000 个训练环境交互步，warm-up 包含在内。
- `global_env_step`、`task_env_step`、`global_critic_updates` 和 `task_critic_updates` 分开记录。
- 评估交互、离线数据收集和蒸馏更新不计入训练环境步。
- Clip 的周期事件按显式环境步触发，不用 critic-update 近似代替。

### 3.2 评估与随机性隔离

- Meta-World 的训练实例、评估实例、reset seeds 和 goal 信息落盘并哈希。
- 评估动作直接使用确定性均值，不先采样再取均值。
- 评估使用独立环境和 RNG；评估前后 Python、NumPy、Torch 和 CUDA 训练 RNG 完全一致。
- 当前任务评估和已见任务边界评估都携带 task position、occurrence ID、policy head 和 learner role。

### 3.3 方法边界行为

- FT：保留 actor、双 Q 和对应 Adam；清空 replay；alpha 重置；target Q 与 online Q 同步。
- Reset：重置双 online Q、双 target Q 和 critic Adam 状态；actor 保留。
- Clip：从第二个任务开始，在每个任务入口及任务内每 200k 环境步裁剪双 online critic 的全部 Linear 权重到奇异值区间 [0.25, 4]；保留 actor、bias 和 Adam moments；裁剪后同步 target Q。
- 同一时刻的入口事件和周期事件只执行一次。
- FAME、Spectral regularization、R&D 分别记录 fast/meta、正则模块、teacher/student 的模型身份和方法专有状态。

### 3.4 Checkpoint 与恢复

- 任务入口和出口保存完整可恢复 checkpoint，包括模型、全部 optimizers、alpha、RNG、sampler/env、replay 和方法专有状态。
- 任务内 0、10k、50k、100k、500k、1M、1.5M 保存轻量分析快照；0 点区分干预前后。
- 从完整 checkpoint 恢复后，下一次环境转移、采样顺序、优化更新和记录时钟必须与不中断运行一致。
- carried-Adam 离线拟合所需的 moments、step、lr、参数映射和完整 optimizer 配置必须可用。

### 3.5 必过 smoke checks

- 五条主序列与 6×6 任务池名称、顺序、重复位置和 head 分配正确。
- 同 seed 的同源分支具有相同 A checkpoint、B warm-up transitions、实例 bank 和初始化 hash。
- 打开记录功能不改变相同 minibatch 下的 loss、参数更新和训练 RNG。
- Reset 和 Clip 的 before/after 权重、Q 输出、target 同步和 optimizer 处理与定义一致。
- DMC walker 任务注册、DMC 接口、重复访问任务和所有七种方法均能完成短程运行。
- 所有 JSONL 可逐行解析；缺失值使用 null/不适用，不用 0 冒充。

## 4. E1–E2：Rethink 迁移矩阵

### 4.1 固定任务池

| ID | 源任务 A | 目标任务 B |
|---|---|---|
| P1 | sweep-into-v2 | push-wall-v2 |
| P2 | push-wall-v2 | sweep-into-v2 |
| P3 | button-press-v2 | button-press-wall-v2 |
| P4 | button-press-wall-v2 | button-press-v2 |
| P5 | reach-v2 | window-close-v2 |
| P6 | window-close-v2 | reach-v2 |

完整任务池为 `sweep-into-v2`、`push-wall-v2`、`window-close-v2`、`button-press-v2`、`button-press-wall-v2`、`reach-v2`。E2 运行全部 36 个 A→B 方向，包括 30 个跨任务格和 6 个自迁移格。

### 4.2 在线运行构成

- 每个源任务独立预训练 seeds 1/2/3，共 18 个 A runs。
- 每个 A checkpoint 分支到 6 个 B，每个 B 运行 FT、Reset、Clip，共 36×3×3=324 个 B runs。
- 三个方法分支共享完整 A checkpoint 和共同 B warm-up transitions。
- A 阶段不执行 reset 或 clip，源预训练在方法间复用。
- Fresh 参照复用六个源任务从初始化学习 1.5M 的曲线；它不是第四个在线方法。
- E2 的计划训练预算为 513M 环境交互步；P1–P6 已包含在矩阵中，不重复训练。

### 4.3 Fig. 1 与迁移指标

完整矩阵分别报告：

- FT − fresh：识别来源依赖的正/负迁移；
- Reset − FT：识别 critic 重置对迁移的改变；
- Clip − FT：识别谱调理对迁移的改变。

每个矩阵格先在相同 direction、seed 和数据协议下形成配对差，再跨三个 seeds 汇总。不能把 36 个方向当作 36 个训练 seeds。

## 5. E3：理论与真实 critic 机制实验

### 5.1 64 状态环形 MRP

- 运行 $\gamma\in\{0,0.9,0.99\}$，随机构造使用 seeds 1、2、3。
- 固定学习核 $K$ 的特征值集合，改变需求方向与 Bellman 传播模态。
- 分别构造匹配即时 TD 幅度和匹配完整价值修正幅度的条件。
- 直接计算理论有限步误差、慢模态质量和预算界，并与数值迭代逐项对齐。
- 同时给出有利与不利于 Clip 的条件，不能只保留支持方法的构造。

该实验是精确矩阵计算，不使用每任务 1.5M 的 RL 预算。输出对应 Fig. 2a–b 和理论附录。

### 5.2 同 correction 离线拟合

- 对 P1–P6 的 parent、FT、Reset、Clip 副本使用相同输入、相同初始 correction、更新预算和批次顺序。
- 保存并报告 $H\in\{0,1,10,50,200,1000\}$ 的绝对 MSE 与归一化误差。
- 先用统一 full-batch GD 校准理论，再运行 fresh Adam 与 carried Adam 条件。
- 保存未中心化 $K=JJ^\top/n$、特征值/特征向量、实际损失缩放和模型/输入哈希。

### 5.3 同真实 target 与动态 target 流

- P1–P6 在 B=10k、100k、500k、1M 窗口复用 FT 参考轨迹。
- 每个窗口保存 1000 次更新的共同 targets、固定 next-action 随机量、target Q、actor、alpha 来源和时间戳。
- 共同 target 流与每个分支自己生成的在线 target 流分别保存、分别命名。
- 计算 $J\Delta\theta$、实际 $\Delta Q$、target drift、optimizer/采样偏差和非线性余项；不能把所有误差统称为非线性。

### 5.4 干预到后续表现的配对链

对 P1–P6 的每个 direction/seed，将 B 入口的干预前后状态与同一分支的后续 B 曲线配对，汇总：

- 即时 Q-jump RMS 与 max jump；
- 谱与 kernel 的变化；
- 同 correction 的 200 步归一化拟合误差；
- 同真实 target 的 200 步 MSE；
- B success-AUC 与最终 success。

这些结果进入 Table 3 和 Fig. 4。相关性只在已测试条件内解释，不声称排除了全部探索、范数或 optimizer 替代解释。

## 6. E4：五条主序列

### 6.1 任务序列

| ID | 任务顺序 | 每个 seed 的训练预算 |
|---|---|---:|
| F1 | button-press → plate-slide-back-side → window-close → plate-slide-side → peg-unplug-side → plate-slide-back → coffee-button → window-open → handle-pull-side → door-close | 15M |
| F2 | plate-slide-back-side → soccer → sweep-into → handle-pull-side → plate-slide-side → peg-unplug-side → door-lock → reach → plate-slide-back → coffee-button | 15M |
| F3 | coffee-push → button-press → reach → peg-unplug-side → reach-wall → door-close → window-open → handle-pull-side → plate-slide-back-side → soccer | 15M |
| D-W6 | walker-stand → walker-walk → walker-run → walker-stand → walker-walk → walker-run | 9M |
| D-C4 | cartpole-balance → cartpole-swingup → cartpole-balance → cartpole-swingup | 6M |

Meta-World 名称在配置中统一使用 `-v2` 后缀。

### 6.2 七种方法与开跑条件

| 方法 | 正式定义 | 开跑前缺口 |
|---|---|---|
| FT | 普通 SAC 连续微调 | 统一主序列入口与记录协议 |
| Reset | critic-only Q reset，并重置 critic Adam | 补齐 optimizer reset 并验证边界 |
| EWC | 原版 EWC，不叠加 Clip | 固定 Fisher、正则参数范围和系数记录 |
| FAME | 单一 FAME-KL | 接入 SAC/Meta-World/DMC 统一协议 |
| Spectral regularization | 独立的 SAC 谱正则 baseline | 实现正式正则公式与系数，不能以 hard clip 代替 |
| R&D | reset-and-distill teacher/student 管线 | 校验数据、teacher/student 角色和 DMC 接口 |
| Clip（ours） | 第二任务起的 critic 双侧谱裁剪 | 补 DMC、分支、环境时钟、每个后续任务入口和统一诊断 |

所有方法在 F1、F2、F3、D-W6、D-C4 上运行 seeds 1、2、3，共 7×5×3=105 个序列配置。按任务数计的名义训练预算为 1.26B 环境交互步，其中 R&D 的 teacher 预算已包含在内；student 的离线蒸馏不伪装成额外在线学习曲线。

### 6.3 主结果指标

Meta-World：

- 每任务 success-AUC 为 $1.5\mathrm{M}^{-1}\int_0^{1.5\mathrm{M}}\mathrm{success}(t)\,dt$；
- 每条序列的位置 2–10 等权平均，第一任务单独报告，不计入迁移获取指标；
- Table 1 每格报告获取 success-AUC / 最终已见任务平均 success，均为 mean ± sample SD。

DMC：

- D-W6 首次获取使用位置 2–3，重访使用位置 4–6；
- D-C4 首次获取使用位置 2，重访使用位置 3–4；
- Table 2 每格报告 first-pass return-AUC / revisit return-AUC / 最终已见任务平均 return；
- 同名任务的不同 occurrence/head 不合并，DMC 与 Meta-World 不合成无量纲总分。

R&D 的获取曲线来自 teacher、保留结果来自部署 student；FAME 的获取曲线来自 fast learner、保留结果来自 meta learner。图例、manifest 和表注必须显式记录角色，不能拼成一个虚构 agent。

## 7. 每个 run 必须保存的原始记录

| 文件 | 最低内容 |
|---|---|
| `run_manifest.json` | method、seed、任务顺序/位置、代码与依赖版本、完整参数、初始化 hash、实例/评估 seeds、预算、角色、状态 |
| `eval_episodes.jsonl` | 全部时钟、训练/评估任务位置、occurrence/head/role、checkpoint、episode/instance/reset seed、return、success、length |
| `train_episodes.jsonl` | 训练 episode 的任务、时钟、return、success、length |
| `train_metrics.jsonl` | loss、alpha、reward/Q/target/TD 分布、梯度与更新范数、buffer size |
| `boundary_events.jsonl` | 边界前后任务、模型/optimizer/alpha/replay/target 的实际处理、父 checkpoint |
| `intervention_events.jsonl` | Clip/Reset 的触发时钟、原因、逐层奇异值、参数位移、Q-jump、target 同步和 optimizer 处理 |
| `checkpoint_events.jsonl` | checkpoint ID、类型、任务位置、时钟、父状态、完整性 |
| `resource_metrics.jsonl` | 训练/评估交互数、各类更新数、墙钟、峰值显存、参数与缓存大小 |

Anchors、target windows 和 checkpoints 按 `PAPER_OUTLINE_20260907.md` 第 6 节的 schema 保存。重型 Jacobian、谱分解和拟合在线关闭，训练后从固定输入、targets、模型和 optimizer 状态离线计算。

## 8. 总预算与去重规则

| 部分 | 计划预算 |
|---|---:|
| 五条主序列 × 七方法 × 三 seeds | 1.26B 训练环境交互 |
| 6×6 rethink：18 个源预训练 + 324 个目标分支 | 513M 训练环境交互 |
| 合计名义配额 | 1.773B 训练环境交互 |

以下内容不得重复计费或重复运行：

- P1–P6 已属于完整 6×6 矩阵；
- fresh 曲线复用六个源任务的独立训练；
- 相同配置的 R&D teacher 可以缓存复用，但账本同时列名义成本和实际成本；
- 评估交互、buffer 收集、蒸馏、meta updates、probe updates 和离线分析不计入训练交互，但分别报告实际成本。

## 9. 明确不运行的旧方案

本轮不运行 H8、E8、CW20、ABC、RPP，不运行 actor-only、norm-only、随机扰动、from-A clip、阈值扫描、BRO、ReDo、Muon 或额外无边界在线组，也不追加大规模 baseline/消融矩阵。

因此本轮不能据实验声称：critic 是唯一原因、第二任务起裁剪优于全程裁剪、当前阈值最优，或已经排除全部探索、范数与 optimizer 解释。

## 10. E5：完成与发布检查

一个正式 run 只有同时满足以下条件才标记为 completed：

- 训练环境步、任务位置和更新计数满足固定预算；
- 所有预定评估点和 50 episodes 均存在；
- manifest、事件文件、必要 checkpoint 和资源记录完整；
- 没有 NaN/Inf、未知退出码、被覆盖路径或方法角色混淆；
- 可从原始记录重算 AUC、最终性能、配对差和图表输入；
- 三 seeds 缺一时不填 0、不用其他 seed 替代，结果保持 incomplete。

发布到 GitHub 时只提交实验源码、测试、配置和本仓库的五份 Markdown。日志、模型、checkpoint、replay、anchors、JSONL、图、表格缓存及其他实验结果只保留在本地或独立制品存储中。
