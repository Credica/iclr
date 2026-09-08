# 投稿论文骨架与记录协议：Bellman Demand and Learning Spectra

2026-09-07 修订。以本版为唯一有效实验范围：五条主序列、每任务 1.5M 环境步、seeds={1,2,3}。旧版 H8/E8/CW20、大规模 baseline 与消融矩阵不再是待执行清单。本次修改文档，不修改训练代码、不启动实验。

执行更新（2026-09-07 晚）：按后续指令，第一批仅启动 P1–P6 的 FT、seeds 1/2/3。独立入口、已实现的环境时钟、多实例评估、原始记录协议及后续全部实验见 `EXPERIMENT_PLAN_20260907.md`；本稿第 7 节保留旧通用入口的审计，不能再据此断言本批 FT 入口也缺少这些功能。其余方法与完整矩阵并未随本批启动。

## 0. 本轮固定决策

| 项目 | 当前决定 |
|---|---|
| 主实验 | F1、F2、F3、D-W6、D-C4 |
| 主实验方法 | FT、Reset、EWC、P&C、Spectral regularization、ReDo、R&D、Clip（ours），共 8 种 |
| Rethink | 原六任务池的完整 6×6 矩阵、六个深入分析方向、64 状态 MRP |
| Rethink 方法 | 仅 FT、Reset、Clip（ours）；不追加其他在线方法或大规模消融 |
| 训练预算 | 所有 RL 任务均为 1,500,000 实际环境交互步，包含 warm-up；源任务 A 和目标任务 B 各 1.5M |
| 随机种子 | 所有正式 RL 运行仅 1、2、3；不再安排其他开发或补充训练 seeds |
| 评估 | 每 10k 环境步评估 50 episodes；每任务结束评估所有已见任务各 50 episodes；不评估未来任务 |
| 统计 | 3 seeds 均值 ± 样本标准差，提供全部 seed 曲线与配对差；不把 episodes/tasks/checkpoints 当额外独立 seeds |

PT-Clip 原先只是 post-transfer clip 的工作缩写，意思是“从第二个任务开始 clip”，不是另外一种算法。现在正文和图表统一写 **Clip（ours，第二任务起）**。

## 1. 主实验：精确任务与 baseline

### 1.1 五条任务序列

Meta-World 名称统一补 `-v2`。F1–F3 仍沿用此前从 FAME 引用的三条公开顺序，但 FAME 不再是本研究 baseline；本研究改为 1.5M/task、3 seeds，属于重新运行的统一协议，不称该论文的严格复现。[任务顺序来源](https://arxiv.org/html/2603.00903v1#A7.SS1)。

| ID | 完整顺序 | 单条流训练预算 |
|---|---|---:|
| F1 | button-press → plate-slide-back-side → window-close → plate-slide-side → peg-unplug-side → plate-slide-back → coffee-button → window-open → handle-pull-side → door-close | 10 × 1.5M = 15M |
| F2 | plate-slide-back-side → soccer → sweep-into → handle-pull-side → plate-slide-side → peg-unplug-side → door-lock → reach → plate-slide-back → coffee-button | 15M |
| F3 | coffee-push → button-press → reach → peg-unplug-side → reach-wall → door-close → window-open → handle-pull-side → plate-slide-back-side → soccer | 15M |
| D-W6 | walker-stand → walker-walk → walker-run → walker-stand → walker-walk → walker-run | 6 × 1.5M = 9M |
| D-C4 | cartpole-balance → cartpole-swingup → cartpole-balance → cartpole-swingup | 4 × 1.5M = 6M |

D-W6/D-C4 是本研究固定的 DMC 持续任务顺序。H8/E8/CW20/ABC/RPP 不进入本轮主实验或附录新训练清单。

统一主干：Meta-World 使用 2×256 ReLU SAC，DMC 使用 2×1024 SAC；actor 按任务位置分配 head，critic 共享。DMC 第二轮分配新的 occurrence head，不复用第一轮 head，因此重访指标解释为“共享 backbone 下同任务重新学习”，不是旧完整策略直接恢复；所有方法采用相同任务/head 语义并记录其适配。Meta-World 每个 task/seed 固定 50 个训练实例与独立的 50 个评估实例，各方法共享列表；训练按 episode 采样，评估固定实例与 reset seeds。本轮不强制 200-episode 终点评估；多实例 sampler 尚待实现验证。DMC 对每个任务固定 50 个评估 reset seeds。

### 1.2 八种方法的定义

| 图表名称 | 本稿采用的定义 | 当前代码状态 |
|---|---|---|
| FT | 普通 SAC 连续微调 | 已有 `--cl_method finetuning` |
| Reset | 沿用前文讨论的 critic-only Q-reset，actor 保留。正式方案重置双 Q、target Q 和对应 critic Adam 状态 | 现有 `--q_reset True` 已重置 critic Adam 并同步 target；旧 weights-only 日志单独标注 |
| EWC | 原版 EWC，不加 clip；当前实现正则 actor | 已有 `--cl_method ewc`；需记录 Fisher、正则范围和系数 |
| P&C | Progress & Compress：active column 学习当前任务，再蒸馏到 knowledge base；compression 用 EWC 保护旧知识，不加 Clip | 正式配置固定 `use_pandc_bc=False, reset_column=True, reset_adaptor=True`；异构 DMC-style spec SAC 更新 smoke 已通过 |
| Spectral regularization | ICLR 2025 的 k=2 layer spectral regularizer；actor 与双 online critic 系数均为 1e-4；多 head actor 只作用共享层与当前 mean/log-std heads，不改未激活 heads；不以 hard clip 冒充 | 已实现 `--cl_method spectral`，使用不消耗训练 RNG 的单步 power iteration；需完成统一协议 smoke test |
| ReDo | Recycling Dormant Neurons：每 1k task-local 环境步按归一化平均绝对激活和固定 tau=0.1 回收；重采样 incoming、清零 outgoing；覆盖 actor 与双 critic | 已实现受影响 Adam moments 清理、双 target Q 同步与事件记录；固定配置，不做阈值/频率扫描 |
| R&D | 完整 reset-and-distill，包括本任务 teacher 与部署 student | `--run` 先完成本机非 R&D，第二阶段自动生成/复用同配置、同 seed、1.5M teacher model+rollout 后蒸馏，不强制要求新完成标记；student 使用显式 artifact root；DMC 异构输入/head 映射已有单元测试 |
| Clip（ours） | 第二任务起的 critic 双侧谱裁剪；具体定义见 §4.1 | MW/DMC 完整主序列、环境时钟、后续各入口和现有 probes 已接入；E2 分支及完整数据契约仍待补 |

P&C 主实验固定使用论文的 EWC compression 路径，不使用仓库可选的 BC compression 变体。每次完成 compression 后重置 active column 及 adaptor；knowledge base、Fisher、compression optimizer 和已见任务计数都属于必须保存的方法状态。[P&C 原论文](https://proceedings.mlr.press/v80/schwarz18a.html)。

R&D 保留原方法的单任务专家训练、训练结束后额外采集专家 rollout、结合历史任务记忆进行顺序蒸馏的流程。五条任务流、1.5M 在线预算和训练/评估实例划分采用本文统一设置，不声称逐项复现原论文实验配置。蒸馏轨迹来自训练 bank，不使用独立评估 bank，也不改用训练 replay buffer。teacher 缓存可复用；队列只按任务/预算/seed 文件名检查 model+rollout 是否齐全，不要求新 completion receipt。完整配置一致性须在导入缓存前另行核对，不能仅凭文件名认定兼容。

Reset 在这里不是 actor+critic 全重置，也不是 R&D。权重+Adam 的正式定义沿用上一版计划；本轮不新增一个 weights-only 训练组，旧 weights-only 结果仅作为历史结果标注。

所有方法在五条流上运行 seeds 1/2/3，即 120 个主序列配置。每方法每任务 1.5M 的主实验名义配额共 1.44B 环境交互，其中包含 R&D 对应的教师训练配额，不再重复加算一份教师预算。R&D 的学生是离线蒸馏，不虚构额外的 student 在线 1.5M 曲线；教师也只训练 1.5M，不读取旧 3M 教师充当同预算结果。配置严格相同的单任务教师可以缓存复用，但账本同时列名义与实际成本。P&C 的 compression 不计作环境交互，但必须单独记录其梯度更新数、样本读取量和墙钟；ReDo 的神经元统计与 recycling 事件单独记账；额外 buffer 收集、评估、distillation/compression/probe 更新分别记账，不声称不同方法计算量相同。

双机 baseline 自动执行顺序：机器 1 为 44 个非 R&D runs → 48 个 Meta-World teacher 前置项 → 9 个 R&D runs；机器 2 为 46 个非 R&D runs → 15 个 DMC teacher 前置项 → 6 个 R&D runs。每卡最多两个进程，本机当前队列全部成功才进入下一队列，两台互不等待；teacher 跨机无重复。具体队列和依赖见实验计划 §6.2.2。

### 1.3 主表直接填数

Meta-World 获取指标：每任务 success-AUC = 1.5M⁻¹∫ success(t)dt，先对每条流位置 2–10 等权平均，再跨 3 seeds 汇总；第一任务结果单列在曲线，不计迁移获取指标。主表使用绝对 AUC，不为主实验另外加一组全任务 fresh 训练。

**Table 1. Acquisition and retention on Meta-World.** 每格为获取 success-AUC / 最终已见任务平均 success，分别报告均值±SD。

获取使用对应流程的当前在线 learner：R&D 为 teacher，P&C 为 active column；最终保留使用部署 student，P&C 对旧任务使用 knowledge base、对当前任务使用 active column。模型身份直接标在表注与图例，不能把不同网络拼成一个虚构 agent。

| Method | F1 | F2 | F3 |
|---|---|---|---|
| FT | -- | -- | -- |
| Reset | -- | -- | -- |
| EWC | -- | -- | -- |
| P&C | -- | -- | -- |
| Spectral regularization | -- | -- | -- |
| ReDo | -- | -- | -- |
| R&D | -- | -- | -- |
| Clip（ours） | -- | -- | -- |

**Table 2. Acquisition, revisiting, and retention on DMC.** 每格给 first-pass return-AUC / revisit return-AUC / 最终已见任务平均 return，分别报告均值±SD。

| Method | D-W6 | D-C4 |
|---|---|---|
| FT | -- | -- |
| Reset | -- | -- |
| EWC | -- | -- |
| P&C | -- | -- |
| Spectral regularization | -- | -- |
| ReDo | -- | -- |
| R&D | -- | -- |
| Clip（ours） | -- | -- |

D-W6 首次获取用位置 2–3，重访用 4–6；D-C4 首次获取用位置 2，重访用 3–4。最终保留按任务位置/head 记录，附录再分首轮/第二轮；不要把同名任务不同访问合并。DMC 与 Meta-World 不合成一个无量纲总分。

## 2. Rethink：任务不变，预算与方法统一缩减

### 2.1 迁移矩阵与六个深入分析方向

机制池：`sweep-into-v2`、`push-wall-v2`、`window-close-v2`、`button-press-v2`、`button-press-wall-v2`、`reach-v2`。

完整 6×6 矩阵：行=源任务、列=目标任务；30 个跨任务格与 6 个自迁移格全部保留。A=1.5M、B=1.5M，seeds=1/2/3，只比较 FT、Reset、Clip。三组从同一个完整 A checkpoint 分支；A 阶段无 reset/clip，源预训练可共享，不重复三遍。

深入分析方向固定为：P1 sweep-into→push-wall；P2 push-wall→sweep-into；P3 button-press→button-press-wall；P4 button-press-wall→button-press；P5 reach→window-close；P6 window-close→reach。

Fresh 参照直接复用六个源任务“从初始化学习 1.5M”的曲线；匹配网络/head 初始化、环境实例、warm-up 和预算后，才可作为相应 B 的 fresh。它不是第四种算法，也不新增独立 RL 训练。不能只凭相同 seed 假定初始化相同。

矩阵预算为 18 个源预训练 ×1.5M + 324 个目标分支 ×1.5M = 513M。六个深入方向属于该矩阵，不再重复运行。主实验和矩阵的名义配额合计 1.953B，已含 R&D 名义教师训练份额，但不含额外数据收集、评估与离线计算；缓存复用后的实际成本另报，时间按实际吞吐估计。

### 2.2 四组机制实验

| ID | 实验内容 | 方法/额外训练约束 | 正文证据 |
|---|---|---|---|
| R1 | 完整 6×6 的 B 获取；FT/Reset/Clip 相对 fresh 的表现及配对差 | 仅已列三组；不再安排 actor 2×2 或额外无边界训练组 | Fig. 1：来源依赖与 critic 干预响应 |
| R2 | 64 状态环形 MRP，γ=0/.9/.99；固定 K 的特征值集合，改变需求和传播模态；分别匹配即时 TD 与完整价值修正幅度 | 精确矩阵计算，不是每任务 1.5M 的 RL 训练；随机构造使用 seeds 1/2/3 | Fig. 2a–b：理论与数值一致性 |
| R3 | P1–P6 同输入/同 correction 与同真实 target 的离线拟合；重放共同 target 流；其他矩阵格做同定义诊断 | 只用 FT/Reset/Clip checkpoint 的分析副本；不增加在线算法 | Fig. 2c–d：真实 critic 的机制对应 |
| R4 | P1–P6 在 B 入口的干预前后 Q/谱/拟合变化，与之后 B 的获取改善对应 | B 入口分支已在 R1 中；B=100k 等中途点只做离线分析，不新增整轮在线续跑 | Fig. 4、Table 3：机制与实际收益对应 |

R3 的冻结-target探针共用输入、target 或初始 correction、更新预算和批次顺序；统一 GD 用于理论校准，fresh/carried Adam 用于优化器敏感性分析。这些是离线测量条件，不是新的在线 baseline。

不再执行上一版 actor-only、norm-only、随机扰动、from-A clip、阈值扫描、BRO、Muon 等额外在线组。ReDo 仅作为固定配置 baseline，不新增阈值或频率扫描。由此相应收窄结论：不能独立声称 critic 是唯一原因、第二任务起优于全程 clip、默认阈值最优，或已经排除全部探索/范数/optimizer 替代解释。

**Table 3. Paired mechanism results on the six fixed transfer directions.** 各单元分别汇总三 seeds，不把六个方向当成六个独立训练 seeds；记录缺失不得填 0。

| Method | 干预即时 Q-jump RMS | 同 correction 的 200 步归一化拟合误差 | 同真实 target 的 200 步 MSE | B success-AUC | B 最终 success |
|---|---|---|---|---|---|
| FT | -- | -- | -- | -- | -- |
| Reset | -- | -- | -- | -- | -- |
| Clip（ours） | -- | -- | -- | -- | -- |

## 3. 论文叙事与场景 novelty

标题：**Rethinking Negative Transfer in Continual Reinforcement Learning: Bellman Demand and Learning Spectra**。

中心问题：继承 critic 的知识为什么有时有益、有时阻碍下一任务？我们的分析对象是任务切换后的有限预算 Bellman 适应，而不只是网络是否“整体健康”。

### 3.1 三个明确的贡献落点

| 贡献 | 本场景中的具体增量 | 对应证据 |
|---|---|---|
| Bellman 条件下的负迁移刻画 | 区分初始 TD、传播后的完整价值修正与有限预算学习速率；分析同一继承 critic 对不同目标任务的差异 | 命题 1、MRP、迁移矩阵、共同 target 流 |
| 价值复用与重新适应的联合分析 | 将继承 Q 的初始化作用与 K 的学习几何作用分开，再分析谱干预的函数改变和后续适应效果 | 命题 2、同 correction/同 target 双探针 |
| 面向任务切换的 critic 谱调理 | 用已知双侧谱投影构成不整体重置 critic 的场景化干预；检验其能否缓解前两项识别的瓶颈 | Clip（ours）、五条长序列与同源配对结果 |

因此可以把方法描述为“**面向 Bellman-demand 适应的 critic 谱调理策略**”；图表仍简称 Clip（ours），不再增加不透明缩写。方法不在线读取未来需求，不称 demand-adaptive clipping，不声称精确保留原 Q 函数。

拟用于论文的主张骨架：
“本文从 Bellman 修正需求与继承 critic 学习几何的联合关系重新分析持续强化学习中的负向迁移，区分价值初始化的复用收益与后续适应的动力学限制，并据此研究任务切换后的 critic 谱调理。我们的分析同时刻画谱干预带来的适应变化与价值函数扰动，实验则检验这两类变化如何对应后续任务获取。”
效果词和强弱结论在新结果完成后补入。

### 3.2 与已有 SingularClip 的关系必须具体

SingularClip 已讨论弱奇异方向上的新需求、任务周期的裁剪，以及持续监督学习和 SAC/BRO 的 RL 实验；其已报告实验并不是本文这套多任务切换控制协议，但“换到 CRL 场景”或“边界 clip”本身不足以构成强创新。本文的增量要落在上面的 Bellman 特定刻画、Q/K 分离和干预—适应证据，不把已有 SVD clamp 算子当新发明。[SingularClip §3–6](https://arxiv.org/html/2608.18319v1)。

这不是把方法降格为无贡献 baseline：算法组件可以已有，本文贡献是具体问题上的分析、策略组织和证据。但仅有同一算子在五条序列上更高的分数，不足以自动证明上述机制 novelty；也不据这一篇近邻核验宣称全领域“首次”。

## 4. 两条主理论与方法定义

### 4.1 Clip（ours，第二任务起）

正式调度协议：A 正常 SAC；B 及以后每个任务入口、任务内每 200k 环境步，对双 online critic 全部 Linear 权重（含输出头）执行：
W=U diag(σ)Vᵀ → U diag(clamp(σ,0.25,4))Vᵀ。
保留 actor、bias、Adam moments；入口裁剪后同步 target，任务内普通 Polyak。任务入口与周期事件重合时只投影一次。

2026-09-08 的主序列入口已改为该环境时钟及所有后续任务入口；每个 1.5M-step 任务（含 warm-up）在 0、200k、400k、600k、800k、1M、1.2M、1.4M 共八次裁剪，A 除外。DMC 的不同 UTD 不改变触发时刻。旧实现仅 B 入口额外 clip、周期按 critic updates，旧结果不能直接改标签。五条完整序列的独立 Clip 队列见 `scripts/run_clip_main.sh`，不是 E1/E2 双任务测试；完整论文记录契约仍须按 §7 验收。

### 4.2 Motivation：Bellman 条件下的有限预算适应

固定策略 MRP，A=I−γP、q*=A⁻¹r、e=q*−q、d=Ae；固定学习核 K 的 semi-gradient 模型满足：

$$e_H=(I-\eta KA)^H e_0.$$

若 P 对称且与 K 共享正交特征基 u_i，令 a_i=1−γp_i，则命题 1 为：

$$\mathcal R_H=\frac{\sum_i |u_i^\top d_0|^2a_i^{-2}(1-\eta\kappa_i a_i)^{2H}}{\sum_i |u_i^\top d_0|^2a_i^{-2}}.$$

初始误差非零；收敛解释要求受激模态 0<ηκ_i a_i<2，单调速率比较用 0≤ηκ_i a_i≤1。该表达把“需要改多少、改什么方向”和“这些方向学多快”放在一起。慢模态预算界、非交换情形与证明放理论附录。标准谱展开本身不作为原创点，原创论证来自具体问题刻画和可区分的预测。

### 4.3 Method：干预收益与函数改变

记 e=q*−q_0，clip 造成 j_c=q_c−q_0，T_0=(I−ηK_0A)^H、T_c=(I−ηK_cA)^H。在每个副本冻结自身核的模型下，命题 2 为：

$$E_0-E_c=\underbrace{\|T_0e\|^2-\|T_ce\|^2}_{G_{\rm geom}}+2\langle T_ce,T_cj_c\rangle-\|T_cj_c\|^2.$$

||T_0e||>||T_ce||+||T_cj_c|| 是改善的充分条件，不是必要条件。函数改变与目标修正同向时也可能有益。以两层线性 critic 的 K=||v||²I+WᵀW 连接权重 clip 与学习核；完整假设和推导见理论底稿。

真实 SAC 不能直接测未知全局 Q*；因此 MRP 检验精确理论，真实网络用共同 targets、局部预测误差和实际拟合检验机制。记录 Adam/小批次偏差和非线性余项，不把它们省略后声称是 SAC 全局收敛定理。

## 5. 正文章节与图表：按此填稿

| 正文章节 | 要写清的内容 | 固定图表/结果填空 |
|---|---|---|
| 1 Introduction | 负迁移问题；Bellman-demand 视角；Q/K 分离；简单干预；三项贡献 | 填一个矩阵现象、一个机制发现、一个主实验结果 |
| 2 Related Work and Preliminaries | CRL 与 R&D/P&C/EWC；谱正则与 SingularClip；SAC/Bellman/K 定义 | 明确组件来源与本文场景增量 |
| 3.1 Source-dependent transfer | 完整 6×6、三方法配对；critic 干预可改变哪些现象 | Fig. 1：FT−fresh、Reset−FT、Clip−FT 三张矩阵；填变化与例外 |
| 3.2 Bellman-demand theory | 命题 1、慢方向需求与传播；精确 MRP | Fig. 2a–b：理论/数值及同谱不同需求结果 |
| 3.3 Real-critic evidence | 六方向同 correction/同 target、共同动态 target 流 | Fig. 2c–d：拟合与残差滞留；填相对 TD/rank/单步诊断的增量 |
| 4 Method and intervention theory | Algorithm 1：Clip（ours）；命题 2；两层机制例子 | 填实现说明，不填未经验证的普遍改善结论 |
| 5.1 Setup | 五条序列、八方法、1.5M、三 seeds、50 eval、资源与统计 | 固定协议；说明与引用论文原协议的差别 |
| 5.2 Main results | F1–F3 获取与保留，D-W/D-C 首次/重访与保留 | Tables 1–2；Fig. 3 五个序列的主学习曲线 |
| 5.3 Mechanism-to-performance link | R4：同源分支的几何/函数/拟合变化及后续 B 获取 | Table 3；Fig. 4：Q-jump、共同需求拟合、后续 AUC 的配对关系 |
| 6 Discussion and Limitations | 小样本统计、局部理论、缺少额外因子对照、方法不利条件与成本 | 填真实边界，不把相关性写成排除所有替代解释 |
| 7 Conclusion | 回答重新理解了什么、干预有效到什么范围 | 一段；不新增主张 |

Fig. 3 必须覆盖全部五条流，不挑最好看的序列；获取曲线图例明确 R&D teacher / P&C active column，student/knowledge base 的部署表现另列，DMC Table 2 采用相同角色说明。完整八方法曲线提供附录。摘要最后按问题—分析—验证—实测效果—边界五句填写。

附录仅包括：A 协议与运行资源；B 完整证明与 MRP；C 五条流全部 seeds/逐任务曲线；D 完整 6×6 与六方向离线诊断；E 数据字典、实现审计和复现检查。上一版额外 baseline/超参数扫描不再保留为“必跑附录”。

## 6. 数据记录：哪些现在能开，哪些需要补

### 6.1 现有 CLI，不能把名称当成已经实现的统一管线

代码依据：`args.py:67–163`、`garage/torch/algos/sac.py:952–1321`、`garage/torch/algos/sac_singular_clip.py:85–121`。

| 当前真实开关 | 本轮建议 | 已有作用与限制 |
|---|---|---|
| `--num_evaluation_episodes 50` | 主实验/rethink 都开 50 | 每个被评估任务 50 episodes；不是全序列总共 50 |
| `--num_evaluation_steps 10000` | 主 baseline 每 10k 环境步评估 | Hessian 在该采样点最后一次优化后计算，随后以同一参数状态评估 |
| `--no_stats False` | 主 baseline 与单任务 teachers 打开 | zero ratio 每 1k；feature rank/weight change 和 Hessian rank 分别每 10k；诊断后恢复训练 RNG |
| `--wandb True`（默认） | 本地原始文件仍为权威记录；W&B 用于在线监控与汇总 | 密钥只存机器本地，不写源码/manifest；显式 `--wandb false` 仍可离线运行，且不影响必须落盘的数据 |
| `--bellman_probe True` | FT/Reset/EWC/Clip 使用现有 probe/checkpoint；Clip 另存裁剪事件 | Clip 已恢复 buffer 收集和 metrics probe；仍不能将这些旧指标当作 §6.2 的完整论文数据契约 |
| `--bellman_probe_size 1024` | 保存输入的旧入口 | 含 Clip 在内，收集每任务最早 1024 transitions，不是随机 replay reservoir |
| `--bellman_probe_interval 100000` | 主 baseline 固定 | 单位为 task-local 环境步，另存 global/task critic updates |
| `--bellman_probe_targets 8` | 旧指标记录参数 | 当前是确定性 sin 噪声形成的 8 组 directions，不能描述为独立 MC targets |
| `--bellman_probe_ridge 0.001` | 固定记录 | 相对 ridge；分析必须同时保留原始尺度 |
| `--bellman_probe_dir <run_root>` | 每个 run 唯一路径 | 目录再接 proc_name；不得覆盖旧结果 |
| `--bellman_spectral_stats True` | baseline 主序列打开 | 仅在 10k、50k、100k、500k、1M、1.5M 固定关键点触发 |
| `--bellman_reference_dir <dir>` | 只在已有离线/探针接口适用时使用 | 固定 transitions，不固定 targets；不能靠它完成共同 target 流实验 |

full-J 探针继续使用 `--bellman_spectral_anchor_size 64`、`--bellman_spectral_fit_lr 0.0003` 和固定 `--bellman_spectral_task_steps 10000 50000 100000 500000 1000000 1500000`；最后一项现在按 task-local 环境步解释。

**没有现成的独立“逐 episode 导出”“完整 checkpoint”“保存共同 target 流”开关。** 下节是需要补齐的记录契约，不是虚构 CLI。只打开上表还不足以得到论文要求的全部数据。

### 6.2 主实验与 rethink 都必须保存的轻量原始数据

以下文件名为拟统一输出 schema，当前尚未全部实现。

| 拟保存文件 | 时机 | 必须字段 | 支持的图表/分析 |
|---|---|---|---|
| `run_manifest.json` | 启动与结束 | method、seed、task 顺序/位置/重复次数、代码与依赖版本、完整参数、初始化 hash、环境实例/评估 seeds、预算、实际事件调度、角色 teacher/student/fast/meta、完成状态 | 排除协议混淆、复现、资源表 |
| `eval_episodes.jsonl` | 每 10k 当前任务；每任务结束全部已见任务 | global_env_step、task_env_step、global/task_critic_updates、train/eval_task_position、occurrence_id、policy_head/role、checkpoint_id、episode/instance/reset_seed、return、success_any、length、deterministic | AUC、return 与 success 分离、遗忘、首次/重访、50 episodes 分布 |
| `train_metrics.jsonl` | 每 1000 环境步汇总已有训练量 | 所有时钟、task、actor loss、Q1/Q2 TD loss、各正则 loss、alpha、reward/Q/target/TD 的均值与分位数、已有梯度和更新范数、buffer size | 学习停滞、Q/TD 尺度与优化状态 |
| `boundary_events.jsonl` | 每次任务切换 | before/after task、Q/actor/optimizer/alpha/replay/target 的实际处理、初始化/父 checkpoint ID | Reset 与 Clip 的真实差异，任务边界对齐 |
| `intervention_events.jsonl` | 每次 Clip 或 Reset 前后 | 实际 env/update 时刻、触发原因、逐层完整奇异值、参数位移、固定输入上 Q1/Q2 前后输出及 RMS/max jump、target sync、optimizer 处理 | Fig. 4、函数扰动、谱变化 |
| `resource_metrics.jsonl` | 周期/结束 | train/eval/teacher/selection 交互数、online/distillation/meta/probe 更新数、训练/评估/记录墙钟、峰值显存、参数和缓存大小 | 不同算法成本与公平比较 |

DMC 没有 success 时填 null/不适用，不填 0。Meta-World success 采用 episode 内任一时刻成功；training “Reward avg.” 与 evaluation average return 是不同字段。R&D 保存 student 的保留结果与 teacher 的获取曲线；P&C 保存 active column 与 knowledge base 各自曲线，主表的部署策略必须注明，不能拼成一个虚构 agent 的成绩。

训练标量尽量复用本来计算出的张量；不为日志额外采样 actor 动作、不额外做 Hessian。所有评估/诊断采用独立 RNG，或完整保存恢复 Python/NumPy/Torch/CUDA RNG；评估环境不得推进训练环境状态。

### 6.3 Checkpoint 与 anchors：训练时保存，重型指标离线算

**共同轻量快照时刻（task-local 环境步）：**
0（边界处理前/后分别标明）、10k（warm-up 后、首个梯度更新前）、50k、100k、500k、1M、1.5M。每次 Clip/Reset 另保存干预前后 critic 权重或足以重建二者的差分，并当场保存固定输入上的 Q-jump。相同点去重，不复制整份 replay。

快照至少包含 actor、online/target 双 Q、alpha、架构/head 信息、所有显式时钟和父 checkpoint ID。EWC 另存 Fisher/参考参数；R&D 另存 teacher/student 状态；P&C 另存 active column、knowledge base、previous knowledge base、Fisher、adaptor 和 compression optimizer 状态；SpectralReg 另存 power-iteration 向量；ReDo 另存激活统计窗口、阈值、调度时钟、recycling masks/事件和 optimizer/target 处理。task0 和 task1 指零基位置时须写清。

用于 carried-Adam 离线拟合的上述选定 probe 点，额外保存双 Q Adam moments、step、lr、参数映射及完整 optimizer 配置；只有模型权重不能重现 inherited optimizer。其他中途快照保持轻量，不保存整份 replay。

**完整可恢复 checkpoint：** 仅在任务出口/入口保存一份共享父状态，包括所有模型、全部 optimizers（含 alpha）、方法专有状态、RNG、sampler/env 状态和必要 replay；入口已清空 replay 时记录清空事实即可。任务中途的轻量快照用于离线分析，不宣称能精确在线续跑。

**Anchors：** 每任务保留 warm-up 中最多 1024 条完整 transitions，字段含 obs/action/reward/next_obs、terminated/truncated、episode/instance、实际行为来源及采样时刻；再保留后期固定预算 reservoir，用独立 RNG，不干扰训练。目标任务之间的输入不强行混成一个 bank。

Rethink 的同源三分支使用共同 warm-up transitions；从不同轨迹划分两个不重叠 panel，各 128 transitions，分支前固定索引和 hash。主实验各方法独立采集的数据不能自动当成共同 anchors；跨方法诊断必须明确指定同一个参考 bank。旧“各方法最先 1024 条”只保证各自固定，不保证彼此可比。

B 入口尚未收集新任务 anchors 时，先保存干预前后模型，再在共同 warm-up bank 上离线评价 Q-jump；该诊断时间是 B=10k，不能冒充环境 step=0 已知指标。参考数据不得进入不应访问该数据的训练或未来任务选择。

### 6.4 Rethink 额外保存：为 Bellman demand、谱和理论图服务

| 数据 | 固定协议 | 后续能计算的指标 |
|---|---|---|
| 未中心化 Jacobian kernel | 每个 panel、每个 critic 保存 K=JJᵀ/n、特征值与特征向量；保存模型/输入/损失缩放以便重算 J，不强制全量存巨型 J | raw trace/rank、条件数、需求模态能量、有限步谱预测；shape 指标另存 |
| 同 correction / 同 target | 保存各模型原始 Q、共同 target、共同 correction、各自初始 residual；合成方向由父模型定义，各副本不重新挑方向 | 区分学习几何效应与初始化/函数变化 |
| 离线拟合曲线 | H={0,1,10,50,200,1000}；统一 full-batch GD 校准，再做 fresh/carried Adam；记录实际 lr、批次顺序、绝对 MSE 和归一化误差 | 有限预算拟合、单步诊断与多步诊断差异 |
| 共同 target 时间序列 | P1–P6 在 B 的 10k/100k/500k/1M 窗口，共用 FT 参考轨迹；记录 1000 次更新的 targets、next-action 随机量、target Q/actor/alpha 来源与时间戳 | target drift、需求持续生成、静态/动态对照 |
| 实际更新与余项 | 上述窗口保存必要参数差/参考快照，计算 JΔθ、实际 ΔQ、u_t=y_{t+1}−y_t、优化器/采样偏差、非线性余项 | 不把 Adam/小批次误差误称非线性误差；检查局部模型可信度 |
| 配对干预信息 | B 入口 parent/FT/Reset/Clip 的权重、Q、K、target、optimizer 状态及后续 B 曲线 | Table 3 和 Fig. 4 的一一对应，而非跨不匹配 runs 拼图 |

共同 target 流不是每个方法自己生成自己的 targets；分支真实在线 target 流也另外记录，两种对象不要混称。参考 FT 轨迹属于已有运行，不增加一个 RL baseline。未来时刻的 target 流用于回溯机制分析，不能用于标称“B 开始即可预测”的指标。

真实网络 q* 未知时，不计算虚假的全局价值修正误差。理论精确量由 MRP 提供；SAC 的 fixed-target surrogate 和实际 online 指标各自命名。需求条件诊断与 TD 大小、rank、Rayleigh/梯度强度及已有 OR 的离线比较不增加在线算法组；无正确 OR 实现时先补分析器，不能把梯度范数改名为 OR。

## 7. 当前实现缺口与正式运行前检查

以下区分 2026-09-08 主序列修复与尚未完成的论文记录/恢复要求；局部测试不等于全部 E0 验收。

| 缺口 | 代码证据与必须的处理 |
|---|---|
| 1.5M 预算 | 共享 SAC 的 exact 路径已按实际 env 数结束 epoch，包含 warm-up，避免末任务多采样；独立训练循环和旧日志仍需分别审计 |
| Clip DMC/probe/分支 | 已开放从头训练的 DMC 和现有 Bellman/spectral probe；E2 同源 checkpoint 分支仍未开放，不将主序列队列冒充 rethink 矩阵 |
| Clip 正式调度 | 已改为 task-local env steps、B 及以后每个入口、事件去重；入口硬同步 target，周期普通 Polyak；测试在 `scripts/test_sac_singular_clip.py` |
| Reset、P&C、谱正则、ReDo、R&D 适配 | Reset Adam reset、SAC SpectralReg、周期 ReDo 和 R&D staged teacher loader 已实现并有针对性单元测试；P&C 已完成异构 DMC-style spec 的真实 SAC 更新，R&D 已完成异构输入切片与目标 head teacher 映射测试；目标机器启动时仍保留常规短程预检 |
| D-W 注册与完整配置 | DMC 名称已和 suite.ALL_TASKS 对齐；Clip manifest 包含 F1/F2/F3 的 10 个位置、D-W6 的 6 个位置、D-C4 的 4 个位置及 occurrence/head 信息 |
| 评估对象与标识 | Clip 已用 current/seen 调度和位置前缀，出口评估先于下一次入口干预；其他方法的评估统一、入口评估和逐 episode schema 仍需补齐 |
| 评估随机性 | 共享 SAC 已保存/恢复全局训练 RNG，DMC train/eval 已拆为独立环境；固定实例/reset-seed bank 和完整恢复仍待验证 |
| 原始评估与可恢复状态缺失 | 现 pkl 只有汇总，旧 checkpoints 缺 replay/RNG/env/alpha optimizer 等；需补 §6 schema 与恢复一致性检查 |
| 现 kernel 尺度不对应理论速度 | `bellman_spectral_stats.py:43–47` 用 JJᵀ/参数数目；正式分析同时保存未中心化 JJᵀ/n 与实际损失缩放，不能直接代入旧数值 |
| 固定实例并不保证 50 条不同轨迹 | `task_sampler.py:534` 固定 task instance；需固定训练/评估实例列表，记录 reset/goal，检查评估多样性。主协议沿用多实例设计，不能仅把 eval 数改成 50 就声称已实现 |

正式跑前只做正确性 smoke checks，不用短程排名筛掉 baseline：核验五条序列/重复任务、每任务预算、三方法共同 A 初始化、边界/reset/clip、评估及诊断 RNG 隔离、逐 episode 数据落盘、干预配对数据、方法专有 checkpoint 和完整恢复。然后生成唯一 run manifest 再运行。上述缺口意味着“文档已改好”不等于“当前命令已经可以直接挂整包实验”。

## 8. 写作与执行边界

本版论文仍是 7 节正文、2 个核心命题、1 个算法、4 张主图、3 张主表。结构检查将 Bellman-demand 问题对应到理论/探针，将谱干预对应到函数扰动/拟合/后续获取；旧版不再支持的因果与消融主张已同步删去。

3 seeds 可以支撑本轮比较和机制探索，但应展示逐 seed 结果与不稳定性，避免强“统计显著/普遍优势”表述。记录完整能减少因缺字段而重跑，不能保证未知 bug 或审稿意见永远不要求补实验。

证明底稿：`MOTIVATION_METHOD_THEORY_20260907.md`。唯一实验执行清单：`EXPERIMENT_PLAN_20260907.md`。旧大规模清单已归档，不再作为仓库内的有效协议；本版覆盖其训练范围、预算、seed、方法与日志协议。
