# 论文大纲：Bellman 需求—学习谱失配与负迁移

更新：2026-09-10。本文档只管论证主线、实验分工与证据边界；逐节撰写说明见 [论文写作框架](PAPER_WRITING_FRAMEWORK_20260910.md)，执行与数据契约见 [实验计划](EXPERIMENT_PLAN_20260907.md)，理论证明底稿见 [理论说明](MOTIVATION_METHOD_THEORY_20260907.md)。

## 0. 本次决策与范围

- 按用户确认，P1–P6 的 18 个 FT workers 与 5 个调度器已停止。保留全部已落盘数据，不重启、不补齐到 1.5M、不删除不支持结果。停止是研究范围调整，不是算法数值崩溃。
- 取消完整 6×6 矩阵及其 324 个 B 分支，不再作为论文完成条件。
- 新机制实验只设四个固定 A→B 方向：两个 Meta-World、两个 DMC。FT seeds 1/2/3 均已启动或排队；新四组同源 Q-reset/Clip 和 fresh-B 仍为后续验证计划，未排队。
- ABC 追加用户指定的两条续跑设计：从旧 FT 的同一个 A 结束 checkpoint 分别运行 Q-reset、Clip-8，继续 B→C。Clip-8 明确为 [0.25,8]，不是 [1/8,8]，也不是每层范数归一化。
- 五条完整主序列、八方法、每任务 1.5M 环境步及朋友的队列均不变；主序列 Clip 仍为 [0.25,4]。
- 首批执行四组 FT seed 1 与 ABC 两条续跑，共六条运行、12M 新环境步；GPU 0/1/2/3 各放 N1/N2/N3/N4，GPU 7 放 ABC 两条。运行状态以新 artifact root 的 status.json 为准。P1–P6 不重启，其他 GPU 进程不停止。

追加批次：四组FT seeds2/3共8条，加ABC FT/Q-reset seeds2/3共4条；新增12条、26M环境步，合计18条、38M。没有ABC seeds2/3旧parent，因此各自FT先训练A=1M环境步，保存验证后派生同seed的Q-reset B→C；不追加Clip seeds2/3。用户随后取消非reset的槽位等待：所有FT种子立即启动，目前GPU0/1/2/3/7的训练数为4/4/3/3/2，共16条；仅ABC Q-reset seeds2/3等待同seed A模型。已有训练保持运行。

## 1. 一条主线，三个贡献

工作标题：**Rethinking Negative Transfer in Continual Reinforcement Learning: Bellman Demand and Learning Spectra**。

中心问题：旧训练形成的 critic 学习谱，何时限制新任务需要的 Bellman 修正，使必要修正在有限预算内滞后，并影响新任务获取？谱干预何时能缓解这个瓶颈，代价是什么？

主线为：**训练历史改变学习几何 → 任务相关需求与弱学习方向失配 → 必要修正滞后 → 获取受阻；干预检验能否打通同一环节。**

“整体谱集中”不是失学的充分条件。不能把“需求落在大特征值快方向”写成失学原因。理论、现象、干预和完整序列都围绕这一个问题，不另开一条 Adam/采样/非线性主线。

| 拟贡献 | 正文位置 | 必须交付的证据 |
|---|---|---|
| C1：用任务条件化的 Bellman 需求—学习谱关系解释整体谱摘要的不足 | §3.1–3.3 | ABC 的 B/C 对照；四方向同输入、同需求的几何比较；不以一个 rank 代替机制 |
| C2：需求加权的有限预算瓶颈理论与真实修正检验 | §3.2–3.4 | 条件性预算界、精确 MRP、预先选定方向上的实际更新与残留 |
| C3：谱干预的几何收益、函数变化和在线效果之间的联系 | §4–5 | 同源 FT/Q-reset/Clip，对照共同需求拟合、Q-jump、target 反馈及后续获取/保留 |

这些是待完成的贡献目标，不是三个已成立的实验结论。SVD clamp 是已有算子；新意不能仅写成“首次裁剪”“边界 Clip”或“换到 CRL”。

## 2. 实验分工：每组只承担明确的问题

| 实验块 | 作用 | 不承担的结论 |
|---|---|---|
| ABC 旧 FT：sweep-into→push-wall→window-close，seed 1 | 开篇案例：B 受阻但 C 可学；检查整体谱与任务条件需求的区别 | 不单独证明因果、不充当三 seed 正式结果 |
| 旧三 seed fresh/FT/Q-reset，3M 历史协议 | 审计后作为受阻/恢复的辅助证据；核对任务确实可以从头学 | 不与 ABC 的 1M updates 或新协议直接合并统计 |
| 四方向 N1–N4 | §3 现象、谱与方向动力学；同批分支进入 §4 干预分析 | 不冒充随机抽样的负迁移普遍性评估 |
| 64 状态 MRP | 精确检验 Bellman 传播、需求方向和预算界 | 不声称真实 SAC 全局收敛或 success 必降 |
| ABC-Q-reset / ABC-Clip-8 | 针对已有受阻 parent 的快速机制干预；B 改善与 C 代价一起看 | 不证明阈值 8 最优；不同 parent 不混作同源 |
| P1–P6 已停止批次 | 已有实际动力学检验与适用边界；完整存档/附录 | 不再承担整篇失学因果链，也不掩盖不支持结果 |
| 五条完整主序列 | 方法的获取、保留、重访和成本 | 主表分数本身不能证明机制 |
| RPP 历史结果 | 仅在审计后能回答不同问题时作补充 | 不为增加实验数量自动加入正文或重跑 |

## 3. 四个固定机制方向及来源

此处“组”指固定任务方向，不是单个 seed，也不是方法数。仅四组，不展开所有交叉组合。

| ID | 域 | 源任务 A → 目标任务 B | 原始依据与确定性 |
|---|---|---|---|
| N1 | Meta-World | sweep-into-v2 → push-wall-v2 | R&D §3.1 / Fig. 2 的明确受阻案例；本地 ABC 也观察到 B 受阻。不能把本地 ABC 单独当 matched-fresh 负迁移对照 |
| N2 | Meta-World | window-open-v2 → sweep-into-v2 | R&D Appendix C 的 Hard 相邻顺序，加 Fig. 19(a) 的 Window→Sweep 组级负迁移；这是有依据的固定候选，不是原文逐 pair、逐 seed 必然失败的保证 |
| N3 | DMC | ball_in_cup-catch → finger-turn_easy | R&D Fig. 14(a) 中该具体 SAC 方向的 return 差为负；方差不可忽略 |
| N4 | DMC | cartpole-swingup → fish-upright | R&D Fig. 14(a) 中该具体 SAC 方向的 return 差为负；方差不可忽略 |

来源：[R&D 正式标题版本](https://arxiv.org/html/2403.05066v3#S3)、[MW 分组与 Hard 顺序](https://arxiv.org/html/2403.05066v3#A3)、[DMC Appendix I](https://arxiv.org/html/2403.05066v3#A9)。核对了 PDF 第 20 页 Fig. 14(a) 和第 23 页 Fig. 19(a)，没有用 PPO 的柱子替代 SAC。论文版本与代码分别锁定，避免将早期标题版本和后期附录混称。

原论文 MW 大规模图是组内随机配对平均；它没有提供“四个固定方向必然负迁移”的保证。N2 的依据比 N1 弱，必须如实标出。新四组是在阅读旧探索结果后选择的机制案例集；保留选择过程及全部 seeds，不用它估计负迁移发生率，不因新结果不符合预期就悄悄换组。

### 3.1 新机制组的协议

按用户最新要求，本批所有新任务统一 1M 实际环境步（含 warm-up），DMC α 以代码为准。任务选择参考 R&D，但 MW 训练长度、task bank 和评估协议与原文不同；这是候选机制检验，不称严格复现，也不保证负迁移。

| 参数 | N1/N2：Meta-World | N3/N4：DMC |
|---|---:|---:|
| A 与 B 各自训练预算 | 1M 实际环境步 | 1M 实际环境步 |
| warm-up | 10k，包含在上述预算 | 10k，包含在上述预算 |
| SAC / optimizer | SAC / Adam | SAC / Adam |
| 网络 | 两层 256 ReLU | 两层 1024 ReLU |
| actor / critic 学习率 | 3e-4 / 3e-4 | 1e-4 / 1e-4 |
| minibatch / UTD | 64 / 1 | 1024 / 0.25 |
| discount / Polyak | 0.99 / 0.005 | 0.99 / 0.005 |
| α | 自动温度，边界重置为 1 | 固定 0.01：按现有代码 |
| 当前FT seeds | 1、2、3；含排队项 | 同左 |
| 当前任务评估 | 每 50k 环境步、50 episodes、含任务出口 | 同左；使用 return，不伪造 success |
| episode horizon | 500 | 1000 |
| 后续计划 Clip（尚未排队） | [0.25,4]，B 入口与每 200k 环境步 | 同左 |

官方代码 HEAD 0fa3158f6f0f41631a57e27b6c3243cffe74f957 的 DMC fixed_alpha=0.01，与论文 Table 3 的 0.2 不一致。用户明确选择代码值 0.01；保持 factory 原值，不自动增加 α 扫描。[论文参数](https://arxiv.org/html/2403.05066v3#A13.SS1)，[官方 factory](https://github.com/hongjoon0805/Reset-Distill/blob/0fa3158f6f0f41631a57e27b6c3243cffe74f957/garage/algo_factory.py)。

DMC 标识使用 suite 元组而非复制 README 中的数字：N3 为 ('ball_in_cup','catch')→('finger','turn_easy')；N4 为 ('cartpole','swingup')→('fish','upright')。已检查本地可创建，观测/动作维数分别为 8/2→12/2、5/1→21/5；这只是构造检查，不是训练通过。异构输入切片、padding、head 以及 fresh 的相同网络布局必须验收。

后续正式对照计划（本次未启动）：每个方向/seed，一份 A parent → FT、Q-reset、Clip 三个 B 分支；另训练同 B 配置的 fresh 参照。Fresh 的整体 actor/critic 从头初始化，不能把保留 actor 的 Q-reset 叫 fresh。同源三分支共用 A、B warm-up、任务 bank 与评估随机种子；fresh 用独立初始化但匹配 B head/网络布局与数据协议。

当前四组FT三seeds规模为4×3×(A+B)=24M环境步；ABC首批4M加seeds2/3新增10M，共计 **38M 新环境步**。若后续批准完整三 seed 三分支+fresh 方案，则四方向总规模为 12 个 A + 36 个 B 分支 + 12 个 fresh-B = 60M 环境步（已跑兼容部分可计入，不是本次追加队列）。不假定已有 1.5M checkpoint 或异构单任务 teacher 兼容。若确有逐项兼容缓存，另报实际去重成本。

## 4. ABC 同源两条续跑：用户指定对照

父模型：[A 结束 checkpoint](bellman_probe_results/sac_transfer_abc_20260907/sac_transfer_abc_s1_1m/checkpoints/task_boundary_task0_step1000000.pt)。

SHA256：0fc3e440b910411cd50ba3b1089b425fa4c7e36325e323c50286ee06a67053db。

| 分支 | B 入口与 C 入口 | 任务内处理 |
|---|---|---|
| ABC-Q-reset | 保留完整 actor/actor Adam；重置双 online Q、critic Adam；target 硬同步 | 普通 SAC，不周期 reset |
| ABC-Clip-8 | 双 online Q 所有 Linear 权重裁剪至 [0.25,8]；保留 bias、actor 和 Adam；target 硬同步 | 每 200k task-local **环境步**裁剪；普通 Polyak，不额外硬同步 |

两条均从同一 A parent 继续 **push-wall→window-close**；C 接各自的 B 结束状态，不在 C 换回旧 FT 的 B checkpoint。按最新要求，新 B/C 每任务 1M **实际环境步**，包含 10k warm-up，共新增 4M 环境步。旧 A parent 的 1M 是 optimizer updates，不能回写成环境步；新 invocation 的环境计数从 B=0 开始，保留更新计数及 parent 来源。新评估每 50k 环境步、50 episodes；旧 FT 是每 100k updates、20 episodes，必须分开标注。

旧 parent 含 actor、双 Q/targets、actor/critic Adam、log_alpha，但缺训练 RNG、环境状态、完整 replay、alpha optimizer；它能定义共同权重起点，不能声称逐位恢复旧 FT 随机轨迹。边界 replay 清空，温度重置为 1；reset 初始化使用独立、记录的 RNG，不声称恢复了没有保存的最初随机 critic。

旧ABC seed1 FT只作历史参照。若要严格比较该seed相同重启协议下的三臂在线因果效应，需要另加matched FT-continuation；seed1仍只指定Q-reset/Clip两条，**不自动追加seed1第三条**。新四组FT三seeds已纳入队列，但Q-reset/Clip/fresh配对仍未启动，不能单凭FT确认负迁移。

已修复 exact-budget 边界分支的 B+C 预算、Clip 加载入口与 fresh critic 的 Adam 状态；ABC 显式 branch_alpha=1。13 项回归检查通过，真实短测与正式运行各用独立目录。启动器为 scripts/launch_mechanism_pilot.py；按任务名解析 DMC 索引，并冻结代码与 parent SHA256。

ABC追加的seeds2/3没有旧A parent：各自FT运行完整A→B→C（3M），Q-reset只从同seed的1M-env A出口分支B→C（2M）。父文件完成校验和hash后释放依赖；源FT可同时继续B/C。新seed的A预算与旧seed1不同，且分支重设RNG、不恢复FT的完整随机轨迹，不能抹掉这些协议差异。实际清单见README追加批次。

## 5. 正文结构与图表

| 小节 | 唯一要回答的问题 | 实验/理论与结果位置 |
|---|---|---|
| 1 Introduction | 已有现象还缺哪一层机制解释？ | ABC 引出任务条件性；三项贡献按已验证程度措辞 |
| 2 Related Work and Preliminaries | 我们解释什么对象，与 R&D/谱方法有何关系？ | SAC、学习核、残差/经验需求/完整修正；方法来源 |
| 3.1 Phenomenon | 任务本身可学时，继承是否阻碍获取？ | Fig. 1：ABC 与 N1–N4 的 FT/fresh 曲线；按 seed 展示 |
| 3.2 Bellman-demand theory | 需求为何在特定方向形成预算瓶颈？ | 命题 1＋预算推论；MRP 精确检验，Fig. 2 |
| 3.3 Task-conditioned geometry | 旧训练如何改变相同 B 输入上的谱，需求落在哪里？ | N1–N4 同输入/同 D 交叉比较；ABC 辅助，Fig. 3a–b |
| 3.4 Actual correction lag | 预先确定的弱方向是否实际持续修正不足？ | N1–N4 动态窗口；P1–P6 的完整旧结果作边界检验，Fig. 3c–d |
| 4.1 Intervention | 如何在任务边界/周期执行谱干预？ | Algorithm 1；默认 Clip-4，ABC Clip-8 单独标记 |
| 4.2 Benefit and cost | 几何收益何时超过函数变化代价？ | 命题 2、线性网络例子、同 target/同 correction 控制 |
| 4.3 Paired mechanism test | 是否先改变方向修正，再改善在线获取？ | 四方向同源 FT/Q-reset/Clip；ABC 两条续跑，Fig. 4 / Table 3 |
| 5.1–5.3 Full sequences | 实际收益、保留/重访代价、不同方法成本怎样？ | Tables 1–2、Fig. 5；全部五条流，不选择性省略 |
| 6 Discussion and Limitations | 哪些机制预测成立、哪些不成立？ | 已停止探索、选择性案例集、局部理论、状态覆盖和方法失败 |
| 7 Conclusion | 重新理解了什么，适用到哪里？ | 只总结正文已经交付的结论 |

固定主图 5 张、主表 3 张。Fig. 1 现象；Fig. 2 理论/MRP；Fig. 3 真实谱与动力学；Fig. 4 干预；Fig. 5 完整序列。详细图表数据清单、每小节写作顺序与验收见写作框架；不为凑数量拆重复实验。

附录：A 协议/资源与任务选择；B 完整证明/MRP；C 五条主序列全部 seeds/位置；D 四方向所有结果与 ABC 的完整 B/C；E P1–P6 旧探索及不支持结果；F 数据字典/实现审计/复现。

## 6. 必需理论及统一对象

在固定输入上，未中心化 full-J kernel 为 K=JJᵀ/n，包含完整 critic 参数；权重奇异谱、feature rank 与 K 谱分别命名。

- 即时残差 d_t=y_t−q_t。
- 经验需求矩阵 D_emp=[d_0,…,d_m]，先保留多列，再取左奇异子空间；不先平均成向量。中心化 D 与 target increments U=[y_(t+1)−y_t] 另报。
- 固定策略 MRP 中 A_B=I−γP，完整价值修正 e=q*−q=A_B⁻¹d，多列记 E=A_B⁻¹D。真实 SAC 的 q* 未知，不能把 D_emp 称为完整 Bellman 算子的子空间或全局价值误差。
- 理论底稿的多列 D 表示完整修正；正文必须换记为 E，以免与经验残差 D_emp 冲突。

### 命题 1：需求、传播与预算

冻结 K 的 semi-gradient 模型：
e_H=(I−ηKA_B)^H e_0。

若 P 对称且与 K 可交换，共同模态满足 β_i=κ_i(1−γp_i)，则：

R_H = Σ_i |u_iᵀd_0|²(1−γp_i)⁻²(1−ηβ_i)^(2H) / Σ_i |u_iᵀd_0|²(1−γp_i)⁻²。

在单调区间 0≤ηβ_i≤1 内，慢模态 β_i≤b 占完整修正能量 m_b，故：
R_H≥m_b(1−ηb)^(2H)。

若 R_H≤ε<m_b，则必要预算
H≥ceil(log(m_b/ε)/[-2log(1−ηb)])，其中 0<ηb<1。

在同一初始 correction 下比较 inherited/fresh，才能把相对误差归因于受控学习几何。非交换情形用合适度量/矩阵幂，不直接套共同特征基公式。这个命题不证明真实 SAC 的策略成功率。

固定目标梯度流的需求加权逆谱积分作为辅助引理；实际记录的 raw/trace-normalized/ridge 指标是明确标注的代理，不换算为 Adam 的真实训练时长。

### 命题 2：干预净收益

在同一固定 Bellman 问题、各分支冻结自身核下，j_c=q_c−q_0，T_0=(I−ηK_0A_B)^H，T_c=(I−ηK_cA_B)^H：

E_0−E_c = ||T_0e||²−||T_ce||² + 2〈T_ce,T_cj_c〉−||T_cj_c||²。

它区分同 correction 的几何收益与函数变化；交叉项不必有害。给出改善的充分条件与不改善构造，不声称 weight clip 必然增大 K 的有效秩或优于 Reset。两层线性网络例子连接权重变化与 K/Q 变化，非线性 SAC 必须实测。

## 7. 已有证据与验收缺口

- ABC 已完成谱分析：C 可在整体谱仍集中的情况下学会；B/C 指定需求的负担有明显差异。是一个案例，不是唯一原因证明。
- P1–P6 已完成 54 个窗口的离线分析；方向修正差异有支持，旧训练相对初始化的普遍谱集中、指标与在线负迁移的稳定对应不获支持。原报告冻结到 B=500k；停机时更晚数据没有自动纳入。
- 旧 Clip 条件重放：共用 FT 外部 target 时 0/36 窗口降低误差面积，不能改写成已证实 Clip 改善学习几何；闭环 target 变化与在线收益须单独检验。
- N1–N4、ABC 两条新分支、完整机制—在线收益桥梁均待执行/验收。四方向文献先验不能替代本地 fresh/FT 结果。
- 论文逻辑框架检查：问题→目标、目标→挑战、挑战→模块在设计上对齐；模块→“已证实贡献”仍有 1 个 CRITICAL 证据缺口（干预到在线获取的机制桥梁），故不提前写肯定式结果。
- 单位、原始数据、全部 seeds、选择理由和不利条件均须可追溯。停止探索训练不等于删除探索事实。

## 8. 主实验保持不变

| ID | 完整顺序（MW 名称加 -v2） | 每条流预算 |
|---|---|---:|
| F1 | button-press → plate-slide-back-side → window-close → plate-slide-side → peg-unplug-side → plate-slide-back → coffee-button → window-open → handle-pull-side → door-close | 15M |
| F2 | plate-slide-back-side → soccer → sweep-into → handle-pull-side → plate-slide-side → peg-unplug-side → door-lock → reach → plate-slide-back → coffee-button | 15M |
| F3 | coffee-push → button-press → reach → peg-unplug-side → reach-wall → door-close → window-open → handle-pull-side → plate-slide-back-side → soccer | 15M |
| D-W6 | walker-stand → walker-walk → walker-run → walker-stand → walker-walk → walker-run | 9M |
| D-C4 | cartpole-balance → cartpole-swingup → cartpole-balance → cartpole-swingup | 6M |

八方法：FT、critic-only Q-reset、actor EWC、P&C（EWC compression）、SpectralReg、ReDo、R&D、Clip-4。每任务 1.5M 环境步，seeds 1/2/3；已提交主序列每 10k 环境步评估 50 episodes，不改标签为 50k。120 个配置、1.44B 名义训练交互含 R&D teacher 份额；额外 rollout、评估与蒸馏成本另报。

获取/保留分开：R&D 的获取来自 teacher，部署保留来自 student；P&C 的获取来自 active column，旧任务保留来自 knowledge base。DMC 重访保留独立 occurrence/head。主实验参数、两台机器分配和命令以实验计划 §6 / README 为准，不在论文大纲复制实现检查清单。

F1–F3 的既有顺序来源见 [FAME Appendix](https://arxiv.org/html/2603.00903v1#A7.SS1)；引用顺序不表示将 FAME 恢复为 baseline。与 [SingularClip](https://arxiv.org/html/2608.18319v1) 的具体差异在写作前逐项复核，不宣称 SVD clamp 算子原创。

本次修改前的 Markdown 已备份在仓库外；日志、模型、图表缓存和分析结果仍不上传 GitHub。本次未 push。
