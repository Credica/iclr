# 实验执行清单：Bellman Demand and Learning Spectra

2026-09-10 修订。论文主线见 [大纲](PAPER_OUTLINE_20260907.md)，逐节内容/理论/分析见 [写作框架](PAPER_WRITING_FRAMEWORK_20260910.md)。本清单取消旧 6×6 配额，保留主实验全部设置；本次新增独立启动器 scripts/launch_mechanism_pilot.py；结果与完成状态不得由计划文字推断。

## 1. 固定实验口径：必须区分四类协议

| 实验组 | 每任务预算 | 评估 | seeds / 方法 |
|---|---|---|---|
| E4 主序列 MAIN | 1.5M 实际环境步，含10k warm-up | 10k 环境步、50 episodes | 1/2/3；八方法；Clip=[0.25,4] |
| E2 N1/N2 新 MW 候选 | 1M 实际环境步，含10k warm-up | 50k 环境步、50 episodes、含出口 | FT seeds1/2/3已运行或排队；新四组Q-reset/Clip/fresh未排队 |
| E2 N3/N4 新 DMC 候选 | 1M 实际环境步，含10k warm-up | 同上 | 同上；DMC固定α=.01、UTD=.25 |
| E-ABC seed1两条旧parent续跑 | 新B/C各1M实际环境步，含10k warm-up；旧A仍按原update时钟 | 新50k环境步、50 episodes、含出口 | seed1共同parent；Q-reset与Clip=[0.25,8] |
| E-ABC seeds2/3追加 | FT的A/B/C各1M环境步；Q-reset共享同seed A，仅训练B/C各1M | 同新50k/50episodes | 两个FT、两个Q-reset；无新增Clip |
| E1 P1–P6 旧探索 | 原1.5M实际env/task，现提前停止 | 原50k环境步、50 episodes | 原18个FT；不补齐、不重启 |

所有方法使用 SAC/Adam，不使用 PPO/Muon。N1/N2 的网络、lr、batch 为2×256、3e-4、64、UTD1、自动α；N3/N4为2×1024、1e-4、1024、UTD.25、固定α=.01。共同γ=.99、Polyak=.005、replay容量1M。新机制任务参考R&D；MW统一缩短为1M、FT已扩展至三seeds，独立task bank与评估方案是本项目改编，不声称严格复现。

**论文/代码差异：** R&D Table3写DMCα=.2，官方HEAD 0fa3158f6f0f41631a57e27b6c3243cffe74f957 的factory写.01。E2按用户最新要求选择代码.01，E4也保持现有.01，不做α扫描。原文MW3M/DMC1M是名义环境预算；本项目将warm-up包含在精确环境计数中。[论文](https://arxiv.org/html/2403.05066v3#A13.SS1)，[代码](https://github.com/hongjoon0805/Reset-Distill/blob/0fa3158f6f0f41631a57e27b6c3243cffe74f957/garage/algo_factory.py)。

统计单位为seed，报告均值、样本SD、每seed曲线与配对差；不把episode、方向、critic或更新窗口当独立seed。DMC只使用return，不填虚假success。相同实验组内使用一致task bank、网络/head布局和评估协议，不跨协议直接拼接AUC。

## 2. 执行状态与顺序

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| E0 | 新机制/ABC恢复、时钟、边界和记录正确性 | 原13项回归和六条真实短测通过；本次追加队列/分支9项检查通过；不代表长程效果 |
| E1（历史） | P1–P6 18个FT | 用户要求停止；18 workers、5 supervisors已退出，数据保留 |
| E2 pilot | 四方向N1–N4各FT seeds1/2/3 | 共12条运行/队列；新四组Q-reset/Clip/fresh未排队 |
| E3 | MRP与离线谱/动态/干预分析 | 部分历史分析已完成，新组待数据 |
| E-ABC | seed1旧A的Q-reset/Clip-8；seeds2/3新FT及同seed A分出的Q-reset | 共6条运行/依赖队列；新Q-reset须等对应A文件完成验证 |
| E4 | 五条主序列×八方法×三seeds | 按既有队列继续；本次不改参数、不停主实验 |
| E5 | 图表/统计/证据审计 | 随结果推进，不预填结论 |

E1的原status记录为failed/KeyboardInterrupt，是人工取消而非数值失败。说明在本地批次 STOPPED_BY_USER_20260910.md；保留原始status，不把停机记录改成completed。已完成的54窗口离线分析只使用其冻结至B=500k的在线对照，不自动把更晚未审计记录混入。

本次E0短测后先执行四条FT seed1与ABC两条，共6条运行/12M新环境步。四组FT各在GPU0/1/2/3，ABC两条在GPU7；不停止GPU7已有他人进程。fresh与新四组Q-reset/Clip不在本次队列。后续完整对照可复用兼容A parent。四个方向与全部seeds事先固定；基础对照用于判断现象和诊断配置，不依据Clip排名暗中筛组。文献不保证每个方向每个seed负迁移，若不出现则报告该结果，不无上限调参到失败。

## 3. E0：新增入口必须通过的检查

追加批次：scripts/extend_mechanism_seeds.py登记12条（四组FT×seeds2/3共8，ABC FT/Q-reset×seeds2/3共4），新增26M环境步，与首批合计18条/38M。原每卡两个槽位的排队规则已按用户最新要求取消：所有非reset立即启动，GPU0/1各4条、GPU2/3各3条、GPU7保留2条，共16条训练；ABC FT先训练各自A，A的模型/optimizer/1M-env/三head和hash验证后才释放同seed Q-reset依赖，不复用seed1父模型。失败不自动重试；缺父模型只阻断对应Q-reset。实际状态见 /home/zqy/plasticity-papers/mechanism_seed23_20260910_v1/actual_status.json；旧queue_state仅为依赖协调器的逻辑状态，提前启动的FT由重复启动保护接管，不会重训或覆盖。

### 3.1 时钟、环境与评估

- E2与E-ABC的新任务均为1M精确环境预算，包含10k warm-up；旧ABC的A parent仍保留历史1M更新标签。明确区分global/task env与critic updates。
- Clip间隔200k按任务内实际环境步，从入口0计数；与UTD无关。不能用10k warm-up后的200k更新替代。
- MW每task/seed固定50训练实例和独立50评估实例，episode horizon500；DMC固定50评估reset seeds、horizon1000。
- 评估确定性均值动作，独立环境；保存恢复Python/NumPy/Torch/CUDA RNG。出口评估已见任务，不评未来head。
- 新机制A出口之前保存完整source；B共同warm-up由同一固定行为策略/独立RNG产生，不因reset耗用RNG而变化。B anchors在warm-up结束才可获得，不冒充step0已知诊断。
- DMC使用suite元组解析任务，验收异构输入切片、输出动作/head、padding和fresh相同布局；不硬编码不同依赖版本的数字ID。
- E2固定α=.01进入manifest，与factory及E4一致。

### 3.2 同源分支与边界

- FT保留actor、双Q、各自Adam；MW边界α重置1，DMC固定α保持；清replay并同步target。
- Q-reset保留actor/actor Adam，重置双online Q与critic Adam，硬同步target；不是完整fresh。
- Clip保留actor、bias和Adam，对双Q全部Linear权重裁剪；入口硬同步target，周期普通Polyak；同一入口/周期重合去重。
- E2为[0.25,4]；E-ABC Clip-8为[0.25,8]，不得全局改默认值。
- 独立记录reset初始化RNG、父checkpoint hash、分支配置、初始Q与前后事件；三seeds不是从同一个seed1 parent变更标签得到的。
- 每个分支必须保存自己B结束模型；ABC的C从各自B继续，不换回旧FT的B。

### 3.3 本次实现与仍然存在的限制

1. factory已允许exact-budget任务边界Clip分支；仍拒绝不受支持的任务内恢复和混合干预。
2. main_garage.py的exact-budget分支现按剩余全部任务计数，ABC为B+C=2M新环境步。
3. fresh online critic不再加载旧critic Adam，actor Adam继续继承；测试覆盖两种分支状态。
4. ABC启动器显式branch_alpha=1，C入口沿普通边界规则再次重置温度。
5. 新启动器冻结main/factory与记录器；不改写锁定P1–P6/MW的旧rethink_ft_recorded.py，不影响旧进程或朋友队列。
6. 原checkpoint缺完整RNG/env/replay的情况必须标明；分析快照不等于可逐位恢复状态。

13项单元/回归测试已通过，含精确时钟、分支优化器、Clip及记录不改变更新/RNG。六条真实10k/task短测与1M/task正式运行分目录，状态以各run的status.json为准；短测不是论文结果。

### 3.4 安全与资源

输出使用新artifact root，禁止覆盖H-ABC/E1/朋友主实验目录。不删除旧缓存或日志。诊断不改变训练RNG；先测窗口文件规模和吞吐，尤其DMC1024网络，记录资源预算。保留原始数据在Git忽略目录；不提交checkpoint、日志、replay或结果。

## 4. E2：四个固定方向

| ID | 源A | 目标B | 证据 |
|---|---|---|---|
| N1 | sweep-into-v2 | push-wall-v2 | 原论文明确案例＋本地ABC受阻线索 |
| N2 | window-open-v2 | sweep-into-v2 | 原Hard相邻顺序＋Window→Sweep组级负迁移；固定pair仍待本地验证 |
| N3 | ball_in_cup-catch | finger-turn_easy | 原Fig.14(a)的具体SAC任务对负均值 |
| N4 | cartpole-swingup | fish-upright | 原Fig.14(a)的具体SAC任务对负均值 |

来源与限制详见大纲§3。仅四方向，不扩展交叉组合，不承诺“一定负迁移”。N3的suite元组为(ball_in_cup,catch)→(finger,turn_easy)；N4为(cartpole,swingup)→(fish,upright)。本地构造检查通过，观测/动作维数分别8/2→12/2、5/1→21/5；这不等于SAC训练通过。

### 4.1 作业数与fresh

当前四方向×三seeds×FT(A→B)=24M环境步，含队列。后续完整证据计划为每方向×3seeds：1个A预训练、FT/Q-reset/Clip三个B分支、1个fresh-B，共60个单任务、60M环境步（包含兼容的已跑部分），不是本次待执行队列。

Fresh从头训练完整actor/critic，使用和该方向B一致的网络/head/输入布局、任务bank、预算、评估规则。不能将主序列teacher、不同head的A训练或旧1.5M曲线直接当成fresh。符合全部条件的缓存才能去重，并保留核验记录；名义预算不预先扣除未知缓存。

### 4.2 指标与判定

- 主获取差：ΔAUC=(1/T_B)∫(FT−fresh)dt；MW用success，DMC用return，各自单位。
- 终段差：最后五个实际评估点的FT−fresh，另报最后一点。
- 方法差：Q-reset−FT、Clip−FT，同direction/seed/parent先配对再汇总。
- 三seed方向不一致标混合；fresh/FT都低不自动解释为历史导致失学。
- N1–N4全部保留，不用新结果事后替换方向或种子。旧P1–P6作为探索批次保留，不估计无偏普遍发生率。

### 4.3 ABC两条补充续跑

共同parent及hash见大纲§4：旧H-ABC的task_boundary_task0_step1000000.pt。只新增Q-reset与Clip-[0.25,8]，seed1，均继续push-wall→window-close，新任务各1M实际环境步（含warm-up）；B/C入口按定义干预，Clip每200k环境步。

parent含模型和actor/critic Adam，但无完整RNG/env/replay/alpha optimizer；新的共同启动协议需落盘。旧FT为100k更新/20episodes的历史参照；新两条为50k环境步/50episodes，不当作完美三臂因果对照。未授权自动加seed1 FT-continuation；如需严格重启对照另行安排。本次启动，实际进度见新run状态文件。

## 5. E3：理论、谱与动态分析

ABC seeds2/3追加：每个FT从头训练A→B→C，共3M；同seed Q-reset等待A=1M环境步的出口checkpoint完成后，从B→C续跑，共2M，两个seed合计10M。父文件/hash另存parents，FT不必等待Q-reset。新seed A为1M环境步，与旧seed1 A=1M更新不同；reset分支重新设RNG，不宣称与FT共享逐步随机轨迹。

### 5.1 精确MRP

64状态对称环形MRP，γ=0/.9/.99，构造seeds1/2/3。固定K全谱改变需求；固定需求改变弱方向速率；分别匹配即时残差与完整修正幅度。检验矩阵迭代、谱公式、预算下界及干预净收益恒等式。保持合法P与明确稳定区间；非交换情形不套共同基公式。包含有利和不利干预构造，不属于在线RL配额。

### 5.2 同输入、同需求谱分析

- 两个不同episode的128点panel，共256个B anchors；固定索引/hash，另保留后期reservoir检查分布敏感性。
- 初始/A历史/A出口/B阶段/fresh/干预后模型：未中心化full-J K=JJᵀ/n；同时记录原始尺度与归一化谱。
- D_emp保留多个残差列；中心化D、target increments分别标记。95%子空间只描述结构，负担和响应不丢尾部。
- 固定D换K、固定K换D；报告方向能量、逆谱负担、有限步响应及same-correction拟合。
- 主relative ridge1e-3，1e-4/1e-2敏感性；原始尺度与shape结论不偷换成真实Adam速度。
- 未来需求只用于回溯，不标为入口可得预测；相邻更新不是独立seed。

### 5.3 动态记录与滞后

当前计划：四组FT三seeds各在B=10k/100k/500k记录，共36窗口；ABC六条运行各在B/C相同时点记录，共36窗口，合计72个1000-update窗口。每个窗口存1001组双critic参数、逐步真实minibatch/target及固定128-input目标/输出路径，端点存Adam/RNG；全局初始化及关键点另存快照。固定输入来自warm-up transitions，不声称episode独立。四方向FT36窗口已纳入队列；该四方向Q-reset/Clip入口24窗口仍未排队。

前100次更新构造需求，第100次更新固定核方向，分析后900次实际修正、方向残差、target流入与非线性余项。DMC1000updates约覆盖4000环境步。逐步参数采用起点＋必要差分/重建数据，不反复保存完整actor/target；记录开销先做smoke估算。

### 5.4 同源干预到在线获取

对每direction/seed对齐Q-jump、target反馈、同D几何、同correction/同真实target的200/1000update拟合、入口真实方向修正与后续B获取。共同外部target重放与各自闭环target分开报告。Adam/采样差异是代数参照，不称失败的因果百分比。

与论文Fig.3/4和Table3对应；P1–P6已有不利Clip重放保留。新的配对证据没有出现之前，不称已证明Clip解除瓶颈。

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

### 6.2 八种方法与开跑条件

| 方法 | 正式定义 | 开跑前缺口 |
|---|---|---|
| FT | 普通 SAC 连续微调 | 统一主序列入口与记录协议 |
| Reset | critic-only Q reset，并重置 critic Adam | 已实现 optimizer reset；保留正式边界验收 |
| EWC | 原版 EWC，不叠加 Clip | 固定 Fisher、正则参数范围和系数记录 |
| P&C | EWC-compression 的 Progress & Compress；任务后重置 active column 与 adaptor | 采样/训练/评估/压缩共用固定旧 KB 侧向连接；压缩目标不随 live KB 更新漂移；状态随 Bellman checkpoint 保存 |
| Spectral regularization | k=2；每层 $(\sigma_{\max}(W)^2-1)^2+\lVert b\rVert_2^4$；actor/双 online critic 均取 1e-4；多 head actor 只正则共享层与当前任务 mean/log-std heads，不改未激活 heads | 已有 `--cl_method spectral` 与单步 power iteration；完成 DMC 环境 smoke test，不能以 hard clip 代替 |
| ReDo | 每 1k task-local 环境步按归一化平均绝对激活（tau=0.1）识别 dormant neurons；重置 incoming、清零 outgoing、清除对应 Adam moments，并同步双 target Q | 周期实现与单元测试已完成；正式长程前保留短程 Meta-World/DMC smoke |
| R&D | reset-and-distill teacher/student 管线 | `--run` 先完成本机非 R&D，再自动生成/复用同配置、同 seed、1.5M teacher 与训练 bank rollout 并蒸馏；完成标记仅作可选追溯；student 每阶段仅评估已见 head，单独记录蒸馏时钟 |
| Clip（ours） | 第二任务起的 critic 双侧谱裁剪 | 主序列入口已支持 MW/DMC、环境时钟、所有后续任务入口及现有 probes；E2 同源 checkpoint 分支和完整 E0 数据契约仍待补 |

所有方法在 F1、F2、F3、D-W6、D-C4 上运行 seeds 1、2、3，共 8×5×3=120 个序列配置。按任务数计的名义训练预算为 1.44B 环境交互步，其中 R&D 的 teacher 预算已包含在内；student 的离线蒸馏不伪装成额外在线学习曲线。

### 6.2.1 Clip 主序列配置（2026-09-08）

`scripts/generate_clip_matrix.py` 与 baseline 生成器共用 §6.1 的五条序列定义，
`scripts/run_clip_main.sh` 单独排队；不混入朋友的 105 个 baseline runs，也不复用
E1/E2 的短序列测试入口。每条序列 seeds 1/2/3，共 **15 runs、120 个任务位置、180M
训练环境步**。DMC 重访保留为不同位置，actor head 按位置递增，不按唯一任务名合并。

固定 Adam、每任务 1.5M（含 10k warm-up）、Clip 范围 [0.25, 4]、起始位置 1。
每个符合条件的任务在 0、200k、400k、600k、800k、1M、1.2M、1.4M 裁剪，
共八次；15 runs 共计划 840 次。周期事件落在该次采样的最后一个 critic 更新后，
随后正常 Polyak、诊断和评估；任务出口评估先于下一任务入口裁剪。日志打开 W&B、
1k 标量、10k feature/weight/Hessian、100k Bellman，以及原固定谱诊断关键点。

启动与 `--prepare-only` 命令见 README 的 Clip 节；GPU 列表可配置，每卡两个进程自动
接续。此处配置完成不表示 E0 全部验收：完整恢复、入口评估、完整逐 episode
原始记录与干预前后 Q-jump 等仍按 §3/写作框架 §9 补齐。旧日志保持旧协议身份，不能改标签。

本次验证：33 项仓库测试通过，含 Clip 调度/预算/状态保持/重复 head 回归，以及 CPU
小网络真实 DMC cartpole 四位置重访的 8k-step 训练、评估和谱日志 smoke。完整尺寸
V100 长程训练未在此验证中执行；队列只做了生成和 dry-run，没有启动正式 runs。

### 6.2.2 双机 baseline 分配（2026-09-08）

不含 Clip（ours）的七个 baseline 共 7×5×3=105 个主序列 runs。固定生成器为
`scripts/generate_baseline_matrix.py`，两台物理机器分别使用
`scripts/run_baselines_machine_1.sh` 与 `scripts/run_baselines_machine_2.sh`；每台机器使用本地 GPU 0–7，每卡最多两个进程，空闲 slot 自动从本机队列补位。机器 1 分配 53 个 runs（440 个任务位置），机器 2 分配 52 个 runs（400 个任务位置）。R&D 的 F1/F2/F3 × seeds 1/2/3 全在机器 1，D-W6/D-C4 × seeds 1/2/3 全在机器 2；分别需要 48 个 Meta-World、15 个 DMC teacher/task/seed 前置项，本机内部去重、跨机无重复 teacher。非 R&D 按全局 run ID 交替分配，但 FT/F1/seed 1 放在机器 2，得到 44/46 个非 R&D runs。每次生成保存逐 run JSON manifest，包括完整任务、seed、命令、机器、阶段和依赖；这个分配不保证两台墙钟耗时相同。

本节主序列队列的所有在线 SAC/teacher 命令显式使用 `--wandb True` 与 50 evaluation episodes。loss、reward、alpha、speed、zero ratio 每 1k 环境步；feature rank、weight change 每 10k；Hessian rank 每 10k 且在同一模型状态后进行 10k evaluation；Bellman probe 每 100k。`--no_stats False` 打开这些统计；诊断和评估前后恢复 Python/NumPy/Torch/CUDA RNG。full-Jacobian Bellman spectral stats 固定在 10k、50k、100k、500k、1M、1.5M，不改成等间隔扫描。probe/checkpoint 与所有生成结果写入仓库外的 run 独立目录。

两个 bash 的 `--run` 自动执行两个阶段，无需第二次手动启动：第一阶段运行 `baseline_non_rnd_machine_N.txt`，机器 1 为 44 runs，机器 2 为 46 runs，共 90 runs；按 FT → Reset → EWC → P&C → Spectral regularization → ReDo 派发，方法之间可以重叠，每个空闲 slot 接续下一个完整序列 run。

本机非 R&D 队列全部成功后自动进入第二阶段：先运行 `baseline_prerequisites_machine_N.txt`（机器 1 为 48 个 MW teacher，机器 2 为 15 个 DMC teacher），全部成功后运行 `baseline_rnd_machine_N.txt`（机器 1 为 9 个 MW student runs，机器 2 为 6 个 DMC student runs）。总顺序为机器 1：44 → 48 → 9，机器 2：46 → 15 → 6。两台机器互不等待，因此一台进入 R&D 时另一台可能仍在跑非 R&D；第一阶段不后台训练 teacher。

每个队列内失败的任务会被记录，其余任务继续；当前队列存在失败则阻止进入下一队列，不自动重试。更新后仅用于尚未启动的批次，不能覆盖运行中 manifest 后盲目重跑。

`baseline_jobs_machine_N.txt` 保留全部 53/52 个 baseline 命令，仅作审计，不应作为单一队列启动以免绕过 teacher 依赖。JSON manifest 为 schema 4，保存完整配置、分阶段计数和执行顺序/队列路径；队列日志分置 `queue/machineN/non_rnd`、`teachers`、`rnd`。`--prepare-only` 仅生成清单，不启动训练。

R&D manifest 仍标记 `requires_teacher_artifacts=true`，但依赖由第二阶段自动处理：按 env type、task、1.5M 预算和 seed 命名的 model、rollout 均存在时复用，否则训练单任务 teacher 并导出。student 通过 `--rd_teacher_root` 读取显式根目录 `<ARTIFACT_ROOT>/teachers/`。新导出的 `complete_<teacher-stem>.json` 仅作可选追溯，不再作为复用前提，也不因缺少它就强制重训。Meta-World 与 DMC 使用各自与 `make_log_name` 一致的文件名，DMC teacher 的局部输入权重映射到序列网络对应输入切片并写入对应 occurrence head。

R&D 保留原方法的独立单任务专家训练、训练结束后新采集专家 rollout、结合历史任务记忆进行顺序蒸馏的流程；不是拿训练 replay buffer 替换专家 rollout。默认额外采集 1M observations，采集交互与 1.5M teacher 在线训练预算分开记账。五条序列、1.5M 预算和独立训练/评估 bank 是本文所有方法共用的实验设置，不声称逐项照搬原论文实验。rollout 只在训练 bank 上采集，不用独立评估 bank 训练 student。

旧缓存允许在配置一致时复用。导入前核对环境/依赖版本、观测动作处理、网络结构、Adam 训练设置、seed、包含 warm-up 的 1.5M 预算、任务实例划分与 rollout 来源。队列只检查文件名和两个文件是否存在，不自动校验完整配置、文件完整性或训练是否完整；同名不等于兼容，旧 3M 或旧单实例 teacher 不可冒充同协议结果。曾存放在 `teachers/fixed-task-banks-v1/` 的同配置缓存可核对后复制到上述目录；脚本不自动搜索、迁移或删除旧文件。更新代码后重新生成队列。

### 6.2.3 Baseline 与 Clip 共用评估实现（2026-09-08 修订）

所有在线 baseline 与 Clip 统一调用 `MTSAC._evaluate_policy`，遵循 §1 的主序列口径：每 10k 环境步仅评估当前位置，任务出口评估全部已见位置，绝不评估未来 head；此处不是 E1/E2 的 50k 入口。P&C 对旧任务使用 KB、当前任务使用 active column；R&D 在每个离线 student 阶段完成后评估已见位置，环境时钟不因蒸馏增长。DMC 不再把各任务挤进同一个 `Evaluation` key。

正式 exact-budget SAC 自动使用 `fixed-task-banks-v1`。MW 按 task name/seed 分别生成 50 个训练实例与 50 个无交集评估实例，训练采用私有 RNG 均匀抽样；评估每轮从固定实例 0–49 开始。DMC 训练/评估环境隔离，每轮重复固定 50 个 evaluation reset seeds。bank 不依赖 method/stream/occurrence，单任务 teacher 与序列模型共享相同划分。`task_banks/` 保存 pickle、reset seeds 与 SHA-256 manifest；`eval_episodes.jsonl` 保存实际 clocks、位置/head/role、episode return/success/length 和 instance/reset seed。确定性评估直接取均值，并恢复训练 RNG。

新批次应使用新 artifact root，不混用修订前曲线。baseline 主队列仍不支持断点续训或自动跳过已完成 runs。上述修改不代表完整 E0、V100 满尺寸双进程显存及长程数值稳定性已全部验收。

此前缓存复用修订后的 CPU 验证：分组执行的 39 项不同回归检查通过。8 项队列/缓存测试覆盖无完成标记的成对文件复用、文件缺失时训练分支、任务/seed/预算文件名不匹配、双机 prepare/dry-run；全部 168 条 baseline/teacher 命令参数与含空格路径均通过检查。环境与算法测试覆盖六种在线 baseline 的真实 DMC 两任务/4k 步训练（小网络，含 Hessian/Bellman）、真实 MW 独立实例 bank 与重放、DMC 50 个固定 reset、teacher rollout 仅使用训练 bank、R&D 小型合成 artifacts 加载及逐阶段已见任务评估、Clip 时钟/入口以及 P&C 等定向检查。未运行正式 1.5M teacher/student 实验或 V100 满尺寸双进程压力测试。

本次分阶段调度通过 9 项队列/缓存测试，含双机实际 launcher 配合 stub queue 验证自动先非 R&D、后 teacher、再 student，以及各阶段失败时阻断下一阶段；全部 168 条命令（105 baseline＋63 teacher）解析、分阶段文件划分和 dry-run 通过。测试核对每个 student 的本地 teacher 覆盖及跨机无重复项。105 个主实验与 63 个 teacher 的训练命令、机器分配与缓存复用修订版一致，仅改变执行阶段顺序；未启动正式训练。

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

R&D 的获取曲线来自 teacher、保留结果来自部署 student；P&C 的获取曲线来自 active column，旧任务保留来自 knowledge base，当前任务来自 active column。图例、manifest 和表注必须显式记录角色，不能拼成一个虚构 agent。

## 7. 后续完整机制实验的数据契约

下表是完整证据目标，不表示本次pilot已逐项产生所有文件。本次实际落盘以README为准：manifest/status、现有results/评估/Bellman记录、mechanism快照与窗口。训练逐episode provenance、后期独立reservoir、每次周期Clip的完整前后快照与细粒度资源JSONL尚未实现，不把汇总日志冒充这些记录。

| 文件/对象 | 最低内容 |
|---|---|
| run_manifest.json | protocol ID、method、seed、任务/位置、parent和源码hash、网络/head/输入布局、完整参数、bank、状态 |
| eval_episodes.jsonl | 全部时钟、训练/评估位置、occurrence/head/role、checkpoint、episode/instance/reset seed、return/success/length |
| train_episodes.jsonl | 训练episode任务、时钟、return/success/length |
| train_metrics.jsonl | 每1k环境步的loss、reward、α、speed、zero ratio及已有Q/target/TD分布 |
| boundary_events.jsonl | 边界前后模型/optimizer/α/replay/target处理与父状态 |
| intervention_events.jsonl | Clip/Reset前后权重谱、Q-jump、实际时钟、触发原因、optimizer/target处理 |
| checkpoint_events.jsonl | 文件ID、完整/轻量类型、时钟、parent与完整性 |
| resource_metrics.jsonl | 训练/评估/采集交互、各类更新、墙钟、显存与存储占用 |

### 7.1 快照与窗口

本次快照覆盖初始化/加载parent、任务切换前后、10k/50k/100k/500k/1M关键点与窗口端点；主SAC另存100k/边界/最终checkpoint。所有新任务止于1M，不生成1.5M/2M/3M训练点。周期Clip现有事件保存权重谱摘要和时钟，不声称保存每次事件前后完整critic。尚无B bank时只能保存模型，后续明确在B=10k anchors上离线评价，不冒充入口已知。

轻量快照至少含actor、online/target双Q、α、架构/head/输入映射、真实时钟与parent。用于carried-Adam分析的选定点另存双Q moments/step/lr/参数映射。完整可恢复状态需全部optimizer、RNG、env/sampler和必要replay；无法恢复的旧快照只标为analysis snapshot。

方法专有状态仍保留：EWC的Fisher/参考参数；R&D的teacher/student；P&C的active/KB/previous-KB/adaptor/Fisher/compression optimizer；SpectralReg的power-iteration状态；ReDo的统计/mask/events；Clip的调度与已触发事件。不得因大纲精简删除实现所需数据。

### 7.2 anchors、目标与频率

本次每任务保留warm-up最早1024个probe transitions；动态panel固定取其中128个，保存obs/action/reward/next_obs/terminal，不包含可声称episode独立的provenance，也没有后期reservoir。独立episode panels/后期分布检验是后续数据契约要求。各方法独立probe不自动等于共同bank。

日志频率：scalar/zero ratio1k；feature/weight/Hessian10k；Bellman100k；在线spectral保留10k/50k/100k/500k/1M，离线重建另计。E2评估50k，Hessian保留10k，不强行改成eval10k。本次main入口不额外产生step-0评估，AUC只积分真实记录区间，不捏造入口分数。E4保持其原10k eval及原固定spectral关键点。

共同target流须记录target Q/actor/α来源、next-action随机量、实际minibatch与参数变化；不是各分支各自重采自己的targets。真实SAC无已知q*，不计算虚构全局价值误差。具体分析验收见写作框架§9。

## 8. 总预算与去重

| 部分 | 名义训练预算 |
|---|---:|
| E4五条主序列×八方法×三seeds（不变） | 1.44B环境步 |
| 当前E2 pilot：四组FT三seeds的A→B | 24M环境步 |
| ABC seed1两条B→C补充 | 4M新环境步，含warm-up |
| ABC seeds2/3：FT完整ABC＋同seed Q-reset的BC | 10M新环境步，A每seed只训练一次 |
| 两批当前运行/排队合计 | 38M环境步（首批12M＋追加26M） |
| 后续完整E2三seed方案：12A＋36B分支＋12fresh-B（未全部排队） | 60M环境步，包含兼容pilot份额 |
| MAIN＋后续完整E2＋ABC名义总规模（不是本次队列） | 1.514B环境步 |
| 已停止E1及旧历史实验 | 已发生的实际成本单列，不计为新的待跑配额 |

取消旧6×6的513M配额及1.953B旧合计。E2不是六任务矩阵的子集缓存复用计划；不同预算/α/head/task bank不可默认复用。相同配置的R&D teacher去重规则仍按§6执行，名义和实际成本都报。

评估、额外rollout、蒸馏、P&C compression、诊断和离线重放分别记账；多列需求/多个窗口不计作额外在线实验个数。

## 9. 不运行的旧方案

不再运行6×6矩阵，不继续P1–P6，不重跑旧ABC seed1的A（新增seeds2/3的A需要各自训练），不启动RPP/H8/E8/CW20、actor-only/norm-only/随机扰动/from-A clip/BRO/Muon/PPO或大规模阈值扫描。ABC范围为seed1两条续跑及seeds2/3的FT/Q-reset；Clip-8不改写MAIN或E2的Clip-4。

不声称critic是唯一原因、默认阈值最优、第二任务起裁剪优于全程裁剪，或所有探索/optimizer替代解释已排除。

## 10. E5：完成与发布

只有预算/任务位置/实际更新满足协议，评估和必要原始记录齐全，没有未知退出/NaN/覆盖，才能标completed。人为取消标cancelled说明，保留原始进程status；部分曲线不补零、不冒充完整预算。

Figs.1–5与Tables1–3每个数可追溯到run/seed/checkpoint/role和分析协议。主图选择须有问题导向，全部四方向和三seeds保留；不删除不支持motivation的结果。

GitHub只提交实验代码、测试、配置与作者Markdown；不再限制为旧“五份Markdown”，新写作框架也属于作者文档。日志、checkpoint、replay、anchors、结果JSONL、图表缓存和本地停机记录留在忽略目录。此次未push。
