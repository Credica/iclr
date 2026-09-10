# 论文写作框架：从 Bellman 需求—学习谱失配到新任务适应

更新：2026-09-10。本版围绕同一条机制链交错组织现象、SAC 更新、简单理论与干预实验。三个主文命题均在下文给出假设和简短证明；已有观测与后续验证分别标明。实现与实验身份参考[实验计划](EXPERIMENT_PLAN_20260907.md)，较完整的固定策略推导参考[理论底稿](MOTIVATION_METHOD_THEORY_20260907.md)。本文的叙事次序、符号与图稿安排以本版为准。

## 1. 核心论点：新任务所需的 Bellman 修正，可能正是继承 critic 难以完成的修正

持续强化学习中的 critic 同时继承了价值预测和学习几何。新任务改变奖励与后续价值需求，但历史训练留下的网络对不同输出方向的响应并不均衡。于是，一个具体矛盾出现了：**新任务必须完成的 Bellman 修正，落在了 critic 响应较弱的方向上。** 网络仍然可以有效改变某些预测，却难以在有限预算内完成当前任务最需要的修正。

本文围绕这一机制解释新任务适应受阻，并以 Clip 检验相应干预：**通过调理 critic 的权重谱，降低固定 Bellman 需求的归一化谱负担，使同一需求获得更强的有效相对谱支撑，进而改善新任务学习。**

这里“更强的谱支撑”有明确含义：同一个需求在干预后学习核上受到的需求加权逆谱惩罚减轻。它可以由相对特征值变化、需求在新特征基上的投影变化，或二者共同产生。固定需求向量本身不需要旋转，全部需求能量也不必跨过某条强弱分界线。

整篇论文围绕下面这条链展开：

> 历史训练形成方向不均衡的 critic → 新任务的 Bellman 需求落入弱学习方向 → 必要修正滞后，并持续面对 SAC 自举目标的变化 → Clip 降低同一需求的谱负担 → 检验实际方向修正与后续任务获取的改善。

已有 ABC 数据已经给出这条链两端的直接观测：受阻任务的需求具有较高谱负担，Clip 降低固定需求负担并提高成功率。论文中间部分要把两端接起来：先证明局部方向响应的规律，再追踪真实 SAC 的需求、修正与目标变化。**研究主张始终是同一个，理论和实验分别负责说明规律、测量规律在实际训练中的作用。**

工作标题可采用：**Rethinking New-Task Adaptation in Continual Reinforcement Learning: Bellman Demand and Learning Spectra**。论文价值落在任务相关的机制解释与干预验证；学习核的标准线性化推导和 SVD clamp 算子本身不作为原创性主张。

## 2. 全文节奏：每一轮实验都提出一个问题，每一个理论结论都产生下一轮检验

| 推进环节 | 读者看到的现象或问题 | 紧接着的理论作用 | 理论之后的实验回答 |
|---|---|---|---|
| 第一环：发现适应差异 | FT 在 B 受阻，在 C 可以学习 | 提出“学习方向是否匹配任务需求”的研究对象 | 展示完整 reward / success 与任务入口几何 |
| 第二环：定位方向障碍 | 整体谱摘要没有解释 B/C 的适应差异 | 命题 1：方向响应与有限预算由特征值和需求投影共同决定 | 测量 B/C 的需求在学习谱上如何分布 |
| 第三环：量化障碍 | 少量弱方向需求也可能代价很高 | 命题 2：归一化谱负担等于局部修正过程的折扣残差能量面积 | 展示需求—谱代价图与累积负担曲线 |
| 第四环：进入真实 SAC | 一次冻结目标的修正如何关联在线学习 | 命题 3：方向残差由旧需求保留、实际修正和目标流入共同决定 | 在同一方向上对齐需求、输出更新与 target 变化 |
| 第五环：干预并回到任务获取 | 降低固定需求负担能否改善适应 | 用前述命题确定 Clip 应改变的环节与测量对象 | 配对几何、方向动力学、actor 更新与在线表现 |

各环节按“观察 → 解释 → 可测预测 → 验证 → 下一问”写成连续正文。理论出现在读者需要解释的地方；实验出现在命题刚刚产生预测的地方。完整效果评估放在机制链之后，回答方法的实际价值。

## 3. 第一环：用任务适应差异提出问题

开篇使用已有 ABC：sweep-into → push-wall → window-close。先展示连续学习曲线，让读者看到 FT 的 B 阶段长期受阻，而后续 C 可以学会；再给出 Clip 对 B 的改善，建立一个有现象、有干预空间的研究问题。

| 已有观测 | FT | Clip | 对叙事的作用 |
|---|---:|---:|---|
| B 末成功率 | 0% | 50% | 给出需要解释的受阻现象与干预收益 |
| C 末成功率 | 100% | 100% | 表明适应行为随任务变化 |
| FT 的 B/C 入口学习核有效秩 | B：4.63；C：3.11 | — | 整体谱摘要尚未告诉我们当前任务需要哪些修正 |

以上为历史 ABC seed 1，Clip 从 B 开始、阈值为 $[0.25,4]$，每任务 1M 次 optimizer updates，原始成功率评估每点 20 episodes。双 critic 的谱统计取均值，不能将两个 critic 当成两个独立训练 seed。

这一节的落点应是：

> 要理解一个继承 critic 为什么在某个任务上学不动，我们需要知道两件事：它容易沿什么方向改变，以及这个任务需要它沿什么方向改变。

B/C 入口采用各自任务的输入与需求，承担任务相关性观察。历史导致的负迁移幅度由匹配的 fresh/FT 对照量化；该对照在实验设计中保留清楚职责，不与 B/C 本身的任务差异混为一个量。

**承接下一节的句子：** SAC 已经在每次 critic 更新中给出了任务所需的修正信号；下面从这条实际更新式出发，确定“方向不匹配”究竟指什么。

## 4. 第二环：从 SAC 更新推出“需求落在弱方向，为何修正缓慢”

### 4.1 Bellman 需求直接来自 SAC 的训练目标

对固定的转移样本集合 $\mathcal B=\{(s_\ell,a_\ell,r_\ell,s'_\ell,z_\ell)\}_{\ell=1}^n$，$z_\ell$ 表示终止标记。SAC 的两个 critic 共享 soft Bellman target：

$$
y_{t,\ell}
=c_r r_\ell+\gamma(1-z_\ell)
\left[
\min_{j=1,2}Q_{\bar\theta_{j,t}}(s'_\ell,a'_{t,\ell})
-\alpha_t\log\pi_{\phi_t}(a'_{t,\ell}\mid s'_\ell)
\right],
\qquad a'_{t,\ell}\sim\pi_{\phi_t}(\cdot\mid s'_\ell).
\tag{1}
$$

其中 $c_r$ 是 reward scale，$\bar\theta_j$ 是 target critic 参数。更新 online critic 时，$y_t$ 停止梯度。记第 $j$ 个 critic 在这组输入上的输出为 $q_{j,t}\in\mathbb R^n$，其即时 Bellman 需求为

$$
d_{j,t}=y_t-q_{j,t},\qquad
L_{Q_j}=\frac{1}{2n}\|q_{j,t}-y_t\|_2^2.
\tag{2}
$$

这使“需求”成为 SAC 实际拟合的对象：它包含奖励、折扣后的双 Q 自举值和熵项。两个 critic 共用 $y_t$，但输出、需求和学习核分别计算。公式对应 [SAC 原论文的算法推导](https://arxiv.org/html/1812.05905v2#S4)与[双 Q 实际算法](https://spinningup.openai.com/en/latest/algorithms/sac.html#key-equations)。本地实现的 MSE 不含 $1/2$；下文 SGD 代理把这一常数吸收进有效步长 $\eta$，不将它直接认作 Adam 的配置学习率。

### 4.2 学习谱刻画同一个 critic 对不同需求方向的响应

以下省略 critic 下标。令 $J=\partial q_\theta/\partial\theta\in\mathbb R^{n\times p}$ 为完整参数 Jacobian，包含偏置，定义未中心化学习核

$$
K=\frac{1}{n}JJ^\top
=U\operatorname{diag}(\lambda_1,\ldots,\lambda_n)U^\top,
\qquad \lambda_i\ge0.
\tag{3}
$$

对式 (2) 做一次全批量 SGD，在局部线性化下有 $\Delta q\simeq\eta Kd$，因而

$$
u_i^\top\Delta q\simeq\eta\lambda_i\,u_i^\top d.
$$

这就是强、弱学习方向的含义：同样的方向需求，在较小 $\lambda_i$ 上得到的局部输出修正较小。权重奇异值、feature rank 和 $K$ 的特征值是不同对象；全文的“学习谱”专指这里的 $K$ 谱。

在同一输入集合上，用 $D=[d^{(1)},\ldots,d^{(m)}]\ne0$ 保留多个需求探针，定义

$$
p_i=\frac{\|u_i^\top D\|_2^2}{\|D\|_F^2},
\qquad \sum_i p_i=1.
\tag{4}
$$

各列分别表征待考察的修正，不把多个 targets 平均成一个训练标签。已有入口分析采用八组确定性动作噪声构造 soft targets，减去同一入口输出后形成需求矩阵；它们是探针，不能当作独立 seeds 或无偏的目标期望。需求二阶矩保留均值，不做默认中心化。

### 4.3 命题 1：弱方向上的重要需求形成有限预算约束

**条件。** 固定输入和 target，使用冻结 Jacobian 的仿射 critic；在这组输入上做全批量 SGD。对多列需求，分别运行同一局部修正模型，再将结果合并记为残差矩阵 $R_H$，$R_0=D$。

于是 $R_{h+1}=(I-\eta K)R_h$，归一化剩余残差精确满足

$$
\mathcal R_H
=\frac{\|R_H\|_F^2}{\|D\|_F^2}
=\sum_i p_i(1-\eta\lambda_i)^{2H}.
\tag{5}
$$

在单调步长区间 $0\le\eta\lambda_i\le1$ 内，选取满足 $0\le\eta\tau\le1$ 的阈值 $\tau$，设弱方向 $\lambda_i\le\tau$ 承载需求质量 $m_\tau=\sum_{\lambda_i\le\tau}p_i$，则

$$
\mathcal R_H\ge m_\tau(1-\eta\tau)^{2H}.
\tag{6}
$$

若要求 $\mathcal R_H\le\delta$，其中 $0<\delta<m_\tau$ 且 $0<\eta\tau<1$，必要预算为

$$
H\ge
\left\lceil
\frac{\log(m_\tau/\delta)}{-2\log(1-\eta\tau)}
\right\rceil.
$$

**简短证明。** 在 $K$ 的正交特征基中，每个残差分量乘以 $1-\eta\lambda_i$，平方求和得到式 (5)。保留弱方向的非负项，并使用 $1-\eta\lambda_i\ge1-\eta\tau$，即得式 (6)；取对数得到预算界。零特征值方向上的初始需求在该模型中保持不变。

**这一步推进的认识。** 相同的谱可以因需求投影不同而具有不同修正难度；同一需求也可以因学习谱不同而具有不同修正难度。决定有限预算障碍的是二者的对应关系。这里的结论是局部 critic 修正结论，真实 SAC 的目标变化与优化器影响在第 6 节接入。

### 4.4 理论之后立即放入 B/C 的方向证据

使用相对特征值 $\tilde\lambda_i=\lambda_i/\bar\lambda$，$\bar\lambda=\operatorname{tr}(K)/n$。已有数据中，FT 的 B/C 入口需求落在 $\tilde\lambda_i<1$ 方向的能量占比分别为 **49.0% / 4.0%**。这里的 1 是各自平均特征值的参照，便于解释分布，不定义普适的失学阈值。

用加权经验分布或需求—谱散点展示这一结果：横轴表示方向响应强度，纵轴或点的权重表示任务在这些方向上的需求。至此，开篇的行为差异获得了具体的方向描述。

**下一问：** 弱方向需求占比仍然把“略弱”和“极弱”混在一起。怎样连续地度量同一任务受到的谱负担？

## 5. 第三环：让“归一化谱负担”具有明确的数学含义

### 5.1 指标同时保留需求方向与相对谱强度

在 $\bar\lambda>0$、$D\ne0$、$\varepsilon>0$ 时，定义

$$
\widetilde P(D,K)
=\sum_i\frac{p_i}{\tilde\lambda_i+\varepsilon}
=\frac{
\operatorname{tr}\!\left[D^\top(\widetilde K+\varepsilon I)^{-1}D\right]
}{\|D\|_F^2},
\qquad \widetilde K=K/\bar\lambda.
\tag{7}
$$

归一化分别去掉需求整体幅度与核整体尺度，使指标聚焦“这个需求与这套相对学习几何是否匹配”。因此，实验同步保留 $\|D\|_F$ 和 $\bar\lambda$，实际速度分析仍使用原始尺度。$D=0$ 表示没有当前修正需求；$\bar\lambda=0$ 表示局部核完全无响应，这两种情形单独报告，不计算上述归一化比值。

### 5.2 命题 2：谱负担等于统一响应尺度下的折扣残差能量面积

沿命题 1 的冻结目标模型，改用梯度流，并按平均核尺度计时：若原始梯度流满足 $dR/dt=-KR$，取 $s=\bar\lambda t$。于是

$$
\frac{dR(s)}{ds}=-\widetilde K R(s),\qquad R(0)=D.
$$

则

$$
\boxed{
\widetilde P(D,K)
=2\int_0^\infty e^{-2\varepsilon s}
\frac{\|R(s)\|_F^2}{\|D\|_F^2}\,ds.
}
\tag{8}
$$

**简短证明。** $R(s)=U\operatorname{diag}(e^{-\tilde\lambda_i s})U^\top D$，所以归一化残差为 $\sum_i p_i e^{-2\tilde\lambda_i s}$。逐项积分，$2\int_0^\infty e^{-2(\tilde\lambda_i+\varepsilon)s}ds=(\tilde\lambda_i+\varepsilon)^{-1}$。

这个结论赋予“负担”直接含义：**统一总体响应尺度后，归一化残差能量在局部修正过程中的折扣面积。** 少量需求若落在极弱方向，也可能贡献很大的面积。$\varepsilon$ 对应这个诊断量的积分折扣，并非 SAC 训练新增的正则项。

已有实测使用 $\varepsilon=10^{-3}$；附录报告 $10^{-4}$ 和 $10^{-2}$ 的敏感性。若 $\varepsilon\to0$ 且需求没有零空间分量，得到未折扣的逆谱面积。指标下降意味着这一局部面积下降，不将其换算成真实 Adam 的训练时长，也不要求所有有限时刻的残差都按同样比例降低。

### 5.3 紧接理论，展示“少量极弱需求为何重要”

已有 FT 的 B/C 入口归一化谱负担分别为 **108.80 / 2.25**。在 FT 的 B 入口，$\tilde\lambda_i<10^{-3}$ 的方向只承载约 **8.82%** 的需求能量，却贡献约 **58.3%** 的总负担。

这一组结果适合紧接式 (8)：它把“新任务需要弱方向”进一步细化成“某些修正需求在局部动力学中具有显著的长尾代价”。图中同时展示需求能量与累积负担，避免只给一个总分。

**下一问：** SAC 的目标会随 target critic、actor 和温度继续变化。上述方向障碍在这个持续更新的过程中如何发挥作用？

## 6. 第四环：把方向瓶颈接入 SAC 的持续自举与策略更新

### 6.1 从真实更新恒等式出发

在同一固定观察集合上，定义

$$
\Delta q_t=q_{t+1}-q_t,\qquad
v_t=y_{t+1}-y_t.
$$

无论使用什么优化器，只要前后在相同输入上计算，就恒有

$$
\boxed{d_{t+1}=d_t-\Delta q_t+v_t.}
\tag{9}
$$

它把实际训练拆成三个可测对象：已有 Bellman 需求、critic 已经完成的输出修正、更新后的目标带来的需求变化。固定输入与可复用的动作噪声让不同时间的测量可以比较；若 actor 改变，相同噪声经新策略得到的动作仍会改变，这部分进入 $v_t$。

SAC 的自举目标让这里的 $v_t$ 具有具体来源。固定 actor、温度、转移与同一 next-action 样本时，由双 critic 的 min 运算可得

$$
\|v_t\|_\infty
\le\gamma\max_{j=1,2}
\|\Delta Q_{\bar\theta_j,t}(s',a')\|_\infty.
\tag{10}
$$

证明仅需 $|\min(a_1,a_2)-\min(b_1,b_2)|\le\max_j|a_j-b_j|$。该式描述 target critic 变化的传播上界；正常 SAC 中 actor 和温度的变化也会进入 $v_t$。Target 的 Polyak 更新发生在参数空间，非线性 $Q$ 输出的变化直接计算。

### 6.2 命题 3：每个学习方向都在保留旧需求，并响应后续目标输入

为沿同一组方向追踪局部窗口，在窗口入口固定参考核 $K_0$ 及其特征基。定义真实输出更新与几何参照之间的差

$$
\xi_t=\Delta q_t-\eta K_0d_t.
$$

则式 (9) 精确改写为

$$
d_{t+1}=(I-\eta K_0)d_t+v_t-\xi_t.
\tag{11}
$$

记 $z_{i,t}=u_i^\top d_t$，$b_{i,t}=u_i^\top(v_t-\xi_t)$，便有

$$
z_{i,H}
=(1-\eta\lambda_i)^H z_{i,0}
+\sum_{t=0}^{H-1}(1-\eta\lambda_i)^{H-1-t}b_{i,t}.
\tag{12}
$$

**简短证明。** 将式 (11) 投影到固定的 $u_i$，得到一阶递推，逐步代入即可。由于 $\xi_t$ 按实际更新定义，式 (11)–(12) 本身是恒等式；把 $\xi_t$ 忽略后作为预测模型，则需要检验该窗口的局部近似是否成立。

**SAC 对 motivation 的补充。** 在 $0\le\eta\lambda_i\le1$ 的参照模型中，弱方向具有更慢的遗忘系数：既有需求保留更久，后续目标输入也通过更长的响应尾部作用于该方向。同号、持续的净输入会加强残差保留；不同符号的输入可以抵消，因此实际积累量由式 (9) 的观测决定。

**实现注。** 本地 SAC 使用 Adam。$\xi_t$ 汇总预条件与动量、训练样本和观察集合的差异、核漂移与非线性影响；$JJ^\top/n$ 提供几何参照，真实 $\Delta q_t$ 检验其解释能力。$\eta$ 在窗口前固定并记录。主文的动态证据始终围绕式 (9) 的需求、修正与目标变化展开。

### 6.3 这一命题需要什么实验

方向在窗口入口确定，并在整个窗口内固定，避免随结果重新选取。对同一需求方向或谱子空间，记录：

| 测量 | 具体对象 | 回答的问题 |
|---|---|---|
| 需求保留 | $u_i^\top d_t$，或固定子空间投影的范数 | 入口的重要需求是否长期没有消除？ |
| 实际修正 | $u_i^\top\Delta q_t$ | critic 是否沿所需方向产生了有效变化？ |
| 目标流入 | $u_i^\top v_t$ | 自举与策略变化是否继续向该方向输入需求？ |
| 局部模型解释力 | 投影后的 $\xi_t$ 与式 (9) 闭合关系 | 原始学习谱在多大程度上解释实际 Adam 更新？ |

使用有符号投影核对式 (9)，用能量展示强弱；不能把各项平方后当成可直接相加的能量分解。特征值接近时优先追踪子空间，避免单个特征向量的符号与基旋转影响解释。

先在冻结 target 的短窗口中隔离修正响应，再在正常 SAC 中记录目标流入。这里是后续方向动力学验证的设计；已有静态谱数据与在线曲线不改标为已经完成该验证。

### 6.4 为什么 critic 的方向修正与策略学习有关

SAC actor 最小化

$$
L_\pi(\phi)
=\mathbb E_{s,\omega}
\left[
\alpha\log\pi_\phi(f_\phi(s,\omega)\mid s)
-Q_{\min}(s,f_\phi(s,\omega))
\right],
\qquad Q_{\min}=\min(Q_1,Q_2).
\tag{13}
$$

其价值项的梯度包含
$-\mathbb E[(\partial f_\phi/\partial\phi)^\top\nabla_a Q_{\min}]$。
因此，critic 同时影响后续的自举目标和 actor 的更新依据：方向修正障碍有一条明确进入强化学习闭环的路径。

相应验证在固定 actor、状态和动作噪声下比较 critic 修正前后的价值项梯度，记录哪个 critic 提供 min 值，再衔接后续 reward / success。双 critic 分别计算几何与方向更新后再汇总。若讨论“策略梯度更准确”，还需要独立价值或动作导数参照；仅凭观察集合上的 Q 值残差不能得到动作导数误差界。

**承接干预的句子：** 前面已经确定要改变的环节——让同一 Bellman 需求在 critic 中获得更有利的响应。接下来检验一个直接作用于继承网络的谱调理方法。

## 7. 第五环：Clip 改变几何，再检验修正与新任务获取

### 7.1 干预从前面的机制自然导出

对双 online critic 的每个 Linear 权重，包含 Q 输出头，执行

$$
W=U_W\operatorname{diag}(\sigma_i)V_W^\top
\quad\longmapsto\quad
W_c=U_W\operatorname{diag}
\bigl(\operatorname{clip}(\sigma_i,\sigma_{\min},\sigma_{\max})\bigr)V_W^\top.
\tag{14}
$$

操作保留各权重矩阵的左右奇异向量、bias、actor 和 Adam 状态，改变权重奇异值。入口干预后 target 硬同步，周期干预后沿用常规 Polyak；阈值和触发口径随实际实验版本说明。当前实现见 [sac_singular_clip.py](garage/torch/algos/sac_singular_clip.py)。

权重谱说明方法直接做了什么；新任务输入上的 $K$ 谱和 $p_i$ 说明它怎样改变需求支撑。非线性 critic 中，权重谱裁剪可以同时改变学习核的特征值和特征基。是否降低 $\widetilde P$ 由同输入、同需求的测量回答，权重奇异值下界不直接等于学习核特征值下界。

Clip 运行时不读取未来任务的 Bellman 需求；需求分析用于解释和检验干预效果。

### 7.2 理论给出一个简单、可测的干预目标

前述命题将方法目标写成

$$
\widetilde P(D,K_c)<\widetilde P(D,K)
\quad\Longleftrightarrow\quad
\text{同一需求的归一化局部折扣残差能量面积降低}.
\tag{15}
$$

在受控的共同特征基下，若需求权重固定，则

$$
\widetilde P(D,K)-\widetilde P(D,K_c)
=\sum_i p_i
\frac{\tilde\lambda_i^{\,c}-\tilde\lambda_i}
{(\tilde\lambda_i+\varepsilon)(\tilde\lambda_i^{\,c}+\varepsilon)}.
$$

这个恒等式说明，改善应看需求加权的净作用：极弱方向上的相对支撑变化尤其重要，强方向的变化也必须计入。实际 FT/Clip 的特征基通常不同，直接使用式 (7) 各自重新投影，不将按大小排序的模态当成相同方向配对。

### 7.3 已有干预证据紧接这一目标

固定 B 的输入集合与 FT 的 B 入口需求，比较干预前后的学习几何：

| 固定需求对照 | FT | Clip | 相对降低 |
|---|---:|---:|---:|
| B 入口 $\widetilde P$ | 108.80 | 55.86 | 48.7% |
| B 结束时对同一入口需求的 $\widetilde P$ | 242.18 | 52.78 | 78.2% |
| B 末成功率 | 0% | 50% | 提高 50 个百分点 |

前两行使用同一 B 需求，末行来自实际在线评估。B 入口干预几何由共同父 checkpoint 上执行同一 Clip 重建；后续几何来自各自实际训练 checkpoint。

正文可以明确写：

> 在已有 ABC 案例中，Clip 将固定 Bellman 需求的入口归一化谱负担降低了 48.7%，并将受阻任务的末次成功率从 0% 提高到 50%。固定入口需求在后续训练中的负担也保持较低，表明该干预改变了与任务修正相关的学习几何。

图中用累积谱负担曲线展示下降来自哪些谱区间，再用整个 B 阶段的固定需求负担轨迹连接后续训练。以完整 reward / success 曲线呈现收益，不将末次 50% 改写成整个阶段稳定达到 50%。

### 7.4 将即时几何变化与真实 SAC 干预接起来

Clip 同时改变 critic 输出和入口 target。记函数位移 $j=q_c-q$，target 位移 $h=y_c-y$，则真实干预后的需求为

$$
d_c=d+h-j.
\tag{16}
$$

因此，验证按两个相接的层次组织：

1. **共同需求几何与响应。** 固定 $D$，测量 $\widetilde P(D,K)$ 与 $\widetilde P(D,K_c)$；短程修正实验若保持同一 correction，分别设置目标 $q_{\mathrm{init}}+d$，而非只固定同一个数值 target。各需求探针独立评估，优化器状态按实际方法保留。
2. **正常 SAC 的闭环作用。** 记录 $j,h$，再按式 (9) 跟踪实际需求、修正和目标变化，衔接 actor 更新及在线获取。

这两个层次共同回答“Clip 先改变了什么，这种变化如何进入后续学习”。Q-reset 作为同源 critic 干预参照，分别测量其几何和函数变化；匹配 fresh 提供目标任务的从头学习参照。所有比较使用各自明确的输入、需求与训练协议。

后续重点验证时序：谱负担变化是否伴随重要方向修正的改善，这种变化如何先于并关联任务获取。已有静态与在线结果承担已完成的证据，方向动力学和 actor 通道承担待补齐的机制测量。

## 8. Introduction 与正文章节按同一条逻辑组织

### 8.1 Introduction 的六段推进

**第一段：建立具体矛盾。** 持续 actor-critic 需要继承知识，也需要适应新任务。新任务的适应要求 critic 完成新的 Bellman 修正，而继承网络对不同修正方向的响应并不均衡。以“新任务所需的修正能否被及时完成”收束。

**第二段：用实验使问题具体。** 介绍 ABC 中 B 受阻、C 可学以及 Clip 改善 B 的现象。读者先看到实际任务获取差异，再进入机制问题。

**第三段：提出缺少的对应关系。** 整体 rank 或谱形状描述 critic，TD 残差幅度描述当前拟合差距；我们需要进一步知道这些需求落在什么学习方向上。自然引出 Bellman 需求—学习谱失配。

**第四段：用最简理论解释障碍。** 一句话交代局部方向响应、有限预算以及需求加权谱负担的残差能量面积含义；接着指出 SAC 自举持续产生目标变化，使 critic 必须不断消化方向需求。此处只讲逻辑，不在 Introduction 展开推导。

**第五段：引出方法与验证链。** 用 Clip 调理继承 critic，检验固定需求负担、实际方向修正和在线获取。给出已核验的代表性效果，再概述理论与实验交错的验证路线。

**第六段：凝练贡献。** 按以下顺序总结最终已交付的内容：任务相关的解释视角；从局部响应、谱负担到 SAC 动态的可测理论联系；同源干预与任务获取证据。贡献句的完成时态与实验范围对应实际完成的分析，不把后续计划写成既成结果。

可用的中心过渡段：

> 新任务带来的挑战可以具体到 critic 必须完成的 Bellman 修正。学习谱刻画这些修正获得多强的局部响应，需求投影决定哪些方向对当前任务重要。我们据此研究需求与弱学习方向的失配，并用归一化谱负担量化其局部修正代价。进一步将方向分析放回 SAC 的持续自举过程，便能同时追踪需求的产生、消除与重新流入。Clip 沿这一思路调理继承 critic，再通过固定需求几何、实际更新和任务获取检验干预效果。

### 8.2 正文章节安排

| 章节 | 内部推进顺序 | 本框架对应位置 |
|---|---|---|
| 1 Introduction | 矛盾 → 现象 → 需求视角 → 理论联系 → Clip → 贡献 | §1、§8.1 |
| 2 SAC and the Adaptation Problem | soft target、双 critic、actor、统一需求与学习核符号；相关方法定位 | §4.1–4.2 的必要定义 |
| 3 Where Does New-Task Adaptation Get Stuck? | ABC 现象 → 命题 1 → B/C 需求分布 → 命题 2 → 需求—谱代价证据 | §3–5 |
| 4 From Spectral Mismatch to SAC Dynamics | 实际残差恒等式 → 命题 3 → 方向更新检验 → target 与 actor 通道 | §6 |
| 5 Conditioning Bellman Adaptation with Clip | 操作 → 固定需求理论目标 → 入口几何 → 实际更新 → 在线收益 | §7 |
| 6 Evaluation of New-Task Acquisition and Retention | 匹配对照、跨 seed 获取、保留与计算成本；检验机制适用范围 | 实验计划中的相应结果 |
| 7 Discussion and Conclusion | 回到任务相关适应能力，概括已验证规律及适用条件 | 中心论点与完成证据 |

主文保留三个短命题，每个命题附近都有观察或检验。证明只用线性化、特征分解、积分和递推展开，不另设脱离 SAC 叙事的长理论章。

### 8.3 理论附录保留什么

- **命题 1–3 的完整证明与条件。** 包含零空间、步长区间、多列需求定义和固定方向追踪细节。
- **谱负担的修正成本解释。** 令 $A=J/\sqrt{n\bar\lambda}$，$\widehat D=D/\|D\|_F$，则
  $$
  \widetilde P
  =\min_Z\left\{\|Z\|_F^2+\varepsilon^{-1}\|AZ-\widehat D\|_F^2\right\}.
  $$
  正规方程给出 $Z^\star=A^\top(AA^\top+\varepsilon I)^{-1}\widehat D$，代回得到式 (7)。这补充“正则化线性修正成本”的解释，主文使用残差能量面积保持一条语言主线。
- **固定策略的 soft Bellman 传播。** 在有限状态动作、固定策略与温度、期望 target、无 target lag 的受控模型中，将熵项并入有效奖励，$A_B=I-\gamma P^\pi$，$e=q^\pi-q$，$d=A_Be$。冻结核半梯度更新给出 $e_{t+1}=(I-\eta KA_B)e_t$；可交换的对称情形用于精确模态核验，其他情形保留矩阵形式。用小型 MRP 检验 Bellman 传播与方向预算；它是补充理论实验，不替代真实 SAC 动态记录。
- **残差与价值的经典联系。** 固定策略下，由 soft Bellman 的 $\gamma$ 压缩性可得 $\|Q-Q^\pi\|_\infty\le\|\mathcal T^\pi Q-Q\|_\infty/(1-\gamma)$，条件为 $0\le\gamma<1$、相关价值函数与有效奖励有界、条件期望存在。它解释 Bellman 修正的价值评估意义；经验 bank 上的 sampled target 残差不等于这个全域范数。
- **权重干预与函数变化。** 简单线性 critic 的 $W\to K$ 构造、共同需求与函数位移的分解放在方法附录。主文使用式 (16) 连接实际 SAC，避免同时引入多套传播算子。

## 9. 图表按照问题递进，直接展示 motivation

目前优先使用真实数据的需求—谱代价图与累积分布。几何示意若不能使“方向响应、任务需求、修正代价”更容易理解，就不放在开篇占据主图空间。

| 图 | 读者首先应看到的关系 | 建议子图 |
|---|---|---|
| Fig. 1：动机总览 | 相似的继承问题为何产生不同任务适应；Clip 改变了什么 | 连续 reward；FT-B / FT-C / Clip-B 的需求—谱代价图，配必要成功率标记 |
| Fig. 2：从方向到负担 | 需求质量与谱位置共同决定局部修正代价 | 命题 1 的受控方向响应；B/C 需求加权分布；累积谱负担与命题 2 面积解释 |
| Fig. 3：SAC 的方向动力学 | 重要需求是否被及时修正，target 又输入了什么 | 同一窗口、同一预设方向的需求、实际 $\Delta q$、$\Delta y$；局部参照误差 |
| Fig. 4：Clip 的干预链 | 同一需求几何如何变化，并怎样关联后续学习 | 入口配对负担；固定需求轨迹；方向修正；完整 reward / success |
| Fig. 5 / 主表：方法价值 | 获取、保留和成本 | 正式匹配协议下的跨 seed 结果与方法比较 |

Fig. 1 已有可继续使用的 PNG：

- [需求—谱代价图与连续 reward](bellman_probe_results/abc_motivation_analysis_20260910/demand_burden_story/10_motivation_demand_spectrum_map.png)。
- [需求分布与累积负担版](bellman_probe_results/abc_motivation_analysis_20260910/demand_burden_story/09_motivation_demand_burden.png)。

代价图横轴为 $\tilde\lambda_i$，纵轴为每个 critic 的 $p_i$，背景等值色场由 $p_i/(\tilde\lambda_i+\varepsilon)$ 解析计算；点来自实际谱分解。背景表达指标代价，不标作实测 loss landscape。FT/Clip 使用同一 B 输入与需求，但分别投影到各自特征基；不同模型之间不按特征值排序连成“同一个方向的迁移箭头”。

所有实测图保留图例、坐标与任务边界。Reward 使用相同的任务内 51 点居中滑动平均，不跨任务边界混合；横轴明确为 optimizer updates。谱分布与累积负担不做平滑。图面只保留标题、符号和必要数值，机制解释由正文与图注承接。后续图稿按上表补齐，未完成的动态面板不使用示意数值替代实验数据。

## 10. 证据、实现与写作落点

### 10.1 已有证据与后续验证各自承担什么

| 状态 | 材料 | 可以承担的结论 |
|---|---|---|
| 已有 | ABC 完整 reward / success、B/C 入口谱与需求 | 展示任务适应差异及任务相关的谱负担 |
| 已有 | 共同 B 需求的 FT/Clip 入口与后续几何 | Clip 降低固定需求的归一化谱负担 |
| 已有 | B 的 FT/Clip 在线结果 | 本案例中受阻任务的学习得到改善 |
| 本版已给出推导 | 三个主文命题及证明 | 局部方向预算、负担面积含义、SAC 需求更新分解 |
| 待验证 | 固定方向的需求、真实输出更新与 target 流入 | 把谱几何与实际 SAC 修正过程接起来 |
| 待汇总或补齐 | 同源干预、匹配 fresh、跨 seed 效果与 actor 通道 | 检验机制对应关系及获取收益的适用范围 |

### 10.2 复现与引用入口

- [ABC 学习谱与需求分析](bellman_probe_results/abc_spectrum_20260910.qJwo3H/ABC_SPECTRAL_CONCLUSIONS.md)。
- [固定 B 需求的谱负担轨迹](bellman_probe_results/abc_motivation_analysis_20260910/fixed_B_demand_trajectory.csv)与[入口比较](bellman_probe_results/abc_motivation_analysis_20260910/entry_comparison.csv)。
- [需求分布与负担图的数据说明](bellman_probe_results/abc_motivation_analysis_20260910/demand_burden_story/figure_notes.json)，包含弱谱区间统计、归一化与聚合口径。
- [原始 reward / success 及实验身份](bellman_probe_results/abc_ft_clip_curves_20260910/README.md)。
- [SAC 本地实现](garage/torch/algos/sac.py)与[入口几何重建](bellman_probe_results/abc_motivation_analysis_20260910/analyze.py)。
- [历史 Clip 源码快照](bellman_probe_results/sac_singular_clip_task1_20260906/source_snapshot/garage/torch/algos/sac_singular_clip.py)。历史 ABC 按 critic updates 调度周期 Clip，当前实现按环境步调度；历史入口逻辑按快照解释，不用当前配置反写历史结果。
- [SAC 算法与 soft policy iteration 原始文献](https://arxiv.org/abs/1812.05905)。已有持续 RL 与谱干预工作的来源及增量在相关工作中逐项核对。

最终论文应让读者沿着同一问题走完全过程：**新任务要求什么修正 → critic 为什么难以完成这些修正 → 这种障碍如何进入 SAC 的持续学习 → Clip 改变了其中哪个环节 → 实际任务获取怎样改善。** 每一段理论都导向可观察量，每一组实验都回答前文提出的问题。
