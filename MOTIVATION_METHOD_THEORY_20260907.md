# Motivation 与 Method 的联合理论：Bellman 需求、有限预算和干预代价

2026-09-07，实验映射已按用户最新范围修订。配套 [PAPER_OUTLINE_20260907.md](PAPER_OUTLINE_20260907.md)：五条主序列、每个 RL 任务 1.5M 环境步、seeds 1/2/3；rethink 仅 FT/Reset/Clip（ours）。旧大规模方案仅作历史参考。下面保留假设、命题与证明；是研究草稿，不是已证明真实 SAC 全局收敛或已获得新的 RL 实验结果。Motivation 解释问题如何出现，Method 对同一个误差对象给出干预收益条件。

## 1. 统一研究对象与假设

先考虑有限状态、固定策略的 discounted Markov reward process，转移矩阵 P、奖励向量 r、0≤γ<1。记

\[
A=I-\gamma P,\qquad q^\star=A^{-1}r,\qquad
e_t=q^\star-q_t,\qquad d_t=r-Aq_t=Ae_t.
\]

e 是完整价值修正，d 是即时 Bellman residual；二者不能混称同一个 demand。为保持符号清楚，本说明中的 D 若使用多列形式，表示若干独立的完整价值修正，而不是将多个目标同时平均训练。

理想学习模型为 q_{t+1}=q_t+ηKd_t，其中 K≽0 为固定学习核。它可由线性 critic 或在一个 checkpoint 处冻结 Jacobian 的 affine surrogate 实现。采用平方损失1/(2n)时，K=JJᵀ/n；也可把1/n吸收进η。不可将centered或除参数数目的诊断kernel直接代入速度公式。

因此

\[
e_{t+1}=(I-\eta KA)e_t,\qquad d_{t+1}=(I-\eta AK)d_t.
\tag{1}
\]

式(1)允许Bellman target随当前q自举更新；它并不要求把target冻结为常数。A=I对应固定target回归。这里不包含target lag、策略更新、双Q取min、动态Adam或minibatch噪声。

若P对称随机，A正定。若K与P还具有共同正交特征基u_i，定义a_i=1−γp_i>0、β_i=κ_i a_i，则

\[
u_i^\top e_H=(1-\eta\beta_i)^H u_i^\top e_0.
\tag{2}
\]

有限H的恒等式不要求稳定；讨论收敛须要求受激模态0<ηβ_i<2。进一步声称“提高β使学习更快”时，使用单调区间0≤ηβ_i≤1，不能仅依赖稳定性。

## 2. Motivation 理论

### M1：有限预算的几何性负迁移条件

比较继承几何K_T和fresh几何K_F，先匹配同一任务、同一初始函数q_0，即同一e_0。一般矩阵形式定义

\[
E_H(K,e)=\|(I-\eta KA)^H e\|_2^2.
\]

若P、K_T、K_F具有共同正交特征基，则

\[
G_H^{T-F}
=E_H(K_T,e_0)-E_H(K_F,e_0)
=\sum_i |u_i^\top e_0|^2
\left[(1-\eta\kappa_i^T a_i)^{2H}
-(1-\eta\kappa_i^F a_i)^{2H}\right].
\tag{M1}
\]

证明：将两组式(2)平方求和后相减。G_H^{T-F}>0表示在该受控模型中，继承几何造成更多有限步价值误差。它不是关于在线success或return的负迁移定理。

若实际初始化也不同，必须使用E_H(K_T,e_T)−E_H(K_F,e_F)。可分解为

\[
\underbrace{E_H(K_T,e_T)-E_H(K_F,e_T)}_{\text{匹配需求后的几何差异}}
+\underbrace{E_H(K_F,e_T)-E_H(K_F,e_F)}_{\text{该参照下的初始化差异}}.
\tag{3}
\]

分解依赖选择哪个核作参照，不是唯一因果分解。它解释为什么实际transfer/reset比较必须同时看初始函数与学习动力学。

**同谱但迁移符号相反的构造。** 取P=.5[[1,1],[1,1]]、γ=.9、u_±=(1,±1)/√2；A的两模态为(.1,1)。令K_T=.01u_+u_+ᵀ+u_-u_-ᵀ，K_F=u_+u_+ᵀ+.01u_-u_-ᵀ。两核的全谱、rank、trace、condition number完全相同。η=1时，需求e_0=u_+使T/F误差分别为.999^(2H)/.9^(2H)；换成e_0=u_-，误差为0/.99^(2H)，符号反转。该构造证明结构摘要甚至未配对的全谱不足以确定适应效果，不证明真实源训练必产生这些核。

### M2：Bellman 传播重加权需求，并给出学习预算下界

由于e_0=A⁻¹d_0，在共同特征基下，归一化价值误差精确为

\[
\mathcal R_H
=\frac{\sum_i |u_i^\top d_0|^2 a_i^{-2}
 (1-\eta\kappa_i a_i)^{2H}}
 {\sum_i |u_i^\top d_0|^2 a_i^{-2}}.
\tag{M2}
\]

Bellman传播同时影响两个量：a_i⁻²改变完整修正的方向能量，κ_i a_i改变学习时间尺度。从旧任务精确解出发且只改变奖励时，d_0=Δr、e_0=Δq*=A⁻¹Δr；改变P或策略时不能直接沿用d_0=Δr。

令慢模态集合S_b={i:β_i≤b}，完整需求在其中的质量为m_b=Σ_{i∈S_b}|u_iᵀe_0|²/||e_0||²。在全部模态0≤ηβ_i≤1、0<ηb<1时，

\[
\mathcal R_H\ge m_b(1-\eta b)^{2H}.
\]

若要求R_H≤ε<m_b，必要更新预算满足

\[
H\ge\left\lceil
\frac{\log(m_b/\epsilon)}{-2\log(1-\eta b)}
\right\rceil.
\tag{4}
\]

证明：慢方向的收缩因子不小于1−ηb，保留这些非负误差项即可得到下界，再取对数。b=0时有不可消除的零空间误差质量m_0。

若继承核有慢方向质量m_b，而fresh在需求支持上均有β_i^F≥c>b，且两组都处于单调区间，则G_H^{T-F}/||e_0||²≥m_b(1−ηb)^(2H)−(1−ηc)^(2H)。右侧为正就给出一个明确的有限预算劣化充分条件。

**理论的motivation含义：** 不把“critic整体坏了”作为解释终点，而是问完整Bellman修正有多少能量落在当前预算消化不了的联合慢模态上。动态需求实验继续检验后续targets是否不断产生这种需求，不预设答案。

### 非交换与真实 SAC 的边界

若P对称但K不与P交换，令B=A^(1/2)KA^(1/2)，则z=A^(1/2)e满足z_{t+1}=(I−ηB)z_t。可在||e||_A²=eᵀAe中得到谱公式，不能直接把它当成普通欧氏Q误差。非对称P使用完整矩阵幂及非正规瞬态分析。

真实SAC在固定anchors上恒有d_{t+1}=d_t+u_t−J_tΔθ_t−r_t，其中u_t=y_{t+1}−y_t、r_t=q_{t+1}−q_t−J_tΔθ_t。若以ηK_td_t为参照，另定义m_t=J_tΔθ_t−ηK_td_t，则

\[
d_{t+1}=(I-\eta K_t)d_t+u_t-m_t-r_t.
\tag{5}
\]

m_t不能遗漏：它包含Adam、minibatch和训练样本与anchors不一致。r_t仅表示非线性余项，不能把所有差异都归给它。

## 3. Method 理论

### S1：干预的有限预算收益等于几何效应加函数变化效应

干预c把初始预测从q_0改为q_c=q_0+j_c，得到自己的局部核K_c。比较时固定同一A、r、η，并在各副本上分别冻结自己的Jacobian。令

\[
T_0=(I-\eta K_0A)^H,\quad T_c=(I-\eta K_cA)^H,
\quad e=q^\star-q_0.
\]

干预后H步的精确surrogate误差为E_c=||T_c(e−j_c)||²。于是

\[
E_0-E_c
=\underbrace{\|T_0e\|^2-\|T_ce\|^2}_{G_{\mathrm{geom}}}
+2\langle T_ce,T_cj_c\rangle-\|T_cj_c\|^2.
\tag{S1}
\]

证明：展开平方。该恒等式不要求K_0、K_c与A交换。

一个不依赖交叉项符号的充分条件是

\[
\boxed{\ \|T_0e\|>\|T_ce\|+\|T_cj_c\|\ }
\quad\Longrightarrow\quad E_c<E_0.
\tag{6}
\]

等价地，G_geom>2ab+b²，其中a=||T_ce||、b=||T_cj_c||。若仅知道||j_c||≤J_c，可用||T_c||₂J_c替代b，得到更保守的充分条件。

**方法的含义：** 在同一个有限预算误差上，几何加速带来的收益足以覆盖干预函数变化的最坏代价时，才获得改进保证。真实交叉项也可能为正，因此“函数冲击”并非总有害；实际收益不必满足这个保守充分条件。

Reset必须用自己的K_reset和j_reset比较：E_reset=||T_reset(e−j_reset)||²。不能只说clip位移较小就断言优于reset，也不能比较两个宽松上界便推导实际排序。

### S2：实际 weight clip 如何同时改变 K 和 Q

取两层无偏置critic Q(x)=vᵀWx，W为n×n、anchors为标准基，两层都可训练。记W=U diag(s_i)Vᵀ，s̄_i=clip_[ℓ,h](s_i)。若v≠0，输出行向量裁剪后v_c=c_vv，其中c_v=clip_[ℓ,h](||v||)/||v||。直接计算得到

\[
K_0=\|v\|^2I+W^\top W,\quad
K_c=c_v^2\|v\|^2I+V\operatorname{diag}(\bar s_i^2)V^\top,
\tag{S2a}
\]

以及实际函数变化

\[
j_c=W_c^\top v_c-W^\top v
=V\operatorname{diag}(c_v\bar s_i-s_i)U^\top v.
\tag{S2b}
\]

证明：每个Q(e_i)对v的梯度为We_i，对W的梯度为ve_iᵀ，故Jacobian Gram为上述两项之和；输出变化由矩阵乘积直接相减。v=0时直接使用j_c=W_cᵀv_c−Wᵀv，不使用未定义的c_v。

式(S2)精确描述干预瞬间，不代表两层继续真实训练时K保持不变。将它代入S1得到的是各自冻结Jacobian的有限步模型；真实训练由R4/R5检验。

若A、K_0、K_c共享特征基，则G_geom=Σ_i[(1−ηa_iκ_0i)^(2H)−(1−ηa_iκ_ci)^(2H)]e_i²。对每个受激方向满足|1−ηa_iκ_ci|≤|1−ηa_iκ_0i|可保证几何项非负；weight condition number下降不能替代该条件。

**明确有利的例子。** W=diag(1,s)、v=(1,0)ᵀ、0<s<ℓ<1<h。clip只将s提升到ℓ，j_c=0，K_0=diag(2,1+s²)、K_c=diag(2,1+ℓ²)。当A=aI、0<ηa≤1/2、需求沿第二方向时，每个H≥1都严格改善。

**明确有害的例子。** 若当前已满足q_0=q*，但j_c≠0且T_cj_c≠0，则无干预误差为0，clip后的误差严格为正。这一阴性条件必须进入R1/R4，而不是只展示有利情况。

### 双侧裁剪与稳定跟踪的作用条件

在上述方阵、单位基、双层同时clip的模型中，ℓ>0给出2ℓ²I≼K_c≼2h²I。若P对称，a_minI≼A≼a_maxI，则联合对称算子A^(1/2)K_cA^(1/2)的谱位于[2ℓ²a_min,2h²a_max]。

当0<η<1/(h²a_max)时，在A范数下有收缩上界c=max{|1−2ηℓ²a_min|,|1−2ηh²a_max|}<1。下界避免该理想模型中的零速模态，上界与步长共同控制离散稳定性。但更小condition number不必得到更小c，更不保证已经学习的函数不受损。

若一段跟踪过程可写成e_{t+1}=S_te_t+z_t，并且||S_t||_A≤c<1，则

\[
\|e_H\|_A\le c^H\|e_0\|_A+
\sum_{t=0}^{H-1}c^{H-1-t}\|z_t\|_A.
\tag{7}
\]

z_t包括新增完整需求、干预造成的预测跳变以及模型/optimizer误差。该界说明更好的收缩需要与这些输入项一起评价，不能以裁剪次数推导最优频率。ReLU gate、输入秩、矩形层及Adam会破坏由weight floor直接推出full-J floor的论证，真实SAC不享有上述无条件下界。

### 深层网络中可以成立的函数冲击上界

对各层1-Lipschitz激活（包括ReLU和线性输出）、偏置保持不变的网络，记原激活h_l(x)。逐层三角不等式给出

\[
|Q_c(x)-Q_0(x)|\le
\sum_{l=1}^L
\left(\prod_{k=l+1}^L\|W_{k,c}\|_2\right)
\|W_{l,c}-W_l\|_2\,\|h_{l-1}(x)\|_2.
\tag{8}
\]

对于SVD clip，||W_{l,c}−W_l||₂=max_i|s̄_li−s_li|。式(8)无需固定ReLU gate；将各anchor的点态界平方求和可给出||j_c||界。但实测ΔQ通常更紧，保持U/V也不等于保持预测或旧知识。

SingularClip是给定奇异值区间内的Frobenius最近投影，这是已有算法的性质，不是本论文新定理；最近参数投影也不等于最小输出冲击或最优Bellman适应。

## 4. 两个附录级联系，不作为未经验证的主贡献

### 未知未来需求：为什么不读取具体 D 的干预也可能有意义

在几何风险中令F_H=TᵀT、C=E[eeᵀ]且trC=1。给定污染模型C=(1−ε)Ĉ+εC_unknown，C_unknown≽0、trC_unknown=1，有

\[
\sup_C\operatorname{tr}(CF_H)
=(1-\epsilon)\operatorname{tr}(\widehat C F_H)
+\epsilon\lambda_{\max}(F_H).
\tag{9}
\]

证明：把未知单位能量集中到F_H的最大特征方向可达上界。它解释当前需求适应和未知方向保障的区别，但仅描述几何风险，不含S1中的函数跳变及其交叉项，也不证明weight clip最小化了该目标。C在此为二阶矩；若讨论非零均值下的完整干预风险，不能遗漏均值与j_c的交叉项。

### Critic 与 policy：一个有明确额外条件的桥梁

对有限动作、固定α>0的熵正则MDP，假设全状态动作域上||Q̂−Q*α||∞≤ε，且实际actor满足α KL(π(·|s)||softmax(Q̂(s,·)/α))≤δ。则

\[
\|V_\alpha^\star-V_\alpha^\pi\|_\infty
\le\frac{2\epsilon+\delta}{1-\gamma}.
\tag{10}
\]

证明：log-sum-exp对∞范数1-Lipschitz，近似Q和真实Q分别产生最多ε差异，actor soft-greedy defect产生δ，再用policy Bellman contraction累积。该式说明还需要value误差覆盖与actor跟随条件。

anchor MSE下降不提供全域ε；SAC的动态α和连续动作还需额外条件；熵正则无限时域值不等于实验中的未正则episode return或success。降低上界也不能直接推出实际return严格上升，因此M1–M4/R5的在线检验仍不可省略。

## 5. 理论、实验与叙事的一一对应

| 理论对象 | 回答的motivation/method问题 | 当前缩减范围内的验证 |
|---|---|---|
| M1与同谱反例 | 为什么相同谱摘要不能决定某个新任务的适应？ | 新骨架 R1 同源 6×6 矩阵；R2 精确 MRP；R3 同 correction 拟合 |
| M2与预算下界 | 哪些Bellman修正会在给定预算内形成瓶颈？ | R2 等Δr/等Δq*；R3 共同 target 流 |
| S1收益分解 | 几何改善能否覆盖clip引起的函数变化？ | R4 同时测 K/ΔQ/实际拟合，并复用 R1 的 FT/Reset/Clip 三组 B 获取曲线 |
| S2权重到K及Q | 实际clip修改了什么，而不只是一个谱指标？ | 两层模型精确检查；三方法干预前后测量。不再要求 head/hidden、范数或随机扰动在线消融 |
| 跟踪界与式(9) | 为什么双侧控制和周期干预可能有效，也可能失败？ | R3 的动态需求及固定方法事件；本轮不做调度/阈值/UTD 在线扫描，不能宣称其最优 |
| 式(10)的限制 | 为什么critic拟合改善不能直接等于任务学会？ | R4 拟合与后续获取对应、五条主流的获取/保留；本轮没有 actor 因子或保留型集成组 |

论文主文建议以M1/M2合并的“Bellman需求条件适应命题”和S1的“干预收益命题”为两个核心结果；S2为连接实际算法的机制引理，其余放附录。命题数量不作为理论强度指标。

最终故事是：**Bellman需求在继承学习几何中形成有限预算瓶颈 → 谱干预改变这种响应，也会改变当前函数 → 只有几何收益与函数变化共同有利时，才应期待改善 → 实验检验这些条件是否出现在真实持续控制中。**

## 6. 归属、证据状态与本次工作范围

固定核谱递推、Bellman resolvent、范数扰动界和soft-policy误差界是标准数学工具。这里的推导是将其组织成可检验的研究理论，不声称开创新的通用收敛理论。

直接近邻：[Spectral Collapse](https://arxiv.org/html/2509.22335v3)已有快慢residual谱分析；[Supervision Complexity](https://arxiv.org/html/2301.12245v1)已有inverse-kernel需求量；[SingularClip](https://arxiv.org/html/2608.18319v1)提供当前裁剪算子及其理论；soft-control背景见[SAC](https://arxiv.org/abs/1801.01290)。式(10)是此处在明确假设下推导的辅助界，不冒充该SAC论文的逐字定理。

当前状态：上述恒等式、界和构造沿用此前的独立代数审查；真实 SAC 中的条件是否满足仍待新骨架的三方法机制实验。本次仅同步文档、范围与实验映射，未修改方法实现或启动训练。论文的场景增量定位为 Bellman 传播下的有限预算适应、继承 Q/K 的作用分离和干预—适应证据，不以重命名已有裁剪算子制造新颖性。tech-paper-template 用于使两条理论链与当前可执行范围对应。

数值核验：20组随机两层模型的Jacobian Gram和实际clip函数变化均符合S2；60组有限H干预收益恒等式的最大绝对误差为2.84e-14。联合谱上下界、两状态相反迁移符号构造、零函数冲击且严格加速的构造均通过检查。这些检查验证代数实现，不替代证明或RL实验。
