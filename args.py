import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Continual RL with Continual World / DMLab')

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument('--task_seq_idx', type=int, default=None, nargs='+', metavar='N',
                        help='task sequence index for CL (default: 0)')
    
    task_group.add_argument('--long_task_seq', type=str, default=None, metavar='N',
                            choices=['hard', 'easy', 'rand'],
                        help='task sequence index for CL (default: 0)')
    
    parser.add_argument('--env_type', default='', required=True, type=str,
                        choices=['metaworld', 'dm_control'], 
                        help='Environment for CL')
    parser.add_argument('--rl_method', default='', required=True, type=str,
                        choices=['ppo', 'sac',], 
                        help='Environment for CL')
    parser.add_argument('--cl_method', default='', type=str, required=True,
                        choices=['bc', 
                                 'ewc',
                                 'finetuning',
                                 'rnd', 
                                 'pandc',
                                 'spectral',],
                        help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='seed (default: 0)')
    parser.add_argument('--sac_optimizer', choices=['adam', 'muon'], default='adam',
                        help='普通 SAC 的优化器；Muon 的输出头、偏置和温度仍使用 Adam')
    parser.add_argument('--muon_lr', type=float, default=3e-4,
                        help='采用 match_rms_adamw 缩放的 Muon 基础学习率')
    parser.add_argument('--muon_momentum', type=float, default=.95)
    parser.add_argument('--muon_ns_steps', type=int, default=5)
    parser.add_argument('--sac_singular_clip', type=str2bool, default=False,
                        help='普通 Adam SAC 的 critic 双边权重谱裁剪')
    parser.add_argument('--singular_clip_min', type=float, default=.25,
                        help='奇异值裁剪下界（默认 %(default)s）；c=16 时设为 0.0625')
    parser.add_argument('--singular_clip_max', type=float, default=4.,
                        help='奇异值裁剪上界（默认 %(default)s）；c=16 时设为 16')
    parser.add_argument('--singular_clip_interval', type=int, default=200000,
                        help='任务内周期裁剪间隔（实际环境步，包含 warm-up）')
    parser.add_argument('--singular_clip_start_task', type=int, default=1,
                        help='零基起始任务位置；该任务及所有后续任务均在入口立即裁剪，默认跳过 A')
    parser.add_argument('--spectral_actor_coef', type=float, default=1e-4,
                        help='SpectralReg actor coefficient (paper default: 1e-4)')
    parser.add_argument('--spectral_critic_coef', type=float, default=1e-4,
                        help='SpectralReg coefficient for each critic (paper default: 1e-4)')
    parser.add_argument('--spectral_power_iterations', type=int, default=1,
                        help='Power iterations per SAC update (paper default: 1)')
    
    parser.add_argument('--wandb', type=str2bool, default=True, metavar='N',
                        help='Use wandb')
    
    parser.add_argument('--cl_reg_coef', type=float, default=1.0, metavar='N',
                        help='regularization strenghth for cl methods (default: 1.0)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='N',
                        help='total number of tasks (default: 10)')
    parser.add_argument('--steps_per_task', type=int, default=int(3e6), metavar='N',
                        help='steps per task (default: int(3e6))')
    parser.add_argument('--exact_sac_task_budget', type=str2bool, default=False,
                        help='Count exactly steps_per_task environment interactions per task, including replay warm-up')
    parser.add_argument('--nepochs_offline', type=int, default=int(5), metavar='N',
                        help='number of epochs for offline training (default: int(100))')

    parser.add_argument('--num_evaluation_steps', type=int, default=int(1e5), metavar='N',
                        help='the number of steps on the interval between evaluation (default: int(1e5))')
    parser.add_argument('--num_evaluation_episodes', type=int, default=50, metavar='N',
                        help='the number of episodes for evaluation (default: 50)')
    parser.add_argument('--expert_buffer_size', type=int, default=10000, metavar='N',
                        help='size of expert buffer for BC (default: 10000)')
    
    # Arguments for PnC
    parser.add_argument('--compress_step', default=int(1e6), type=int, help="(default: 1e6)")
    parser.add_argument('--use_pandc_bc', type=str2bool, default=False, metavar='N',
                        help='Use BC in P&C')
    parser.add_argument('--reset_column', type=str2bool, default=False, metavar='N',
                        help='Reset column in P&C')
    parser.add_argument('--reset_adaptor', type=str2bool, default=False, metavar='N',
                        help='Reset adaptor in P&C')
    parser.add_argument('--zero_alpha', type=str2bool, default=False, metavar='N',
                        help='Set alpha to zero in P&C adaptor')
    
    
    parser.add_argument('--use_exploration', type=str2bool, default=False, metavar='N',
                        help='Use exploration technique')
    parser.add_argument('--q_reset', type=str2bool, default='False', metavar='N',
                        help='Reset Q function')
    parser.add_argument('--policy_reset', type=str2bool, default='False', metavar='N',
                        help='Reset policy')
    parser.add_argument('--first_task', type=str, default=None, metavar='N',
                        help='Name of first task')
    parser.add_argument('--first_task_steps', type=int, default=int(3e6),
                        help='Training steps encoded in PPO warm-start model names')
    parser.add_argument('--bc_kl', default='reverse', type=str,
                        choices=['forward', 'reverse'], 
                        help='The direction of the behaviour cloning loss')
    parser.add_argument('--distill_kl', default='forward', type=str,
                        choices=['forward', 'reverse'], 
                        help='The direction of the distillation loss')
    parser.add_argument('--device_type', default=None, type=int,
                        choices=[None, 0, 1, 2, 3, 4, 5, 6, 7], 
                        help='None: use cpu, 0~3: use GPU with device number #')
    parser.add_argument('--proc_name', default='Negative-Transfer-Test', type=str, required=True,
                        help='(default=%(default)s)')
    parser.add_argument('--replay_buffer_size', default=int(1e6), type=int,
                        help="R&D buffer size")
    parser.add_argument('--replay_k', default=5, type=int,
                        help='Number of HER transitions to add for each regular transition')
    parser.add_argument('--infer', default=False, type=str2bool,
                        help="Use InFeR loss if True")
    parser.add_argument('--ReDo', default=False, type=str2bool,
                        help="Use ReDo if True")
    parser.add_argument('--redo_interval', default=1000, type=int,
                        help='Environment-step interval for ReDo recycling')
    parser.add_argument('--redo_tau', default=0.1, type=float,
                        help='Normalized activation-score threshold for ReDo')
    parser.add_argument('--reset_offline_actor', default=False, type=str2bool,
                        help="Reset the offline actor in R&D")
    parser.add_argument('--rd_teacher_steps', default=int(3e6), type=int,
                        help='Training steps used by the single-task R&D teachers')
    parser.add_argument('--rd_teacher_root', default='.', type=str,
                        help='Root containing models/sac_models and rollouts/sac_rollouts for R&D teachers')
    parser.add_argument('--save_single_task_artifacts', default=False,
                        type=str2bool,
                        help='Save model/buffer/rollout even when probes are enabled')
    parser.add_argument('--wasserstein', default=0, type=float,
                        help="Use Wasserstein loss if positive")
    
    parser.add_argument('--fixed_alpha', default=None, type=float,
                        help='SAC alpha tuning')
    parser.add_argument('--lr_clip_range', default=0.2, type=float,
                        help='PPO likelihood-ratio clip range')
    parser.add_argument('--ppo_gae_lambda', default=0.95, type=float,
                        help='GAE lambda used by PPO')


    parser.add_argument('--crelu', type=str2bool, default='False', metavar='N',
                        help='Use CReLU')
    
    parser.add_argument('--no_stats', type=str2bool, default=True, metavar='N',
                        help='Do not store the statistics')
    parser.add_argument('--scalar_log_interval', type=int, default=1000,
                        help='Environment steps between loss/reward/alpha/speed/zero-ratio logs')
    parser.add_argument('--feature_stats_interval', type=int, default=10000,
                        help='Environment steps between feature-rank and weight-change logs')
    parser.add_argument('--hessian_stats_interval', type=int, default=10000,
                        help='Environment steps between Hessian-rank logs')

    parser.add_argument('--bellman_probe', type=str2bool, default=False,
                        help='Measure critic feature/Bellman subspace alignment')
    parser.add_argument('--bellman_probe_size', type=int, default=1024,
                        help='Fixed transition probes retained per task')
    parser.add_argument('--bellman_probe_interval', type=int, default=100000,
                        help='Environment steps between Bellman probes')
    parser.add_argument('--bellman_probe_targets', type=int, default=8,
                        help='Stochastic soft Bellman targets per probe')
    parser.add_argument('--bellman_probe_ridge', type=float, default=1e-3,
                        help='Relative ridge used for Bellman coverage')
    parser.add_argument('--bellman_probe_dir', type=str,
                        default='bellman_probe_results',
                        help='Root directory for probe metrics and checkpoints')
    parser.add_argument('--bellman_spectral_stats', type=str2bool,
                        default=False,
                        help='Measure full-Jacobian Bellman spectrum and fitting speed')
    parser.add_argument('--bellman_spectral_anchor_size', type=int, default=64,
                        help='Transitions used by each full-Jacobian measurement')
    parser.add_argument('--bellman_spectral_fit_lr', type=float, default=3e-4,
                        help='Fresh-Adam learning rate for cloned critic fitting')
    parser.add_argument('--bellman_reference_dir', type=str, default=None,
                        help='Optional directory of fixed per-task Bellman references')
    parser.add_argument('--bellman_spectral_task_steps', type=int, nargs='+',
                        default=[10000, 50000, 100000, 500000, 1000000,
                                 1500000],
                        help='Task-local steps for full Bellman spectral statistics')
    parser.add_argument('--pbsr', type=str2bool, default=False,
                        help='Train SAC critics with PBSR-head')
    parser.add_argument('--pbsr_coef', type=float, default=0.1,
                        help='Target PBSR-to-TD critic gradient-norm ratio')
    parser.add_argument('--pbsr_anchor_size', type=int, default=64,
                        help='Current-task transitions per PBSR update')
    parser.add_argument('--pbsr_targets', type=int, default=8,
                        help='Soft-Bellman directions per PBSR update')
    parser.add_argument('--pbsr_ridge', type=float, default=1e-3,
                        help='Ridge on the trace-normalized PBSR head kernel')
    parser.add_argument('--pbsr_update_interval', type=int, default=100,
                        help='Critic updates between PBSR regularization steps')
    parser.add_argument('--pbsr_train_task_count', type=int, default=1,
                        help='Leading sequence tasks trained with PBSR')
    parser.add_argument('--pbsr_train_task_indices', type=int, nargs='*',
                        default=None,
                        help='Explicit task indices trained with PBSR')
    parser.add_argument('--pbsr_variant', type=str, default='v1',
                        choices=['v1', 'ppo_v2', 'actual_demand',
                                 'spectral'],
                        help='PBSR objective used by PPO')
    parser.add_argument('--pbsr_v2_actor', type=str2bool, default=True,
                        help='Apply demand-complete PBSR to the PPO actor')
    parser.add_argument('--pbsr_v2_horizons', type=int, nargs='+',
                        default=[1, 3, 5],
                        help='Bellman horizons in the PPO-v2 reserve bank')
    parser.add_argument('--pbsr_v2_bandwidth', type=float, default=0.5,
                        help='Minimum relative tangent singular bandwidth')
    parser.add_argument('--ppo_value_reference_dir', type=str, default=None,
                        help='Directory of fixed PPO value-demand references')
    parser.add_argument('--ppo_value_reference_mode', type=str,
                        default='load', choices=['create', 'load'],
                        help='Load fixed transition-only PPO references')
    parser.add_argument('--ppo_bolt', type=str2bool, default=False,
                        help='Use inverse-burden metric rotation for PPO value Adam')
    parser.add_argument('--ppo_bolt_rank', type=int, default=4,
                        help='Active parameter-space rank for PPO BOLT')
    parser.add_argument('--ppo_bolt_rho', type=float, default=0.5,
                        help='Fixed PPO BOLT metric anisotropy')
    parser.add_argument('--ppo_bolt_ridge', type=float, default=0.1,
                        help='Ridge on the trace-normalized PPO value kernel')
    parser.add_argument('--ppo_bolt_calibration_size', type=int, default=32,
                        help='Disjoint calibration and held-out rows for PPO BOLT')
    parser.add_argument('--ppo_bolt_update_interval', type=int, default=1000,
                        help='Value optimizer steps between PPO BOLT refreshes')
    parser.add_argument('--ppo_bolt_history_columns', type=int, default=24,
                        help='Raw Bellman pullback columns retained by PPO BOLT')
    parser.add_argument('--ppo_bolt_horizons', type=int, nargs='+',
                        default=[1, 3, 5],
                        help='Observed value-learning horizons used by PPO BOLT')
    parser.add_argument('--ppo_bolt_train_task_indices', type=int, nargs='*',
                        default=None,
                        help='Explicit sequence task indices using PPO BOLT')
    parser.add_argument('--ppo_bolt_update_mode', type=str,
                        default='innovation',
                        choices=['innovation', 'full_moment'],
                        help='Adam component transformed by PPO BOLT')
    parser.add_argument('--ppo_bolt_reset_first_moment', type=str2bool,
                        default=False,
                        help='Reset Adam first moment at a task boundary')
    parser.add_argument('--ppo_spectral_advantage', type=str2bool,
                        default=False,
                        help='Use MC returns in slow value-NTK directions')
    parser.add_argument('--ppo_spectral_advantage_ridge', type=float,
                        default=0.1,
                        help='Relative value-NTK ridge for MC interpolation')
    parser.add_argument('--ppo_spectral_advantage_block_size', type=int,
                        default=128,
                        help='Trajectory block size for spectral interpolation')
    parser.add_argument('--ppo_spectral_advantage_train_task_indices',
                        type=int, nargs='*', default=None,
                        help='Sequence task indices using spectral advantage')
    parser.add_argument('--branch_checkpoint', type=str, default=None,
                        help='Task-A checkpoint used to start directly on task B')
    parser.add_argument('--branch_task_step', type=int, default=0,
                        help='Local task progress represented by a branch checkpoint')
    parser.add_argument('--branch_alpha', type=float, default=None,
                        help='Entropy coefficient restored at a branch checkpoint')
    parser.add_argument('--branch_online_critic_source', type=str,
                        default='inherited', choices=['inherited', 'fresh'],
                        help='Online critic weights used at a task-boundary branch')
    parser.add_argument('--branch_target_critic_source', type=str,
                        default='inherited', choices=['inherited', 'fresh'],
                        help='Target critic weights used at a task-boundary branch')
    parser.add_argument('--branch_run_steps', type=int, default=None,
                        help='Optional number of post-branch gradient steps to run')
    parser.add_argument('--plasticity_injection_mode', type=str,
                        default='none', choices=['none', 'fixed', 'burden'],
                        help='SAC critic plasticity-injection width policy')
    parser.add_argument('--plasticity_injection_width', type=int, default=256,
                        help='Active branch width for fixed injection')
    parser.add_argument('--plasticity_injection_widths', type=int, nargs='+',
                        default=[32, 64, 128, 256],
                        help='Candidate widths for Bellman-burden selection')
    parser.add_argument('--plasticity_injection_rows', type=int, default=64,
                        help='Current replay transitions used to select width')
    parser.add_argument('--plasticity_injection_targets', type=int, default=8,
                        help='Deterministic soft Bellman demands for selection')
    parser.add_argument('--plasticity_injection_ridge', type=float,
                        default=0.001,
                        help='Fresh-kernel-relative inverse-burden ridge')
    parser.add_argument('--plasticity_injection_task_indices', type=int,
                        nargs='*', default=None,
                        help='Sequence task indices receiving injection')
    parser.add_argument('--demand_aligned_reserve', type=str2bool,
                        default=False,
                        help='Use the SAC demand-aligned spectral reserve')
    parser.add_argument('--dar_rows', type=int, default=64,
                        help='New-task replay rows used to align the reserve')
    parser.add_argument('--dar_hidden_dim', type=int, default=256,
                        help='Hidden width of each reserve feature block')
    parser.add_argument('--dar_feature_dim', type=int, default=64,
                        help='Number of new function-space directions')
    parser.add_argument('--dar_targets', type=int, default=8,
                        help='Soft Bellman demand ensemble size')
    parser.add_argument('--dar_ridge', type=float, default=0.001,
                        help='Relative ridge in the inverse burden')
    parser.add_argument('--dar_trace_ratio', type=float, default=1.0,
                        help='Reserve kernel trace relative to inherited trace')
    parser.add_argument('--dar_capacity_price', type=float, default=0.0,
                        help='KKT marginal price below which capacity is reused')
    parser.add_argument('--dar_alignment_steps', type=int, default=200,
                        help='Optimization steps for demand alignment')
    parser.add_argument('--dar_alignment_lr', type=float, default=0.001,
                        help='Learning rate for demand alignment')
    parser.add_argument('--dar_task_indices', type=int, nargs='*',
                        default=None,
                        help='Sequence task indices evaluated for reserve reuse')
    # v2 参数独立命名，避免改变正在运行的原版 DAR 实验语义。
    parser.add_argument('--dsr_v2', type=str2bool, default=False,
                        help='启用分阶段需求对齐的 DSR v2')
    parser.add_argument('--dsr_v2_task_steps', type=int, nargs='+',
                        default=[100000, 500000, 800000],
                        help='每个新任务内重新估计需求的 critic 更新步数')
    parser.add_argument('--dsr_v2_rows', type=int, default=128,
                        help='校准和独立留出集合各自的 transition 数')
    parser.add_argument('--dsr_v2_min_gain', type=float, default=0.01,
                        help='留出 burden 相对等 trace 随机分支的最小改善比例')
    # Bellman 几何方法独立启用，默认不改变已有 SAC 实验。
    parser.add_argument('--bellman_geometry', type=str2bool, default=False,
                        help='启用完整 critic Jacobian 的 Bellman 几何正则')
    parser.add_argument('--geometry_coefficient', type=float, default=0.01,
                        help='几何辅助损失的最大系数')
    parser.add_argument('--geometry_isotropic_fraction', type=float, default=0.1,
                        help='需求协方差中的均匀先验比例')
    parser.add_argument('--geometry_ridge', type=float, default=0.01,
                        help='原始核求解的绝对 ridge，不按 trace 归一化')
    parser.add_argument('--geometry_scale_penalty', type=float, default=0.01,
                        help='核整体尺度的惩罚系数')
    parser.add_argument('--geometry_max_grad_ratio', type=float, default=0.1,
                        help='辅助梯度范数相对原 TD 梯度的上限')
    parser.add_argument('--geometry_target_samples', type=int, default=8,
                        help='构造辅助目标均值的下一动作采样次数')
    parser.add_argument('--geometry_probe_size', type=int, default=32,
                        help='每次更新计算完整 Jacobian 的转移数')
    parser.add_argument('--geometry_first_task', type=int, default=1,
                        help='开始使用几何正则的任务序号，从 0 开始')
    # 更新响应方法独立于旧几何正则，实验配置与源码一起保存。
    parser.add_argument('--bellman_response', type=str2bool, default=False,
                        help='启用基于可微 Adam 虚拟更新的 Bellman 响应修正')
    parser.add_argument('--response_step_ratio', type=float, default=0.05,
                        help='辅助参数位移相对真实 TD 参数位移的上限')
    parser.add_argument('--response_rows', type=int, default=64,
                        help='校准集及检查集各自的转移数')
    parser.add_argument('--response_target_samples', type=int, default=8,
                        help='辅助 Bellman 目标的下一动作采样次数')
    parser.add_argument('--response_update_interval', type=int, default=1,
                        help='响应修正间隔，以 critic 更新次数计')
    parser.add_argument('--response_first_task', type=int, default=1,
                        help='开始响应修正的任务序号，从 0 开始')
    parser.add_argument('--train_task_count', type=int, default=None,
                        help='Number of sequence tasks to execute')
    
    args = parser.parse_args()

    return args
