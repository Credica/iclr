import metaworld
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import CReLU

from garage.torch.policies import TanhGaussianMLPPolicy, GaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.value_functions import GaussianMLPValueFunction

from garage.torch.algos import BC_SAC, BC_PPO
from garage.torch.algos import RND_SAC, RND_PPO
from garage.torch.algos import (
    Finetuning_SAC, Finetuning_PPO, SpectralRegularizedSAC)


from garage.torch.algos import EWC_SAC, EWC_PPO
from garage.torch.algos import P_and_C_SAC, P_and_C_PPO

    
def get_algo(args, spec, n_tasks, train_envs, test_envs, train_info, env_seq):
    batch_size, epoch_cycles = train_info
    env_type, rl_method = args.env_type, args.rl_method
    geometry_enabled = getattr(args, 'bellman_geometry', False)
    response_enabled = getattr(args, 'bellman_response', False)
    muon_enabled = getattr(args, 'sac_optimizer', 'adam') == 'muon'
    singular_clip_enabled = getattr(args, 'sac_singular_clip', False)
    spectral_enabled = args.cl_method == 'spectral'
    if spectral_enabled:
        incompatible = ('bellman_geometry', 'bellman_response', 'pbsr', 'dsr_v2',
                        'demand_aligned_reserve', 'q_reset', 'policy_reset', 'ReDo',
                        'crelu', 'infer', 'wasserstein',
                        'use_exploration', 'sac_singular_clip')
        if (muon_enabled or rl_method != 'sac' or args.branch_checkpoint or
                args.first_task or
                any(getattr(args, name, False) for name in incompatible) or
                getattr(args, 'plasticity_injection_mode', 'none') != 'none'):
            raise ValueError(
                'SpectralReg 仅支持从头训练、无其他干预的普通 Adam SAC')
    if singular_clip_enabled:
        incompatible = ('bellman_geometry', 'bellman_response', 'pbsr', 'dsr_v2',
                        'demand_aligned_reserve', 'q_reset', 'policy_reset', 'ReDo',
                        'crelu', 'infer', 'wasserstein', 'bellman_spectral_stats',
                        'use_exploration')
        if (muon_enabled or rl_method != 'sac' or args.cl_method != 'finetuning' or
                env_type != 'metaworld' or args.branch_checkpoint or args.first_task or
                any(getattr(args, name, False) for name in incompatible) or
                getattr(args, 'plasticity_injection_mode', 'none') != 'none'):
            raise ValueError('谱裁剪入口仅支持从头训练的普通 Adam MetaWorld SAC')
    if muon_enabled:
        # 此入口只比较普通 SAC 的优化器，避免混入已有塑性方法。
        incompatible = ('bellman_geometry', 'bellman_response', 'pbsr', 'dsr_v2',
                        'demand_aligned_reserve', 'q_reset', 'policy_reset', 'ReDo',
                        'crelu', 'infer', 'wasserstein',
                        'bellman_spectral_stats', 'use_exploration')
        if (rl_method != 'sac' or args.cl_method != 'finetuning' or
                env_type != 'metaworld' or
                any(getattr(args, name, False) for name in incompatible) or
                getattr(args, 'plasticity_injection_mode', 'none') != 'none'):
            raise ValueError('Muon 入口只支持无附加干预的 MetaWorld finetuning SAC')
    if response_enabled and (geometry_enabled or rl_method != 'sac' or
                             args.cl_method != 'finetuning' or env_type != 'metaworld'):
        raise ValueError('响应方法仅支持单独启用的 MetaWorld finetuning SAC')
    if geometry_enabled and (rl_method != 'sac' or
                             args.cl_method != 'finetuning' or
                             env_type != 'metaworld'):
        raise ValueError('当前 Bellman 几何入口支持 MetaWorld 的 finetuning SAC')
    if env_type == 'metaworld' or env_type == "dm_control":

        if rl_method == 'sac':
            
            hidden_nonlinearity = CReLU if args.crelu else nn.ReLU

            hidden_sizes = [256, 256] if env_type == "metaworld" else [1024, 1024]

            policy = TanhGaussianMLPPolicy(
                env_spec=spec,
                n_tasks=n_tasks,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None,
                min_std=np.exp(-20.),
                max_std=np.exp(2.),
                infer=args.infer, 
                adaptor=(args.cl_method == 'pandc'),
                zero_alpha = ((args.cl_method == 'pandc') and args.zero_alpha),
                ReDo = args.ReDo,
                no_stats = args.no_stats
            )
            
            hidden_nonlinearity = CReLU if args.crelu else F.relu

            qf1 = ContinuousMLPQFunction(env_spec=spec,
                                         infer=args.infer,
                                         hidden_sizes=hidden_sizes,
                                         hidden_nonlinearity=hidden_nonlinearity,
                                         ReDo=args.ReDo,
                                         no_stats=args.no_stats)

            qf2 = ContinuousMLPQFunction(env_spec=spec,
                                         infer=args.infer,
                                         hidden_sizes=hidden_sizes,
                                         hidden_nonlinearity=hidden_nonlinearity,
                                         ReDo=args.ReDo,
                                         no_stats=args.no_stats)
            replay_buffer = PathBuffer(capacity_in_transitions=int(1e6),)
            
            if isinstance(spec,list):
                max_episode_length = spec[0].max_episode_length
            else:
                max_episode_length = spec.max_episode_length
            sampler = LocalSampler(agents=policy,
                                    envs=train_envs,
                                    max_episode_length=max_episode_length,
                                    n_workers=1,)
            
            if env_type == "metaworld":
                sac_kwargs = {
                'policy': policy,
                'qf1': qf1,
                'qf2': qf2,
                'sampler': sampler,
                'seed': args.seed,
                'gradient_steps_per_itr': batch_size,
                'eval_env': test_envs,
                'env_spec': spec,
                'steps_per_epoch': epoch_cycles,
                'replay_buffer': replay_buffer,
                'use_exploration': args.use_exploration,
                'q_reset': args.q_reset,
                'policy_reset': args.policy_reset,
                'first_task': args.first_task,
                'crelu': args.crelu,
                'num_tasks': 1,
                'num_evaluation_episodes': args.num_evaluation_episodes,
                'log_name': args.proc_name,
                'use_wandb': args.wandb,
                'infer': args.infer,
                'wasserstein': args.wasserstein,
                'ReDo': args.ReDo,
                'redo_interval': args.redo_interval,
                'redo_tau': args.redo_tau,
                'no_stats': args.no_stats,
                'scalar_log_interval': args.scalar_log_interval,
                'feature_stats_interval': args.feature_stats_interval,
                'hessian_stats_interval': args.hessian_stats_interval,
                'bellman_probe': args.bellman_probe,
                'bellman_probe_size': args.bellman_probe_size,
                'bellman_probe_interval': args.bellman_probe_interval,
                'bellman_probe_targets': args.bellman_probe_targets,
                'bellman_probe_ridge': args.bellman_probe_ridge,
                'bellman_probe_dir': args.bellman_probe_dir,
                'bellman_spectral_stats': args.bellman_spectral_stats,
                'bellman_spectral_anchor_size': args.bellman_spectral_anchor_size,
                'bellman_spectral_fit_lr': args.bellman_spectral_fit_lr,
                'bellman_reference_dir': args.bellman_reference_dir,
                'bellman_spectral_task_steps': args.bellman_spectral_task_steps,
                'pbsr': args.pbsr,
                'pbsr_coef': args.pbsr_coef,
                'pbsr_anchor_size': args.pbsr_anchor_size,
                'pbsr_targets': args.pbsr_targets,
                'pbsr_ridge': args.pbsr_ridge,
                'pbsr_update_interval': args.pbsr_update_interval,
                'pbsr_train_task_count': args.pbsr_train_task_count,
                'task_names': env_seq,
                'exact_task_budget': args.exact_sac_task_budget,
                'branch_checkpoint': args.branch_checkpoint,
                'branch_task_step': args.branch_task_step,
                'branch_alpha': args.branch_alpha,
                'branch_online_critic_source': args.branch_online_critic_source,
                'branch_target_critic_source': args.branch_target_critic_source,
                'plasticity_injection_mode': args.plasticity_injection_mode,
                'plasticity_injection_width': args.plasticity_injection_width,
                'plasticity_injection_widths': args.plasticity_injection_widths,
                'plasticity_injection_rows': args.plasticity_injection_rows,
                'plasticity_injection_targets': args.plasticity_injection_targets,
                'plasticity_injection_ridge': args.plasticity_injection_ridge,
                'plasticity_injection_task_indices': args.plasticity_injection_task_indices,
                'dsr_v2': args.dsr_v2,
                'dsr_v2_kwargs': {
                    'task_steps': args.dsr_v2_task_steps,
                    'rows': args.dsr_v2_rows,
                    'min_gain': args.dsr_v2_min_gain,
                    'hidden_dim': args.dar_hidden_dim,
                    'feature_dim': args.dar_feature_dim,
                    'targets': args.dar_targets,
                    'alignment_steps': args.dar_alignment_steps,
                    'alignment_lr': args.dar_alignment_lr,
                    'trace_ratio': args.dar_trace_ratio,
                    'ridge': args.dar_ridge,
                },
                'demand_aligned_reserve': args.demand_aligned_reserve,
                'dar_rows': args.dar_rows,
                'dar_hidden_dim': args.dar_hidden_dim,
                'dar_feature_dim': args.dar_feature_dim,
                'dar_targets': args.dar_targets,
                'dar_ridge': args.dar_ridge,
                'dar_trace_ratio': args.dar_trace_ratio,
                'dar_capacity_price': args.dar_capacity_price,
                'dar_alignment_steps': args.dar_alignment_steps,
                'dar_alignment_lr': args.dar_alignment_lr,
                'dar_task_indices': args.dar_task_indices,
            }
            
            elif env_type == "dm_control":
                sac_kwargs = {
                'policy': policy,
                'qf1': qf1,
                'qf2': qf2,
                'sampler': sampler,
                'seed': args.seed,
                'gradient_steps_per_itr': batch_size // 4,
                'eval_env': test_envs,
                'env_spec': spec,
                'steps_per_epoch': epoch_cycles,
                'replay_buffer': replay_buffer,
                'use_exploration': args.use_exploration,
                'q_reset': args.q_reset,
                'policy_reset': args.policy_reset,
                'first_task': args.first_task,
                'crelu': args.crelu,
                'num_tasks': 1,
                'num_evaluation_episodes': args.num_evaluation_episodes,
                'log_name': args.proc_name,
                'use_wandb': args.wandb,
                'infer': args.infer,
                'wasserstein': args.wasserstein,
                'ReDo': args.ReDo,
                'redo_interval': args.redo_interval,
                'redo_tau': args.redo_tau,
                'no_stats': args.no_stats,
                'scalar_log_interval': args.scalar_log_interval,
                'feature_stats_interval': args.feature_stats_interval,
                'hessian_stats_interval': args.hessian_stats_interval,
                'bellman_probe': args.bellman_probe,
                'bellman_probe_size': args.bellman_probe_size,
                'bellman_probe_interval': args.bellman_probe_interval,
                'bellman_probe_targets': args.bellman_probe_targets,
                'bellman_probe_ridge': args.bellman_probe_ridge,
                'bellman_probe_dir': args.bellman_probe_dir,
                'bellman_spectral_stats': args.bellman_spectral_stats,
                'bellman_spectral_anchor_size': args.bellman_spectral_anchor_size,
                'bellman_spectral_fit_lr': args.bellman_spectral_fit_lr,
                'bellman_reference_dir': args.bellman_reference_dir,
                'bellman_spectral_task_steps': args.bellman_spectral_task_steps,
                'pbsr': args.pbsr,
                'pbsr_coef': args.pbsr_coef,
                'pbsr_anchor_size': args.pbsr_anchor_size,
                'pbsr_targets': args.pbsr_targets,
                'pbsr_ridge': args.pbsr_ridge,
                'pbsr_update_interval': args.pbsr_update_interval,
                'pbsr_train_task_count': args.pbsr_train_task_count,
                'task_names': env_seq,
                'exact_task_budget': args.exact_sac_task_budget,
                'branch_checkpoint': args.branch_checkpoint,
                'branch_task_step': args.branch_task_step,
                'branch_alpha': args.branch_alpha,
                'branch_online_critic_source': args.branch_online_critic_source,
                'branch_target_critic_source': args.branch_target_critic_source,
                'plasticity_injection_mode': args.plasticity_injection_mode,
                'plasticity_injection_width': args.plasticity_injection_width,
                'plasticity_injection_widths': args.plasticity_injection_widths,
                'plasticity_injection_rows': args.plasticity_injection_rows,
                'plasticity_injection_targets': args.plasticity_injection_targets,
                'plasticity_injection_ridge': args.plasticity_injection_ridge,
                'plasticity_injection_task_indices': args.plasticity_injection_task_indices,
                'dsr_v2': args.dsr_v2,
                'dsr_v2_kwargs': {
                    'task_steps': args.dsr_v2_task_steps,
                    'rows': args.dsr_v2_rows,
                    'min_gain': args.dsr_v2_min_gain,
                    'hidden_dim': args.dar_hidden_dim,
                    'feature_dim': args.dar_feature_dim,
                    'targets': args.dar_targets,
                    'alignment_steps': args.dar_alignment_steps,
                    'alignment_lr': args.dar_alignment_lr,
                    'trace_ratio': args.dar_trace_ratio,
                    'ridge': args.dar_ridge,
                },
                'demand_aligned_reserve': args.demand_aligned_reserve,
                'dar_rows': args.dar_rows,
                'dar_hidden_dim': args.dar_hidden_dim,
                'dar_feature_dim': args.dar_feature_dim,
                'dar_targets': args.dar_targets,
                'dar_ridge': args.dar_ridge,
                'dar_trace_ratio': args.dar_trace_ratio,
                'dar_capacity_price': args.dar_capacity_price,
                'dar_alignment_steps': args.dar_alignment_steps,
                'dar_alignment_lr': args.dar_alignment_lr,
                'dar_task_indices': args.dar_task_indices,

                'policy_lr': 1e-4,
                'qf_lr': 1e-4,
                'fixed_alpha': 0.01,
                'buffer_batch_size': 1024,
                'multi_input': True,
            }

            if args.cl_method == 'finetuning':
                if singular_clip_enabled:
                    from garage.torch.algos.sac_singular_clip import FinetuningSACSingularClip
                    algo = FinetuningSACSingularClip(
                        **sac_kwargs, singular_clip_min=args.singular_clip_min,
                        singular_clip_max=args.singular_clip_max,
                        singular_clip_interval=args.singular_clip_interval,
                        singular_clip_start_task=args.singular_clip_start_task)
                elif muon_enabled:
                    from garage.torch.algos.sac_muon import FinetuningSACMuon
                    algo = FinetuningSACMuon(
                        **sac_kwargs, muon_lr=args.muon_lr,
                        muon_momentum=args.muon_momentum, muon_ns_steps=args.muon_ns_steps)
                elif response_enabled:
                    from garage.torch.algos.sac_bellman_response import (
                        FinetuningSACBellmanResponse)
                    response_kwargs = {name: getattr(args, name) for name in (
                        'response_step_ratio', 'response_rows',
                        'response_target_samples', 'response_update_interval',
                        'response_first_task')}
                    algo = FinetuningSACBellmanResponse(**sac_kwargs, **response_kwargs)
                elif geometry_enabled:
                    # 复用 Finetuning/MTSAC 的评估及温度逻辑，只替换 critic 目标。
                    from garage.torch.algos.sac_bellman_geometry import (
                        FinetuningSACBellmanGeometry)
                    geometry_kwargs = {
                        name: getattr(args, name)
                        for name in (
                            'geometry_coefficient', 'geometry_isotropic_fraction',
                            'geometry_ridge', 'geometry_scale_penalty',
                            'geometry_max_grad_ratio', 'geometry_target_samples',
                            'geometry_probe_size', 'geometry_first_task')}
                    algo = FinetuningSACBellmanGeometry(
                        **sac_kwargs, **geometry_kwargs)
                else:
                    algo = Finetuning_SAC(**sac_kwargs)

            if args.cl_method == 'spectral':
                algo = SpectralRegularizedSAC(
                    **sac_kwargs,
                    actor_coef=args.spectral_actor_coef,
                    critic_coef=args.spectral_critic_coef,
                    power_iterations=args.spectral_power_iterations)

            if args.cl_method == 'bc':
                algo = BC_SAC(**sac_kwargs,
                            cl_reg_coef=args.cl_reg_coef, 
                            expert_buffer_size=args.expert_buffer_size)
            if args.cl_method == 'rnd':
                algo = RND_SAC(**sac_kwargs,
                                    cl_reg_coef=args.cl_reg_coef,
                                    expert_buffer_size=args.expert_buffer_size,
                                    replay_buffer_size=args.replay_buffer_size,
                                    env_seq=env_seq,
                                    nepochs_offline=args.nepochs_offline,
                                    bc_kl=args.bc_kl,
                                    distill_kl=args.distill_kl,
                                    reset_offline_actor=args.reset_offline_actor,
                                    teacher_steps=args.rd_teacher_steps,
                                    teacher_root=args.rd_teacher_root)
            
            if args.cl_method == 'ewc':
                algo = EWC_SAC(**sac_kwargs, cl_reg_coef=args.cl_reg_coef)
            if args.cl_method == 'pandc':
                algo = P_and_C_SAC(**sac_kwargs, cl_reg_coef=args.cl_reg_coef, 
                                   compress_step=args.compress_step, 
                                   bc=args.use_pandc_bc, reset_column=args.reset_column, 
                                   reset_adaptor=args.reset_adaptor)
        elif rl_method == 'ppo':

            hidden_nonlinearity = CReLU if args.crelu else torch.tanh

            hidden_sizes = (128, 128) if env_type == 'metaworld' else (1024, 1024)

            policy = GaussianMLPPolicy(
                env_spec=spec,
                n_tasks=n_tasks,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None,
                infer=args.infer,
                adaptor=(args.cl_method == 'pandc'),
                zero_alpha = ((args.cl_method == 'pandc') and args.zero_alpha),
                ReDo = args.ReDo,
                no_stats = args.no_stats,
            )

            value_function = GaussianMLPValueFunction(env_spec=spec,
                                                    hidden_sizes=hidden_sizes,
                                                    hidden_nonlinearity=hidden_nonlinearity,
                                                    output_nonlinearity=None,
                                                    infer=args.infer,)

            if isinstance(spec,list):
                max_episode_length = spec[0].max_episode_length
            else:
                max_episode_length = spec.max_episode_length
            
            sampler = LocalSampler(agents=policy,
                                    envs=train_envs,
                                    max_episode_length=max_episode_length,
                                    n_workers=1,)

            if env_type == 'metaworld':
                ppo_kwargs = {
                'env_spec': spec,
                'policy': policy,
                'value_function': value_function,
                'eval_env': test_envs,
                'sampler': sampler,
                'seed': args.seed,
                'discount': 0.99,
                'gae_lambda': args.ppo_gae_lambda,
                'center_adv': True,
                'lr_clip_range': args.lr_clip_range,
                'log_name': args.proc_name,
                'num_evaluation_episodes': args.num_evaluation_episodes,
                'q_reset': args.q_reset,
                'policy_reset': args.policy_reset,
                'first_task': args.first_task,
                'first_task_steps': args.first_task_steps,
                'use_wandb': args.wandb,
                'crelu': args.crelu,
                'infer': args.infer,
                'wasserstein': args.wasserstein,
                'ReDo': args.ReDo,
                'no_stats': args.no_stats,
                'multi_input': False,
                'bellman_spectral_stats': args.bellman_spectral_stats,
                'bellman_spectral_anchor_size': args.bellman_spectral_anchor_size,
                'bellman_spectral_fit_lr': args.bellman_spectral_fit_lr,
                'bellman_spectral_ridge': args.bellman_probe_ridge,
                'bellman_spectral_task_steps': args.bellman_spectral_task_steps,
                'bellman_probe_dir': args.bellman_probe_dir,
                'ppo_value_reference_dir': args.ppo_value_reference_dir,
                'ppo_value_reference_mode': args.ppo_value_reference_mode,
                'pbsr': args.pbsr,
                'pbsr_coef': args.pbsr_coef,
                'pbsr_anchor_size': args.pbsr_anchor_size,
                'pbsr_targets': args.pbsr_targets,
                'pbsr_ridge': args.pbsr_ridge,
                'pbsr_update_interval': args.pbsr_update_interval,
                'pbsr_train_task_count': args.pbsr_train_task_count,
                'pbsr_train_task_indices': args.pbsr_train_task_indices,
                'pbsr_variant': args.pbsr_variant,
                'pbsr_v2_actor': args.pbsr_v2_actor,
                'pbsr_v2_horizons': args.pbsr_v2_horizons,
                'pbsr_v2_bandwidth': args.pbsr_v2_bandwidth,
                'ppo_bolt': args.ppo_bolt,
                'ppo_bolt_rank': args.ppo_bolt_rank,
                'ppo_bolt_rho': args.ppo_bolt_rho,
                'ppo_bolt_ridge': args.ppo_bolt_ridge,
                'ppo_bolt_calibration_size': (
                    args.ppo_bolt_calibration_size),
                'ppo_bolt_update_interval': args.ppo_bolt_update_interval,
                'ppo_bolt_history_columns': args.ppo_bolt_history_columns,
                'ppo_bolt_horizons': args.ppo_bolt_horizons,
                'ppo_bolt_train_task_indices': (
                    args.ppo_bolt_train_task_indices),
                'ppo_bolt_update_mode': args.ppo_bolt_update_mode,
                'ppo_bolt_reset_first_moment': (
                    args.ppo_bolt_reset_first_moment),
                'ppo_spectral_advantage': args.ppo_spectral_advantage,
                'ppo_spectral_advantage_ridge': (
                    args.ppo_spectral_advantage_ridge),
                'ppo_spectral_advantage_block_size': (
                    args.ppo_spectral_advantage_block_size),
                'ppo_spectral_advantage_train_task_indices': (
                    args.ppo_spectral_advantage_train_task_indices),
                'task_names': env_seq,

            }
                
            elif env_type == 'dm_control':
                ppo_kwargs = {
                'env_spec': spec,
                'policy': policy,
                'value_function': value_function,
                'eval_env': test_envs,
                'sampler': sampler,
                'seed': args.seed,
                'discount': 0.99,
                'gae_lambda': args.ppo_gae_lambda,
                'center_adv': True,
                'lr_clip_range': args.lr_clip_range,
                'log_name': args.proc_name,
                'num_evaluation_episodes': args.num_evaluation_episodes,
                'q_reset': args.q_reset,
                'policy_reset': args.policy_reset,
                'first_task': args.first_task,
                'first_task_steps': args.first_task_steps,
                'use_wandb': args.wandb,
                'crelu': args.crelu,
                'infer': args.infer,
                'wasserstein': args.wasserstein,
                'ReDo': args.ReDo,
                'no_stats': args.no_stats,
                'policy_lr':3e-4,
                'value_lr':3e-4,
                'max_optimization_epochs':64,
                'minibatch_size':1024,
                'multi_input': True,
                'bellman_spectral_stats': args.bellman_spectral_stats,
                'bellman_spectral_anchor_size': args.bellman_spectral_anchor_size,
                'bellman_spectral_fit_lr': args.bellman_spectral_fit_lr,
                'bellman_spectral_ridge': args.bellman_probe_ridge,
                'bellman_spectral_task_steps': args.bellman_spectral_task_steps,
                'bellman_probe_dir': args.bellman_probe_dir,
                'ppo_value_reference_dir': args.ppo_value_reference_dir,
                'ppo_value_reference_mode': args.ppo_value_reference_mode,
                'pbsr': args.pbsr,
                'pbsr_coef': args.pbsr_coef,
                'pbsr_anchor_size': args.pbsr_anchor_size,
                'pbsr_targets': args.pbsr_targets,
                'pbsr_ridge': args.pbsr_ridge,
                'pbsr_update_interval': args.pbsr_update_interval,
                'pbsr_train_task_count': args.pbsr_train_task_count,
                'pbsr_train_task_indices': args.pbsr_train_task_indices,
                'pbsr_variant': args.pbsr_variant,
                'pbsr_v2_actor': args.pbsr_v2_actor,
                'pbsr_v2_horizons': args.pbsr_v2_horizons,
                'pbsr_v2_bandwidth': args.pbsr_v2_bandwidth,
                'ppo_bolt': args.ppo_bolt,
                'ppo_bolt_rank': args.ppo_bolt_rank,
                'ppo_bolt_rho': args.ppo_bolt_rho,
                'ppo_bolt_ridge': args.ppo_bolt_ridge,
                'ppo_bolt_calibration_size': (
                    args.ppo_bolt_calibration_size),
                'ppo_bolt_update_interval': args.ppo_bolt_update_interval,
                'ppo_bolt_history_columns': args.ppo_bolt_history_columns,
                'ppo_bolt_horizons': args.ppo_bolt_horizons,
                'ppo_bolt_train_task_indices': (
                    args.ppo_bolt_train_task_indices),
                'ppo_bolt_update_mode': args.ppo_bolt_update_mode,
                'ppo_bolt_reset_first_moment': (
                    args.ppo_bolt_reset_first_moment),
                'ppo_spectral_advantage': args.ppo_spectral_advantage,
                'ppo_spectral_advantage_ridge': (
                    args.ppo_spectral_advantage_ridge),
                'ppo_spectral_advantage_block_size': (
                    args.ppo_spectral_advantage_block_size),
                'ppo_spectral_advantage_train_task_indices': (
                    args.ppo_spectral_advantage_train_task_indices),
                'task_names': env_seq,

            }


            if args.cl_method == 'finetuning':
                algo = Finetuning_PPO(**ppo_kwargs)
            
            if args.cl_method == 'ewc':
                algo = EWC_PPO(**ppo_kwargs, cl_reg_coef=args.cl_reg_coef)
         
            if args.cl_method == 'pandc':
                algo = P_and_C_PPO(**ppo_kwargs, cl_reg_coef=args.cl_reg_coef, 
                                   compress_step=args.compress_step, 
                                   bc=args.use_pandc_bc, 
                                   reset_column=args.reset_column, 
                                   reset_adaptor=args.reset_adaptor)
            if args.cl_method == 'bc':
                algo = BC_PPO(**ppo_kwargs,
                            cl_reg_coef=args.cl_reg_coef, 
                            expert_buffer_size=args.expert_buffer_size)
            if args.cl_method == 'rnd':
                algo = RND_PPO(**ppo_kwargs,
                                    cl_reg_coef=args.cl_reg_coef,
                                    expert_buffer_size=args.expert_buffer_size,
                                    replay_buffer_size=args.replay_buffer_size,
                                    env_seq=env_seq,
                                    nepochs_offline=args.nepochs_offline,
                                    bc_kl=args.bc_kl,
                                    distill_kl=args.distill_kl,
                                    teacher_steps=args.rd_teacher_steps)
    
    


    return algo
