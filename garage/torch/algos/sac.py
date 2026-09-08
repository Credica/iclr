"""This modules creates a sac model in PyTorch."""
from collections import deque
import copy
import json
import random

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

# yapf: disable
from garage import log_performance, obtain_evaluation_episodes, StepType
from garage.np.algos import RLAlgorithm
from garage.torch import as_torch_dict, global_device, state_dict_to, np_to_torch
from garage.torch.algos.bellman_spectral_stats import (
    BellmanSpectralStats, deterministic_soft_bellman_directions)
from garage.torch.algos.pbsr import PBSRHead
from garage.torch.algos.sac_plasticity_injection import (
    select_plasticity_width)
from garage.torch.algos.sac_demand_aligned_reserve import (
    SACDemandAlignedReserve)
from garage.torch.algos.sac_dsr_v2 import (
    SACDSRV2, restore_branches as restore_dsr_v2_branches,
    zero_head_gradients, step_heads)
from garage.torch.algos.spectral_regularization import (
    LayerSpectralRegularizer)
from garage.torch._functions import list_to_tensor, zero_optim_grads, weight_deviation, weight_hessian, feature_rank
from garage.torch.q_functions.continuous_mlp_q_function import (
    PlasticityInjectionBranch)
from time import time

import wandb
import os
import pickle

from time import time

# yapf: enable


def _capture_training_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_training_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])

def load_model(model, model_name, first_task, seed):
    # Load policy
    copied_model = copy.deepcopy(model)
    name = model_name.format(first_task, seed)
    loaded_state_dict = torch.load('./models/sac_models/'+name, map_location=global_device())

    copied_model_state_dict = copied_model.state_dict()
    new_state_dict = {}

    for k in loaded_state_dict.keys():
        
        if loaded_state_dict[k].shape == copied_model_state_dict[k].shape:
            new_state_dict[k] = loaded_state_dict[k]
        else:
            new_state_dict[k] = copied_model_state_dict[k]

    copied_model_state_dict.update(new_state_dict)
    model.load_state_dict(copied_model_state_dict)

    return


def select_branch_critic_states(checkpoint, random_qf1, random_qf2,
                                online_source, target_source):
    """Select online and target critic states for a boundary branch."""
    inherited = (checkpoint['qf1'], checkpoint['qf2'])
    fresh = (random_qf1, random_qf2)
    online_qf1, online_qf2 = {
        'inherited': inherited,
        'fresh': fresh,
    }[online_source]
    target_qf1, target_qf2 = {
        'inherited': inherited,
        'fresh': fresh,
    }[target_source]
    return {
        'qf1': online_qf1,
        'qf2': online_qf2,
        'target_qf1': target_qf1,
        'target_qf2': target_qf2,
    }


def critic_first_batch_stats(q1_pred, q2_pred, target_q1, target_q2,
                             soft_bootstrap, bellman_target, rewards):
    """Summarize the unmodified critic quantities on a first task batch."""
    values = {
        'online_q1_mean': q1_pred.mean(),
        'online_q2_mean': q2_pred.mean(),
        'online_min_q_mean': torch.min(q1_pred, q2_pred).mean(),
        'target_q1_mean': target_q1.mean(),
        'target_q2_mean': target_q2.mean(),
        'target_min_q_mean': torch.min(target_q1, target_q2).mean(),
        'soft_bootstrap_mean': soft_bootstrap.mean(),
        'bellman_target_mean': bellman_target.mean(),
        'reward_mean': rewards.mean(),
        'qf1_td_abs_mean': (bellman_target - q1_pred).abs().mean(),
        'qf2_td_abs_mean': (bellman_target - q2_pred).abs().mean(),
    }
    return {name: float(value.detach()) for name, value in values.items()}


class SAC(RLAlgorithm):
    """A SAC Model in Torch.

    Based on Soft Actor-Critic and Applications:
        https://arxiv.org/abs/1812.05905

    Soft Actor-Critic (SAC) is an algorithm which optimizes a stochastic
    policy in an off-policy way, forming a bridge between stochastic policy
    optimization and DDPG-style approaches.
    A central feature of SAC is entropy regularization. The policy is trained
    to maximize a trade-off between expected return and entropy, a measure of
    randomness in the policy. This has a close connection to the
    exploration-exploitation trade-off: increasing entropy results in more
    exploration, which can accelerate learning later on. It can also prevent
    the policy from prematurely converging to a bad local optimum.

    Args:
        policy (garage.torch.policy.Policy): Policy/Actor/Agent that is being
            optimized by SAC.
        qf1 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        qf2 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        replay_buffer (ReplayBuffer): Stores transitions that are previously
            collected by the sampler.
        sampler (garage.sampler.Sampler): Sampler.
        env_spec (EnvSpec): The env_spec attribute of the environment that the
            agent is being trained in.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If None, defaults to
            `env_spec.max_episode_length`.
        gradient_steps_per_itr (int): Number of optimization steps that should
        gradient_steps_per_itr(int): Number of optimization steps that should
            occur before the training step is over and a new batch of
            transitions is collected by the sampler.
        fixed_alpha (float): The entropy/temperature to be used if temperature
            is not supposed to be learned.
        target_entropy (float): target entropy to be used during
            entropy/temperature optimization. If None, the default heuristic
            from Soft Actor-Critic Algorithms and Applications is used.
        initial_log_entropy (float): initial entropy/temperature coefficient
            to be used if a fixed_alpha is not being used (fixed_alpha=None),
            and the entropy/temperature coefficient is being learned.
        discount (float): Discount factor to be used during sampling and
            critic/q_function optimization.
        buffer_batch_size (int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size (int): The minimum number of transitions that need to
            be in the replay buffer before training can begin.
        target_update_tau (float): coefficient that controls the rate at which
            the target q_functions update over optimization iterations.
        policy_lr (float): learning rate for policy optimizers.
        qf_lr (float): learning rate for q_function optimizers.
        reward_scale (float): reward scale. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer (torch.optim.Optimizer): optimizer to be used for
            policy/actor, q_functions/critics, and temperature/entropy
            optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.
        num_evaluation_episodes (int): The number of evaluation episodes used
            for computing eval stats at the end of every epoch.
        eval_env (Environment): environment used for collecting evaluation
            episodes. If None, a copy of the train env is used.
        use_deterministic_evaluation (bool): True if the trained policy
            should be evaluated deterministically.
        temporal_regularization_factor (float): coefficient that determines
            the temporal regularization penalty as defined in CAPS as lambda_t
        spatial_regularization_factor (float): coefficient that determines
            the spatial regularization penalty as defined in CAPS as lambda_s
        spatial_regularization_eps (float): sigma of the normal distribution
            from with spatial regularization observations are drawn,
            in caps this is defined as epsilon_s
    """

    def __init__(
            self,
            env_spec,
            policy,
            qf1,
            qf2,
            replay_buffer,
            sampler,
            *,  # Everything after this is numbers.
            seed=0,
            max_episode_length_eval=None,
            gradient_steps_per_itr,
            fixed_alpha=None,
            target_entropy=None,
            initial_log_entropy=0.,
            discount=0.99,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            target_update_tau=5e-3,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1.0,
            optimizer=torch.optim.Adam,
            steps_per_epoch=1,
            num_evaluation_episodes=10,
            eval_env=None,
            use_deterministic_evaluation=True,
            use_exploration = False,
            q_reset = False,
            policy_reset = False,
            first_task = None,
            temporal_regularization_factor=0.,
            spatial_regularization_factor=0.,
            spatial_regularization_eps=1.,
            log_name = None, 
            use_wandb=True,
            infer = False,
            crelu=False,
            wasserstein = 0, 
            ReDo = False, 
            redo_interval=1000,
            redo_tau=0.1,
            no_stats=False, 
            scalar_log_interval=1000,
            feature_stats_interval=10000,
            hessian_stats_interval=10000,
            multi_input=False,
            bellman_probe=False,
            bellman_probe_size=1024,
            bellman_probe_interval=100000,
            bellman_probe_targets=8,
            bellman_probe_ridge=1e-3,
            bellman_probe_dir='bellman_probe_results',
            bellman_spectral_stats=False,
            bellman_spectral_anchor_size=64,
            bellman_spectral_fit_lr=3e-4,
            bellman_reference_dir=None,
            bellman_spectral_task_steps=(10000, 50000, 100000, 500000,
                                         1000000, 1500000),
            pbsr=False,
            pbsr_coef=0.1,
            pbsr_anchor_size=64,
            pbsr_targets=8,
            pbsr_ridge=1e-3,
            pbsr_update_interval=100,
            pbsr_train_task_count=1,
            task_names=None,
            exact_task_budget=False,
            branch_checkpoint=None,
            branch_task_step=0,
            branch_alpha=None,
            branch_online_critic_source='inherited',
            branch_target_critic_source='inherited',
            plasticity_injection_mode='none',
            plasticity_injection_width=256,
            plasticity_injection_widths=(32, 64, 128, 256),
            plasticity_injection_rows=64,
            plasticity_injection_targets=8,
            plasticity_injection_ridge=1e-3,
            plasticity_injection_task_indices=None,
            demand_aligned_reserve=False,
            dar_rows=64,
            dar_hidden_dim=256,
            dar_feature_dim=64,
            dar_targets=8,
            dar_ridge=1e-3,
            dar_trace_ratio=1.0,
            dar_capacity_price=0.0,
            dar_alignment_steps=200,
            dar_alignment_lr=1e-3,
            dar_task_indices=None,
            dsr_v2=False,
            dsr_v2_kwargs=None,
            spectral_regularization=False,
            spectral_actor_coef=1e-4,
            spectral_critic_coef=1e-4,
            spectral_power_iterations=1):

        self._qf1 = qf1
        self._qf2 = qf2
        self.replay_buffer = replay_buffer
        self._tau = target_update_tau
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._initial_log_entropy = initial_log_entropy
        self._gradient_steps = gradient_steps_per_itr
        self._optimizer = optimizer
        self._num_evaluation_episodes = num_evaluation_episodes
        self._eval_env = eval_env
        self._seed = seed
        self._infer = infer
        self._wasserstein = (wasserstein > 0)
        self._ReDo = ReDo
        self._redo_interval = int(redo_interval)
        self._redo_tau = float(redo_tau)
        if self._redo_interval < 1 or self._redo_tau < 0:
            raise ValueError('ReDo interval must be positive and tau non-negative.')
        self._no_stats = no_stats
        self._scalar_log_interval = int(scalar_log_interval)
        self._feature_stats_interval = int(feature_stats_interval)
        self._hessian_stats_interval = int(hessian_stats_interval)
        if min(self._scalar_log_interval, self._feature_stats_interval,
               self._hessian_stats_interval) < 1:
            raise ValueError('Logging intervals must be positive environment-step counts.')
        self._multi_input = multi_input
        self._bellman_probe = bellman_probe
        self._bellman_probe_size = bellman_probe_size
        self._bellman_probe_interval = bellman_probe_interval
        self._bellman_probe_targets = bellman_probe_targets
        self._bellman_probe_ridge = bellman_probe_ridge
        self._bellman_spectral_stats_enabled = bool(bellman_spectral_stats)
        self._pbsr_enabled = bool(pbsr)
        self._pbsr_update_interval = int(pbsr_update_interval)
        self._pbsr_train_task_count = int(pbsr_train_task_count)
        self._pbsr_last_stats = {}
        self._pbsr = (
            PBSRHead(
                anchor_size=pbsr_anchor_size,
                target_count=pbsr_targets,
                ridge=pbsr_ridge,
                gradient_ratio=pbsr_coef,
                seed=seed)
            if self._pbsr_enabled else None)
        self._bellman_probe_buffers = {}
        self._bellman_probe_task_start_step = 0
        self._task_names = task_names
        self._branch_task_step = branch_task_step
        self._plasticity_injection_mode = plasticity_injection_mode
        self._plasticity_injection_width = int(plasticity_injection_width)
        self._plasticity_injection_widths = tuple(
            int(width) for width in plasticity_injection_widths)
        self._plasticity_injection_rows = int(plasticity_injection_rows)
        self._plasticity_injection_targets = int(
            plasticity_injection_targets)
        self._plasticity_injection_ridge = float(plasticity_injection_ridge)
        self._plasticity_injection_task_indices = (
            None if plasticity_injection_task_indices is None else
            set(int(index) for index in plasticity_injection_task_indices))
        self._plasticity_injection_records = []
        self._plasticity_injected_tasks = set()
        self._demand_aligned_reserve_enabled = bool(
            demand_aligned_reserve)
        self._dar_rows = int(dar_rows)
        self._dar_task_indices = (
            None if dar_task_indices is None else
            set(int(index) for index in dar_task_indices))
        self._dar_prepared_tasks = set()
        self._dar_records = []
        # v2 使用独立开关，原版 DAR 的实现和实验入口保持可用。
        self._dsr_v2_enabled = bool(dsr_v2)
        if self._dsr_v2_enabled and (demand_aligned_reserve or
                plasticity_injection_mode != 'none' or q_reset):
            raise ValueError('DSR v2 请单独启用，不能同时使用 DAR、PI 或 q_reset')
        self._dsr_v2 = SACDSRV2(
            num_tasks=len(task_names) if task_names else 1,
            **(dsr_v2_kwargs or {}))
        self._spectral_regularization_enabled = bool(
            spectral_regularization)
        self._spectral_actor_coef = float(spectral_actor_coef)
        self._spectral_critic_coef = float(spectral_critic_coef)
        if self._spectral_actor_coef < 0 or self._spectral_critic_coef < 0:
            raise ValueError('Spectral regularization coefficients must be non-negative.')
        self._spectral_regularizer = (
            LayerSpectralRegularizer(spectral_power_iterations)
            if self._spectral_regularization_enabled else None)
        self._spectral_regularization_last_stats = {}
        self._demand_aligned_reserve = SACDemandAlignedReserve(
            capacity_price=dar_capacity_price,
            hidden_dim=dar_hidden_dim,
            feature_dim=dar_feature_dim,
            target_count=dar_targets,
            relative_ridge=dar_ridge,
            trace_ratio=dar_trace_ratio,
            alignment_steps=dar_alignment_steps,
            alignment_lr=dar_alignment_lr,
        )
        if self._pbsr_update_interval < 1:
            raise ValueError('pbsr_update_interval must be positive.')
        if self._pbsr_train_task_count < 1:
            raise ValueError('pbsr_train_task_count must be positive.')
        self._critic_optimizer_steps = 0
        self._restored_branch_alpha = None

        # Total number of CL tasks
        self.masks = None

        self._log_name = log_name
        self._use_wandb = use_wandb

        if self._bellman_probe:
            self._bellman_probe_run_dir = os.path.join(
                bellman_probe_dir, self._log_name)
            self._bellman_probe_checkpoint_dir = os.path.join(
                self._bellman_probe_run_dir, 'checkpoints')
            os.makedirs(self._bellman_probe_checkpoint_dir, exist_ok=True)
            self._bellman_probe_metrics_path = os.path.join(
                self._bellman_probe_run_dir, 'metrics.jsonl')
            self._bellman_spectral_probe = (
                BellmanSpectralStats(
                    run_dir=self._bellman_probe_run_dir,
                    anchor_size=bellman_spectral_anchor_size,
                    target_count=bellman_probe_targets,
                    relative_ridge=bellman_probe_ridge,
                    fit_learning_rate=bellman_spectral_fit_lr,
                    reference_dir=bellman_reference_dir,
                    task_steps=bellman_spectral_task_steps,
                    seed=seed)
                if self._bellman_spectral_stats_enabled else None)
        if self._pbsr_enabled:
            pbsr_run_dir = os.path.join(bellman_probe_dir, self._log_name)
            os.makedirs(pbsr_run_dir, exist_ok=True)
            pbsr_config = {
                'method': 'pbsr_head',
                'probe_source': 'current_task_replay_minibatch',
                'probe_definition': (
                    'one_step_random_value_head_policy_mixture_'
                    'random_cumulant'),
                'policy_mixture': [
                    'current_actor_mean', 'deterministic_noisy_actor',
                    'task_independent_action'],
                'bellman_horizons': [1],
                'anchor_size': pbsr_anchor_size,
                'target_count': pbsr_targets,
                'normalized_kernel': 'HHt_over_d_then_trace_normalized',
                'probe_column_normalization': 'center_then_unit_l2',
                'ridge': pbsr_ridge,
                'gradient_ratio': pbsr_coef,
                'update_interval': self._pbsr_update_interval,
                'train_task_count': self._pbsr_train_task_count,
                'future_task_reference_used_for_training': False,
            }
            with open(os.path.join(
                    pbsr_run_dir, 'pbsr_config.json'), 'w') as config_file:
                json.dump(pbsr_config, config_file, indent=2)
            print('PBSR_CONFIG', json.dumps(pbsr_config), flush=True)
        if self._spectral_regularization_enabled:
            spectral_config = {
                'method': 'spectral_regularization',
                'formula': 'sum_l[(sigma_max(W_l)^2-1)^2+||b_l||_2^4]',
                'actor_coefficient': self._spectral_actor_coef,
                'critic_coefficient': self._spectral_critic_coef,
                'power_iterations': self._spectral_regularizer.power_iterations,
                'regularized_networks': [
                    'actor_shared_and_current_task_head',
                    'online_qf1', 'online_qf2'],
                'target_networks_regularized': False,
            }
            print('SPECTRAL_REG_CONFIG', json.dumps(spectral_config), flush=True)
        self._min_buffer_size = min_buffer_size
        self._steps_per_epoch = steps_per_epoch
        self._buffer_batch_size = buffer_batch_size
        self._discount = discount
        self._reward_scale = reward_scale
        if isinstance(env_spec,list):
            max_episode_length = env_spec[0].max_episode_length
        else:
            max_episode_length = env_spec.max_episode_length
        self.max_episode_length = max_episode_length
        self._max_episode_length_eval = max_episode_length
        self._beta = 0.05
        self._rho = 0.00001

        if max_episode_length_eval is not None:
            self._max_episode_length_eval = max_episode_length_eval
        self._use_deterministic_evaluation = use_deterministic_evaluation
        self._use_exploration = use_exploration
        self._q_reset = q_reset
        self._policy_reset = policy_reset
        self._first_task = first_task
        self._exact_task_budget = bool(exact_task_budget)

        self._temporal_regularization_factor = temporal_regularization_factor
        self._spatial_regularization_factor = spatial_regularization_factor
        self._spatial_regularization_dist = torch.distributions.Normal(
            0, spatial_regularization_eps)

        self.policy = policy
        self.env_spec = env_spec
        self.replay_buffer = replay_buffer

        self._sampler = sampler

        self._reward_scale = reward_scale
        # use 2 target q networks
        self._target_qf1 = copy.deepcopy(self._qf1)
        self._target_qf2 = copy.deepcopy(self._qf2)
        self._policy_optimizer = self._make_network_optimizer(self.policy, self._policy_lr)
        self._qf1_optimizer = self._make_network_optimizer(self._qf1, self._qf_lr)
        self._qf2_optimizer = self._make_network_optimizer(self._qf2, self._qf_lr)
        
        # automatic entropy coefficient tuning
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                if isinstance(self.env_spec, list):
                    self._target_entropy = [-np.prod(spec.action_space.shape).item() for spec in self.env_spec]
                    
                else:
                    self._target_entropy = -np.prod(
                            self.env_spec.action_space.shape).item()

        self._reset_alpha()
        
        self.episode_rewards = deque(maxlen=30)
        self.recent_trajectory = RecentTrajectory(maxlen=10000)

        self.global_step = 0
        self.global_env_step = 0
        self._task_env_start_step = 0
        self._last_scalar_log_env_step = -1
        self._last_bellman_probe_env_step = -1
        self._last_spectral_probe_env_step = -1
        self._last_redo_env_step = -1
        # self.global_step = -self._gradient_steps
        self.seq_idx = 0
        self.start_time = time()
        self.begin = self.start_time
        self.results = {}

        self.results['Running avg. of episode return'] = []
        self.results['Policy loss'] = []
        self.results['Q loss'] = []
        self.results['Alpha'] = []
        self.results['Speed (it/s)'] = []
        self.results['Environment speed (steps/s)'] = []
        self.results['Global env step'] = []
        self.results['Task env step'] = []
        self.results['Global critic updates'] = []
        self.results['ReDo events'] = []
        if self._pbsr_enabled:
            self.results['PBSR Qf1 TD loss'] = []
            self.results['PBSR Qf2 TD loss'] = []
            self.results['PBSR Qf1 loss'] = []
            self.results['PBSR Qf2 loss'] = []
            self.results['PBSR Qf1 coefficient'] = []
            self.results['PBSR Qf2 coefficient'] = []
            self.results['PBSR Qf1 slowest30 energy'] = []
            self.results['PBSR Qf2 slowest30 energy'] = []

        if self._no_stats == False:
        
            self.results['Policy zero ratio'] = []
            self.results['Qf1 zero ratio'] = []
            self.results['Qf2 zero ratio'] = []
            self.results['Policy feature rank'] = []
            self.results['Qf1 feature rank'] = []
            self.results['Qf2 feature rank'] = []
            self.results['Policy hessian rank'] = []
            self.results['Qf1 hessian rank'] = []
            self.results['Qf2 hessian rank'] = []
            self.results['Policy weight change'] = []
            self.results['Qf1 weight change'] = []
            self.results['Qf2 weight change'] = []
        

        self._random_qf1_state_dict = copy.deepcopy(self._qf1.state_dict())
        self._random_qf2_state_dict = copy.deepcopy(self._qf2.state_dict())
        self._random_policy_state_dict = copy.deepcopy(self.policy.state_dict())
        
        # For ReDo
        self._random_qf1 = copy.deepcopy(self._qf1)
        self._random_qf2 = copy.deepcopy(self._qf2)
        self.random_policy = copy.deepcopy(self.policy)

        random_policy_network = self.random_policy._module._shared_mean_log_std_network
        

        if infer:
            self._infer_target_policy = copy.deepcopy(self.policy)
            self._infer_target_qf1 = copy.deepcopy(self._qf1)
            self._infer_target_qf2 = copy.deepcopy(self._qf2)
            self._infer_alpha = 1.
            self._infer_beta = 10.
            print("Use InFeR Loss")

        if wasserstein:
            self._wasserstein_target_policy = copy.deepcopy(self.policy)
            self._wasserstein_target_qf1 = copy.deepcopy(self._qf1)
            self._wasserstein_target_qf2 = copy.deepcopy(self._qf2)
            self._wasserstein_lambda = wasserstein
            print("Use Wasserstein Regularization")

        if self._q_reset:
            print('Reset Q when task is changed')
        
        if self._policy_reset:
            print('Reset policy when task is changed')

        if branch_checkpoint is not None:
            checkpoint = torch.load(branch_checkpoint, map_location='cpu')
            if checkpoint.get('plasticity_injections'):
                self._restore_plasticity_injections(
                    checkpoint['plasticity_injections'])
            if checkpoint.get('dsr_v2') is not None:
                if not self._dsr_v2_enabled:
                    raise ValueError('恢复 DSR v2 检查点时必须启用 --dsr_v2 True')
                self._dsr_v2.load_state_dict(checkpoint['dsr_v2'])
                for qf, target_qf, key in (
                        (self._qf1, self._target_qf1, 'qf1'),
                        (self._qf2, self._target_qf2, 'qf2')):
                    restore_dsr_v2_branches(qf, target_qf, checkpoint[key])
            self.policy.load_state_dict(checkpoint['policy'])
            if branch_task_step > 0:
                branch_states = {
                    'qf1': checkpoint['qf1'],
                    'qf2': checkpoint['qf2'],
                    'target_qf1': checkpoint.get(
                        'target_qf1', checkpoint['qf1']),
                    'target_qf2': checkpoint.get(
                        'target_qf2', checkpoint['qf2']),
                }
                online_source = 'checkpoint'
                target_source = 'checkpoint'
            else:
                online_source = (
                    'fresh' if self._q_reset
                    else branch_online_critic_source)
                target_source = (
                    'fresh' if self._q_reset
                    else branch_target_critic_source)
                branch_states = select_branch_critic_states(
                    checkpoint,
                    self._random_qf1_state_dict,
                    self._random_qf2_state_dict,
                    online_source,
                    target_source)
            self._qf1.load_state_dict(branch_states['qf1'])
            self._qf2.load_state_dict(branch_states['qf2'])
            self._target_qf1.load_state_dict(branch_states['target_qf1'])
            self._target_qf2.load_state_dict(branch_states['target_qf2'])

            if 'policy_optimizer' in checkpoint:
                self._policy_optimizer.load_state_dict(
                    checkpoint['policy_optimizer'])
            if 'qf1_optimizer' in checkpoint:
                self._qf1_optimizer.load_state_dict(
                    checkpoint['qf1_optimizer'])
            if 'qf2_optimizer' in checkpoint:
                self._qf2_optimizer.load_state_dict(
                    checkpoint['qf2_optimizer'])

            self.global_step = checkpoint['global_step']
            if branch_task_step > 0:
                self.seq_idx = int(checkpoint['seq_idx'])
            else:
                self.seq_idx = int(checkpoint.get('seq_idx', 0)) + 1
            self._bellman_probe_task_start_step = (
                self.global_step - branch_task_step)
            self._bellman_probe_buffers = checkpoint.get('probe_buffers', {})
            self._critic_optimizer_steps = (
                int(checkpoint.get('critic_optimizer_steps', 0))
                if branch_task_step > 0 else 0)
            for env in self._sampler._envs:
                env.cur_step = branch_task_step
                env.cur_seq_idx = self.seq_idx

            restored_alpha = branch_alpha
            if restored_alpha is None and 'log_alpha' in checkpoint:
                restored_alpha = float(
                    torch.as_tensor(checkpoint['log_alpha']).exp().item())
            self._restored_branch_alpha = restored_alpha
            if restored_alpha is not None and self._use_automatic_entropy_tuning:
                self._log_alpha = list_to_tensor(
                    [restored_alpha]).log().requires_grad_()
                self._alpha_optimizer = self._optimizer(
                    [self._log_alpha], lr=self._policy_lr)

            print('Loaded branch checkpoint:', branch_checkpoint)
            print('Branch task/step:', self.seq_idx, branch_task_step)
            if branch_task_step == 0:
                print('BRANCH_CRITIC_INITIALIZATION', json.dumps({
                    'online_source': online_source,
                    'target_source': target_source,
                    'q_reset_alias': bool(self._q_reset),
                }))
            print('Branch replay buffer starts empty.')

        if self._first_task is not None:
            if 'DMC' in self._first_task:
                policy_model_name = 'policy_dm_control_sac_{}_1000000_{}.pt'
                qf1_model_name = 'qf1_dm_control_sac_{}_1000000_{}.pt'
                target_qf1_model_name = 'target_qf1_dm_control_sac_{}_1000000_{}.pt'
                qf2_model_name = 'qf2_dm_control_sac_{}_1000000_{}.pt'
                target_qf2_model_name = 'target_qf2_dm_control_sac_{}_1000000_{}.pt'

                
            else:
                policy_model_name = 'policy_metaworld_sac_{}_3000000_{}.pt'
                qf1_model_name = 'qf1_metaworld_sac_{}_3000000_{}.pt'
                target_qf1_model_name = 'target_qf1_metaworld_sac_{}_3000000_{}.pt'
                qf2_model_name = 'qf2_metaworld_sac_{}_3000000_{}.pt'
                target_qf2_model_name = 'target_qf1_metaworld_sac_{}_3000000_{}.pt'
                if crelu:
                    policy_model_name = 'policy_CReLU_metaworld_sac_{}_3000000_{}.pt'
                    qf1_model_name = 'qf1_CReLU_metaworld_sac_{}_3000000_{}.pt'
                    target_qf1_model_name = 'target_qf1_CReLU_metaworld_sac_{}_3000000_{}.pt'
                    qf2_model_name = 'qf2_CReLU_metaworld_sac_{}_3000000_{}.pt'
                    target_qf2_model_name = 'target_qf2_CReLU_metaworld_sac_{}_3000000_{}.pt'
                if wasserstein:
                    policy_model_name = 'policy_Wasserstein_0.1_metaworld_sac_{}_3000000_{}.pt'
                    qf1_model_name = 'qf1_Wasserstein_0.1_metaworld_sac_{}_3000000_{}.pt'
                    target_qf1_model_name = 'target_qf1_Wasserstein_0.1_metaworld_sac_{}_3000000_{}.pt'
                    qf2_model_name = 'qf2_Wasserstein_0.1_metaworld_sac_{}_3000000_{}.pt'
                    target_qf2_model_name = 'target_qf2_Wasserstein_0.1_metaworld_sac_{}_3000000_{}.pt'
            
                
                
            load_model(self.policy, policy_model_name, self._first_task, self._seed)
            load_model(self._qf1, qf1_model_name, self._first_task, self._seed)
            load_model(self._target_qf1, target_qf1_model_name, self._first_task, self._seed)
            load_model(self._qf2, qf2_model_name, self._first_task, self._seed)
            load_model(self._target_qf2, target_qf2_model_name, self._first_task, self._seed)

            
                

            if self._q_reset:
                print('############################################################')
                print('                        Q-reset!!!!!                        ')
                print('############################################################')
                self._qf1.load_state_dict(self._random_qf1_state_dict)
                self._qf2.load_state_dict(self._random_qf2_state_dict)

                self._target_qf1 = copy.deepcopy(self._qf1)
                self._target_qf2 = copy.deepcopy(self._qf2)
            
            if self._policy_reset:
                print('############################################################')
                print('                     Policy-reset!!!!!                      ')
                print('############################################################')
                self.policy.load_state_dict(self._random_policy_state_dict)


    def _append_plasticity_branch(self, qf, target_qf, optimizer,
                                  max_width, active_width, branch_seed):
        branch = PlasticityInjectionBranch(
            qf._obs_dim + qf._action_dim,
            max_width=max_width,
            active_width=active_width,
            seed=branch_seed).to(next(qf.parameters()).device)
        qf.append_plasticity_injection(branch)
        target_qf.append_plasticity_injection(copy.deepcopy(branch))
        optimizer.add_param_group({
            'params': branch.trainable_parameters(),
        })
        return branch

    def _restore_plasticity_injections(self, records):
        for record in records:
            self._append_plasticity_branch(
                self._qf1, self._target_qf1, self._qf1_optimizer,
                record['max_width'], record['selected_width'],
                record['qf1_seed'])
            self._append_plasticity_branch(
                self._qf2, self._target_qf2, self._qf2_optimizer,
                record['max_width'], record['selected_width'],
                record['qf2_seed'])
            self._plasticity_injected_tasks.add(int(record['task']))
        self._plasticity_injection_records = copy.deepcopy(records)

    def _make_network_optimizer(self, model, lr):
        """默认完全沿用原优化器；Muon 子类仅替换网络参数的优化器。"""
        return self._optimizer(model.parameters(), lr=lr)

    def _plasticity_injection_due(self, seq_idx):
        if self._plasticity_injection_mode == 'none' or seq_idx == 0:
            return False
        if seq_idx in self._plasticity_injected_tasks:
            return False
        return (self._plasticity_injection_task_indices is None or
                seq_idx in self._plasticity_injection_task_indices)

    def _apply_plasticity_injection(self, samples, seq_idx):
        observations = samples['observation']
        actions = samples['action']
        device = observations.device
        with torch.no_grad():
            alpha = self._get_log_alpha(samples).exp()
            q1_directions, q2_directions = (
                deterministic_soft_bellman_directions(
                    samples, seq_idx, self._qf1, self._qf2,
                    self._target_qf1, self._target_qf2, self.policy, alpha,
                    self._discount, self._reward_scale,
                    self._plasticity_injection_targets))
            q1_before = self._qf1(
                observations, actions, seq_idx=seq_idx).clone()
            q2_before = self._qf2(
                observations, actions, seq_idx=seq_idx).clone()
            target_q1_before = self._target_qf1(
                observations, actions, seq_idx=seq_idx).clone()
            target_q2_before = self._target_qf2(
                observations, actions, seq_idx=seq_idx).clone()

        max_width = max(
            (self._plasticity_injection_width,) +
            self._plasticity_injection_widths)
        qf1_seed = self._seed * 100003 + seq_idx * 1009 + 7901
        qf2_seed = qf1_seed + 53
        branch1 = PlasticityInjectionBranch(
            self._qf1._obs_dim + self._qf1._action_dim,
            max_width=max_width,
            active_width=max_width,
            seed=qf1_seed).to(device)
        branch2 = PlasticityInjectionBranch(
            self._qf2._obs_dim + self._qf2._action_dim,
            max_width=max_width,
            active_width=max_width,
            seed=qf2_seed).to(device)

        fresh_qf1 = copy.deepcopy(self._random_qf1).to(device)
        fresh_qf2 = copy.deepcopy(self._random_qf2).to(device)
        fixed_width = (
            self._plasticity_injection_width
            if self._plasticity_injection_mode == 'fixed' else None)
        selection = select_plasticity_width(
            self._qf1, self._qf2, fresh_qf1, fresh_qf2,
            branch1, branch2, observations, actions, seq_idx,
            q1_directions, q2_directions,
            self._plasticity_injection_widths,
            self._plasticity_injection_ridge,
            fixed_width=fixed_width)
        selected_width = selection['selected_width']

        self._qf1.append_plasticity_injection(branch1)
        self._qf2.append_plasticity_injection(branch2)
        self._target_qf1.append_plasticity_injection(copy.deepcopy(branch1))
        self._target_qf2.append_plasticity_injection(copy.deepcopy(branch2))
        self._qf1_optimizer.add_param_group({
            'params': branch1.trainable_parameters(),
        })
        self._qf2_optimizer.add_param_group({
            'params': branch2.trainable_parameters(),
        })

        with torch.no_grad():
            q1_after = self._qf1(
                observations, actions, seq_idx=seq_idx)
            q2_after = self._qf2(
                observations, actions, seq_idx=seq_idx)
            target_q1_after = self._target_qf1(
                observations, actions, seq_idx=seq_idx)
            target_q2_after = self._target_qf2(
                observations, actions, seq_idx=seq_idx)

        record = {
            'mode': self._plasticity_injection_mode,
            'task': int(seq_idx),
            'task_name': self._task_names[seq_idx],
            'global_step': int(self.global_step),
            'rows': int(len(observations)),
            'target_count': self._plasticity_injection_targets,
            'relative_ridge': self._plasticity_injection_ridge,
            'max_width': max_width,
            'selected_width': selected_width,
            'qf1_seed': qf1_seed,
            'qf2_seed': qf2_seed,
            'qf1_output_drift': float((q1_after - q1_before).abs().max()),
            'qf2_output_drift': float((q2_after - q2_before).abs().max()),
            'target_qf1_output_drift': float(
                (target_q1_after - target_q1_before).abs().max()),
            'target_qf2_output_drift': float(
                (target_q2_after - target_q2_before).abs().max()),
        }
        record.update(selection)
        self._plasticity_injection_records.append(record)
        self._plasticity_injected_tasks.add(seq_idx)

        if self._bellman_probe:
            artifact_path = os.path.join(
                self._bellman_probe_run_dir,
                'plasticity_injection_task{}.json'.format(seq_idx))
            with open(artifact_path, 'w') as artifact_file:
                json.dump(record, artifact_file, indent=2)
        print('PLASTICITY_INJECTION', json.dumps(record), flush=True)

    def _demand_aligned_reserve_due(self, seq_idx):
        """判断当前新任务是否需要进行一次谱储备准备。"""
        if not self._demand_aligned_reserve_enabled or seq_idx == 0:
            return False
        if seq_idx in self._dar_prepared_tasks:
            return False
        return (self._dar_task_indices is None or
                seq_idx in self._dar_task_indices)

    def _prepare_demand_aligned_reserve(self, samples, seq_idx):
        """在新任务第一次 critic 更新前对齐或复用学习谱。"""
        alpha = self._get_log_alpha(samples).exp().detach()
        result = self._demand_aligned_reserve.prepare(
            samples=samples,
            task_idx=seq_idx,
            qf1=self._qf1,
            qf2=self._qf2,
            target_qf1=self._target_qf1,
            target_qf2=self._target_qf2,
            qf1_optimizer=self._qf1_optimizer,
            qf2_optimizer=self._qf2_optimizer,
            policy=self.policy,
            alpha=alpha,
            discount=self._discount,
            reward_scale=self._reward_scale,
        )
        record = {
            'method': 'demand_aligned_reserve',
            'task': int(seq_idx),
            'task_name': self._task_names[seq_idx],
            'global_step': int(self.global_step),
            'rows': int(len(samples['observation'])),
            'qf1': result['qf1'],
            'qf2': result['qf2'],
        }
        self._dar_records.append(record)
        self._dar_prepared_tasks.add(seq_idx)

        if self._bellman_probe:
            artifact_path = os.path.join(
                self._bellman_probe_run_dir,
                'demand_aligned_reserve_task{}.json'.format(seq_idx))
            with open(artifact_path, 'w') as artifact_file:
                json.dump(record, artifact_file, indent=2)
        print('DEMAND_ALIGNED_RESERVE', json.dumps(record), flush=True)

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        self.recent_policy_state_dict = copy.deepcopy(self.policy.state_dict())
        self.recent_qf1_state_dict = copy.deepcopy(self._qf1.state_dict())
        self.recent_qf2_state_dict = copy.deepcopy(self._qf2.state_dict())
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        # 从 checkpoint 恢复时，吞吐分子只统计本次 train 调用新增的更新。
        speed_start_step = self.global_step
        speed_start_env_step = self.global_env_step
        speed_start_time = time()
        for env in self._sampler._envs:
            env.reset()
            
        for _ in trainer.step_epochs():
            # Exact-budget runs end an epoch on actual interactions, not on
            # collection calls: the 10k warm-up is one call, not one ordinary
            # 500/1000-step batch. Legacy update-budget runs retain their loop.
            epoch_env_end = (self.global_env_step + self._steps_per_epoch *
                             trainer._train_args.batch_size
                             if self._exact_task_budget else None)
            collection_index = 0
            while (self.global_env_step < epoch_env_end if self._exact_task_budget
                   else collection_index < self._steps_per_epoch):
                collection_index += 1
                if self._exact_task_budget:
                    next_task = self._sampler._envs[0].cur_seq_idx
                    if next_task != self.seq_idx:
                        if self._task_names and next_task >= len(self._task_names):
                            raise RuntimeError('Exact SAC budget exceeds the task sequence')
                        # Defer the incoming intervention until after the
                        # outgoing task's final evaluation and diagnostics.
                        self.task_change(self.seq_idx)
                        self.seq_idx = next_task
                if not (self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                if self._exact_task_budget:
                    batch_size = min(
                        batch_size or trainer._train_args.batch_size,
                        epoch_env_end - self.global_env_step)
                
                if self._use_exploration and batch_size is not None:
                    if self._multi_input:
                        discount_tensor = torch.full((1000,), self._discount)
                        filter = torch.cumprod(discount_tensor, dim=0) / self._discount
                    else:
                        discount_tensor = torch.full((500,), self._discount)
                        filter = torch.cumprod(discount_tensor, dim=0) / self._discount
                    
                    episodes_list = []
                    return_list = []
                    for seq in range(self.seq_idx+1):
                        episodes = trainer.obtain_samples(
                            trainer.step_itr, seq, batch_size)
                        episodes_list.append(episodes)
                        ret = 0
                        for path in episodes:
                            rewards = path['rewards']
                            ret = np.sum(rewards * filter.numpy())
                        return_list.append(ret.item())
                        
                    best_ret_arg = np.argsort(np.array(return_list))[-1]
                    
                    trainer.step_episode = episodes_list[best_ret_arg]

                else:
                    trainer.step_episode = trainer.obtain_samples(
                            trainer.step_itr, self.seq_idx, batch_size)
                
                path_returns = []
                collected_env_steps = 0
                for path in trainer.step_episode:
                    replay_path = dict(
                        observation=path['observations'],
                        action=path['actions'],
                        reward=path['rewards'].reshape(-1, 1),
                        next_observation=path['next_observations'],
                        terminal=np.array([
                            step_type == StepType.TERMINAL
                            for step_type in path['step_types']
                        ]).reshape(-1, 1))
                    self.replay_buffer.add_path(replay_path)
                    if self._bellman_probe:
                        self._append_bellman_probe(replay_path, self.seq_idx)
                    path_returns.append(path['rewards'])
                    collected_env_steps += len(path['rewards'])
                    self.recent_trajectory.append(path)
                assert len(path_returns) == len(trainer.step_episode)
                self.global_env_step += collected_env_steps
                if self._exact_task_budget and self.global_env_step > epoch_env_end:
                    raise RuntimeError('Sampler overshot the exact SAC evaluation budget')
                self.episode_rewards.append(np.mean(path_returns))

                

                for gradient_index in range(self._gradient_steps):
                    self._is_last_collection_update = (
                        gradient_index == self._gradient_steps - 1)
                    policy_loss, qf1_loss, qf2_loss = self.train_once(self.seq_idx)
                    self.global_step += 1
                    task_step = self.global_env_step - self._task_env_start_step
                    redo_due = (
                        self._ReDo and task_step > 0 and
                        task_step % self._redo_interval == 0 and
                        gradient_index == self._gradient_steps - 1 and
                        self.global_env_step != self._last_redo_env_step)
                    if redo_due:
                        redo_event = self.ReDo(self.seq_idx)
                        self._last_redo_env_step = self.global_env_step
                        self.results['ReDo events'].append(redo_event)
                        if self._use_wandb:
                            wandb.log({
                                'Global env step': self.global_env_step,
                                'Task env step': task_step,
                                'ReDo recycled policy neurons':
                                    redo_event['policy_recycled'],
                                'ReDo recycled Qf1 neurons':
                                    redo_event['qf1_recycled'],
                                'ReDo recycled Qf2 neurons':
                                    redo_event['qf2_recycled'],
                            })
                    interval_probe_due = (
                        task_step > 0 and
                        task_step % self._bellman_probe_interval == 0 and
                        gradient_index == self._gradient_steps - 1 and
                        self.global_env_step != self._last_bellman_probe_env_step)
                    spectral_probe_due = (
                        self._bellman_spectral_stats_enabled and
                        gradient_index == self._gradient_steps - 1 and
                        self.global_env_step != self._last_spectral_probe_env_step and
                        self._bellman_spectral_probe.should_run(
                            'interval', task_step))
                    if (self._bellman_probe and
                            (interval_probe_due or spectral_probe_due)):
                        self._run_bellman_probe('interval', self.seq_idx)
                        if spectral_probe_due:
                            self._last_spectral_probe_env_step = self.global_env_step
                        if interval_probe_due:
                            self._last_bellman_probe_env_step = self.global_env_step
                            self._save_bellman_probe_checkpoint(
                                self.seq_idx, 'interval')
                    with torch.no_grad():
                        alpha = self._log_alpha.exp()
                    end_time = time()
                    
                    scalar_due = (
                        task_step > 0 and
                        task_step % self._scalar_log_interval == 0 and
                        gradient_index == self._gradient_steps - 1 and
                        self.global_env_step != self._last_scalar_log_env_step)
                    if scalar_due:
                        self._last_scalar_log_env_step = self.global_env_step
                        training_speed = ((self.global_step - speed_start_step) /
                                          max(end_time - speed_start_time, 1e-9))
                        environment_speed = (
                            (self.global_env_step - speed_start_env_step) /
                            max(end_time - speed_start_time, 1e-9))
                        reward_avg = (sum(self.episode_rewards) /
                                      len(self.episode_rewards))
                        wandb_metrics = {
                            'Global env step': self.global_env_step,
                            'Task env step': task_step,
                            'Global critic updates': self.global_step,
                            'Running avg. of episode return': reward_avg,
                            'Policy loss': policy_loss.item(),
                            'Q loss': (qf1_loss + qf2_loss).item(),
                            'Alpha': alpha.item(),
                            'Speed (it/s)': training_speed,
                            'Environment speed (steps/s)': environment_speed,
                        }

                        if self._no_stats == False:

                            # Dormant neurons
                            def recent_mean(values):
                                window = list(values)[-1000:]
                                return float(np.mean(window)) if window else float('nan')

                            policy_zero_cnt = recent_mean(
                                self.policy._stats['zero_ratio'])
                            qf1_zero_cnt = recent_mean(
                                self._qf1._stats['zero_ratio'])
                            qf2_zero_cnt = recent_mean(
                                self._qf2._stats['zero_ratio'])
                            wandb_metrics.update({
                                'Policy zero ratio': policy_zero_cnt,
                                'Qf1 zero ratio': qf1_zero_cnt,
                                'Qf2 zero ratio': qf2_zero_cnt,
                            })

                            feature_due = (
                                task_step % self._feature_stats_interval == 0)
                            hessian_due = (
                                task_step % self._hessian_stats_interval == 0)
                            if feature_due or hessian_due:
                                diagnostics_rng_state = (
                                    _capture_training_rng_state())
                                recent_obs = self.recent_trajectory.observation
                                recent_samples = self.recent_trajectory.samples

                            if hessian_due:
                                qf1_loss_hess, qf2_loss_hess = (
                                    self._critic_objective(
                                        recent_samples, self.seq_idx))
                                action_dists, new_actions, log_pi_new_actions = (
                                    self._get_policy_output(
                                        recent_obs, self.seq_idx))
                                policy_loss_hess = self._actor_objective(
                                    recent_samples, new_actions,
                                    log_pi_new_actions, seq_idx=self.seq_idx)
                                policy_loss_hess += (
                                    self._caps_regularization_objective(
                                        action_dists, recent_samples,
                                        self.seq_idx))
                                qf1_last_weight = self._qf1._output_layers[0][0].weight
                                qf2_last_weight = self._qf2._output_layers[0][0].weight
                                policy_last_weight = self.policy._module._shared_mean_log_std_network._output_layers[2*self.seq_idx][0].weight
                                qf1_hessian_rank = feature_rank(
                                    weight_hessian(qf1_loss_hess,
                                                   qf1_last_weight), 1e-5)
                                qf2_hessian_rank = feature_rank(
                                    weight_hessian(qf2_loss_hess,
                                                   qf2_last_weight), 1e-5)
                                policy_hessian_rank = feature_rank(
                                    weight_hessian(policy_loss_hess,
                                                   policy_last_weight), 1e-5)
                                wandb_metrics.update({
                                    'Policy hessian rank': policy_hessian_rank,
                                    'Qf1 hessian rank': qf1_hessian_rank,
                                    'Qf2 hessian rank': qf2_hessian_rank,
                                })
                                self.results['Policy hessian rank'].append(
                                    policy_hessian_rank)
                                self.results['Qf1 hessian rank'].append(
                                    qf1_hessian_rank)
                                self.results['Qf2 hessian rank'].append(
                                    qf2_hessian_rank)

                            if feature_due:
                                if not hessian_due:
                                    with torch.no_grad():
                                        self._critic_objective(
                                            recent_samples, self.seq_idx)
                                        self._get_policy_output(
                                            recent_obs, self.seq_idx)
                                eps = 0.001
                                normalizer = np.sqrt(max(len(recent_obs), 1))
                                policy_feature_rank = feature_rank(
                                    self.policy._feature / normalizer, eps)
                                qf1_feature_rank = feature_rank(
                                    self._qf1._feature / normalizer, eps)
                                qf2_feature_rank = feature_rank(
                                    self._qf2._feature / normalizer, eps)

                                policy_state_dict = copy.deepcopy(
                                    self.policy.state_dict())
                                qf1_state_dict = copy.deepcopy(
                                    self._qf1.state_dict())
                                qf2_state_dict = copy.deepcopy(
                                    self._qf2.state_dict())
                                policy_dev = weight_deviation(
                                    policy_state_dict,
                                    self.recent_policy_state_dict)
                                qf1_dev = weight_deviation(
                                    qf1_state_dict,
                                    self.recent_qf1_state_dict)
                                qf2_dev = weight_deviation(
                                    qf2_state_dict,
                                    self.recent_qf2_state_dict)
                                self.recent_policy_state_dict = policy_state_dict
                                self.recent_qf1_state_dict = qf1_state_dict
                                self.recent_qf2_state_dict = qf2_state_dict
                                wandb_metrics.update({
                                    'Policy feature rank': policy_feature_rank,
                                    'Qf1 feature rank': qf1_feature_rank,
                                    'Qf2 feature rank': qf2_feature_rank,
                                    'Policy weight change': policy_dev.item(),
                                    'Qf1 weight change': qf1_dev.item(),
                                    'Qf2 weight change': qf2_dev.item(),
                                })
                                self.results['Policy feature rank'].append(
                                    policy_feature_rank)
                                self.results['Qf1 feature rank'].append(
                                    qf1_feature_rank)
                                self.results['Qf2 feature rank'].append(
                                    qf2_feature_rank)
                                self.results['Policy weight change'].append(
                                    policy_dev.item())
                                self.results['Qf1 weight change'].append(
                                    qf1_dev.item())
                                self.results['Qf2 weight change'].append(
                                    qf2_dev.item())
                            if feature_due or hessian_due:
                                _restore_training_rng_state(
                                    diagnostics_rng_state)

                        if self._use_wandb:
                            wandb.log(wandb_metrics)


                        self.results['Running avg. of episode return'].append(reward_avg)
                        self.results['Policy loss'].append(policy_loss.item())
                        self.results['Q loss'].append((qf1_loss + qf2_loss).item())
                        self.results['Alpha'].append(alpha.item())
                        self.results['Speed (it/s)'].append(training_speed)
                        self.results['Environment speed (steps/s)'].append(
                            environment_speed)
                        self.results['Global env step'].append(
                            self.global_env_step)
                        self.results['Task env step'].append(task_step)
                        self.results['Global critic updates'].append(
                            self.global_step)
                        if self._pbsr_enabled:
                            pbsr_active = (
                                self.seq_idx < self._pbsr_train_task_count and
                                bool(self._pbsr_last_stats))
                            q1_pbsr = (
                                self._pbsr_last_stats['qf1']
                                if pbsr_active else {})
                            q2_pbsr = (
                                self._pbsr_last_stats['qf2']
                                if pbsr_active else {})
                            self.results['PBSR Qf1 TD loss'].append(
                                q1_pbsr.get('td_loss', float('nan')))
                            self.results['PBSR Qf2 TD loss'].append(
                                q2_pbsr.get('td_loss', float('nan')))
                            self.results['PBSR Qf1 loss'].append(
                                q1_pbsr.get('loss', float('nan')))
                            self.results['PBSR Qf2 loss'].append(
                                q2_pbsr.get('loss', float('nan')))
                            self.results['PBSR Qf1 coefficient'].append(
                                q1_pbsr.get('coefficient', float('nan')))
                            self.results['PBSR Qf2 coefficient'].append(
                                q2_pbsr.get('coefficient', float('nan')))
                            self.results[
                                'PBSR Qf1 slowest30 energy'].append(
                                    q1_pbsr.get(
                                        'slowest30_energy', float('nan')))
                            self.results[
                                'PBSR Qf2 slowest30 energy'].append(
                                    q2_pbsr.get(
                                        'slowest30_energy', float('nan')))

                        if self._no_stats == False:
                            self.results['Policy zero ratio'].append(policy_zero_cnt)
                            self.results['Qf1 zero ratio'].append(qf1_zero_cnt)
                            self.results['Qf2 zero ratio'].append(qf2_zero_cnt)
                        

                        print('ENV_STEP: {} UPDATE: {} '.format(
                            self.global_env_step, self.global_step),
                            'policy loss: {:.2f} '.format(policy_loss.item()),
                            'Q loss: {:.6f} '.format(
                                (qf1_loss + qf2_loss).item()),
                            'Alpha: {:.7f}'.format(alpha.item()),
                            'Reward avg.: {:.7f}'.format(reward_avg),
                            'Speed: {:.1f} env-steps/s'.format(
                                environment_speed))
                
                next_task = getattr(self._sampler._envs[0], "cur_seq_idx")
                # v2 的精确预算会到达最后一个环境的结束边界，最终评估仍属于末任务。
                dsr_v2_finished = ((self._dsr_v2_enabled or getattr(self, '_muon_enabled', False) or
                                    getattr(self, '_singular_clip_enabled', False) or
                                    getattr(self, '_bellman_response_enabled', False) or
                                    self._exact_task_budget) and self._task_names and
                                   next_task >= len(self._task_names))
                if (self.seq_idx != next_task and not dsr_v2_finished and
                        not self._exact_task_budget):
                    print('Task change')
                    print('Current task number =',self.seq_idx)
                    # NOTE: Must call self.task_change before changing self.seq_idx
                    self.task_change(self.seq_idx)
                    self.seq_idx = getattr(self._sampler._envs[0], "cur_seq_idx")
                    print('Next task number =',self.seq_idx)
            
            evaluation_rng_state = _capture_training_rng_state()
            try:
                last_return = self._evaluate_policy(trainer.step_itr)
            finally:
                _restore_training_rng_state(evaluation_rng_state)
            self.save_results()
            trainer.step_itr += 1

        if self._bellman_probe:
            final_task_idx = self.seq_idx
            self._run_bellman_probe('final', final_task_idx)
            self._save_bellman_probe_checkpoint(final_task_idx, 'final')

        return np.mean(last_return)

    def _append_bellman_probe(self, path, seq_idx):
        if seq_idx not in self._bellman_probe_buffers:
            self._bellman_probe_buffers[seq_idx] = {
                key: value[:0].copy() for key, value in path.items()
            }

        probe = self._bellman_probe_buffers[seq_idx]
        remaining = self._bellman_probe_size - len(probe['observation'])
        take = min(remaining, len(path['observation']))
        if take > 0:
            for key, value in path.items():
                probe[key] = np.concatenate((probe[key], value[:take]))

    @staticmethod
    def _bellman_energy_rank(singular_values):
        energy = singular_values.square()
        cutoff = 0.99 * energy.sum()
        return int(torch.searchsorted(torch.cumsum(energy, dim=0), cutoff).item() + 1)

    def _run_bellman_probe(self, event, current_task_idx):
        feature_blocks = []
        demand_blocks = []
        task_td_abs = {}
        task_rows = {}
        alpha_values = self._log_alpha.exp().detach()

        with torch.no_grad():
            for task_idx in sorted(self._bellman_probe_buffers):
                probe = self._bellman_probe_buffers[task_idx]
                obs = np_to_torch(probe['observation'])
                actions = np_to_torch(probe['action'])
                rewards = np_to_torch(probe['reward']).flatten()
                next_obs = np_to_torch(probe['next_observation'])
                terminals = np_to_torch(probe['terminal']).flatten()

                q_pred = self._qf1(obs, actions, seq_idx=task_idx).flatten()
                features = self._qf1._feature.detach().clone()
                task_demands = []
                alpha = (alpha_values[task_idx]
                         if alpha_values.numel() > 1
                         else alpha_values.reshape(()))
                next_action_dist = self.policy(next_obs, task_idx)[0]
                base_dist = next_action_dist._normal.base_dist
                action_axis = torch.arange(
                    1, base_dist.loc.shape[-1] + 1,
                    device=base_dist.loc.device,
                    dtype=base_dist.loc.dtype).unsqueeze(0)

                for target_idx in range(self._bellman_probe_targets):
                    noise = torch.sin((target_idx + 1) * action_axis)
                    pre_tanh = base_dist.loc + base_dist.scale * noise
                    next_actions = torch.tanh(pre_tanh)
                    next_log_pi = next_action_dist.log_prob(
                        value=next_actions, pre_tanh_value=pre_tanh)
                    target_q1 = self._target_qf1(
                        next_obs, next_actions, seq_idx=task_idx).flatten()
                    target_q2 = self._target_qf2(
                        next_obs, next_actions, seq_idx=task_idx).flatten()
                    target = rewards * self._reward_scale + (
                        1. - terminals) * self._discount * (
                            torch.min(target_q1, target_q2) - alpha * next_log_pi)
                    task_demands.append(target - q_pred)

                task_demand = torch.stack(task_demands, dim=1)
                feature_blocks.append(features)
                demand_blocks.append(task_demand)
                task_td_abs[str(task_idx)] = task_demand.abs().mean().item()
                task_rows[str(task_idx)] = len(obs)

            features = torch.cat(feature_blocks, dim=0)
            demands = torch.cat(demand_blocks, dim=0)
            features = features - features.mean(dim=0, keepdim=True)
            demands = demands - demands.mean(dim=0, keepdim=True)

            n_rows = features.shape[0]
            feature_u, feature_s, _ = torch.linalg.svd(
                features / np.sqrt(n_rows), full_matrices=False)
            demand_u, demand_s, _ = torch.linalg.svd(
                demands / np.sqrt(n_rows), full_matrices=False)
            feature_rank = self._bellman_energy_rank(feature_s)
            demand_rank = self._bellman_energy_rank(demand_s)

            overlap = torch.matmul(
                feature_u[:, :feature_rank].T,
                demand_u[:, :demand_rank])
            alignment = overlap.square().sum() / demand_rank

            feature_gram = torch.matmul(features.T, features) / n_rows
            ridge = (self._bellman_probe_ridge *
                     torch.trace(feature_gram) / feature_gram.shape[0])
            rhs = torch.matmul(features.T, demands) / n_rows
            weights = torch.linalg.solve(
                feature_gram + ridge * torch.eye(
                    feature_gram.shape[0], device=feature_gram.device), rhs)
            unexplained = demands - torch.matmul(features, weights)
            coverage = 1. - unexplained.square().sum() / demands.square().sum()
            stable_rank = feature_s.square().sum() / feature_s[0].square()

        task_step = self.global_env_step - self._task_env_start_step
        task_critic_step = (
            self.global_step - self._bellman_probe_task_start_step)
        metric = {
            'event': event,
            'global_step': self.global_step,
            'global_env_step': self.global_env_step,
            'global_critic_updates': self.global_step,
            'current_task': current_task_idx,
            'current_task_name': self._task_names[current_task_idx],
            'task_step': task_step,
            'task_env_step': task_step,
            'task_critic_updates': task_critic_step,
            'feature_rank_99': feature_rank,
            'feature_stable_rank': stable_rank.item(),
            'bellman_demand_rank_99': demand_rank,
            'bellman_alignment': alignment.item(),
            'bellman_coverage': coverage.item(),
            'bellman_misalignment': 1. - coverage.item(),
            'task_td_abs': task_td_abs,
            'task_rows': task_rows,
            'feature_singular_values': feature_s.cpu().tolist(),
            'bellman_singular_values': demand_s.cpu().tolist(),
        }
        if (self._bellman_spectral_stats_enabled and
                self._bellman_spectral_probe.should_run(event, task_step)):
            current_alpha = (
                alpha_values[current_task_idx]
                if alpha_values.numel() > 1
                else alpha_values.reshape(()))
            metric.update(self._bellman_spectral_probe.run(
                event=event,
                global_step=self.global_env_step,
                task_step=task_step,
                task_idx=current_task_idx,
                task_name=self._task_names[current_task_idx],
                probe=self._bellman_probe_buffers[current_task_idx],
                qf1=self._qf1,
                qf2=self._qf2,
                target_qf1=self._target_qf1,
                target_qf2=self._target_qf2,
                policy=self.policy,
                alpha=current_alpha,
                discount=self._discount,
                reward_scale=self._reward_scale))
        with open(self._bellman_probe_metrics_path, 'a') as probe_file:
            probe_file.write(json.dumps(metric) + '\n')
        print('BELLMAN_PROBE', json.dumps({
            key: value for key, value in metric.items()
            if key not in ('feature_singular_values', 'bellman_singular_values')
        }))
        return metric

    def _save_bellman_probe_checkpoint(self, seq_idx, event):
        checkpoint_path = os.path.join(
            self._bellman_probe_checkpoint_dir,
            '{}_task{}_env{}_update{}.pt'.format(
                event, seq_idx, self.global_env_step, self.global_step))
        torch.save({
            'global_step': self.global_step,
            'global_env_step': self.global_env_step,
            'seq_idx': seq_idx,
            'policy': self.policy.state_dict(),
            'qf1': self._qf1.state_dict(),
            'qf2': self._qf2.state_dict(),
            'target_qf1': self._target_qf1.state_dict(),
            'target_qf2': self._target_qf2.state_dict(),
            'policy_optimizer': self._policy_optimizer.state_dict(),
            'qf1_optimizer': self._qf1_optimizer.state_dict(),
            'qf2_optimizer': self._qf2_optimizer.state_dict(),
            'log_alpha': self._log_alpha.detach().cpu(),
            'critic_optimizer_steps': self._critic_optimizer_steps,
            'plasticity_injections': self._plasticity_injection_records,
            'demand_aligned_reserve': self._dar_records,
            'dsr_v2': (self._dsr_v2.state_dict()
                       if self._dsr_v2_enabled else None),
            'branch_task_step': self._branch_task_step,
            'probe_buffers': self._bellman_probe_buffers,
            'bellman_response': (self.response_checkpoint_state()
                                 if hasattr(self, 'response_checkpoint_state') else None),
            'muon': (self.muon_checkpoint_state()
                     if hasattr(self, 'muon_checkpoint_state') else None),
            'singular_clip': (self.singular_clip_checkpoint_state()
                             if hasattr(self, 'singular_clip_checkpoint_state') else None),
            'spectral_regularization': (
                self.spectral_regularization_checkpoint_state()
                if self._spectral_regularization_enabled else None),
            'pandc': (self.pandc_checkpoint_state()
                      if hasattr(self, 'pandc_checkpoint_state') else None),
        }, checkpoint_path)

    def spectral_regularization_checkpoint_state(self):
        """Return the method-specific state needed for exact continuation."""
        if not self._spectral_regularization_enabled:
            return None
        return {
            'actor_coefficient': self._spectral_actor_coef,
            'critic_coefficient': self._spectral_critic_coef,
            'regularizer': self._spectral_regularizer.state_dict(),
            'last_stats': copy.deepcopy(
                self._spectral_regularization_last_stats),
        }

    @staticmethod
    def _spectral_actor_layer_is_active(name, seq_idx):
        """Exclude inactive task heads from the current actor objective."""
        marker = '_output_layers.'
        if marker not in name:
            return True
        output_index = int(name.split(marker, 1)[1].split('.', 1)[0])
        return output_index in (2 * int(seq_idx), 2 * int(seq_idx) + 1)

    def train_once(self, seq_idx, itr=None, paths=None):
        """Complete 1 training iteration of SAC.

        Args:
            itr (int): Iteration number. This argument is deprecated.
            paths (list[dict]): A list of collected paths.
                This argument is deprecated.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        del itr
        del paths
        if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
            if self._dsr_v2_enabled:
                stage = self._dsr_v2.due_stage(seq_idx, self._critic_optimizer_steps)
                rows = self._dsr_v2.rows
                available = self.replay_buffer.n_transitions_stored
                if stage is not None and available >= 2 * rows:
                    # 同一次无放回抽样后分成两组，保证校准集和留出集不共用 transition。
                    indices = np.random.choice(available, 2 * rows, replace=False)
                    calibration = as_torch_dict(self.replay_buffer.sample_transitions(
                        rows, idx=indices[:rows]))
                    heldout = as_torch_dict(self.replay_buffer.sample_transitions(
                        rows, idx=indices[rows:]))
                    record = self._dsr_v2.prepare(
                        calibration, heldout, seq_idx, stage, self._critic_optimizer_steps,
                        (self._qf1, self._qf2), (self._target_qf1, self._target_qf2),
                        (self._qf1_optimizer, self._qf2_optimizer), self.policy,
                        self._get_log_alpha(calibration).exp().detach(),
                        self._discount, self._reward_scale)
                    record['global_step'] = int(self.global_step)
                    print('DSR_V2', json.dumps(record), flush=True)
                    if self._bellman_probe:
                        with open(os.path.join(self._bellman_probe_run_dir,
                                               'dsr_v2_events.jsonl'), 'a') as event_file:
                            event_file.write(json.dumps(record) + '\n')
            if self._demand_aligned_reserve_due(seq_idx):
                reserve_samples = self.replay_buffer.sample_transitions(
                    self._dar_rows)
                self._prepare_demand_aligned_reserve(
                    as_torch_dict(reserve_samples), seq_idx)
            if self._plasticity_injection_due(seq_idx):
                injection_samples = self.replay_buffer.sample_transitions(
                    self._plasticity_injection_rows)
                self._apply_plasticity_injection(
                    as_torch_dict(injection_samples), seq_idx)
            samples = self.replay_buffer.sample_transitions(
                self._buffer_batch_size)
            samples = as_torch_dict(samples)

            policy_loss, qf1_loss, qf2_loss = self.optimize_policy(samples, seq_idx)
            self._update_targets()

        return policy_loss, qf1_loss, qf2_loss

    def _get_log_alpha(self, samples_data):
        """Return the value of log_alpha.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        This function exists in case there are versions of sac that need
        access to a modified log_alpha, such as multi_task sac.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: log_alpha

        """
        del samples_data
        log_alpha = self._log_alpha
        return log_alpha

    def _temperature_objective(self, log_pi, samples_data, seq_idx=None):
        """Compute the temperature/alpha coefficient loss.

        Args:
            log_pi(torch.Tensor): log probability of actions that are sampled
                from the replay buffer. Shape is (1, buffer_batch_size).
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: the temperature/alpha coefficient loss.

        """
        alpha_loss = 0
        if self._use_automatic_entropy_tuning:
            if isinstance(self.env_spec, list):
                alpha_loss = (-(self._get_log_alpha(samples_data)) *
                            (log_pi.detach() + self._target_entropy[seq_idx])).mean()
            else:
                alpha_loss = (-(self._get_log_alpha(samples_data)) *
                            (log_pi.detach() + self._target_entropy)).mean()
        return alpha_loss

    def _infer_loss(self, pred_network, target_network):
        pred = pred_network.get_feature_prediction()
        target = target_network.get_feature_prediction().clone()
        return F.mse_loss(pred, self._infer_beta * target)
    
    def wasserstein_reg_loss(self, model, target):
        target.eval()
        loss = 0
        for p1, p2 in zip(model.parameters(), target.parameters()):
            sorted, _ = p1.flatten().sort()
            target_sorted, _ = p2.flatten().sort()
            loss += F.mse_loss(sorted, target_sorted)
        return self._wasserstein_lambda * loss

    @staticmethod
    def _zero_optimizer_mask(optimizer, parameter, row_mask=None,
                             column_mask=None):
        state = optimizer.state.get(parameter, {})
        for value in state.values():
            if not torch.is_tensor(value) or value.shape != parameter.shape:
                continue
            if row_mask is not None:
                value[row_mask] = 0
            if column_mask is not None:
                value[:, column_mask] = 0

    def _redo_network(self, network, optimizer):
        features = getattr(network, '_features', None)
        if not features or len(features) != len(network._layers):
            return 0
        masks = []
        for feature in features:
            detached = feature.detach().abs()
            reduce_dims = tuple(range(detached.ndim - 1))
            mean_activation = detached.mean(dim=reduce_dims)
            score = mean_activation / mean_activation.mean().clamp_min(1e-9)
            masks.append(score <= self._redo_tau)

        recycled = 0
        with torch.no_grad():
            # First reinitialize every dormant unit's incoming parameters.
            for layer_idx, (layer, mask) in enumerate(
                    zip(network._layers, masks)):
                if not bool(mask.any()):
                    continue
                recycled += int(mask.sum().item())
                linear = layer[0]
                fresh_weight = torch.empty_like(linear.weight)
                network._hidden_w_init(fresh_weight)
                linear.weight[mask] = fresh_weight[mask]
                fresh_bias = torch.empty_like(linear.bias)
                network._hidden_b_init(fresh_bias)
                linear.bias[mask] = fresh_bias[mask]
                self._zero_optimizer_mask(
                    optimizer, linear.weight, row_mask=mask)
                self._zero_optimizer_mask(
                    optimizer, linear.bias, row_mask=mask)
            # Then zero outgoing connections, after all incoming resets, so a
            # deeper-layer reset cannot reintroduce a previous dormant unit.
            for layer_idx, mask in enumerate(masks):
                if not bool(mask.any()):
                    continue
                if layer_idx + 1 < len(network._layers):
                    outgoing = network._layers[layer_idx + 1][0].weight
                    outgoing[:, mask] = 0
                    self._zero_optimizer_mask(
                        optimizer, outgoing, column_mask=mask)
                else:
                    for output_layer in network._output_layers:
                        outgoing = output_layer[0].weight
                        outgoing[:, mask] = 0
                        self._zero_optimizer_mask(
                            optimizer, outgoing, column_mask=mask)
        return recycled

    def ReDo(self, seq_idx):
        """Recycle dormant actor/critic neurons using normalized activation."""
        policy_network = (
            self.policy._module._shared_mean_log_std_network)
        event = {
            'global_env_step': self.global_env_step,
            'task_env_step': self.global_env_step - self._task_env_start_step,
            'task_index': seq_idx,
            'tau': self._redo_tau,
            'interval': self._redo_interval,
            'policy_recycled': self._redo_network(
                policy_network, self._policy_optimizer),
            'qf1_recycled': self._redo_network(
                self._qf1, self._qf1_optimizer),
            'qf2_recycled': self._redo_network(
                self._qf2, self._qf2_optimizer),
        }
        self._target_qf1.load_state_dict(self._qf1.state_dict())
        self._target_qf2.load_state_dict(self._qf2.state_dict())
        print('REDO_EVENT', json.dumps(event), flush=True)
        return event


    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions, seq_idx=None):
        """Compute the Policy/Actor loss.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
            new_actions (torch.Tensor): Actions resampled from the policy based
                based on the Observations, obs, which were sampled from the
                replay buffer. Shape is (action_dim, buffer_batch_size).
            log_pi_new_actions (torch.Tensor): Log probability of the new
                actions on the TanhNormal distributions that they were sampled
                from. Shape is (1, buffer_batch_size).

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from the Policy/Actor.

        """
        obs = samples_data['observation']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
        min_q_new_actions = torch.min(self._qf1(obs, new_actions, seq_idx),
                                      self._qf2(obs, new_actions, seq_idx))
        policy_objective = ((alpha * log_pi_new_actions) -
                            min_q_new_actions.flatten()).mean()
        
        return policy_objective

    def _critic_objective(self, samples_data, seq_idx,
                          return_predictions=False):
        """Compute the Q-function/critic loss.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        actions = samples_data['action']
        rewards = samples_data['reward'].flatten()
        terminals = samples_data['terminal'].flatten()
        next_obs = samples_data['next_observation']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        q1_pred = self._qf1(obs, actions, seq_idx=seq_idx)
        q2_pred = self._qf2(obs, actions, seq_idx=seq_idx)

        new_next_actions_dist = self.policy(next_obs, seq_idx)[0]
        new_next_actions_pre_tanh, new_next_actions = (
            new_next_actions_dist.rsample_with_pre_tanh_value())
        new_log_pi = new_next_actions_dist.log_prob(
            value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)

        
        qf1 = self._target_qf1(next_obs, new_next_actions, seq_idx=seq_idx)
        qf2 = self._target_qf2(next_obs, new_next_actions, seq_idx=seq_idx)
        
        target_q_values = torch.min(qf1,qf2).flatten() - (alpha * new_log_pi)
        
        
        with torch.no_grad():
            q_target = rewards * self._reward_scale + (
                1. - terminals) * self._discount * target_q_values

        if self._critic_optimizer_steps == 0:
            first_batch = critic_first_batch_stats(
                q1_pred.flatten(), q2_pred.flatten(), qf1.flatten(),
                qf2.flatten(), target_q_values, q_target, rewards)
            first_batch.update({
                'global_step': self.global_step,
                'task': int(seq_idx),
                'alpha': float(alpha.detach()),
            })
            print('CRITIC_FIRST_BATCH', json.dumps(first_batch), flush=True)

        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        if return_predictions:
            return qf1_loss, qf2_loss, q1_pred, q2_pred, q_target
        return qf1_loss, qf2_loss

    def _caps_regularization_objective(self, action_dists, samples_data, seq_idx):
        """Compute the spatial and temporal regularization loss as in CAPS.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
            action_dists (torch.distribution.Distribution): Distributions
                returned from the policy after feeding through observations.

        Returns:
            torch.Tensor: combined regularization loss
        """
        # torch.tensor is callable and the recommended way to create a scalar
        # tensor
        # pylint: disable=not-callable

        if self._temporal_regularization_factor:
            next_action_dists = self.policy(
                samples_data['next_observation'], seq_idx)[0]
            temporal_loss = self._temporal_regularization_factor * torch.mean(
                torch.cdist(action_dists.mean, next_action_dists.mean, p=2))
        else:
            temporal_loss = torch.tensor(0.)

        if self._spatial_regularization_factor:
            obs = samples_data['observation']
            noisy_action_dists = self.policy(
                obs + self._spatial_regularization_dist.sample(obs.shape), seq_idx)[0]
            spatial_loss = self._spatial_regularization_factor * torch.mean(
                torch.cdist(action_dists.mean, noisy_action_dists.mean, p=2))
        else:
            spatial_loss = torch.tensor(0.)

        return temporal_loss + spatial_loss

    def _update_targets(self):
        """Update parameters in the target q-functions."""
        target_qfs = [self._target_qf1, self._target_qf2]
        qfs = [self._qf1, self._qf2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                if self._dsr_v2_enabled and not param.requires_grad:
                    # 固定特征应逐位保持一致，避免反复 Polyak 运算引入舍入漂移。
                    t_param.data.copy_(param.data)
                    continue
                t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                                   param.data * self._tau)

    def optimize_policy(self, samples_data, seq_idx):
        """Optimize the policy q_functions, and temperature coefficient.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        critic_values = self._critic_objective(
            samples_data, seq_idx, return_predictions=True)
        qf1_loss, qf2_loss, q1_pred, q2_pred, q_target = critic_values

        if self._infer:
            obs = samples_data['observation']
            actions = samples_data['action']
            with torch.no_grad():
                _ = self._infer_target_qf1(obs, actions, seq_idx=seq_idx)
                _ = self._infer_target_qf2(obs, actions, seq_idx=seq_idx)
            qf1_loss += self._infer_alpha * self._infer_loss(self._qf1, self._infer_target_qf1)
            qf2_loss += self._infer_alpha * self._infer_loss(self._qf2, self._infer_target_qf2)

        if self._wasserstein:
            qf1_loss += self.wasserstein_reg_loss(self._qf1, self._wasserstein_target_qf1)
            qf2_loss += self.wasserstein_reg_loss(self._qf2, self._wasserstein_target_qf2)

        if self._spectral_regularization_enabled:
            spectral_diagnostics_due = (
                self._critic_optimizer_steps < 10 or
                self._critic_optimizer_steps % 1000 == 0)
            qf1_regularizer, qf1_spectral_stats = (
                self._spectral_regularizer.loss(
                    self._qf1, 'qf1', spectral_diagnostics_due))
            qf2_regularizer, qf2_spectral_stats = (
                self._spectral_regularizer.loss(
                    self._qf2, 'qf2', spectral_diagnostics_due))
            qf1_loss = (
                qf1_loss + self._spectral_critic_coef * qf1_regularizer)
            qf2_loss = (
                qf2_loss + self._spectral_critic_coef * qf2_regularizer)

        pbsr_due = (
            self._pbsr_enabled and
            seq_idx < self._pbsr_train_task_count and
            self._critic_optimizer_steps % self._pbsr_update_interval == 0)
        if pbsr_due:
            pbsr_diagnostics_due = (
                self._critic_optimizer_steps < 10 or
                self._critic_optimizer_steps % 1000 == 0)
            with torch.no_grad():
                pbsr_alpha = self._get_log_alpha(samples_data).exp()
            q1_directions, q2_directions = self._pbsr.direction_banks(
                samples_data, seq_idx, q1_pred, q2_pred,
                self._target_qf1, self._target_qf2, self.policy, pbsr_alpha,
                self._discount, self._reward_scale)
            qf1_loss, q1_pbsr_stats = self._pbsr.critic_loss(
                self._qf1, samples_data, seq_idx, q1_directions, qf1_loss,
                compute_diagnostics=pbsr_diagnostics_due)
            qf2_loss, q2_pbsr_stats = self._pbsr.critic_loss(
                self._qf2, samples_data, seq_idx, q2_directions, qf2_loss,
                compute_diagnostics=pbsr_diagnostics_due)
            self._pbsr_last_stats = {
                'global_step': int(self.global_step),
                'critic_optimizer_step': int(self._critic_optimizer_steps + 1),
                'task': int(seq_idx),
                'qf1': q1_pbsr_stats,
                'qf2': q2_pbsr_stats,
            }
            if pbsr_diagnostics_due:
                print(
                    'PBSR_UPDATE', json.dumps(self._pbsr_last_stats),
                    flush=True)

        zero_optim_grads(self._qf1_optimizer)
        if self._dsr_v2_enabled:
            zero_head_gradients(self._qf1)
        qf1_loss.backward()
        self._qf1_optimizer.step()
        if self._dsr_v2_enabled:
            step_heads(self._qf1)

        zero_optim_grads(self._qf2_optimizer)
        if self._dsr_v2_enabled:
            zero_head_gradients(self._qf2)
        qf2_loss.backward()
        self._qf2_optimizer.step()
        if self._dsr_v2_enabled:
            step_heads(self._qf2)

        self._critic_optimizer_steps += 1
        # 新方法在真实双 critic TD 更新后、actor 更新前施加独立参数修正。
        # 原有算法没有此钩子，继续执行完全相同的优化路径。
        if hasattr(self, '_after_critic_update'):
            self._after_critic_update(samples_data, seq_idx)

        # action_dists = self.policy(obs, seq_idx)[0]
        # new_actions_pre_tanh, new_actions = (
        #     action_dists.rsample_with_pre_tanh_value())
        # log_pi_new_actions = action_dists.log_prob(
        #     value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        action_dists, new_actions, log_pi_new_actions = self._get_policy_output(obs, seq_idx)

        policy_loss = self._actor_objective(samples_data, new_actions,
                                            log_pi_new_actions, seq_idx=seq_idx)
        policy_loss += self._caps_regularization_objective(
            action_dists, samples_data, seq_idx)
        policy_loss += self.cl_reg_loss(seq_idx)
        

        if self._infer:
            with torch.no_grad():
                _ = self._infer_target_policy(obs, seq_idx)[0]
            policy_loss += self._infer_alpha * self._infer_loss(self.policy, self._infer_target_policy)

        if self._wasserstein:
            policy_loss += self.wasserstein_reg_loss(self.policy, self._wasserstein_target_policy)

        if self._spectral_regularization_enabled:
            actor_regularizer, actor_spectral_stats = (
                self._spectral_regularizer.loss(
                    self.policy, 'actor', spectral_diagnostics_due,
                    module_filter=lambda name, module:
                    self._spectral_actor_layer_is_active(name, seq_idx)))
            policy_loss = (
                policy_loss + self._spectral_actor_coef * actor_regularizer)
            if spectral_diagnostics_due:
                self._spectral_regularization_last_stats = {
                    'global_step': int(self.global_step),
                    'critic_optimizer_step': int(self._critic_optimizer_steps),
                    'task': int(seq_idx),
                    'actor': actor_spectral_stats,
                    'qf1': qf1_spectral_stats,
                    'qf2': qf2_spectral_stats,
                }

        zero_optim_grads(self._policy_optimizer)
        policy_loss.backward()
        self._policy_optimizer.step()

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data, seq_idx=seq_idx)
            zero_optim_grads(self._alpha_optimizer)
            alpha_loss.backward()
            self._alpha_optimizer.step()

        return policy_loss, qf1_loss, qf2_loss

    def _get_policy_output(self, obs, seq_idx):

        action_dists = self.policy(obs, seq_idx)[0]
        new_actions_pre_tanh, new_actions = (
            action_dists.rsample_with_pre_tanh_value())
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh)
        
        return action_dists, new_actions, log_pi_new_actions

    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """
        eval_episodes = obtain_evaluation_episodes(
            self.policy,
            self._eval_env,
            seq_idx = -1, # dummy
            max_episode_length = self._max_episode_length_eval,
            num_eps=self._num_evaluation_episodes,
            deterministic=self._use_deterministic_evaluation)
        last_return = log_performance(epoch,
                                      eval_episodes,
                                      discount=self._discount,
                                      results=self.results, 
                                      use_wandb=self._use_wandb)
        return last_return

    def _reset_alpha(self):
        
        if self._use_automatic_entropy_tuning:
            self._log_alpha = list_to_tensor([self._initial_log_entropy
                                              ]).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                              lr=self._policy_lr)
        else:
            self._log_alpha = list_to_tensor([self._fixed_alpha]).log()

    def task_change(self, seq_idx):
        bellman_probe_metric = None
        if self._bellman_probe:
            bellman_probe_metric = self._run_bellman_probe(
                'task_boundary', seq_idx)
            self._save_bellman_probe_checkpoint(seq_idx, 'task_boundary')

        self.on_task_start(seq_idx)

        self.replay_buffer.clear()
        self._critic_optimizer_steps = 0
        self._reset_alpha()
        self.recent_trajectory.clear()

        if self._q_reset:
            self._qf1.load_state_dict(self._random_qf1_state_dict)
            self._qf2.load_state_dict(self._random_qf2_state_dict)
            # The formal critic-reset baseline resets both critic parameters
            # and their Adam moments. Carrying the old moments would define a
            # different weights-only reset intervention.
            self._qf1_optimizer = self._make_network_optimizer(
                self._qf1, self._qf_lr)
            self._qf2_optimizer = self._make_network_optimizer(
                self._qf2, self._qf_lr)
            print('CRITIC_RESET', json.dumps({
                'completed_task_position': int(seq_idx),
                'next_task_position': int(seq_idx + 1),
                'critic_parameters': 'initialization',
                'critic_adam': 'reset',
                'target_critics': 'hard_sync_after_reset',
                'global_critic_updates': int(self.global_step),
            }), flush=True)

        qf1_state_dict = copy.deepcopy(self._qf1.state_dict())
        qf2_state_dict = copy.deepcopy(self._qf2.state_dict())

        self._target_qf1.load_state_dict(qf1_state_dict)
        self._target_qf2.load_state_dict(qf2_state_dict)

        if self._bellman_probe:
            self._bellman_probe_task_start_step = self.global_step
        self._task_env_start_step = self.global_env_step
        return bellman_probe_metric
    
    def save_models(self, log_name = None):

        if log_name is None:
            log_name = self._log_name

        #  Save models into 'models/sac_models'
        os.makedirs('models/sac_models', exist_ok=True)

        for net, name in zip(self.networks, self.networks_names):
            torch.save(
                net.state_dict(),
                os.path.join('models', 'sac_models',
                             name + '_' + log_name + '.pt'))

        
    def save_buffers(self, log_name = None):

        if log_name is None:
            log_name = self._log_name

        # Save buffers into 'buffers/sac_buffers'
        os.makedirs('buffers/sac_buffers', exist_ok=True)

        buffer_data = self.replay_buffer.get_all_transitions()
        buffer_data = as_torch_dict(buffer_data)
        path = os.path.join(
            'buffers', 'sac_buffers', log_name + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(buffer_data, f)
    
    def save_rollouts(self, log_name = None, buffer_size = int(1e6)):
        
        buffer = dict()
        seq_idx = 0
        # Preserve R&D's fresh post-training expert rollouts (not replay-buffer
        # samples). Our shared held-out evaluation split is an experiment-setting
        # adaptation: collect on training instances, outside the CL step counter.
        rollout_env = self._sampler._envs[0].envs[0]
        observations = []
        obs_len = 0

        while obs_len < buffer_size:
            episode_batch = obtain_evaluation_episodes(
                self.policy,
                rollout_env,
                seq_idx,
                self._max_episode_length_eval,
                num_eps=self._num_evaluation_episodes,
                deterministic=self._use_deterministic_evaluation)
            observation = episode_batch.observations
            observations.append(observation)
            obs_len += len(observation)
        
        buffer['observation'] = np.concatenate((observations))
        buffer['observation'] = torch.Tensor(buffer['observation'][:buffer_size, :])
        buffer['observation'] = buffer['observation'].to(global_device())
        
        assert buffer['observation'].shape[0] == buffer_size

        os.makedirs('rollouts/sac_rollouts', exist_ok=True)
        
        path = './rollouts/sac_rollouts/rollouts_' + log_name + '.pkl'
        with open(path, "wb") as file:
            pickle.dump(buffer, file)

    
    def save_results(self, log_name = None):

        if log_name is None:
            log_name = self._log_name
        
        path = './logs/' + log_name + '.pkl'

        if not os.path.exists('logs/'):
            os.makedirs('logs/')

        with open(path, 'wb') as f:
            pickle.dump(self.results, f)

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy, self._qf1, self._qf2, self._target_qf1,
            self._target_qf2
        ]
    @property
    def networks_names(self):

        return [
            'policy', 'qf1', 'qf2', 'target_qf1', 'target_qf2'
        ]

    def cl_reg_network(self):
        return [self.policy]
    def on_task_start(self, seq_idx):
        pass
    def on_test_start(self, seq_idx):
        pass
    def on_test_end(self, seq_idx):
        pass
    def cl_reg_loss(self, seq_idx):
        return 0

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)

        if self._infer:
            self._infer_target_policy.to(device)
            self._infer_target_qf1.to(device)
            self._infer_target_qf2.to(device)

        if self._wasserstein:
            self._wasserstein_target_policy.to(device)
            self._wasserstein_target_qf1.to(device)
            self._wasserstein_target_qf2.to(device)
        
        if self._ReDo:
            self.random_policy.to(device)
            self._random_qf1.to(device)
            self._random_qf2.to(device)

        if not self._use_automatic_entropy_tuning:
            self._log_alpha = list_to_tensor([self._fixed_alpha
                                              ]).log().to(device)
        else:
            self._log_alpha = self._log_alpha.detach().to(
                device).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                                    lr=self._policy_lr)
            self._alpha_optimizer.load_state_dict(
                state_dict_to(self._alpha_optimizer.state_dict(), device))
            self._qf1_optimizer.load_state_dict(
                state_dict_to(self._qf1_optimizer.state_dict(), device))
            self._qf2_optimizer.load_state_dict(
                state_dict_to(self._qf2_optimizer.state_dict(), device))
            self._policy_optimizer.load_state_dict(
                state_dict_to(self._policy_optimizer.state_dict(), device))


class RecentTrajectory:
    def __init__(self, maxlen = 10000):
        self._observation = []
        self._action = []
        self._reward = []
        self._next_observation = []
        self._terminal = []

        self.maxlen = maxlen

    def _concat(self, x, y):
        if x is None:
            return y
        else:
            return np.concatenate([x, y])[-self.maxlen:]

    def append(self, path):
        obs, action = path['observations'], path['actions']
        reward = path['rewards']
        next_obs = path['next_observations']
        terminal = np.array([
                        step_type == StepType.TERMINAL
                        for step_type in path['step_types']
                    ]).reshape(-1, 1)
        
        

        self._observation.append(obs)
        self._action.append(action)
        self._reward.append(reward)
        self._next_observation.append(next_obs)
        self._terminal.append(terminal)
    
    @property
    def observation(self):
        observation = np.concatenate(self._observation)[-self.maxlen:]
        return np_to_torch(observation)
    
    @property
    def action(self):
        action = np.concatenate(self._action)[-self.maxlen:]
        return np_to_torch(action)
    
    @property
    def reward(self):
        reward = np.concatenate(self._reward)[-self.maxlen:]
        return np_to_torch(reward)

    @property
    def next_observation(self):
        next_observation = np.concatenate(self._next_observation)[-self.maxlen:]
        return np_to_torch(next_observation)
    
    @property
    def terminal(self):
        terminal = np.concatenate(self._terminal)[-self.maxlen:]
        return np_to_torch(terminal)
    
    @property
    def samples(self):
        dic = dict(
            observation = self.observation,
            action = self.action,
            reward = self.reward,
            terminal = self.terminal,
            next_observation = self.next_observation
        )
        return dic
    
    def clear(self):
        self._observation = []
        self._action = []
        self._reward = []
        self._next_observation = []
        self._terminal = []
