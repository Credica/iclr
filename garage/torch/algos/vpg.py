"""Vanilla Policy Gradient (REINFORCE)."""
import collections
import copy
import json

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import log_performance
from garage import (EpisodeBatch, log_multitask_performance,
                    obtain_evaluation_episodes)
from garage.np import discount_cumsum
from garage.np.algos import RLAlgorithm
from garage.torch import compute_advantages, filter_valids, as_torch_dict
from garage.torch._functions import np_to_torch, zero_optim_grads, feature_rank, weight_deviation, weight_hessian
from garage.torch import global_device, state_dict_to
from garage.torch.algos.ppo_value_spectral_stats import PPOValueSpectralStats
from garage.torch.algos.pbsr import PBSRValueHead
from garage.torch.algos.ppo_pbsr_v2 import PPOPBSRV2
from garage.torch.algos.ppo_critic_regularizers import (
    PPOActualDemandBurden,
    PPOValueLayerSpectralRegularizer,
)
from garage.torch.algos.ppo_bolt import (
    PPOBellmanBurdenController,
    build_multistep_value_targets,
)
from garage.torch.algos.ppo_spectral_advantage import spectral_trust_advantage
from garage.torch.optimizers import OptimizerWrapper
from time import time

import wandb
import pickle
import os

def load_model(model, model_name, first_task, first_task_steps, seed):
    # Load policy
    copied_model = copy.deepcopy(model)
    name = model_name.format(first_task, first_task_steps, seed)
    loaded_state_dict = torch.load('./models/ppo_models/'+name, map_location=global_device())

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


class VPG(RLAlgorithm):
    """Vanilla Policy Gradient (REINFORCE).
    VPG, also known as Reinforce, trains stochastic policy in an on-policy way.
    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        sampler (garage.sampler.Sampler): Sampler.
        policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
            value function.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
    """

    def __init__(
        self,
        env_spec,
        policy,
        value_function,
        sampler,
        seed = 0,
        policy_optimizer=None,
        vf_optimizer=None,
        num_train_per_epoch=1,
        discount=0.99,
        gae_lambda=1,
        center_adv=True,
        positive_adv=False,
        policy_ent_coeff=0.0,
        use_softplus_entropy=False,
        stop_entropy_gradient=False,
        entropy_method='no_entropy',
        eval_env=None,
        num_evaluation_episodes=10,
        use_deterministic_evaluation=True,
        log_name=None,
        q_reset=False,
        policy_reset=False,
        first_task = None, 
        first_task_steps=int(3e6),
        use_wandb=True,
        infer=False,
        crelu=False,
        wasserstein=0, 
        ReDo=False,
        no_stats=False, 
        multi_input=False,
        bellman_spectral_stats=False,
        bellman_spectral_anchor_size=64,
        bellman_spectral_fit_lr=5e-4,
        bellman_spectral_ridge=1e-3,
        bellman_spectral_task_steps=(15000, 60000, 105000, 510000,
                                     1005000, 1500000),
        bellman_probe_dir='bellman_probe_results',
        ppo_value_reference_dir=None,
        ppo_value_reference_mode='load',
        pbsr=False,
        pbsr_coef=0.1,
        pbsr_anchor_size=64,
        pbsr_targets=8,
        pbsr_ridge=1e-3,
        pbsr_update_interval=100,
        pbsr_train_task_count=1,
        pbsr_train_task_indices=None,
        pbsr_variant='v1',
        pbsr_v2_actor=True,
        pbsr_v2_horizons=(1, 3, 5),
        pbsr_v2_bandwidth=0.5,
        ppo_bolt=False,
        ppo_bolt_rank=4,
        ppo_bolt_rho=0.5,
        ppo_bolt_ridge=0.1,
        ppo_bolt_calibration_size=32,
        ppo_bolt_update_interval=1000,
        ppo_bolt_history_columns=24,
        ppo_bolt_horizons=(1, 3, 5),
        ppo_bolt_train_task_indices=None,
        ppo_bolt_update_mode='innovation',
        ppo_bolt_reset_first_moment=False,
        ppo_spectral_advantage=False,
        ppo_spectral_advantage_ridge=0.1,
        ppo_spectral_advantage_block_size=128,
        ppo_spectral_advantage_train_task_indices=None,
        task_names=None):
        
        self._discount = discount
        self.policy = policy
        if isinstance(env_spec,list):
            max_episode_length = env_spec[0].max_episode_length
        else:
            max_episode_length = env_spec.max_episode_length
        self.max_episode_length = max_episode_length

        self._value_function = value_function
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._policy_ent_coeff = policy_ent_coeff
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._n_samples = num_train_per_epoch
        self._env_spec = env_spec
        
        self._eval_env = eval_env
        self._seed = seed
        self._num_evaluation_episodes = num_evaluation_episodes
        self._use_deterministic_evaluation = use_deterministic_evaluation
        self._max_episode_length_eval = max_episode_length

        self._value_reset = q_reset
        self._policy_reset = policy_reset
        self._first_task = first_task
        self._infer = infer
        self._wasserstein = (wasserstein > 0)
        self._ReDo = ReDo
        self._no_stats = no_stats
        self._multi_input = multi_input
        self._bellman_spectral_stats_enabled = bool(bellman_spectral_stats)
        self._task_names = task_names
        self._pbsr_enabled = bool(pbsr)
        self._pbsr_variant = pbsr_variant
        self._pbsr_update_interval = int(pbsr_update_interval)
        self._pbsr_train_task_count = int(pbsr_train_task_count)
        if pbsr_train_task_indices is None:
            self._pbsr_active_tasks = set(range(self._pbsr_train_task_count))
        else:
            self._pbsr_active_tasks = set(
                int(task_idx) for task_idx in pbsr_train_task_indices)
        self._pbsr_value_optimizer_steps = 0
        self._pbsr_policy_optimizer_steps = 0
        self._pbsr_last_stats = {}
        self._pbsr_last_policy_stats = {}
        self._ppo_bolt_enabled = bool(ppo_bolt)
        self._ppo_bolt_horizons = tuple(ppo_bolt_horizons)
        self._ppo_bolt_last_stats = {}
        self._ppo_spectral_advantage_enabled = bool(ppo_spectral_advantage)
        self._ppo_spectral_advantage_ridge = float(
            ppo_spectral_advantage_ridge)
        self._ppo_spectral_advantage_block_size = int(
            ppo_spectral_advantage_block_size)
        if ppo_spectral_advantage_train_task_indices is None:
            self._ppo_spectral_advantage_active_tasks = (
                set(range(len(task_names)))
                if self._ppo_spectral_advantage_enabled else set())
        else:
            self._ppo_spectral_advantage_active_tasks = set(
                int(task_idx)
                for task_idx in ppo_spectral_advantage_train_task_indices)
        self._ppo_spectral_advantage_last_stats = {}
        if self._pbsr_enabled and self._pbsr_variant == 'v1':
            self._pbsr = PBSRValueHead(
                anchor_size=pbsr_anchor_size,
                target_count=pbsr_targets,
                ridge=pbsr_ridge,
                gradient_ratio=pbsr_coef,
                seed=seed)
        elif self._pbsr_enabled and self._pbsr_variant == 'ppo_v2':
            self._pbsr = PPOPBSRV2(
                anchor_size=pbsr_anchor_size,
                probe_count=pbsr_targets,
                horizons=pbsr_v2_horizons,
                ridge=pbsr_ridge,
                gradient_ratio=pbsr_coef,
                minimum_bandwidth=pbsr_v2_bandwidth,
                seed=seed,
                train_actor=pbsr_v2_actor)
        elif self._pbsr_enabled and self._pbsr_variant == 'actual_demand':
            self._pbsr = PPOActualDemandBurden(
                anchor_size=pbsr_anchor_size,
                ridge=pbsr_ridge,
                gradient_ratio=pbsr_coef)
        elif self._pbsr_enabled and self._pbsr_variant == 'spectral':
            self._pbsr = PPOValueLayerSpectralRegularizer(
                gradient_ratio=pbsr_coef)
        else:
            self._pbsr = None

        self._log_name=log_name
        self._use_wandb = use_wandb
        self._environment_steps = 0
        self._task_start_environment_step = 0
        self._last_value_probe_task_step = None
        if self._bellman_spectral_stats_enabled:
            spectral_run_dir = os.path.join(bellman_probe_dir, self._log_name)
            reference_dir = (
                ppo_value_reference_dir
                if ppo_value_reference_dir is not None
                else os.path.join(bellman_probe_dir, 'ppo_value_references'))
            os.makedirs(spectral_run_dir, exist_ok=True)
            self._ppo_value_spectral_probe = PPOValueSpectralStats(
                run_dir=spectral_run_dir,
                reference_dir=reference_dir,
                reference_mode=ppo_value_reference_mode,
                anchor_size=bellman_spectral_anchor_size,
                relative_ridge=bellman_spectral_ridge,
                fit_learning_rate=bellman_spectral_fit_lr,
                discount=self._discount,
                task_steps=bellman_spectral_task_steps,
                seed=seed)
        if self._pbsr_enabled:
            pbsr_run_dir = os.path.join(bellman_probe_dir, self._log_name)
            os.makedirs(pbsr_run_dir, exist_ok=True)
            method_names = {
                'v1': 'pbsr_value_head',
                'ppo_v2': 'ppo_pbsr_demand_complete_v2',
                'actual_demand': 'ppo_actual_value_demand_burden',
                'spectral': 'ppo_value_layer_spectral_regularization',
            }
            probe_definitions = {
                'v1': 'one_step_random_value_head_random_cumulant',
                'ppo_v2': 'raw_input_rff_multihorizon_value_and_cumulant',
                'actual_demand': 'detached_monte_carlo_return_residual',
                'spectral': None,
            }
            kernel_definitions = {
                'v1': 'HHt_over_d_then_trace_normalized',
                'ppo_v2':
                    'centered_linear_head_weight_tangent_trace_normalized',
                'actual_demand': 'raw_linear_head_tangent_with_bias',
                'spectral': None,
            }
            pbsr_config = {
                'method': method_names[self._pbsr_variant],
                'algorithm': 'ppo',
                'probe_source': (
                    None if self._pbsr_variant == 'spectral'
                    else 'current_task_on_policy_transitions'),
                'probe_definition': probe_definitions[self._pbsr_variant],
                'bellman_horizons': (
                    [1] if self._pbsr_variant == 'v1'
                    else (list(pbsr_v2_horizons)
                          if self._pbsr_variant == 'ppo_v2' else [])),
                'anchor_size': (
                    None if self._pbsr_variant == 'spectral'
                    else pbsr_anchor_size),
                'target_count': (
                    pbsr_targets
                    if self._pbsr_variant in ('v1', 'ppo_v2') else None),
                'normalized_kernel': kernel_definitions[self._pbsr_variant],
                'probe_column_normalization': (
                    'center_then_unit_l2'
                    if self._pbsr_variant in ('v1', 'ppo_v2') else None),
                'ridge': pbsr_ridge,
                'gradient_ratio': pbsr_coef,
                'update_interval': self._pbsr_update_interval,
                'train_task_count': len(self._pbsr_active_tasks),
                'train_task_indices': sorted(self._pbsr_active_tasks),
                'actor_geometry': bool(
                    self._pbsr_variant == 'ppo_v2' and pbsr_v2_actor),
                'demand_covariance_completion': (
                    'analytic_minimum_bandwidth_on_centered_complement'
                    if self._pbsr_variant == 'ppo_v2' else None),
                'minimum_bandwidth': (
                    pbsr_v2_bandwidth
                    if self._pbsr_variant == 'ppo_v2' else None),
                'spectral_exponent': (
                    2 if self._pbsr_variant == 'spectral' else None),
                'spectral_weight_target': (
                    1 if self._pbsr_variant == 'spectral' else None),
                'spectral_bias_target': (
                    0 if self._pbsr_variant == 'spectral' else None),
                'spectral_layers': (
                    'all_value_linear_layers'
                    if self._pbsr_variant == 'spectral' else None),
                'future_task_reference_used_for_training': False,
            }
            with open(os.path.join(
                    pbsr_run_dir, 'pbsr_config.json'), 'w') as config_file:
                json.dump(pbsr_config, config_file, indent=2)
            print('PBSR_CONFIG', json.dumps(pbsr_config), flush=True)
        

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)
        self._episode_reward_mean = collections.deque(maxlen=10)
        self._sampler = sampler

        if policy_optimizer:
            self._policy_optimizer = policy_optimizer
        else:
            self._policy_optimizer = OptimizerWrapper(torch.optim.Adam, policy)
        if vf_optimizer:
            self._vf_optimizer = vf_optimizer
        else:
            self._vf_optimizer = OptimizerWrapper(torch.optim.Adam,
                                                  value_function)

        if self._ppo_bolt_enabled:
            self._ppo_bolt = PPOBellmanBurdenController(
                value_function=self._value_function,
                optimizer=self._vf_optimizer._optimizer,
                rank=ppo_bolt_rank,
                rho=ppo_bolt_rho,
                ridge=ppo_bolt_ridge,
                calibration_size=ppo_bolt_calibration_size,
                update_interval=ppo_bolt_update_interval,
                history_columns=ppo_bolt_history_columns,
                active_tasks=ppo_bolt_train_task_indices,
                reset_first_moment=ppo_bolt_reset_first_moment)
            self._ppo_bolt.start_task(0)
            bolt_run_dir = os.path.join(bellman_probe_dir, self._log_name)
            os.makedirs(bolt_run_dir, exist_ok=True)
            bolt_config = {
                'method': 'ppo_bolt_inverse_burden',
                'algorithm': 'ppo',
                'value_demands': ['monte_carlo'] + [
                    '{}_step'.format(horizon)
                    for horizon in self._ppo_bolt_horizons],
                'rank': int(ppo_bolt_rank),
                'rho': float(ppo_bolt_rho),
                'ridge': float(ppo_bolt_ridge),
                'calibration_size': int(ppo_bolt_calibration_size),
                'heldout_size': int(ppo_bolt_calibration_size),
                'row_split': 'disjoint_current_minibatch',
                'update_interval': int(ppo_bolt_update_interval),
                'history_columns': int(ppo_bolt_history_columns),
                'train_task_indices': (
                    None if ppo_bolt_train_task_indices is None
                    else list(ppo_bolt_train_task_indices)),
                'optimizer_update_mode': ppo_bolt_update_mode,
                'optimizer_metric_spectrum': 'fixed_1_plus_minus_rho',
                'first_moment_at_task_boundary': (
                    'reset' if ppo_bolt_reset_first_moment else 'retain'),
                'second_moment_at_task_boundary': 'retain',
                'actor_modified': False,
            }
            with open(os.path.join(
                    bolt_run_dir, 'ppo_bolt_config.json'), 'w') as config_file:
                json.dump(bolt_config, config_file, indent=2)
            print('PPO_BOLT_CONFIG', json.dumps(bolt_config), flush=True)

        if self._ppo_spectral_advantage_enabled:
            spectral_advantage_config = {
                'method': 'ppo_spectral_trust_advantage',
                'control_direction': 'on_policy_monte_carlo_advantage',
                'critic_role': 'spectral_trust_for_gae_bootstrap',
                'kernel': 'exact_full_value_mean_tanh_mlp_ntk',
                'filter': 'rho_times_inverse_K_plus_rho_I',
                'relative_ridge': self._ppo_spectral_advantage_ridge,
                'block_size': self._ppo_spectral_advantage_block_size,
                'train_task_indices': sorted(
                    self._ppo_spectral_advantage_active_tasks),
            }
            spectral_advantage_run_dir = os.path.join(
                bellman_probe_dir, self._log_name)
            os.makedirs(spectral_advantage_run_dir, exist_ok=True)
            with open(os.path.join(
                    spectral_advantage_run_dir,
                    'ppo_spectral_advantage_config.json'),
                    'w') as config_file:
                json.dump(spectral_advantage_config, config_file, indent=2)
            print('PPO_SPECTRAL_ADVANTAGE_CONFIG', json.dumps(
                spectral_advantage_config), flush=True)

        self._old_policy = copy.deepcopy(self.policy)

        self.global_step = 0
        self.seq_idx = 0
        self.start_time = time()
        self.begin = self.start_time
        self.results = {}

        self.results['Running avg. of episode return'] = []
        self.results['Policy loss'] = []
        self.results['Value loss'] = []
        self.results['KL'] = []
        self.results['Speed (it/s)'] = []
        if self._ppo_bolt_enabled:
            self.results['BOLT heldout burden before'] = []
            self.results['BOLT heldout burden after'] = []
            self.results['BOLT heldout burden ratio'] = []
            self.results['BOLT demand rank'] = []
            self.results['BOLT history rank'] = []
        if self._ppo_spectral_advantage_enabled:
            self.results['Spectral advantage MC weight'] = []
            self.results['Spectral advantage correction fraction'] = []
            self.results['Spectral advantage value NTK rank'] = []
            self.results['Spectral advantage value NTK top1 mass'] = []
        if self._pbsr_enabled:
            self.results['PBSR Value primary loss'] = []
            self.results['PBSR Value loss'] = []
            self.results['PBSR Value coefficient'] = []
            self.results['PBSR Value slowest30 energy'] = []
            if self._pbsr_variant == 'actual_demand':
                self.results['PBSR Value residual mean square'] = []
                self.results['PBSR Value kernel mean eigenvalue'] = []
            if self._pbsr_variant == 'spectral':
                self.results['PBSR Value mean top singular value'] = []
            if self._pbsr_variant == 'ppo_v2':
                self.results['PBSR Value current burden'] = []
                self.results['PBSR Value reserve burden'] = []
                self.results['PBSR Value effective rank'] = []
                self.results['PBSR Policy primary loss'] = []
                self.results['PBSR Policy loss'] = []
                self.results['PBSR Policy coefficient'] = []
                self.results['PBSR Policy current burden'] = []
                self.results['PBSR Policy reserve burden'] = []
                self.results['PBSR Policy effective rank'] = []

        if self._no_stats == False:
            self.results['Policy dormant ratio'] = []
            self.results['Value dormant ratio'] = []
            
            self.results['Policy feature rank'] = []
            self.results['Value feature rank'] = []
            self.results['Policy hessian rank'] = []
            self.results['Value hessian rank'] = []
            self.results['Policy weight change'] = []
            self.results['Value weight change'] = []

        self._random_policy_state_dict = copy.deepcopy(self.policy.state_dict())
        self._random_vf_state_dict = copy.deepcopy(self._value_function.state_dict())

        # For ReDo
        self.random_policy = copy.deepcopy(self.policy)
        self._random_value_function = copy.deepcopy(self._value_function)

        if infer:
            self._infer_target_policy = copy.deepcopy(self.policy)
            self._infer_target_vf = copy.deepcopy(self._value_function)
            self._infer_alpha = 1.
            self._infer_beta = 10.
            print("Use InFeR Loss")
        
        if wasserstein:
            self._wasserstein_target_policy = copy.deepcopy(self.policy)
            self._wasserstein_target_vf = copy.deepcopy(self._value_function)
            self._wasserstein_lambda = wasserstein
            print("Use Wasserstein Regularization")


        if self._first_task is not None:
            if 'DMC' in self._first_task:
                policy_name = 'policy_dm_control_ppo_{}_{}_{}.pt'
                old_policy_name = 'old_policy_dm_control_ppo_{}_{}_{}.pt'
                vf_name = 'vf_dm_control_ppo_{}_{}_{}.pt'

            else:
                policy_name = 'policy_metaworld_ppo_{}_{}_{}.pt'
                old_policy_name = 'old_policy_metaworld_ppo_{}_{}_{}.pt'
                vf_name = 'vf_metaworld_ppo_{}_{}_{}.pt'

                if crelu:
                    policy_name = 'policy_CReLU_metaworld_ppo_{}_{}_{}.pt'
                    old_policy_name = 'old_policy_CReLU_metaworld_ppo_{}_{}_{}.pt'
                    vf_name = 'vf_CReLU_metaworld_ppo_{}_{}_{}.pt'

                if wasserstein:
                    policy_name = 'policy_Wasserstein_0.1_metaworld_ppo_{}_{}_{}.pt'
                    old_policy_name = 'old_policy_Wasserstein_0.1_metaworld_ppo_{}_{}_{}.pt'
                    vf_name = 'vf_Wasserstein_0.1_metaworld_ppo_{}_{}_{}.pt'
            
            load_model(self.policy, policy_name, self._first_task,
                       first_task_steps, self._seed)
            load_model(self._old_policy, old_policy_name, self._first_task,
                       first_task_steps, self._seed)
            load_model(self._value_function, vf_name, self._first_task,
                       first_task_steps, self._seed)
            
        
        if self._value_reset:
            print('############################################################')
            print('                     Value-reset!!!!!                       ')
            print('############################################################')

            self._value_function.load_state_dict(self._random_vf_state_dict)
        
        if self._policy_reset:
            print('############################################################')
            print('                     Policy-reset!!!!!                      ')
            print('############################################################')
            self.policy.load_state_dict(self._random_policy_state_dict)



    @staticmethod
    def _check_entropy_configuration(entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')

    @property
    def discount(self):
        """Discount factor used by the algorithm.
        Returns:
            float: discount factor.
        """
        return self._discount

    def _train_once(self, itr, eps):
        """Train the algorithm once.
        Args:
            itr (int): Iteration number.
            eps (EpisodeBatch): A batch of collected paths.
        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.
        """
        obs = np_to_torch(eps.padded_observations)
        rewards = np_to_torch(eps.padded_rewards)
        returns = np_to_torch(
            np.stack([
                discount_cumsum(reward, self.discount)
                for reward in eps.padded_rewards
            ]))
        valids = eps.lengths
        with torch.no_grad():
            baselines = self._value_function(obs, seq_idx=self.seq_idx)
        
        self._episode_reward_mean.append(rewards.mean().item())

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs, self.seq_idx)
            rewards += self._policy_ent_coeff * policy_entropies

        obs_flat = np_to_torch(eps.observations)
        actions_flat = np_to_torch(eps.actions)
        rewards_flat = np_to_torch(eps.rewards)
        next_obs_flat = np_to_torch(eps.next_observations)
        terminals_flat = np_to_torch(eps.terminals.astype(np.float32))
        returns_flat = torch.cat(filter_valids(returns, valids))
        advs_flat = self._compute_advantage(
            rewards, valids, baselines, observations=obs,
            returns=returns, seq_idx=self.seq_idx)

        next_environment_steps = self._environment_steps + len(obs_flat)
        task_step = (
            next_environment_steps - self._task_start_environment_step)
        task_changed = (
            self.seq_idx != getattr(self._sampler._envs[0], 'cur_seq_idx'))
        spectral_event = 'task_boundary' if task_changed else 'interval'
        spectral_due = (
            self._bellman_spectral_stats_enabled and
            self._ppo_value_spectral_probe.should_run(
                spectral_event, task_step))
        path_ends = torch.zeros_like(terminals_flat)
        path_end_indices = torch.as_tensor(
            np.cumsum(valids) - 1, dtype=torch.long,
            device=path_ends.device)
        path_ends[path_end_indices] = 1.
        bolt_target_bank = None
        if self._ppo_bolt_enabled:
            with torch.no_grad():
                next_values = self._value_function(
                    next_obs_flat, seq_idx=self.seq_idx).flatten()
            bolt_target_bank = build_multistep_value_targets(
                rewards_flat, next_values, terminals_flat, path_ends,
                returns_flat, self._ppo_bolt_horizons, self._discount)
        pbsr_reserve_directions = None
        if (
                self._pbsr_enabled and
                self._pbsr_variant == 'ppo_v2' and
                self.seq_idx in self._pbsr_active_tasks):
            pbsr_reserve_directions = self._pbsr.build_reserve_bank(
                obs_flat, actions_flat, next_obs_flat, terminals_flat,
                path_ends, self._discount, self.seq_idx)
        self._train(
            obs_flat, actions_flat, rewards_flat, next_obs_flat,
            terminals_flat, returns_flat, advs_flat, self.seq_idx,
            pbsr_reserve_directions, bolt_target_bank)


        self._old_policy.load_state_dict(self.policy.state_dict())

        # 매 step 마다 evaluate 할 필요가 없을듯?
        undiscounted_returns = 0
        # with torch.no_grad():
        policy_loss = self._compute_loss_with_adv(
            obs_flat, actions_flat, rewards_flat, advs_flat, self.seq_idx)
        vf_loss = self._value_function.compute_loss(
            obs_flat, returns_flat, seq_idx=self.seq_idx)
        kl = self._compute_kl_constraint(obs, self.seq_idx)
        policy_entropy = self._compute_policy_entropy(obs, self.seq_idx)

        end_time = time()


        if self._no_stats == False:
            policy_zero_cnt = sum(self.policy._stats['dormant'][-1000:]) / 1000
            value_zero_cnt = sum(self._value_function._stats['dormant'][-1000:]) / 1000

            policy_zero_cnt = sum(self.policy._stats['dormant'][-1000:]) / 1000
            value_zero_cnt = sum(self._value_function._stats['dormant'][-1000:]) / 1000

            epsilon = 1e-5

            value_last_weight = self._value_function.module._mean_module._output_layers[0][0].weight
            policy_last_weight = self.policy._module._mean_module._output_layers[self.seq_idx][0].weight

            value_hessian = weight_hessian(vf_loss, value_last_weight)
            policy_hessian = weight_hessian(policy_loss, policy_last_weight)

            value_hessian_rank = feature_rank(value_hessian, 1e-5)
            policy_hessian_rank = feature_rank(policy_hessian, 1e-5)

            print("policy / value hessian rank: ", policy_hessian_rank, value_hessian_rank)

            n = obs_flat.shape[0]

            value_normalized_feature = self._value_function._feature / np.sqrt(n)
            policy_normalized_feature = self.policy._feature.flatten(0, 1) / np.sqrt(n)

            value_feature_rank = feature_rank(value_normalized_feature, 1e-3)
            policy_feature_rank = feature_rank(policy_normalized_feature, 1e-5)

            print("policy / value feature rank: ", policy_feature_rank, value_feature_rank)

            vf_state_dict = copy.deepcopy(self._value_function.state_dict())
            policy_state_dict = copy.deepcopy(self.policy.state_dict())

        
            value_dev = weight_deviation(vf_state_dict, self.recent_vf_state_dict)
            policy_dev = weight_deviation(policy_state_dict, self.recent_policy_state_dict)

            print("policy / value weight deviation: ", policy_dev.item(), value_dev.item())

            self.recent_vf_state_dict = vf_state_dict
            self.recent_policy_state_dict = policy_state_dict

        if self._use_wandb:
            if self._no_stats == False:
                wandb.log({
                    'Running avg. of episode return': sum(self._episode_reward_mean) / len(self._episode_reward_mean),
                    'Policy loss': policy_loss.item(),
                    'Value loss': (vf_loss).item(),
                    'KL': (kl).item(),
                    'Speed (it/s)' : ((itr+1) / (end_time - self.start_time)),
                    'Policy dormant ratio': policy_zero_cnt,
                    'Value dormant ratio': value_zero_cnt,
                    'Value feature rank': value_feature_rank,
                    'Policy hessian rank': policy_hessian_rank,
                    'Value hessian rank': value_hessian_rank,
                    'Policy weight change': policy_dev.item(),
                    'Value weight change': value_dev.item(),
                })
            else:
                wandb.log({
                    'Running avg. of episode return': sum(self._episode_reward_mean) / len(self._episode_reward_mean),
                    'Policy loss': policy_loss.item(),
                    'Value loss': (vf_loss).item(),
                    'KL': (kl).item(),
                    'Speed (it/s)' : ((itr+1) / (end_time - self.start_time)),
                })

        self.results['Running avg. of episode return'].append(sum(self._episode_reward_mean) / len(self._episode_reward_mean))
        self.results['Policy loss'].append(policy_loss.item())
        self.results['Value loss'].append((vf_loss).item())
        self.results['KL'].append((kl).item())
        self.results['Speed (it/s)'].append(((itr+1)*2000 / (end_time - self.start_time)))
        if self._ppo_bolt_enabled:
            bolt_stats = self._ppo_bolt_last_stats
            self.results['BOLT heldout burden before'].append(
                bolt_stats.get('heldout_burden_before', float('nan')))
            self.results['BOLT heldout burden after'].append(
                bolt_stats.get('heldout_burden_after', float('nan')))
            self.results['BOLT heldout burden ratio'].append(
                bolt_stats.get('heldout_burden_ratio', float('nan')))
            self.results['BOLT demand rank'].append(
                bolt_stats.get('demand_rank', float('nan')))
            self.results['BOLT history rank'].append(
                bolt_stats.get('history_rank', float('nan')))
        if self._ppo_spectral_advantage_enabled:
            spectral_advantage_stats = self._ppo_spectral_advantage_last_stats
            self.results['Spectral advantage MC weight'].append(
                spectral_advantage_stats.get('mc_weight', float('nan')))
            self.results['Spectral advantage correction fraction'].append(
                spectral_advantage_stats.get(
                    'correction_fraction', float('nan')))
            self.results['Spectral advantage value NTK rank'].append(
                spectral_advantage_stats.get(
                    'kernel_entropy_rank', float('nan')))
            self.results['Spectral advantage value NTK top1 mass'].append(
                spectral_advantage_stats.get(
                    'kernel_top1_mass', float('nan')))
        if self._pbsr_enabled:
            pbsr_active = (
                self.seq_idx in self._pbsr_active_tasks and
                bool(self._pbsr_last_stats))
            pbsr_stats = self._pbsr_last_stats if pbsr_active else {}
            self.results['PBSR Value primary loss'].append(
                pbsr_stats.get(
                    'value_loss',
                    pbsr_stats.get('primary_loss', float('nan'))))
            self.results['PBSR Value loss'].append(
                pbsr_stats.get('loss', float('nan')))
            self.results['PBSR Value coefficient'].append(
                pbsr_stats.get('coefficient', float('nan')))
            self.results['PBSR Value slowest30 energy'].append(
                pbsr_stats.get('slowest30_energy', float('nan')))
            if self._pbsr_variant == 'actual_demand':
                self.results['PBSR Value residual mean square'].append(
                    pbsr_stats.get('residual_mean_square', float('nan')))
                self.results['PBSR Value kernel mean eigenvalue'].append(
                    pbsr_stats.get('kernel_mean_eigenvalue', float('nan')))
            if self._pbsr_variant == 'spectral':
                self.results['PBSR Value mean top singular value'].append(
                    pbsr_stats.get(
                        'mean_top_singular_value', float('nan')))
            if self._pbsr_variant == 'ppo_v2':
                policy_stats = (
                    self._pbsr_last_policy_stats if pbsr_active else {})
                self.results['PBSR Value current burden'].append(
                    pbsr_stats.get('current_burden', float('nan')))
                self.results['PBSR Value reserve burden'].append(
                    pbsr_stats.get('reserve_burden', float('nan')))
                self.results['PBSR Value effective rank'].append(
                    pbsr_stats.get('kernel_effective_rank', float('nan')))
                self.results['PBSR Policy primary loss'].append(
                    policy_stats.get('primary_loss', float('nan')))
                self.results['PBSR Policy loss'].append(
                    policy_stats.get('loss', float('nan')))
                self.results['PBSR Policy coefficient'].append(
                    policy_stats.get('coefficient', float('nan')))
                self.results['PBSR Policy current burden'].append(
                    policy_stats.get('current_burden', float('nan')))
                self.results['PBSR Policy reserve burden'].append(
                    policy_stats.get('reserve_burden', float('nan')))
                self.results['PBSR Policy effective rank'].append(
                    policy_stats.get(
                        'kernel_effective_rank', float('nan')))

        if self._no_stats == False:
            self.results['Policy dormant ratio'].append(policy_zero_cnt)
            self.results['Value dormant ratio'].append(value_zero_cnt)
            self.results['Policy feature rank'].append(policy_feature_rank)
            self.results['Value feature rank'].append(value_feature_rank)
            self.results['Policy hessian rank'].append(policy_hessian_rank)
            self.results['Value hessian rank'].append(value_hessian_rank)
            self.results['Policy weight change'].append(policy_dev.item())
            self.results['Value weight change'].append(value_dev.item())

        self._environment_steps = next_environment_steps
        self._last_value_probe_task_step = task_step
        if spectral_due:
            self._ppo_value_spectral_probe.run(
                event=spectral_event,
                global_step=self._environment_steps,
                task_step=task_step,
                task_idx=self.seq_idx,
                task_name=self._task_names[self.seq_idx],
                value_function=self._value_function,
                policy=self.policy)

        print('STEP: {} '.format(itr),'policy loss: {:.6f} '.format(policy_loss.item()), 'Value loss: {:.6f} '.format((vf_loss).item()), 'Reward avg.: {:.6f}'.format(sum(self._episode_reward_mean) / len(self._episode_reward_mean)), 'Speed: {:.1f} it/s'.format((itr+1)*2000 / (end_time - self.start_time)))
            
        if self.seq_idx != getattr(self._sampler._envs[0], "cur_seq_idx"):
            print('Task change')
            print('Current task number =',self.seq_idx)
            # NOTE: Must call self.task_change before changeing self.seq_idx
            self.task_change(self.seq_idx)
            self.seq_idx = getattr(self._sampler._envs[0], "cur_seq_idx")
            self._task_start_environment_step = self._environment_steps
            print('Next task number =',self.seq_idx)
        
        
        return np.mean(self._episode_reward_mean)

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.
        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.
        Returns:
            float: The average return in last epoch cycle.
        """
        last_return = None
        self.recent_policy_state_dict = copy.deepcopy(self.policy.state_dict())
        self.recent_vf_state_dict = copy.deepcopy(self._value_function.state_dict())

        for _ in trainer.step_epochs():
            for _ in range(self._n_samples):
                eps = trainer.obtain_episodes(trainer.step_itr, seq_idx=self.seq_idx)
                last_return = self._train_once(trainer.step_itr, eps)

            self.save_results()
            trainer.step_itr += 1
            if trainer.step_itr % 10 == 0: self._evaluate_policy(trainer.step_itr)

        if self._bellman_spectral_stats_enabled:
            final_task_idx = min(self.seq_idx, len(self._task_names) - 1)
            self._ppo_value_spectral_probe.run(
                event='final',
                global_step=self._environment_steps,
                task_step=self._last_value_probe_task_step,
                task_idx=final_task_idx,
                task_name=self._task_names[final_task_idx],
                value_function=self._value_function,
                policy=self.policy)

        return last_return

    def _train(self, obs, actions, rewards, next_observations, terminals,
               returns, advs, seq_idx, pbsr_reserve_directions=None,
               bolt_target_bank=None):
        r"""Train the policy and value function with minibatch.
        Args:
            obs (torch.Tensor): Observation from the environment with shape
                :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment with shape
                :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advs (torch.Tensor): Advantage value at each step with shape
                :math:`(N, )`.
        """
        if pbsr_reserve_directions is None:
            pbsr_reserve_directions = torch.zeros(
                obs.shape[0], 1, dtype=obs.dtype, device=obs.device)
        if bolt_target_bank is None:
            bolt_target_bank = torch.zeros(
                obs.shape[0], 1, dtype=obs.dtype, device=obs.device)
        for dataset in self._policy_optimizer.get_minibatch(
                obs, actions, rewards, advs, pbsr_reserve_directions):
            self._train_policy(*dataset, seq_idx=seq_idx)
        for dataset in self._vf_optimizer.get_minibatch(
                obs, returns, actions, rewards, next_observations, terminals,
                pbsr_reserve_directions, bolt_target_bank):
            self._train_value_function(*dataset, seq_idx=seq_idx)

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

    def _train_policy(self, obs, actions, rewards, advantages,
                      pbsr_reserve_directions, seq_idx):
        r"""Train the policy.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N, )`.
        Returns:
            torch.Tensor: Calculated mean scalar value of policy loss (float).
        """
        # pylint: disable=protected-access
        zero_optim_grads(self._policy_optimizer._optimizer)
        loss = self._compute_loss_with_adv(obs, actions, rewards, advantages, seq_idx)
        loss += self.cl_reg_loss(seq_idx)
        if self._infer:
            with torch.no_grad():
                _ = self._infer_target_policy(obs, seq_idx)[0]
            loss += self._infer_loss(self.policy, self._infer_target_policy)
        if self._wasserstein:
            loss += self.wasserstein_reg_loss(self.policy, self._wasserstein_target_policy)
        pbsr_due = (
            self._pbsr_enabled and
            self._pbsr_variant == 'ppo_v2' and
            self._pbsr.train_actor and
            seq_idx in self._pbsr_active_tasks and
            self._pbsr_policy_optimizer_steps %
            self._pbsr_update_interval == 0)
        if pbsr_due:
            diagnostics_due = (
                self._pbsr_policy_optimizer_steps < 10 or
                self._pbsr_policy_optimizer_steps % 1000 == 0)
            loss, pbsr_stats = self._pbsr.policy_loss(
                self.policy, self._old_policy, obs, actions, advantages,
                pbsr_reserve_directions, seq_idx, loss,
                self._lr_clip_range,
                compute_diagnostics=diagnostics_due)
            self._pbsr_last_policy_stats = dict(pbsr_stats)
            self._pbsr_last_policy_stats.update({
                'environment_step': int(self._environment_steps),
                'policy_optimizer_step': int(
                    self._pbsr_policy_optimizer_steps + 1),
                'task': int(seq_idx),
            })
            if diagnostics_due:
                print(
                    'PBSR_V2_POLICY_UPDATE',
                    json.dumps(self._pbsr_last_policy_stats), flush=True)
        loss.backward()
        self._policy_optimizer.step()
        self._pbsr_policy_optimizer_steps += 1

        return loss

    def _train_value_function(self, obs, returns, actions, rewards,
                              next_observations, terminals,
                              pbsr_reserve_directions, bolt_target_bank,
                              seq_idx=None):
        r"""Train the value function.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            returns (torch.Tensor): Acquired returns
                with shape :math:`(N, )`.
        Returns:
            torch.Tensor: Calculated mean scalar value of value function loss
                (float).
        """

        # pylint: disable=protected-access
        zero_optim_grads(self._vf_optimizer._optimizer)
        loss = self._value_function.compute_loss(obs, returns, seq_idx=seq_idx)
        if self._infer:
            with torch.no_grad():
                _ = self._infer_target_vf.compute_loss(obs, returns, seq_idx=seq_idx)
            loss += self._infer_loss(self._value_function, self._infer_target_vf)
        if self._wasserstein:
            loss += self.wasserstein_reg_loss(self._value_function, self._wasserstein_target_vf)
        pbsr_due = (
            self._pbsr_enabled and
            seq_idx in self._pbsr_active_tasks and
            self._pbsr_value_optimizer_steps %
            self._pbsr_update_interval == 0)
        if pbsr_due:
            diagnostics_due = (
                self._pbsr_value_optimizer_steps < 10 or
                self._pbsr_value_optimizer_steps % 1000 == 0)
            if self._pbsr_variant == 'v1':
                loss, pbsr_stats = self._pbsr.value_loss(
                    self._value_function, obs, actions, rewards,
                    next_observations, terminals, seq_idx, loss,
                    self._discount, compute_diagnostics=diagnostics_due)
            elif self._pbsr_variant == 'ppo_v2':
                loss, pbsr_stats = self._pbsr.value_loss(
                    self._value_function, obs, returns, rewards,
                    next_observations, terminals, pbsr_reserve_directions,
                    seq_idx, loss, self._discount,
                    compute_diagnostics=diagnostics_due)
            elif self._pbsr_variant == 'actual_demand':
                loss, pbsr_stats = self._pbsr.value_loss(
                    self._value_function, obs, returns, seq_idx, loss,
                    compute_diagnostics=diagnostics_due)
            else:
                loss, pbsr_stats = self._pbsr.value_loss(
                    self._value_function, loss,
                    compute_diagnostics=diagnostics_due)
            self._pbsr_last_stats = dict(pbsr_stats)
            self._pbsr_last_stats.update({
                'environment_step': int(self._environment_steps),
                'value_optimizer_step': int(
                    self._pbsr_value_optimizer_steps + 1),
                'task': int(seq_idx),
            })
            if diagnostics_due:
                update_names = {
                    'v1': 'PBSR_VALUE_UPDATE',
                    'ppo_v2': 'PBSR_V2_VALUE_UPDATE',
                    'actual_demand': 'ACTUAL_DEMAND_VALUE_UPDATE',
                    'spectral': 'SPECTRAL_VALUE_UPDATE',
                }
                print(update_names[self._pbsr_variant],
                      json.dumps(self._pbsr_last_stats), flush=True)
        loss.backward()
        if self._ppo_bolt_enabled:
            bolt_stats = self._ppo_bolt.maybe_refresh(
                obs, bolt_target_bank, seq_idx)
            if bolt_stats is not None:
                self._ppo_bolt_last_stats = dict(bolt_stats)
                self._ppo_bolt_last_stats.update({
                    'environment_step': int(self._environment_steps),
                })
                print('PPO_BOLT_VALUE_UPDATE', json.dumps(
                    self._ppo_bolt_last_stats), flush=True)
        self._vf_optimizer.step()
        if self._ppo_bolt_enabled:
            self._ppo_bolt.step_complete()
        self._pbsr_value_optimizer_steps += 1

        return loss

    def _compute_loss(self, obs, actions, rewards, valids, baselines, seq_idx):
        r"""Compute mean value of loss.
        Notes: P is the maximum episode length (self.max_episode_length)
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, P, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, P)`.
            valids (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, P)`.
        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).
        """
        obs_flat = torch.cat(filter_valids(obs, valids))
        actions_flat = torch.cat(filter_valids(actions, valids))
        rewards_flat = torch.cat(filter_valids(rewards, valids))
        returns = np_to_torch(
            np.stack([
                discount_cumsum(reward, self.discount)
                for reward in rewards.detach().cpu().numpy()
            ]))
        advantages_flat = self._compute_advantage(
            rewards, valids, baselines, observations=obs,
            returns=returns, seq_idx=seq_idx)

        return self._compute_loss_with_adv(obs_flat, actions_flat,
                                           rewards_flat, advantages_flat, seq_idx)

    def _compute_loss_with_adv(self, obs, actions, rewards, advantages, seq_idx):
        r"""Compute mean value of loss.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
        Returns:
            torch.Tensor: Calculated negative mean scalar value of objective.
        """
        objectives = self._compute_objective(advantages, obs, actions, rewards, seq_idx)

        if self._entropy_regularzied:
            policy_entropies = self._compute_policy_entropy(obs, seq_idx)
            objectives += self._policy_ent_coeff * policy_entropies

        return -objectives.mean()

    def _compute_advantage(self, rewards, valids, baselines,
                           observations=None, returns=None, seq_idx=None):
        r"""Compute mean value of loss.
        Notes: P is the maximum episode length (self.max_episode_length)
        Args:
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, P)`.
            valids (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, P)`.
        Returns:
            torch.Tensor: Calculated advantage values given rewards and
                baselines with shape :math:`(N \dot [T], )`.
        """
        advantages = compute_advantages(self._discount, self._gae_lambda,
                                        self.max_episode_length, baselines,
                                        rewards)
        advantage_flat = torch.cat(filter_valids(advantages, valids))

        spectral_active = (
            self._ppo_spectral_advantage_enabled and
            seq_idx in self._ppo_spectral_advantage_active_tasks)
        if spectral_active:
            monte_carlo = returns - baselines
            advantage_flat, stats = spectral_trust_advantage(
                self._value_function, observations, valids, advantages,
                monte_carlo, seq_idx,
                relative_ridge=self._ppo_spectral_advantage_ridge,
                block_size=self._ppo_spectral_advantage_block_size)
            self._ppo_spectral_advantage_last_stats = stats
            print('PPO_SPECTRAL_ADVANTAGE', json.dumps(stats), flush=True)

        if self._center_adv:
            means = advantage_flat.mean()
            variance = advantage_flat.var()
            advantage_flat = (advantage_flat - means) / (variance + 1e-8)

        if self._positive_adv:
            advantage_flat -= advantage_flat.min()

        return advantage_flat

    def _compute_kl_constraint(self, obs, seq_idx):
        r"""Compute KL divergence.
        Compute the KL divergence between the old policy distribution and
        current policy distribution.
        Notes: P is the maximum episode length (self.max_episode_length)
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.
        Returns:
            torch.Tensor: Calculated mean scalar value of KL divergence
                (float).
        """
        with torch.no_grad():
            old_dist = self._old_policy(obs, seq_idx)[0]

        new_dist = self.policy(obs, seq_idx)[0]

        kl_constraint = torch.distributions.kl.kl_divergence(
            old_dist, new_dist)

        return kl_constraint.mean()

    def _compute_policy_entropy(self, obs, seq_idx):
        r"""Compute entropy value of probability distribution.
        Notes: P is the maximum episode length (self.max_episode_length)
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.
        Returns:
            torch.Tensor: Calculated entropy values given observation
                with shape :math:`(N, P)`.
        """
        if self._stop_entropy_gradient:
            with torch.no_grad():
                policy_entropy = self.policy(obs, seq_idx)[0].entropy()
        else:
            policy_entropy = self.policy(obs, seq_idx)[0].entropy()

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_objective(self, advantages, obs, actions, rewards, seq_idx):
        r"""Compute objective value.
        Args:
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.
        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.
        """
        del rewards
        log_likelihoods = self.policy(obs, seq_idx)[0].log_prob(actions)

        return log_likelihoods * advantages
    
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
        eval_eps = []
        for seq_idx, eval_env in enumerate(self._eval_env):

            self.on_test_start(seq_idx)
            eps = obtain_evaluation_episodes(
                    self.policy,
                    eval_env,
                    seq_idx,
                    self._max_episode_length_eval,
                    num_eps=self._num_evaluation_episodes,
                    deterministic=self._use_deterministic_evaluation)
            

            
            eval_eps.append(eps)
            
            self.on_test_end(seq_idx)

            if isinstance(self._env_spec, list):
                last_return = log_performance(epoch,
                                      eps,
                                      discount=self._discount,
                                      results=self.results, 
                                      use_wandb=self._use_wandb)

        if not isinstance(self._env_spec, list):
            eval_eps = EpisodeBatch.concatenate(*eval_eps)
            last_return = log_multitask_performance(epoch, eval_eps,
                                                    self._discount,
                                                    self.results, 
                                                    use_wandb=self._use_wandb)
        return last_return

    def ReDo(self, seq_idx):

        policy_network = self.policy._module._mean_module
        value_network = self._value_function.module._mean_module

        random_policy_network = self.random_policy._module._mean_module
        random_value_network = self._random_value_function.module._mean_module

        network_list = [policy_network, value_network]
        random_network_list = [random_policy_network, random_value_network]

        for network_idx, (random_network, network) in enumerate(zip(random_network_list, network_list)):
            random_layers = random_network._layers
            layers = network._layers
            pre_idx = -1
            for idx, (random_layer, layer) in enumerate(zip(random_layers, layers)):
                
                with torch.no_grad():
                    if pre_idx!=-1:
                        pre_zero_idx = network._stats['dormant_idx'][pre_idx]
                        temp = 1 - pre_zero_idx.float()
                        temp = temp.unsqueeze(0)
                        layer[0].weight.data *= temp

                    zero_idx = network._stats['dormant_idx'][idx]
                    mask = zero_idx.float().unsqueeze(-1)

                    layer[0].weight.data = (1-mask)*layer[0].weight.data + mask*random_layer[0].weight.data
                    mask = mask.squeeze()
                    layer[0].bias.data = (1-mask)*layer[0].bias.data + mask*random_layer[0].bias.data

                    pre_idx = idx
            
            zero_idx = network._stats['dormant_idx'][pre_idx]

            if network_idx == 0:
                next_seq_idx = seq_idx + 1
                network._output_layers[next_seq_idx][0].weight.data[:, zero_idx] = 0
            else:
                network._output_layers[0][0].weight.data[:, zero_idx] = 0
    
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
    def task_change(self, seq_idx):
        self.on_task_start(seq_idx)

        if self._ppo_bolt_enabled:
            self._ppo_bolt.start_task(seq_idx + 1)
            self._ppo_bolt_last_stats = {}

        if self._ReDo and (seq_idx+1) < len(self._eval_env):
            self.ReDo(seq_idx)

        if self._policy_reset:
            self.policy.load_state_dict(self._random_policy_state_dict)
            self._old_policy.load_state_dict(self._random_policy_state_dict)
            print("Policy reset")
        if self._value_reset:
            self._value_function.load_state_dict(self._random_vf_state_dict)
            print("Value function reset")
    
    def save_models(self, log_name = None):

        if log_name is None:
            log_name = self._log_name

        #  model을 그냥 models에 저장한다
        if not os.path.exists('models/'):
            os.makedirs('models')
        if not os.path.exists('models/ppo_models'):
            os.makedirs('models/ppo_models')
        
        for net, name in zip(self.networks, self.networks_names):
            torch.save(net.state_dict(), './models/ppo_models/' + name + '_' + log_name + '.pt')

    def save_rollouts(self, log_name = None, buffer_size = int(1e6)):
        
        buffer = dict()
        seq_idx = 0
        eval_env = self._eval_env[0]
        
        episode_batch = obtain_evaluation_episodes(
                self.policy,
                eval_env,
                seq_idx,
                self._max_episode_length_eval,
                num_eps=self._num_evaluation_episodes,
                deterministic=self._use_deterministic_evaluation)
        buffer['observation'] = episode_batch.observations
        while buffer['observation'].shape[0] < buffer_size:
            episode_batch = obtain_evaluation_episodes(
                self.policy,
                eval_env,
                seq_idx,
                self._max_episode_length_eval,
                num_eps=self._num_evaluation_episodes,
                deterministic=self._use_deterministic_evaluation)
            observation = episode_batch.observations
            buffer['observation'] = np.concatenate((buffer['observation'], observation))
        buffer['observation'] = torch.Tensor(buffer['observation'][:buffer_size, :])
        buffer['observation'] = buffer['observation'].to(global_device())
        
        assert buffer['observation'].shape[0] == buffer_size

        if not os.path.exists('rollouts'):
            os.makedirs('rollouts')
        if not os.path.exists('rollouts/ppo_rollouts'):
            os.makedirs('rollouts/ppo_rollouts')


        path = './rollouts/ppo_rollouts/' + log_name + '.pkl'
        with open(path, "wb") as file:
            pickle.dump(buffer, file)

    def save_results(self, log_name = None):

        if log_name is None:
            log_name = self._log_name
        
        if not os.path.exists('logs/'):
            os.makedirs('logs/')

        path = './logs/' + log_name + '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy, self._old_policy, self._value_function
        ]
    
    @property
    def networks_names(self):
        return [
            'policy', 'old_policy', 'vf'
        ]

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
            self._infer_target_vf.to(device)
        if self._wasserstein:
            self._wasserstein_target_policy.to(device)
            self._wasserstein_target_vf.to(device)
        if self._ReDo:
            self.random_policy.to(device)
            self._random_value_function.to(device)
