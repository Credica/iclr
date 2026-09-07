"""This modules creates a MTSAC model in PyTorch."""
# yapf: disable
import numpy as np
import torch

from garage import (EpisodeBatch, log_multitask_performance, log_performance,
                    obtain_evaluation_episodes)
from garage.torch import global_device
from garage.torch.algos import SAC

import wandb

# yapf: enable


class MTSAC(SAC):
    """A MTSAC Model in Torch.

    This MTSAC implementation uses is the same as SAC except for a small change
    called "disentangled alphas". Alpha is the entropy coefficient that is used
    to control exploration of the agent/policy. Disentangling alphas refers to
    having a separate alpha coefficients for every task learned by the policy.
    The alphas are accessed by using a the one-hot encoding of an id that is
    assigned to each task.

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
        env_spec (EnvSpec): The env_spec attribute of the environment that the
            agent is being trained in.
        sampler (garage.sampler.Sampler): Sampler.
        num_tasks (int): The number of tasks being learned.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If None, defaults to
            `max_episode_length`.
        eval_env (Environment): The environment used for collecting evaluation
            episodes.
        gradient_steps_per_itr (int): Number of optimization steps that should
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
        discount (float): The discount factor to be used during sampling and
            critic/q_function optimization.
        buffer_batch_size (int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size (int): The minimum number of transitions that need to
            be in the replay buffer before training can begin.
        target_update_tau (float): A coefficient that controls the rate at
            which the target q_functions update over optimization iterations.
        policy_lr (float): Learning rate for policy optimizers.
        qf_lr (float): Learning rate for q_function optimizers.
        reward_scale (float): Reward multiplier. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer (torch.optim.Optimizer): Optimizer to be used for
            policy/actor, q_functions/critics, and temperature/entropy
            optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.
        num_evaluation_episodes (int): The number of evaluation episodes used
            for computing eval stats at the end of every epoch.
        use_deterministic_evaluation (bool): True if the trained policy
            should be evaluated deterministically.

    """

    def __init__(
        self,
        policy,
        qf1,
        qf2,
        replay_buffer,
        env_spec,
        sampler,
        *,
        num_tasks,
        eval_env,
        gradient_steps_per_itr,
        seed=0,
        max_episode_length_eval=None,
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
        num_evaluation_episodes=100,
        use_deterministic_evaluation=True,
        use_exploration = False,
        q_reset = False,
        policy_reset = False,
        first_task = None,
        crelu=False,
        log_name = None,
        use_wandb = True,
        infer = False,
        wasserstein = 0,
        ReDo = False,
        redo_interval=1000,
        redo_tau=0.1,
        no_stats = False,
        scalar_log_interval=1000,
        feature_stats_interval=10000,
        hessian_stats_interval=10000,
        multi_input = False,
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
        spectral_power_iterations=1,
    ):

        super().__init__(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            replay_buffer=replay_buffer,
            sampler=sampler,
            env_spec=env_spec,
            seed=seed,
            max_episode_length_eval=max_episode_length_eval,
            gradient_steps_per_itr=gradient_steps_per_itr,
            fixed_alpha=fixed_alpha,
            target_entropy=target_entropy,
            initial_log_entropy=initial_log_entropy,
            discount=discount,
            buffer_batch_size=buffer_batch_size,
            min_buffer_size=min_buffer_size,
            target_update_tau=target_update_tau,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            reward_scale=reward_scale,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
            num_evaluation_episodes=num_evaluation_episodes,
            eval_env=eval_env,
            use_deterministic_evaluation=use_deterministic_evaluation, 
            use_exploration=use_exploration,
            q_reset=q_reset,
            policy_reset=policy_reset,
            first_task=first_task,
            crelu=crelu,
            log_name=log_name, 
            use_wandb=use_wandb,
            infer=infer,
            wasserstein=wasserstein, 
            ReDo=ReDo,
            redo_interval=redo_interval,
            redo_tau=redo_tau,
            no_stats=no_stats,
            scalar_log_interval=scalar_log_interval,
            feature_stats_interval=feature_stats_interval,
            hessian_stats_interval=hessian_stats_interval,
            multi_input=multi_input,
            bellman_probe=bellman_probe,
            bellman_probe_size=bellman_probe_size,
            bellman_probe_interval=bellman_probe_interval,
            bellman_probe_targets=bellman_probe_targets,
            bellman_probe_ridge=bellman_probe_ridge,
            bellman_probe_dir=bellman_probe_dir,
            bellman_spectral_stats=bellman_spectral_stats,
            bellman_spectral_anchor_size=bellman_spectral_anchor_size,
            bellman_spectral_fit_lr=bellman_spectral_fit_lr,
            bellman_reference_dir=bellman_reference_dir,
            bellman_spectral_task_steps=bellman_spectral_task_steps,
            pbsr=pbsr,
            pbsr_coef=pbsr_coef,
            pbsr_anchor_size=pbsr_anchor_size,
            pbsr_targets=pbsr_targets,
            pbsr_ridge=pbsr_ridge,
            pbsr_update_interval=pbsr_update_interval,
            pbsr_train_task_count=pbsr_train_task_count,
            task_names=task_names,
            exact_task_budget=exact_task_budget,
            branch_checkpoint=branch_checkpoint,
            branch_task_step=branch_task_step,
            branch_alpha=branch_alpha,
            branch_online_critic_source=branch_online_critic_source,
            branch_target_critic_source=branch_target_critic_source,
            plasticity_injection_mode=plasticity_injection_mode,
            plasticity_injection_width=plasticity_injection_width,
            plasticity_injection_widths=plasticity_injection_widths,
            plasticity_injection_rows=plasticity_injection_rows,
            plasticity_injection_targets=plasticity_injection_targets,
            plasticity_injection_ridge=plasticity_injection_ridge,
            plasticity_injection_task_indices=(
                plasticity_injection_task_indices),
            demand_aligned_reserve=demand_aligned_reserve,
            dar_rows=dar_rows,
            dar_hidden_dim=dar_hidden_dim,
            dar_feature_dim=dar_feature_dim,
            dar_targets=dar_targets,
            dar_ridge=dar_ridge,
            dar_trace_ratio=dar_trace_ratio,
            dar_capacity_price=dar_capacity_price,
            dar_alignment_steps=dar_alignment_steps,
            dar_alignment_lr=dar_alignment_lr,
            dar_task_indices=dar_task_indices,
            dsr_v2=dsr_v2,
            dsr_v2_kwargs=dsr_v2_kwargs,
            spectral_regularization=spectral_regularization,
            spectral_actor_coef=spectral_actor_coef,
            spectral_critic_coef=spectral_critic_coef,
            spectral_power_iterations=spectral_power_iterations)
        self._num_tasks = num_tasks
        self._eval_env = eval_env
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        self._seed = seed
        self._log_name = log_name
        self._use_wandb = use_wandb


        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                if isinstance(self.env_spec, list):
                    self._target_entropy_list = [-np.prod(spec.action_space.shape).item() for spec in self.env_spec]
                    
                else:
                    self._target_entropy = -np.prod(
                            self.env_spec.action_space.shape).item()
            initial_log_alpha = (
                np.log(self._restored_branch_alpha)
                if self._restored_branch_alpha is not None
                else self._initial_log_entropy)
            self._log_alpha = torch.Tensor([initial_log_alpha] *
                                           self._num_tasks).requires_grad_()
            self._alpha_optimizer = optimizer([self._log_alpha] *
                                              self._num_tasks,
                                              lr=self._policy_lr)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha] *
                                           self._num_tasks).log()
        self._epoch_mean_success_rate = []
        self._epoch_median_success_rate = []

    # def _get_log_alpha(self, samples_data):
    #     """Return the value of log_alpha.

    #     Args:
    #         samples_data (dict): Transitions(S,A,R,S') that are sampled from
    #             the replay buffer. It should have the keys 'observation',
    #             'action', 'reward', 'terminal', and 'next_observations'.

    #     Note:
    #         samples_data's entries should be torch.Tensor's with the following
    #         shapes:
    #             observation: :math:`(N, O^*)`
    #             action: :math:`(N, A^*)`
    #             reward: :math:`(N, 1)`
    #             terminal: :math:`(N, 1)`
    #             next_observation: :math:`(N, O^*)`

    #     Raises:
    #         ValueError: If the number of tasks, num_tasks passed to
    #             this algorithm doesn't match the length of the task
    #             one-hot id in the observation vector.

    #     Returns:
    #         torch.Tensor: log_alpha. shape is (1, self.buffer_batch_size)

    #     """
    #     obs = samples_data['observation']
    #     log_alpha = self._log_alpha
    #     one_hots = obs[:, -self._num_tasks:]
    #     if (log_alpha.shape[0] != one_hots.shape[1]
    #             or one_hots.shape[1] != self._num_tasks
    #             or log_alpha.shape[0] != self._num_tasks):
    #         raise ValueError(
    #             'The number of tasks in the environment does '
    #             'not match self._num_tasks. Are you sure that you passed '
    #             'The correct number of tasks?')
    #     ret = torch.mm(one_hots, log_alpha.unsqueeze(0).t()).squeeze()
    #     return ret

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

            if isinstance(self.env_spec, list):
                last_return = log_performance(epoch,
                                      eps,
                                      discount=self._discount,
                                      results=self.results, 
                                      use_wandb=self._use_wandb)


        
        
        if not isinstance(self.env_spec, list):
            eval_eps = EpisodeBatch.concatenate(*eval_eps)
            last_return = log_multitask_performance(epoch, eval_eps,
                                                    self._discount,
                                                    self.results,
                                                    use_wandb=self._use_wandb)
        return last_return


    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        super().to(device)
        if device is None:
            device = global_device()
        if not self._use_automatic_entropy_tuning:
            self._log_alpha = torch.Tensor([self._fixed_alpha] *
                                           self._num_tasks).log().to(device)
        else:
            initial_log_alpha = (
                np.log(self._restored_branch_alpha)
                if self._restored_branch_alpha is not None
                else self._initial_log_entropy)
            self._log_alpha = torch.Tensor(
                [initial_log_alpha] *
                self._num_tasks).to(device).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                                    lr=self._policy_lr)
