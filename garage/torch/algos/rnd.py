import numpy as np
import torch
from torch import nn

from garage.torch.algos import MTSAC, PPO
from garage.torch.algos.sac import (
    _capture_training_rng_state, _restore_training_rng_state)
from garage.torch.policies import TanhGaussianMLPPolicy, GaussianMLPPolicy
from garage.replay_buffer import ExpertBuffer
from garage.torch import as_torch_dict
from garage.torch._functions import zero_optim_grads
from garage.torch import as_torch_dict, global_device
import copy
import random
from tqdm import tqdm
from time import time
import pickle
from pathlib import Path

import wandb


class RND_SAC(MTSAC):
    def __init__(self, cl_reg_coef=1.0, expert_buffer_size=10000,
                 replay_buffer_size=int(1e6), nepochs_offline=5,
                 env_seq=None, bc_kl='reverse', distill_kl='forward',
                 reset_offline_actor=False, teacher_steps=int(3e6),
                 teacher_root='.',
                 **sac_kwargs):
        super().__init__(**sac_kwargs)
        self._cl_reg_coef=cl_reg_coef
        self._expert_buffer_size=expert_buffer_size
        self._replay_buffer_size=replay_buffer_size
        self._nepochs_offline = nepochs_offline
        self._reset_offline_actor=reset_offline_actor
        self._teacher_steps = int(teacher_steps)
        self._teacher_root = Path(teacher_root).expanduser().resolve()
        self._env_seq=env_seq
        self._bc_kl = bc_kl
        self._distill_kl=distill_kl

        print('BC: ', bc_kl, ' Distill: ', distill_kl)

        self._expert_buffer=ExpertBuffer(capacity=expert_buffer_size, per_task=self._multi_input)
        self.BATCH_SIZE=128
        self._num_task_seen=0

        self.results = {}

        self.results['Policy loss'] = []
        self.results['BC loss'] = []
        self.results['Speed (it/s)'] = []
        self.results['Distillation update'] = []
        self.results['Distillation task index'] = []

        
    
    # Load first task
    # And then start training from second task

    def train(self, trainer):

        tasknum = len(self._eval_env)
        global_step = 0
        self._distillation_updates = 0
        self._learner_role = 'rnd_offline_student'
        self.seq_idx = 0

        target_task_name = self._env_seq[0]
        
        # Skip first task & load policy
        target_policy_state_dict = torch.load(
            self._teacher_model_path(target_task_name),
            map_location=global_device())
        self._copy_teacher_policy_state(
            self.policy, target_policy_state_dict, seq_idx=0)
        
        self.load_target_policy_and_buffer(0, self._replay_buffer_size)
        evaluation_rng_state = _capture_training_rng_state()
        try:
            last_return = self._evaluate_policy(trainer.step_itr, at_boundary=True)
        finally:
            _restore_training_rng_state(evaluation_rng_state)
        self.on_task_start(0)
        self.save_results()
        trainer.step_itr += 1

        task_list = list(range(1,tasknum))

        for seq_idx in task_list:
            self.seq_idx = seq_idx
            target_policy = self.load_target_policy_and_buffer(seq_idx, self._replay_buffer_size)
            total_observations, total_target_means, total_target_log_stds = self.make_single_task_targets(target_policy, seq_idx)
            num_iter = 0

            if self._reset_offline_actor:
                print('Reset the offline actor in R&D')
                self.policy.load_state_dict(self._random_policy_state_dict)
            
            for epoch in range(self._nepochs_offline):
                idx_list = list(range(len(total_observations)))
                random.shuffle(idx_list)
                pbar = tqdm(
                    range(0,len(total_observations), self.BATCH_SIZE), 
                    desc = 'Epoch '+str(epoch+1), 
                    ascii = ' =',
                    leave=True)

                for i in pbar:
                    start = i
                    end = start + self.BATCH_SIZE
                    if end > len(total_observations):
                        end = len(total_observations)
                    idxs = idx_list[start:end]
                    obs, target_means, target_log_stds = total_observations[idxs], total_target_means[idxs], total_target_log_stds[idxs]

                    action_info = self.policy(obs, seq_idx)[1]
                    means, log_stds = action_info['mean'], action_info['log_std']

                    if self._distill_kl == 'forward':
                        policy_loss = self.kl_loss((target_means, target_log_stds), (means, log_stds))
                    elif self._distill_kl == 'reverse':
                        policy_loss = self.kl_loss((means, log_stds), (target_means, target_log_stds))

                    bc_loss = self.cl_reg_loss(seq_idx)
                    loss = policy_loss + bc_loss

                    zero_optim_grads(self._policy_optimizer)
                    loss.backward()
                    self._policy_optimizer.step()

                    end_time = time()

                    global_step += 1
                    self._distillation_updates = global_step

                    if global_step % 1000 == 0:
                        if self._use_wandb:
                            wandb.log({
                                'Learner role': 'R&D offline student',
                                'Distillation update': global_step,
                                'Distillation task index': seq_idx,
                                'Policy loss': policy_loss.item(),
                                'BC loss': bc_loss.item(),
                                'Speed (it/s)' : (global_step / max(
                                    end_time - self.start_time, 1e-9))
                            })

                        self.results['Policy loss'].append(policy_loss.item())
                        self.results['BC loss'].append(bc_loss.item())
                        self.results['Speed (it/s)'].append(
                            global_step / max(
                                end_time - self.start_time, 1e-9))
                        self.results['Distillation update'].append(global_step)
                        self.results['Distillation task index'].append(seq_idx)

                
            evaluation_rng_state = _capture_training_rng_state()
            try:
                last_return = self._evaluate_policy(trainer.step_itr, at_boundary=True)
            finally:
                _restore_training_rng_state(evaluation_rng_state)
            self.save_results()
            
            self.on_task_start(seq_idx)
            trainer.step_itr += 1

        return np.mean(last_return)
                
    
    def on_task_start(self, seq_idx):

        assert self._num_task_seen == seq_idx

        
        obs_len = len(self.observations)
        idx = np.random.choice(list(range(obs_len)), size=self._expert_buffer_size)
        obs = self.observations[idx]
        self._expert_buffer.add_observation_batch(obs, self.policy, seq_idx)

        self._num_task_seen += 1

    
    def cl_reg_loss(self, seq_idx):
        
        if seq_idx == 0:
            return torch.zeros(1).to(global_device())
        
        if self._multi_input:
            loss = 0
            for task_idx in range(self._num_task_seen):
                obs, target_means, target_log_stds = self._expert_buffer.sample_expert_batch(self.BATCH_SIZE, task_idx=task_idx)
                action_info = self.policy(obs, task_idx)[1]
                means, log_stds = action_info['mean'], action_info['log_std']

                if self._bc_kl == 'reverse':
                    loss += self.kl_loss((means, log_stds), (target_means, target_log_stds)) * self._cl_reg_coef
                elif self._bc_kl == 'forward':
                    loss += self.kl_loss((target_means, target_log_stds),(means, log_stds)) * self._cl_reg_coef

                
        else:
            obs, target_means, target_log_stds, task_idxs = self._expert_buffer.sample_expert_batch(self.BATCH_SIZE)

            action_info = self.policy(obs, task_idxs)[1]
            means, log_stds = action_info['mean'], action_info['log_std']

            if self._bc_kl == 'reverse':
                loss = self.kl_loss((means, log_stds), (target_means, target_log_stds)) * self._cl_reg_coef
            elif self._bc_kl == 'forward':
                loss = self.kl_loss((target_means, target_log_stds),(means, log_stds)) * self._cl_reg_coef

        return loss
    
    def load_target_policy_and_buffer(self, seq_idx, replay_buffer_size):

        target_task_name = self._env_seq[seq_idx]

        # Load policy
        target_policy = copy.deepcopy(self.policy)
        loaded_state_dict = torch.load(
            self._teacher_model_path(target_task_name),
            map_location=global_device())
        self._copy_teacher_policy_state(
            target_policy, loaded_state_dict, seq_idx=seq_idx)

        _device = "cpu"
        if global_device() != None:
            _device = global_device()

        with open(self._teacher_rollout_path(target_task_name), "rb") as file:
            data = pickle.load(file)
        self.observations = data['observation'][:replay_buffer_size].to(_device)

        return target_policy

    
    def make_single_task_targets(self, target_policy, seq_idx):
        observations = self.observations

        means = []
        log_stds = []

        for i in range(0,len(observations),self.BATCH_SIZE):
            start = i
            if i+self.BATCH_SIZE > len(observations):
                end = len(observations)
            else:
                end = i+self.BATCH_SIZE
            with torch.no_grad():
                
                action_info = target_policy(
                    observations[start:end], seq_idx)[1]
                mean, log_std = action_info['mean'], action_info['log_std']
                means.append(mean)
                log_stds.append(log_std)
        
        mean_targets = torch.cat(means)
        log_std_targets = torch.cat(log_stds)

        return observations, mean_targets, log_std_targets

    def _teacher_stem(self, task_name):
        env_type = ('dm_control' if task_name.startswith('DMControl-')
                    else 'metaworld')
        return '{}_sac_{}_{}_{}'.format(
            env_type, task_name, self._teacher_steps, self._seed)

    def _teacher_model_path(self, task_name):
        return (self._teacher_root / 'models' / 'sac_models' /
                ('policy_' + self._teacher_stem(task_name) + '.pt'))

    def _teacher_rollout_path(self, task_name):
        return (self._teacher_root / 'rollouts' / 'sac_rollouts' /
                ('rollouts_' + self._teacher_stem(task_name) + '.pkl'))

    @staticmethod
    def _output_key_for_task(key, seq_idx):
        marker = '._output_layers.'
        if marker not in key:
            return key
        prefix, suffix = key.split(marker, 1)
        head_text, remainder = suffix.split('.', 1)
        teacher_head = int(head_text)
        if teacher_head not in (0, 1):
            return None
        return '{}{}{}.{}'.format(
            prefix, marker, 2 * seq_idx + teacher_head, remainder)

    def _copy_teacher_policy_state(self, policy, teacher_state, seq_idx):
        """Copy a one-task teacher into one head of a sequence policy.

        DMC sequence policies concatenate heterogeneous observation spaces.
        For their first shared layer, place the teacher columns in the slice
        selected by the target task's zero-padding module.
        """
        student_state = policy.state_dict()
        copied = 0
        for teacher_key, teacher_value in teacher_state.items():
            student_key = self._output_key_for_task(teacher_key, seq_idx)
            if student_key is None or student_key not in student_state:
                continue
            student_value = student_state[student_key]
            if teacher_value.shape == student_value.shape:
                student_state[student_key] = teacher_value
                copied += 1
                continue
            first_layer = '._layers.0.linear.weight'
            if (first_layer in teacher_key and teacher_value.ndim == 2 and
                    student_value.ndim == 2 and
                    teacher_value.shape[0] == student_value.shape[0] and
                    hasattr(policy, '_zero_pad_per_task')):
                left = policy._zero_pad_per_task[seq_idx].padding[0]
                right = left + teacher_value.shape[1]
                if right <= student_value.shape[1]:
                    expanded = student_value.clone()
                    expanded[:, left:right] = teacher_value
                    student_state[student_key] = expanded
                    copied += 1
        if copied == 0:
            raise ValueError('No compatible R&D teacher policy parameters found')
        policy.load_state_dict(student_state)
        

    @staticmethod
    def kl_loss(dist1, dist2):
        """
        Computes KL(dist1 || dist2)
        """
        mu1, log_std1 = dist1
        mu2, log_std2 = dist2

        std1 = log_std1.exp()
        std2 = log_std2.exp()

        mu_loss = 0.5 * ((1/(std2**2)) * (mu1-mu2).pow(2)).sum(dim=-1).mean()
        std_loss = 0.5 * (((std1/std2)**2 - 2*(log_std1 - log_std2))).sum(dim=-1).mean()

        return mu_loss + std_loss

    
class RND_PPO(PPO):

    def __init__(self, cl_reg_coef=1.0, expert_buffer_size=10000,
                 replay_buffer_size=int(1e6), nepochs_offline=5,
                 env_seq=None, bc_kl='reverse', distill_kl='forward',
                 teacher_steps=int(3e6), **ppo_kwargs):
        super().__init__(**ppo_kwargs)

        self._cl_reg_coef=cl_reg_coef
        self._expert_buffer_size=expert_buffer_size
        self._replay_buffer_size=replay_buffer_size
        self._nepochs_offline = nepochs_offline
        self._env_seq=env_seq
        self._bc_kl = bc_kl
        self._distill_kl=distill_kl
        self._teacher_steps = int(teacher_steps)

        print('BC: ', bc_kl, ' Distill: ', distill_kl)

        self._expert_buffer=ExpertBuffer(capacity=expert_buffer_size, per_task=self._multi_input)
        self.BATCH_SIZE=512
        self.TOTAL_OBS_SIZE = int(1e5)
        self._num_task_seen=0

        self.results = {}

        self.results['Policy loss'] = []
        self.results['BC loss'] = []
        self.results['Speed (it/s)'] = []
        


    def train(self, trainer):
        
        tasknum = len(self._eval_env)
        global_step = 0

        
        target_task_name = self._env_seq[0]
        if 'DMC' in target_task_name:
            task_list = list(range(tasknum))
        else:# Skip first task & load policy
            task_list = list(range(1, tasknum))
            model_name = 'policy_metaworld_ppo_{}_{}_{}.pt'.format(
                target_task_name, self._teacher_steps, self._seed)
        
            target_policy_state_dict = torch.load('./models/ppo_models/'+model_name, map_location=global_device())
            policy_state_dict = self.policy.state_dict()

            target_policy_state_dict = {k: v for k, v in target_policy_state_dict.items() if k in policy_state_dict}
            policy_state_dict.update(target_policy_state_dict)
            self.policy.load_state_dict(policy_state_dict)
            
            target_policy = self.load_target_policy_and_buffer(0)
            total_observations, total_target_means, total_target_log_stds = self.make_single_task_targets(target_policy, 0)
            self.on_task_start(total_observations, 0)

        for seq_idx in task_list:
            target_policy = self.load_target_policy_and_buffer(seq_idx)
            total_observations, total_target_means, total_target_log_stds = self.make_single_task_targets(target_policy, seq_idx)
            num_iter = 0

            for epoch in range(self._nepochs_offline):
                idx_list = list(range(len(total_observations)))
                random.shuffle(idx_list)
                pbar = tqdm(
                    range(0,len(total_observations), self.BATCH_SIZE), 
                    desc = 'Epoch '+str(epoch+1), 
                    ascii = ' =',
                    leave=True)

                for i in pbar:
                    start = i
                    end = start + self.BATCH_SIZE
                    if end > len(total_observations):
                        end = len(total_observations)
                    idxs = idx_list[start:end]
                    obs, target_means, target_log_stds = total_observations[idxs], total_target_means[idxs], total_target_log_stds[idxs]

                    action_info = self.policy(obs, seq_idx)[1]
                    means, log_stds = action_info['mean'], action_info['log_std']

                    if self._distill_kl == 'forward':
                        policy_loss = self.kl_loss((target_means, target_log_stds), (means, log_stds))
                    elif self._distill_kl == 'reverse':
                        policy_loss = self.kl_loss((means, log_stds), (target_means, target_log_stds))
                    bc_loss = self.cl_reg_loss(seq_idx)
                    loss = policy_loss + bc_loss

                    zero_optim_grads(self._policy_optimizer._optimizer)
                    loss.backward()
                    self._policy_optimizer.step()

                    end_time = time()

                    global_step += 1

                    if global_step % 1000 == 0:
                        
                        if self._use_wandb:
                            wandb.log({
                                'Policy loss': policy_loss.item(),
                                'BC loss': bc_loss.item(),
                                'Speed (it/s)' : (self.global_step / (end_time - self.start_time))
                            })

                        self.results['Policy loss'].append(policy_loss.item())
                        self.results['BC loss'].append(bc_loss.item())
                        self.results['Speed (it/s)'].append((self.global_step / (end_time - self.start_time)))

            last_return = self._evaluate_policy(trainer.step_itr)
            self.save_results()
            
            self.on_task_start(total_observations, seq_idx)
                
    
    def on_task_start(self, observations, seq_idx):

        assert self._num_task_seen == seq_idx

        idx = np.random.choice(list(range(self.TOTAL_OBS_SIZE)), size=self._expert_buffer_size)
        obs = observations[idx]
        self._expert_buffer.add_observation_batch(obs, self.policy, seq_idx)


        self._num_task_seen += 1

    
    def cl_reg_loss(self, seq_idx):
        if seq_idx == 0:
            return torch.zeros(1).to(global_device())
        
        if self._multi_input:
            loss = 0
            for task_idx in range(self._num_task_seen):
                obs, target_means, target_log_stds = self._expert_buffer.sample_expert_batch(self.BATCH_SIZE, task_idx=task_idx)
                action_info = self.policy(obs, task_idx)[1]
                means, log_stds = action_info['mean'], action_info['log_std']

                if self._bc_kl == 'reverse':
                    loss += self.kl_loss((means, log_stds), (target_means, target_log_stds)) * self._cl_reg_coef
                elif self._bc_kl == 'forward':
                    loss += self.kl_loss((target_means, target_log_stds),(means, log_stds)) * self._cl_reg_coef

                
        else:
            obs, target_means, target_log_stds, task_idxs = self._expert_buffer.sample_expert_batch(self.BATCH_SIZE)

            action_info = self.policy(obs, task_idxs)[1]
            means, log_stds = action_info['mean'], action_info['log_std']

            if self._bc_kl == 'reverse':
                loss = self.kl_loss((means, log_stds), (target_means, target_log_stds)) * self._cl_reg_coef
            elif self._bc_kl == 'forward':
                loss = self.kl_loss((target_means, target_log_stds),(means, log_stds)) * self._cl_reg_coef

        return loss
    
    def load_target_policy_and_buffer(self, seq_idx):

        target_task_name = self._env_seq[seq_idx]
        model_name = 'policy_metaworld_ppo_{}_{}_{}.pt'.format(
            target_task_name, self._teacher_steps, self._seed)

        # Load policy
        target_policy = copy.deepcopy(self.policy)
        loaded_state_dict = torch.load('./models/ppo_models/'+model_name, map_location=global_device())
        
        target_policy_state_dict = target_policy.state_dict()

        loaded_state_dict = {k: v for k, v in loaded_state_dict.items() if k in target_policy_state_dict}
        target_policy_state_dict.update(loaded_state_dict)
        target_policy.load_state_dict(target_policy_state_dict)


        return target_policy

    
    def make_single_task_targets(self, target_policy, seq_idx):
        
        batch_size = 512

        buffer_name = 'metaworld_ppo_{}_{}_{}.pkl'.format(
            self._env_seq[seq_idx], self._teacher_steps, self._seed)
            
        _device = "cpu"
        if global_device() != None:
            _device = global_device()

        with open('./rollouts/ppo_rollouts/' + buffer_name, "rb") as file:
            data = pickle.load(file)
        observations = data['observation'][:self._replay_buffer_size].to(_device)

        means = []
        log_stds = []

        for i in range(0,self._replay_buffer_size,batch_size):
            start = i
            if i+batch_size > self._replay_buffer_size:
                end = self._replay_buffer_size
            else:
                end = i+batch_size
            with torch.no_grad():
                
                action_info = target_policy(observations[start:end], 0)[1]
                mean, log_std = action_info['mean'], action_info['log_std']
                means.append(mean)
                log_stds.append(log_std)
        
        mean_targets = torch.cat(means)
        log_std_targets = torch.cat(log_stds)

        return observations, mean_targets, log_std_targets

    @staticmethod
    def kl_loss(dist1, dist2):
        """
        Computes KL(dist1 || dist2)
        """
        mu1, log_std1 = dist1
        mu2, log_std2 = dist2

        std1 = log_std1.exp()
        std2 = log_std2.exp()

        mu_loss = 0.5 * ((1/(std2**2)) * (mu1-mu2).pow(2)).sum(dim=-1).mean()
        std_loss = 0.5 * (((std1/std2)**2 - 2*(log_std1 - log_std2))).sum(dim=-1).mean()

        return mu_loss + std_loss
