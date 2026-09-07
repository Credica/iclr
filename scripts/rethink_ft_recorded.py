#!/usr/bin/env python3
"""Opt-in P1--P6 FT runner. Environment-clock protocol; existing SAC updates.

This does not change main_garage.py or any existing running experiment.
Heavy Jacobian/kernel/fitting analysis is OFFLINE, using the recorded weights,
inputs, exact targets and Adam states. No spectral intervention is applied.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import traceback

import akro
import metaworld
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage import EnvSpec
from garage.replay_buffer import PathBuffer
from garage.torch import as_torch_dict, set_gpu_mode
from garage.torch.algos.finetuning import Finetuning_SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction

PAIRS = {
    'P1': ['sweep-into-v2', 'push-wall-v2'],
    'P2': ['push-wall-v2', 'sweep-into-v2'],
    'P3': ['button-press-v2', 'button-press-wall-v2'],
    'P4': ['button-press-wall-v2', 'button-press-v2'],
    'P5': ['reach-v2', 'window-close-v2'],
    'P6': ['window-close-v2', 'reach-v2'],
}
TRANSITION_KEYS = ('observation', 'action', 'reward', 'next_observation', 'terminal')


def stable_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def rng_fingerprint():
    state = rng_state()
    digest = hashlib.sha256(repr(state['python']).encode())
    digest.update(state['numpy'][1].tobytes())
    digest.update(repr((state['numpy'][0], state['numpy'][2:])).encode())
    for value in [state['torch']] + state['cuda']:
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


@contextlib.contextmanager
def isolated_rng(seed=None):
    state = rng_state()
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        yield
    finally:
        restore_rng(state)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    os.replace(str(temporary), str(path))


def atomic_torch(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, str(temporary))
    os.replace(str(temporary), str(path))


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def vectors(models):
    return np.concatenate([p.detach().reshape(-1).cpu().numpy()
                           for model in models for p in model.parameters()])


def stack_records(records):
    return {key: np.stack([r[key] for r in records]) for key in records[0]}


def summary_values(tensor):
    array = tensor.detach().cpu().numpy().reshape(-1)
    return dict(mean=float(array.mean()), std=float(array.std()),
                quantiles=np.quantile(array, [0, .1, .5, .9, 1]).tolist())


class RecordedFT(Finetuning_SAC):
    """Read-only taps around the unmodified SAC objective and Adam update."""
    capture = False
    capture_update = False

    def _critic_objective(self, samples_data, seq_idx, return_predictions=False):
        if self.capture_update:
            self._before_critic_parameters = [vectors([model]) for model in (self._qf1, self._qf2)]
        values = super()._critic_objective(samples_data, seq_idx, return_predictions=True)
        if self.capture:
            self.last_batch = {key: value.detach().cpu().numpy().copy()
                               for key, value in samples_data.items()}
            self.last_predictions = {
                'q1': values[2].detach().cpu().numpy().copy(),
                'q2': values[3].detach().cpu().numpy().copy(),
                'target': values[4].detach().cpu().numpy().copy()}
        return values if return_predictions else values[:2]

    def _after_critic_update(self, samples_data, seq_idx):
        if self.capture_update:
            self.last_gradient_norms = [float(torch.sqrt(sum(
                (p.grad.detach() ** 2).sum() for p in model.parameters()
                if p.grad is not None)).cpu()) for model in (self._qf1, self._qf2)]
            self.last_update_norms = [float(np.linalg.norm(vectors([model])-before))
                                      for model, before in zip((self._qf1, self._qf2),
                                                               self._before_critic_parameters)]


def build_algorithm(spec, seed, device='cuda'):
    set_gpu_mode(device == 'cuda', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy = TanhGaussianMLPPolicy(spec, n_tasks=2, hidden_sizes=(256, 256),
                                  hidden_nonlinearity=nn.ReLU, no_stats=True)
    # Same fresh output-head initialization across task positions. These remain
    # separate parameters, not tied heads; A never optimizes the unused B head.
    outputs = policy._module._shared_mean_log_std_network._output_layers
    for offset in (0, 1):
        outputs[2 + offset].load_state_dict(outputs[offset].state_dict())
    q1 = ContinuousMLPQFunction(spec, hidden_sizes=(256, 256), hidden_nonlinearity=F.relu)
    q2 = ContinuousMLPQFunction(spec, hidden_sizes=(256, 256), hidden_nonlinearity=F.relu)
    algorithm = RecordedFT(
        policy=policy, qf1=q1, qf2=q2, env_spec=spec, sampler=None,
        replay_buffer=PathBuffer(capacity_in_transitions=1000000),
        num_tasks=1, eval_env=[], gradient_steps_per_itr=500, seed=seed,
        no_stats=True, use_wandb=False, bellman_probe=False,
        bellman_spectral_stats=False, q_reset=False, policy_reset=False,
        task_names=None, buffer_batch_size=64, policy_lr=3e-4, qf_lr=3e-4,
        discount=.99, target_update_tau=.005, initial_log_entropy=0.)
    algorithm.to(torch.device(device))
    return algorithm


class TargetWindow:
    """1000 actual updates plus both endpoints; fixed-noise anchor targets.

    Critic vectors reconstruct J delta-theta/nonlinearity offline. Exact online
    minibatches/targets distinguish Adam and sampling error from nonlinearity.
    Target values, distribution parameters and noise reconstruct reference y_t
    without storing another full actor/target checkpoint every update.
    """
    def __init__(self, run, nominal_start):
        self.run = run
        self.start = nominal_start
        self.index = 0
        self.pending = []
        self.root = run.root / 'target_windows' / ('B_env_%07d' % nominal_start)
        self.root.mkdir(parents=True)
        self.n = run.args.window_updates
        self.q_models = [run.algo._qf1, run.algo._qf2]
        q_vector = vectors(self.q_models)
        self.weights = np.lib.format.open_memmap(
            str(self.root / 'critic_parameters.npy'), mode='w+', dtype=np.float32,
            shape=(self.n + 1, len(q_vector)))
        self.weights[0] = q_vector
        self.bank = as_torch_dict({k: run.anchors[k] for k in TRANSITION_KEYS})
        generator = np.random.RandomState(stable_seed(run.args.seed, run.tasks[1], 'target-noise'))
        self.epsilon = torch.as_tensor(generator.normal(size=(len(run.anchors['action']), 4)),
                                       dtype=torch.float32, device=run.device)
        np.save(str(self.root / 'fixed_next_action_epsilon.npy'), self.epsilon.cpu().numpy())
        atomic_torch(self.root / 'start.pt', run.checkpoint_payload(full=False))
        atomic_json(self.root / 'manifest.json', dict(
            nominal_start_env_step=nominal_start, updates=self.n, status='running',
            target_definition='reward + gamma*(1-terminal)*(min(target_Q1,target_Q2)-alpha*log_pi)',
            target_noise='one independent Gaussian draw per anchor, common across all times and analysis branches',
            source='FT online actor, target twin critics and alpha at each recorded clock',
            tau=.005, loss='mean((Q-y)^2)', raw_kernel='J J^T / n; GD operator I - 2*lr*K',
            anchor_file='../../anchors/task_1_warmup.npz',
            anchor_indices=run.panel_indices.tolist(),
            parameter_layout=[dict(model=name, parameter=name_p, shape=list(p.shape), size=p.numel())
                              for name, model in zip(('qf1', 'qf2'), self.q_models)
                              for name_p, p in model.named_parameters()]))
        self.pending.append(self.reference_row())

    def reference_row(self):
        run, algo, bank = self.run, self.run.algo, self.bank
        with torch.no_grad():
            dist = algo.policy(bank['next_observation'], 1)[0]
            normal = dist._normal.base_dist
            z = normal.loc + normal.scale * self.epsilon
            action = torch.tanh(z)
            log_pi = dist.log_prob(action, pre_tanh_value=z)
            q1 = algo._target_qf1(bank['next_observation'], action, seq_idx=1).flatten()
            q2 = algo._target_qf2(bank['next_observation'], action, seq_idx=1).flatten()
            alpha = algo._log_alpha.exp()
            target = bank['reward'].flatten() + .99 * (1-bank['terminal'].flatten()) * (
                torch.minimum(q1, q2) - alpha * log_pi)
            result = dict(mu=normal.loc, std=normal.scale, next_action=action,
                          next_action_pre_tanh=z, log_pi=log_pi,
                          target_q1=q1, target_q2=q2, target=target,
                          q1=algo._qf1(bank['observation'], bank['action'], seq_idx=1).flatten(),
                          q2=algo._qf2(bank['observation'], bank['action'], seq_idx=1).flatten(),
                          alpha=alpha)
            result = {k: v.cpu().numpy().copy() for k, v in result.items()}
        result.update(kind='reference', index=self.index, **run.clocks())
        return result

    def after_update(self):
        self.pending.append(dict(kind='online_minibatch', index=self.index,
                                 samples=self.run.algo.last_batch,
                                 predictions=self.run.algo.last_predictions,
                                 **self.run.clocks()))
        self.index += 1
        self.weights[self.index] = vectors(self.q_models)
        self.pending.append(self.reference_row())
        if self.index % 100 == 0 or self.index == self.n:
            atomic_torch(self.root / ('rows_through_%04d.pt' % self.index), self.pending)
            self.pending = []
            self.weights.flush()
        if self.index == self.n:
            atomic_torch(self.root / 'end.pt', self.run.checkpoint_payload(full=False))
            atomic_json(self.root / 'complete.json', dict(updates=self.index, **self.run.clocks()))
            del self.weights
            return True
        return False


class Run:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        for name in ('checkpoints', 'anchors', 'target_windows', 'task_banks'):
            (self.root / name).mkdir()
        self.tasks = PAIRS[args.pair]
        self.device = torch.device(args.device)
        self.start_time = time.time()
        self.task = 0
        self.task_env_step = 0
        self.global_env_step = 0
        self.global_updates = 0
        self.task_updates = 0
        self.eval_interactions = 0
        self.eval_seconds = 0.
        self.record_seconds = 0.
        self.episode = 0
        self.env = None
        self.eval_envs = []
        self.banks = []
        self.window = None
        self.last_checkpoint = None
        self.train_losses = []
        self.logs = {}
        self.manifest = dict(
            schema_version=1, method='FT', pair=args.pair, seed=args.seed, tasks=self.tasks,
            args=vars(args), status='initializing', pid=os.getpid(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            source_root=str(Path(__file__).resolve().parents[1]),
            python=sys.version, torch=torch.__version__, numpy=np.__version__,
            metaworld=metaworld.__file__,
            protocol=dict(budget_unit='actual train environment interactions, warmup included',
                          actor='2x256 ReLU, separate position heads, identical fresh head initialization',
                          critic='shared twin 2x256 ReLU', optimizer='Adam', lr=.0003,
                          gamma=.99, tau=.005, batch_size=64, buffer_capacity=1000000,
                          initial_alpha=1., target_entropy=-4.,
                          collect_update_cadence='500 env steps, then 500 updates after warmup',
                          warmup_behavior='uniform actions, shared by task/seed across positions/branches',
                          bootstrap_time_limit=True, train_instances=50, eval_instances=50,
                          eval_policy='deterministic tanh Gaussian location, separate env/RNG',
                          evaluation_scope='current every interval including 0; all seen at task end',
                          full_checkpoint='task boundaries only; intermediate checkpoints are analysis snapshots',
                          heavy_online_jacobian=False, recording=True,
                          fresh_reference_caveat='same initial weights/head and warmup; transferred backbone differs'))
        self.status('initializing')

    def log(self, name, value):
        if name not in self.logs:
            self.logs[name] = (self.root / (name + '.jsonl')).open('a', buffering=1)
        self.logs[name].write(json.dumps(value, allow_nan=False) + '\n')

    def clocks(self):
        return dict(global_env_step=self.global_env_step, task_env_step=self.task_env_step,
                    global_critic_updates=self.global_updates, task_critic_updates=self.task_updates,
                    train_task_position=self.task, train_task=self.tasks[self.task],
                    policy_head=self.task, occurrence_id=self.task, policy_role='online')

    def status(self, state, **extra):
        self.manifest.update(status=state, **extra)
        atomic_json(self.root / 'run_manifest.json', self.manifest)
        atomic_json(self.root / 'status.json', dict(status=state, pid=os.getpid(),
                    elapsed_seconds=time.time()-self.start_time, **self.clocks(), **extra))

    def prepare(self):
        for task_name in self.tasks:
            print('PREPARING_INSTANCES', task_name, flush=True)
            with isolated_rng():
                train_seed = stable_seed(self.args.seed, task_name, 'train-instances')
                eval_seed = stable_seed(self.args.seed, task_name, 'eval-instances')
                train_benchmark = metaworld.MT1(task_name, seed=train_seed)
                eval_benchmark = metaworld.MT1(task_name, seed=eval_seed)
                train_tasks = train_benchmark.train_tasks
                eval_tasks = eval_benchmark.train_tasks  # MT1.test_tasks is empty in this version.
                assert len(train_tasks) == len(eval_tasks) == 50
                assert len(set(t.data for t in train_tasks)) == 50
                assert not (set(t.data for t in train_tasks) & set(t.data for t in eval_tasks))
                cls = train_benchmark.train_classes[task_name]
                env = cls()
                self.eval_envs.append(env)
                bank = dict(train=train_tasks, evaluation=eval_tasks, env_class=cls,
                            train_seed=train_seed, evaluation_seed=eval_seed,
                            reset_seeds=[stable_seed(self.args.seed, task_name, 'eval-reset', i)
                                         for i in range(50)])
                self.banks.append(bank)
                payload = pickle.dumps(bank, protocol=4)
                (self.root / 'task_banks' / (task_name + '.pkl')).write_bytes(payload)
                self.manifest.setdefault('instance_banks', {})[task_name] = dict(
                    sha256=hashlib.sha256(payload).hexdigest(), train_seed=train_seed,
                    evaluation_seed=eval_seed, reset_seeds=bank['reset_seeds'])
        env = self.eval_envs[0]
        assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
        spec = EnvSpec(akro.from_gym(env.observation_space), akro.from_gym(env.action_space),
                       max_episode_length=500)
        self.algo = build_algorithm(spec, self.args.seed, self.args.device)
        self.manifest['initialization_hashes'] = {name: state_hash(model)
                                                for name, model in zip(self.algo.networks_names, self.algo.networks)}
        atomic_torch(self.root / 'checkpoints' / 'initial_models.pt',
                     {name: cpu_tree(model.state_dict()) for name, model in
                      zip(self.algo.networks_names, self.algo.networks)})
        self.status('prepared')

    def reset_episode(self):
        self.instance = int(self.sample_rng.randint(50))
        self.reset_seed = int(self.sample_rng.randint(2**32))
        with isolated_rng(self.reset_seed):
            self.env.seed(self.reset_seed)
            self.env.set_task(self.banks[self.task]['train'][self.instance])
            self.obs = np.asarray(self.env.reset(), dtype=np.float32)
        self.ep_length, self.ep_return, self.ep_success = 0, 0., False

    def begin_task(self, position):
        self.task = position
        self.algo.seq_idx = position
        self.task_env_step = self.task_updates = self.episode = 0
        self.warmup = []
        self.reservoir = []
        self.reservoir_seen = 0
        self.anchors = None
        name = self.tasks[position]
        self.sample_rng = np.random.RandomState(stable_seed(self.args.seed, name, 'episode-sampler'))
        self.action_rng = np.random.RandomState(stable_seed(self.args.seed, name, 'warmup-actions'))
        self.replay_rng = np.random.RandomState(stable_seed(self.args.seed, name, 'replay'))
        self.reservoir_rng = np.random.RandomState(stable_seed(self.args.seed, name, 'reservoir'))
        if position:
            parent = self.last_checkpoint
            self.algo.task_change(position - 1)
            self.log('boundary_events', dict(
                event='A_to_B', parent_checkpoint=parent, actor='carried, select untouched B head',
                online_critics='carried unchanged', critic_optimizers='carried Adam',
                actor_optimizer='carried Adam', alpha='reset to 1, fresh alpha Adam',
                replay='cleared', targets='hard sync to online critics', clip=False, reset=False,
                **self.clocks()))
        with isolated_rng():
            if self.env is not None:
                self.env.close()
            self.env = self.banks[position]['env_class']()
        self.reset_episode()
        self.save_checkpoint(full=True)
        self.evaluate(position, 'task_entry')
        self.status('running')

    def checkpoint_payload(self, full=False):
        algo = self.algo
        payload = dict(schema_version=1, method='FT', args=vars(self.args), tasks=self.tasks,
                       clocks=self.clocks(), parent_checkpoint=self.last_checkpoint,
                       models={name: cpu_tree(model.state_dict()) for name, model in
                               zip(algo.networks_names, algo.networks)},
                       optimizers={name: cpu_tree(getattr(algo, attr).state_dict()) for name, attr in
                                   [('policy', '_policy_optimizer'), ('qf1', '_qf1_optimizer'),
                                    ('qf2', '_qf2_optimizer'), ('alpha', '_alpha_optimizer')]},
                       log_alpha=cpu_tree(algo._log_alpha),
                       rng=rng_state(), full_boundary_state=full,
                       optimizer_parameter_names={name: [key for key, _ in model.named_parameters()]
                                                  for name, model in zip(algo.networks_names[:3], algo.networks[:3])},
                       architecture=dict(hidden_sizes=[256, 256], heads=2, activation='ReLU'),
                       loss='mean squared error; uncentered K=JJ^T/n; actual gradient scale=2/n')
        if full:
            payload.update(
                replay=algo.replay_buffer,
                sampler=dict(episode=self.episode, instance=self.instance, reset_seed=self.reset_seed,
                             observation=self.obs.copy(), episode_length=self.ep_length,
                             episode_return=self.ep_return, episode_success=self.ep_success),
                independent_rng={key: getattr(self, key).get_state() for key in
                                 ('sample_rng', 'action_rng', 'replay_rng', 'reservoir_rng')},
                environment=pickle.dumps(self.env, protocol=4),
                next_operation='task_boundary' if self.task_env_step == self.args.steps_per_task else 'collect',
                reservoir=self.reservoir, reservoir_seen=self.reservoir_seen,
                initial_models='initial_models.pt', banks='../task_banks')
        return payload

    def save_checkpoint(self, full=False):
        start = time.time()
        name = 'task_%d_env_%07d%s.pt' % (self.task, self.task_env_step, '_full' if full else '')
        atomic_torch(self.root / 'checkpoints' / name, self.checkpoint_payload(full))
        self.last_checkpoint = name
        self.log('checkpoint_events', dict(file=name, full=full, **self.clocks()))
        self.record_seconds += time.time() - start

    def evaluate(self, position, reason):
        start = time.time()
        before_rng = rng_fingerprint()
        env, bank = self.eval_envs[position], self.banks[position]
        results = []
        print('EVAL_START', self.tasks[position], self.clocks(), flush=True)
        with isolated_rng(), torch.no_grad():
            training = self.algo.policy.training
            self.algo.policy.eval()
            try:
                for index in range(self.args.eval_episodes):
                    reset_seed = bank['reset_seeds'][index]
                    random.seed(reset_seed)
                    np.random.seed(reset_seed)
                    env.seed(reset_seed)
                    env.set_task(bank['evaluation'][index])
                    observation = env.reset()
                    first_obs = np.asarray(observation).copy()
                    total, success = 0., False
                    for step in range(self.args.episode_length):
                        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                        action = self.algo.policy(obs, position)[0].mean[0].cpu().numpy()
                        observation, reward, terminal, info = env.step(action)
                        total += float(reward)
                        success = success or bool(info.get('success', False))
                        self.eval_interactions += 1
                        if terminal:
                            break
                    row = dict(**self.clocks(), eval_task=self.tasks[position], eval_task_position=position,
                               eval_policy_head=position, eval_occurrence_id=position,
                               checkpoint_id=(self.last_checkpoint if self.last_checkpoint.startswith(
                                   'task_%d_env_%07d' % (self.task, self.task_env_step)) else None),
                               last_saved_checkpoint=self.last_checkpoint,
                               policy_version='update_%d' % self.global_updates,
                               reason=reason, episode=index, instance=index, reset_seed=reset_seed,
                               initial_observation=first_obs.tolist(),
                               goal=np.asarray(env._target_pos).tolist(),
                               return_value=total, success_any=success, length=step + 1,
                               deterministic=True)
                    self.log('eval_episodes', row)
                    results.append(row)
            finally:
                self.algo.policy.train(training)
        assert before_rng == rng_fingerprint(), 'Evaluation changed training RNG'
        aggregate = dict(**self.clocks(), eval_task_position=position, eval_task=self.tasks[position],
                         reason=reason, episodes=len(results),
                         training_rng_preserved=True,
                         average_return=float(np.mean([r['return_value'] for r in results])),
                         success_rate=float(np.mean([r['success_any'] for r in results])))
        self.log('eval_summary', aggregate)
        print('EVAL_DONE', json.dumps(aggregate), flush=True)
        self.eval_seconds += time.time() - start

    def collect(self, steps, warmup=False):
        records = []
        for _ in range(steps):
            if self.ep_length >= self.args.episode_length:
                self.reset_episode()
            if warmup:
                action = self.action_rng.uniform(-1, 1, size=4).astype(np.float32)
            else:
                with torch.no_grad():
                    obs = torch.as_tensor(self.obs, device=self.device).unsqueeze(0)
                    action = self.algo.policy(obs, self.task)[0].sample()[0].cpu().numpy()
            next_obs, reward, terminal, info = self.env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            self.task_env_step += 1
            self.global_env_step += 1
            self.ep_length += 1
            self.ep_return += float(reward)
            self.ep_success = self.ep_success or bool(info.get('success', False))
            truncated = self.ep_length >= self.args.episode_length and not terminal
            record = dict(observation=self.obs.copy(), action=action.copy(),
                          reward=np.asarray([reward], dtype=np.float32),
                          next_observation=next_obs.copy(), terminal=np.asarray([terminal], dtype=np.float32),
                          terminated=np.bool_(terminal), truncated=np.bool_(truncated),
                          episode=np.int64(self.episode), instance=np.int64(self.instance),
                          reset_seed=np.int64(self.reset_seed), task_env_step=np.int64(self.task_env_step),
                          global_env_step=np.int64(self.global_env_step),
                          behavior=np.int8(0 if warmup else 1))
            records.append(record)
            if warmup:
                self.warmup.append(record)
            else:
                self.reservoir_seen += 1
                if len(self.reservoir) < 1024:
                    self.reservoir.append(record)
                else:
                    index = int(self.reservoir_rng.randint(self.reservoir_seen))
                    if index < 1024:
                        self.reservoir[index] = record
            self.obs = next_obs
            if terminal or truncated:
                self.log('train_episodes', dict(**self.clocks(), episode=self.episode,
                         instance=self.instance, reset_seed=self.reset_seed,
                         return_value=self.ep_return, success_any=self.ep_success, length=self.ep_length))
                self.episode += 1
                self.ep_length = self.args.episode_length  # reset before next action, not before boundary snapshot
        batch = stack_records(records)
        self.algo.replay_buffer.add_path({key: batch[key] for key in TRANSITION_KEYS})

    def save_warmup(self):
        bank = stack_records(self.warmup)
        path = self.root / 'anchors' / ('task_%d_warmup.npz' % self.task)
        np.savez_compressed(str(path), **bank)
        rng = np.random.RandomState(stable_seed(self.args.seed, self.tasks[self.task], 'anchor-panels'))
        episodes = rng.permutation(np.unique(bank['episode']))
        assert len(episodes) >= 2
        first, second = np.array_split(episodes, 2)
        panels = [rng.choice(np.flatnonzero(np.isin(bank['episode'], group)), self.args.panel_size,
                             replace=False) for group in (first, second)]
        assert not set(bank['episode'][panels[0]]) & set(bank['episode'][panels[1]])
        self.panel_indices = np.concatenate(panels)
        self.anchors = {key: values[self.panel_indices] for key, values in bank.items()}
        anchor_indices = rng.choice(len(self.warmup), min(1024, len(self.warmup)), replace=False)
        np.savez_compressed(str(self.root / 'anchors' / ('task_%d_panels.npz' % self.task)),
                            panel_0=panels[0], panel_1=panels[1], anchor_1024=anchor_indices)
        atomic_json(self.root / 'anchors' / ('task_%d_manifest.json' % self.task), dict(
            full_warmup_count=len(self.warmup), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            panels='trajectory-disjoint, independently sampled without replacement',
            panel_size=self.args.panel_size, behavior={'0': 'uniform warmup', '1': 'SAC stochastic actor'},
            shared_across_branches=True, **self.clocks()))
        self.warmup = []

    def log_training(self):
        algo = self.algo
        row = dict(**self.clocks(), buffer_size=algo.replay_buffer.n_transitions_stored,
                   alpha=float(algo._log_alpha.detach().exp().cpu()), regularizer_loss=0.,
                   gradient_norms_last_update=getattr(algo, 'last_gradient_norms', None),
                   critic_update_norms_last_update=getattr(algo, 'last_update_norms', None),
                   statistics_scope='loss means over updates since last log; Q/TD/reward summaries from last minibatch')
        if self.train_losses:
            means = torch.stack(self.train_losses).mean(0).cpu().tolist()
            row.update(actor_loss=means[0], q1_loss=means[1], q2_loss=means[2],
                       aggregated_updates=len(self.train_losses))
            self.train_losses = []
            pred = algo.last_predictions
            row['last_minibatch'] = {key: summary_values(torch.as_tensor(value))
                                     for key, value in pred.items()}
            row['last_minibatch']['reward'] = summary_values(torch.as_tensor(algo.last_batch['reward']))
            for key in ('q1', 'q2'):
                row['last_minibatch'][key + '_td'] = summary_values(torch.as_tensor(
                    pred[key].reshape(-1) - pred['target'].reshape(-1)))
        self.log('train_metrics', row)
        self.log('resource_metrics', dict(**self.clocks(), wall_seconds=time.time()-self.start_time,
                 eval_interactions=self.eval_interactions, eval_seconds=self.eval_seconds,
                 checkpoint_seconds=self.record_seconds,
                 cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.args.device == 'cuda' else 0))
        self.status('running')
        if self.task_env_step % 10000 == 0:
            print('PROGRESS', json.dumps(row), flush=True)

    def train(self):
        self.prepare()
        for task in range(2):
            self.begin_task(task)
            while self.task_env_step < self.args.warmup_steps:
                self.collect(min(self.args.collect_steps, self.args.warmup_steps-self.task_env_step), warmup=True)
                if self.task_env_step % 1000 == 0:
                    self.log_training()
            self.save_warmup()
            self.save_checkpoint()
            while self.task_env_step < self.args.steps_per_task:
                if task == 1 and self.task_env_step in self.args.window_starts:
                    assert self.window is None
                    self.window = TargetWindow(self, self.task_env_step)
                count = min(self.args.collect_steps, self.args.steps_per_task-self.task_env_step)
                self.collect(count)
                for update in range(count):
                    algo = self.algo
                    algo.capture = self.window is not None or update == count-1
                    algo.capture_update = update == count-1
                    idx = self.replay_rng.randint(algo.replay_buffer.n_transitions_stored, size=64)
                    samples = as_torch_dict(algo.replay_buffer.sample_transitions(64, idx=idx))
                    losses = algo.optimize_policy(samples, task)
                    algo._update_targets()
                    self.task_updates += 1
                    self.global_updates += 1
                    algo.global_step = self.global_updates
                    self.train_losses.append(torch.stack([loss.detach() for loss in losses]))
                    if self.window is not None and self.window.after_update():
                        self.window = None
                if self.task_env_step % 1000 == 0 or self.task_env_step == self.args.steps_per_task:
                    self.log_training()
                endpoint = self.task_env_step == self.args.steps_per_task
                if self.task_env_step in self.args.snapshot_steps or endpoint:
                    self.save_checkpoint(full=endpoint)
                    if self.reservoir:
                        np.savez_compressed(str(self.root / 'anchors' / (
                            'task_%d_reservoir_env_%07d.npz' % (task, self.task_env_step))),
                            **stack_records(self.reservoir))
                if self.task_env_step % self.args.eval_interval == 0 or endpoint:
                    self.evaluate(task, 'task_end_current' if endpoint else 'periodic_current')
                if endpoint:
                    for previous in range(task):
                        self.evaluate(previous, 'task_end_retention')
            assert self.task_env_step == self.args.steps_per_task
            assert self.task_updates == self.args.steps_per_task-self.args.warmup_steps
        assert self.window is None
        self.status('completed', finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        print('COMPLETED', self.clocks(), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', choices=sorted(PAIRS), required=True)
    parser.add_argument('--seed', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--steps-per-task', type=int, default=1500000)
    parser.add_argument('--warmup-steps', type=int, default=10000)
    parser.add_argument('--collect-steps', type=int, default=500)
    parser.add_argument('--eval-interval', type=int, default=50000)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--episode-length', type=int, default=500)
    parser.add_argument('--panel-size', type=int, default=128)
    parser.add_argument('--window-updates', type=int, default=1000)
    parser.add_argument('--window-starts', type=int, nargs='*', default=[10000, 100000, 500000, 1000000])
    parser.add_argument('--snapshot-steps', type=int, nargs='*', default=[50000, 100000, 500000, 1000000])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    assert args.steps_per_task > args.warmup_steps > 0
    assert args.warmup_steps % args.collect_steps == 0
    assert args.steps_per_task % args.collect_steps == 0
    assert args.eval_interval % args.collect_steps == 0
    assert args.warmup_steps >= 2 * args.episode_length
    assert 1 <= args.eval_episodes <= 50
    assert all(x >= args.warmup_steps and x+args.window_updates <= args.steps_per_task
               and x % args.collect_steps == 0 for x in args.window_starts)
    if not args.smoke:
        assert (args.steps_per_task, args.warmup_steps, args.collect_steps, args.eval_interval,
                args.eval_episodes, args.episode_length, args.panel_size, args.window_updates) == (
                    1500000, 10000, 500, 50000, 50, 500, 128, 1000)
    return args


def main():
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = Run(args)
    try:
        run.train()
    except BaseException:
        run.status('failed', error=traceback.format_exc())
        raise
    finally:
        for handle in run.logs.values():
            handle.close()


if __name__ == '__main__':
    main()
