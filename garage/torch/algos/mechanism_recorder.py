"""Opt-in observational recording for the four-case pilot (no training changes).

Windows contain 1000 actual SAC updates, fixed-input/fixed-noise target paths,
the actual sampled targets, and twin-critic parameter paths. They are empirical
Bellman-demand observations, not a representation of the full Bellman operator.
"""
import contextlib
import copy
import json
import random

import numpy as np
import torch

from garage.torch import as_torch_dict


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return copy.deepcopy(value)


def save_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


@contextlib.contextmanager
def observational(models):
    """Reference forwards must not consume training RNG or activation stats."""
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
    attrs, lists, seen = [], [], set()
    for model in models:
        for module in model.modules():
            for name in ('_feature', '_features'):
                attrs.append((module, name, hasattr(module, name), getattr(module, name, None)))
            stats = getattr(module, '_stats', {})
            for values in stats.values():
                if isinstance(values, list) and id(values) not in seen:
                    seen.add(id(values))
                    lists.append((values, len(values)))
    try:
        with torch.no_grad():
            yield
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3]:
            torch.cuda.set_rng_state_all(rng[3])
        for module, name, existed, value in attrs:
            if existed:
                setattr(module, name, value)
            elif hasattr(module, name):
                delattr(module, name)
        for values, length in lists:
            del values[length:]


class MechanismRecorder:
    def __init__(self, algo, root, args, window_updates=1000):
        if not args.exact_sac_task_budget or not args.bellman_probe:
            raise ValueError('Mechanism recording requires exact SAC budgets and probe buffers')
        self.algo, self.root, self.args = algo, root, args
        root.mkdir(parents=True, exist_ok=False)
        self.models = [algo.policy, algo._qf1, algo._qf2,
                       algo._target_qf1, algo._target_qf2]
        self.length = window_updates
        self.done, self.snapshots = set(), set()
        self.window, self.last_batch = None, None
        self.objective = algo._critic_objective
        self.update = algo.train_once
        self.change = algo.task_change
        algo._critic_objective = self.capture_objective
        algo.train_once = self.train_once
        algo.task_change = self.task_change
        self.snapshot('loaded_parent' if args.branch_checkpoint else 'initialization', algo.seq_idx)
        save_json(root / 'protocol.json', dict(
            schema=1, args=vars(args), window_updates=window_updates,
            window_start_env_steps=[10000, 100000, 500000],
            window_tasks='all post-A task positions', anchor_rows=128,
            anchor_selection='first 128 retained warm-up transitions, fixed thereafter',
            limitations='anchors are not episode-independent; old ABC parent lacks RNG/replay',
            env_clock='new invocation only; legacy parent global_step is optimizer updates'))

    def clocks(self, task, after=False):
        a = self.algo
        return dict(task=int(task), global_env_step=int(a.global_env_step),
                    task_env_step=int(a.global_env_step-a._task_env_start_step),
                    global_update=int(a.global_step)+int(after),
                    task_critic_updates=int(a._critic_optimizer_steps))

    def snapshot(self, name, task):
        a = self.algo
        folder = self.root / 'checkpoints'
        folder.mkdir(exist_ok=True)
        payload = dict(**self.clocks(task),
                       models={n: cpu(m.state_dict()) for n, m in zip(
                           ('policy', 'qf1', 'qf2', 'target_qf1', 'target_qf2'), self.models)},
                       optimizers={n: cpu(o.state_dict()) for n, o in (
                           ('policy', a._policy_optimizer), ('qf1', a._qf1_optimizer),
                           ('qf2', a._qf2_optimizer))},
                       log_alpha=cpu(a._log_alpha),
                       rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                torch=torch.get_rng_state(),
                                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []))
        if a._use_automatic_entropy_tuning:
            payload['optimizers']['alpha'] = cpu(a._alpha_optimizer.state_dict())
        torch.save(payload, str(folder / ('task%d_%s_env%d_update%d.pt' % (
            task, name, a.global_env_step, a.global_step))))

    def vector(self):
        return np.concatenate([p.detach().cpu().numpy().ravel()
                               for m in self.models[1:3] for p in m.parameters()])

    def reference(self, task):
        a, b = self.algo, self.bank
        with observational(self.models):
            dist = a.policy(b['next_observation'], task)[0]
            normal = dist._normal.base_dist
            z = normal.loc + normal.scale * self.epsilon
            action = z.tanh()
            log_pi = dist.log_prob(action, pre_tanh_value=z)
            target_q1 = a._target_qf1(b['next_observation'], action, seq_idx=task).flatten()
            target_q2 = a._target_qf2(b['next_observation'], action, seq_idx=task).flatten()
            alpha = a._get_log_alpha(b).exp()
            target = b['reward'].flatten()*a._reward_scale + a._discount*(
                1-b['terminal'].flatten())*(torch.min(target_q1, target_q2)-alpha*log_pi)
            return cpu(dict(target=target, target_q1=target_q1, target_q2=target_q2,
                            q1=a._qf1(b['observation'], b['action'], seq_idx=task).flatten(),
                            q2=a._qf2(b['observation'], b['action'], seq_idx=task).flatten(),
                            mu=normal.loc, std=normal.scale, alpha=alpha))

    def capture_objective(self, samples, seq_idx, return_predictions=False):
        values = self.objective(samples, seq_idx, return_predictions=True)
        if self.window is not None:
            self.last_batch = cpu(dict(samples=samples, q1=values[2], q2=values[3], target=values[4]))
        return values if return_predictions else values[:2]

    def start_window(self, task, step):
        self.window = self.root / ('task%d_env%d_window' % (task, step))
        self.window.mkdir()
        anchors = {k: np.asarray(v)[:128].copy()
                   for k, v in self.algo._bellman_probe_buffers[task].items()}
        # At later windows use the same stored task warm-up input panel.
        anchor_path = self.root / ('task%d_anchors.pt' % task)
        if anchor_path.exists():
            anchors = torch.load(str(anchor_path), map_location='cpu')
        else:
            torch.save(anchors, str(anchor_path))
        self.bank = as_torch_dict(anchors)
        with observational(self.models):
            shape = self.algo.policy(self.bank['next_observation'], task)[0]._normal.base_dist.loc.shape
        rng = np.random.RandomState(self.args.seed + 1009*task)
        self.epsilon = torch.as_tensor(rng.normal(size=tuple(shape)), dtype=torch.float32,
                                       device=self.bank['observation'].device)
        np.save(str(self.window / 'epsilon.npy'), self.epsilon.cpu().numpy())
        vector = self.vector()
        self.weights = np.lib.format.open_memmap(str(self.window / 'critic_parameters.npy'),
            mode='w+', dtype=np.float32, shape=(self.length+1, len(vector)))
        self.weights[0] = vector
        self.index = 0
        self.rows = [dict(kind='reference', index=0, values=self.reference(task), **self.clocks(task))]
        save_json(self.window / 'manifest.json', dict(status='running', updates=self.length,
            clocks=self.clocks(task), parameter_layout=[dict(model=n, name=k, shape=list(p.shape))
                for n, m in zip(('qf1', 'qf2'), self.models[1:3]) for k, p in m.named_parameters()]))
        self.snapshot('window_start', task)
        print('MECHANISM_WINDOW_START', str(self.window), flush=True)

    def train_once(self, task, *args, **kwargs):
        a = self.algo
        step = int(a.global_env_step-a._task_env_start_step)
        if (task, step) not in self.snapshots and step in (10000, 50000, 100000, 500000, 1000000):
            self.snapshot('before_collection_updates', task)
            self.snapshots.add((task, step))
        if task > 0 and step in (10000, 100000, 500000) and (task, step) not in self.done:
            if self.window is not None:
                raise RuntimeError('Overlapping mechanism windows')
            self.done.add((task, step))
            self.start_window(task, step)
        result = self.update(task, *args, **kwargs)
        if self.window is not None:
            if self.last_batch is None:
                raise RuntimeError('Window did not capture an actual critic update')
            self.rows.append(dict(kind='online_minibatch', index=self.index,
                                  values=self.last_batch, **self.clocks(task, after=True)))
            self.index += 1
            self.weights[self.index] = self.vector()
            self.rows.append(dict(kind='reference', index=self.index,
                                  values=self.reference(task), **self.clocks(task, after=True)))
            self.last_batch = None
            if self.index % 100 == 0 or self.index == self.length:
                torch.save(self.rows, str(self.window / ('rows_through_%04d.pt' % self.index)))
                self.rows = []
                self.weights.flush()
            if self.index == self.length:
                self.snapshot('window_end', task)
                save_json(self.window / 'complete.json', dict(updates=self.index, **self.clocks(task, True)))
                print('MECHANISM_WINDOW_COMPLETE', str(self.window), flush=True)
                del self.weights
                self.window = None
        return result

    def task_change(self, task):
        self.snapshot('task_exit_before_intervention', task)
        result = self.change(task)
        self.snapshot('task_entry_after_intervention', task+1)
        return result
