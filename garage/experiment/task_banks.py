"""Versioned train/evaluation banks shared by formal SAC and its teachers."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from garage import Wrapper
from garage.envs import GymEnv, TaskNameWrapper, normalize
from garage.sampler.env_update import NewEnvUpdate

PROTOCOL = 'fixed-task-banks-v1'
BANK_SIZE = 50


def stable_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


@contextmanager
def isolated_numpy(rng):
    state = np.random.get_state()
    np.random.set_state(rng.get_state())
    try:
        yield
    finally:
        rng.set_state(np.random.get_state())
        np.random.set_state(state)


def make_task_bank(env_type, task_name, seed):
    bank = dict(protocol=PROTOCOL, env_type=env_type, task=task_name, seed=int(seed),
        train_seed=stable_seed(seed, task_name, 'train-instances'),
        evaluation_seed=stable_seed(seed, task_name, 'eval-instances'),
        reset_seeds=[stable_seed(seed, task_name, 'eval-reset', i) for i in range(BANK_SIZE)])
    if env_type == 'metaworld':
        import metaworld
        with isolated_numpy(np.random.RandomState(bank['train_seed'])):
            train = metaworld.MT1(task_name, seed=bank['train_seed'])
            evaluation = metaworld.MT1(task_name, seed=bank['evaluation_seed'])
        bank.update(train=train.train_tasks, evaluation=evaluation.train_tasks,
                    env_class=train.train_classes[task_name])
        assert len(bank['train']) == len(bank['evaluation']) == BANK_SIZE
        train_vectors = {pickle.loads(t.data)['rand_vec'].tobytes() for t in bank['train']}
        eval_vectors = {pickle.loads(t.data)['rand_vec'].tobytes() for t in bank['evaluation']}
        if (len(train_vectors) != BANK_SIZE or len(eval_vectors) != BANK_SIZE or
                train_vectors & eval_vectors):
            raise ValueError('Train/evaluation task banks must be unique and disjoint')
    elif env_type != 'dm_control':
        raise ValueError('Unsupported task bank environment: ' + env_type)
    return bank


class BankedEnv(Wrapper):
    """Private reset RNG; fixed ordered evaluation bank restarted each round."""

    def __init__(self, bank, evaluation=False):
        self.bank = bank
        self.bank_role = 'evaluation' if evaluation else 'train'
        self._evaluation = evaluation
        self._episode_index = 0
        self._selection_rng = np.random.RandomState(stable_seed(
            bank['seed'], bank['task'], self.bank_role, 'reset-selection'))
        self._physics_rng = np.random.RandomState(stable_seed(
            bank['seed'], bank['task'], self.bank_role, 'physics'))
        with isolated_numpy(self._physics_rng):
            if bank['env_type'] == 'metaworld':
                raw = bank['env_class']()
                raw.set_task(bank[self.bank_role][0])
                env = GymEnv(raw, max_episode_length=raw.max_path_length)
            else:
                from garage.envs.dm_control import DMControlEnv
                _, domain, task = bank['task'].split('-', 2)
                raw = DMControlEnv.from_suite(domain, task)
                env = raw
        self._raw = raw
        super().__init__(normalize(TaskNameWrapper(env, task_name=bank['task'])))

    def start_evaluation(self, num_episodes):
        if not self._evaluation or not 0 < num_episodes <= BANK_SIZE:
            raise ValueError('Evaluation must use at most the fixed 50 held-out resets')
        self._episode_index = 0

    def reset(self):
        if self._evaluation:
            # A round uses every instance once; rollout collection may cycle,
            # but formal evaluation always explicitly restarts at index zero.
            instance = self._episode_index % BANK_SIZE
            reset_seed = self.bank['reset_seeds'][instance]
        else:
            instance = int(self._selection_rng.randint(BANK_SIZE))
            reset_seed = int(self._selection_rng.randint(2**32))
        self._episode_index += 1
        self._physics_rng.seed(reset_seed)
        with isolated_numpy(self._physics_rng):
            if self.bank['env_type'] == 'metaworld':
                self._raw.set_task(self.bank[self.bank_role][instance])
            self._raw.seed(reset_seed)
            obs, info = self._env.reset()
        info.update(instance_id=instance, reset_seed=reset_seed, bank_role=self.bank_role)
        return obs, info

    def step(self, action):
        with isolated_numpy(self._physics_rng):
            return self._env.step(action)


def build_task_banks(env_type, task_names, seed, directory):
    """Return legacy-compatible train/eval updates and persist replayable banks.

    Banks are keyed only by task name and seed, not method/stream/position.
    Revisited tasks and cached one-task teachers therefore share the same bank.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    banks, records = {}, {}
    for task_name in dict.fromkeys(task_names):
        bank = make_task_bank(env_type, task_name, seed)
        payload = pickle.dumps(bank, protocol=4)
        path = root / (task_name + '.pkl')
        temporary = path.with_suffix('.pkl.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)
        banks[task_name] = bank
        records[task_name] = dict(sha256=hashlib.sha256(payload).hexdigest(),
            file=path.name, train_seed=bank['train_seed'],
            evaluation_seed=bank['evaluation_seed'], reset_seeds=bank['reset_seeds'],
            train_instances=BANK_SIZE if env_type == 'metaworld' else None,
            evaluation_instances=BANK_SIZE if env_type == 'metaworld' else None)
    manifest = dict(protocol=PROTOCOL, seed=int(seed), env_type=env_type, banks=records,
                    task_positions=list(task_names),
                    training_instance_selection='uniform_private_rng',
                    evaluation_order='fixed_0_to_49_restarted_every_round')
    temporary = root / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(root / 'manifest.json')
    train, evaluation = [], []
    for name in task_names:
        bank = banks[name]
        if env_type == 'metaworld':
            train.append(NewEnvUpdate(lambda bank=bank: BankedEnv(bank)))
            evaluation.append(NewEnvUpdate(lambda bank=bank: BankedEnv(bank, True)))
        else:
            train.append(BankedEnv(bank))
            evaluation.append(BankedEnv(bank, True))
    return train, evaluation
