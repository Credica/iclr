"""Regression tests for shared evaluation, P&C execution, and held-out banks."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import akro
import numpy as np
import torch

from garage import EnvSpec
from garage.experiment.task_banks import BankedEnv, make_task_bank, build_task_banks
from garage.replay_buffer import PathBuffer
from garage.torch.algos.finetuning import Finetuning_SAC, SpectralRegularizedSAC
from garage.torch.algos.ewc import EWC_SAC
from garage.torch.algos.mtsac import MTSAC
from garage.torch.algos.pandc import P_and_C_SAC
from garage.torch.algos.rnd import RND_SAC
from garage.torch.algos.sac_singular_clip import FinetuningSACSingularClip
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


def make_algo(kind=Finetuning_SAC, count=4, spec=None, **overrides):
    if spec is None:
        spec = [EnvSpec(akro.Box(-np.inf, np.inf, shape=(3,)),
                        akro.Box(-1., 1., shape=(2,)), max_episode_length=10)] * count
    no_stats = overrides.get('no_stats', True)
    kwargs = dict(
        policy=TanhGaussianMLPPolicy(spec, n_tasks=count, hidden_sizes=(8, 8),
                                    adaptor=kind is P_and_C_SAC, no_stats=no_stats),
        qf1=ContinuousMLPQFunction(spec, hidden_sizes=(8, 8), no_stats=no_stats),
        qf2=ContinuousMLPQFunction(spec, hidden_sizes=(8, 8), no_stats=no_stats),
        env_spec=spec, sampler=None, replay_buffer=PathBuffer(30000),
        num_tasks=1, eval_env=[object()] * count, gradient_steps_per_itr=1,
        no_stats=True, use_wandb=False, bellman_probe=False, seed=7,
        q_reset=False, policy_reset=False, buffer_batch_size=8, fixed_alpha=.01,
        multi_input=isinstance(spec, list), task_names=['task'] * count)
    if kind is P_and_C_SAC:
        kwargs.update(compress_step=16, reset_column=True, reset_adaptor=True)
    kwargs.update(overrides)
    algo = kind(**kwargs)
    algo.to()
    return algo


class BaselineProtocolChecks(unittest.TestCase):
    def test_teacher_rollout_export_never_uses_held_out_bank(self):
        bank = make_task_bank('dm_control', 'DMControl-cartpole-balance', 7)
        train, evaluation = BankedEnv(bank), BankedEnv(bank, True)
        algo = make_algo(count=1, spec=[train.spec], eval_env=[evaluation],
                         num_evaluation_episodes=1, max_episode_length_eval=2)
        algo._sampler = SimpleNamespace(_envs=[SimpleNamespace(envs=[train])])
        previous_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with patch.object(evaluation, 'reset', side_effect=AssertionError('held-out leak')):
                    algo.save_rollouts(log_name='teacher_test', buffer_size=16)
                with Path('rollouts/sac_rollouts/rollouts_teacher_test.pkl').open('rb') as stream:
                    rollout = pickle.load(stream)
                self.assertEqual(rollout['observation'].shape, (16, 5))
                self.assertGreater(train._episode_index, 0)
                self.assertEqual(evaluation._episode_index, 0)
                os.chdir(previous_cwd)
        finally:
            os.chdir(previous_cwd)
            train.close()
            evaluation.close()

    def test_six_online_baselines_train_across_real_dmc_boundary_with_banks(self):
        from garage.experiment import CLTaskSampler
        from garage.sampler import LocalSampler
        from garage.trainer import Trainer
        cases = [(Finetuning_SAC, {}), (Finetuning_SAC, {'q_reset': True}),
                 (EWC_SAC, {}), (P_and_C_SAC, {}), (SpectralRegularizedSAC, {}),
                 (Finetuning_SAC, {'ReDo': True})]
        for kind, flags in cases:
            with self.subTest(kind=kind.__name__, flags=flags), \
                    tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                training, evaluation = CLTaskSampler(None, 2000, 7,
                    env_type='dm_control', bank_dir=Path(directory) / 'banks').sample([3, 5])
                algo = make_algo(kind, count=2, spec=[e.spec for e in evaluation],
                    eval_env=evaluation, task_names=['balance', 'swingup'],
                    min_buffer_size=1000, exact_task_budget=True, steps_per_epoch=1,
                    no_stats=False, num_evaluation_episodes=1, max_episode_length_eval=2,
                    feature_stats_interval=1000, hessian_stats_interval=1000,
                    bellman_probe=True, bellman_probe_dir=directory, log_name='smoke',
                    bellman_probe_size=8, bellman_probe_interval=1000, **flags)
                if kind is EWC_SAC:
                    algo.SAMPLE_SIZE, algo.BATCH_SIZE = 16, 8
                algo._evaluation_dir = directory
                algo._sampler = LocalSampler(agents=algo.policy, envs=training,
                    max_episode_length=1000, n_workers=1)
                algo.save_results = Mock()
                trainer = Trainer(None)
                trainer.setup(algo, evaluation[0])
                trainer.enable_logging = False
                trainer.save = Mock()
                try:
                    self.assertTrue(np.isfinite(trainer.train(n_epochs=4, batch_size=1000)))
                    self.assertEqual(algo.global_env_step, 4000)
                    self.assertEqual(algo.seq_idx, 1)
                    self.assertEqual(len(algo.results['Policy hessian rank']), 4)
                    self.assertEqual(len(algo._bellman_probe_buffers), 2)
                    self.assertTrue(np.isfinite(algo.results['Policy zero ratio']).all())
                    rows = [json.loads(line) for line in (Path(directory) /
                        'eval_episodes.jsonl').read_text().splitlines()]
                    self.assertEqual([r['eval_task_position'] for r in rows], [0, 0, 1, 0, 1])
                    self.assertEqual([r['Global env step'] for r in rows],
                                     [1000, 2000, 3000, 4000, 4000])
                finally:
                    for env in evaluation + training[0].envs:
                        env.close()

    def test_all_methods_share_clip_current_and_seen_evaluation(self):
        for kind in (Finetuning_SAC, EWC_SAC, P_and_C_SAC,
                     SpectralRegularizedSAC, RND_SAC, FinetuningSACSingularClip):
            self.assertIs(kind._evaluate_policy, MTSAC._evaluate_policy)
            algo = make_algo(kind)
            algo.seq_idx = 2
            algo._sampler = SimpleNamespace(_envs=[SimpleNamespace(cur_seq_idx=2)])
            module = 'garage.torch.algos.mtsac.'
            with patch(module + 'obtain_evaluation_episodes') as obtain, \
                    patch(module + 'log_performance', return_value=[1.]) as log:
                algo._evaluate_policy(0)
                self.assertEqual([c[0][2] for c in obtain.call_args_list], [2])
                obtain.reset_mock()
                algo._sampler._envs[0].cur_seq_idx = 3
                algo._evaluate_policy(1)
                self.assertEqual([c[0][2] for c in obtain.call_args_list], [0, 1, 2])
                self.assertEqual([c[1]['prefix'] for c in log.call_args_list],
                    ['test/2/task/', 'test/0/task/', 'test/1/task/', 'test/2/task/'])
                if kind is P_and_C_SAC:
                    self.assertIs(obtain.call_args_list[0][0][0], algo.policy_kb)
                    self.assertIs(obtain.call_args_list[-1][0][0], algo.policy)

    def test_pandc_sampling_training_compression_use_same_frozen_policy(self):
        torch.manual_seed(13)
        algo = make_algo(P_and_C_SAC)
        obs = torch.randn(8, 3)
        policy_parameters = {id(p) for p in algo.policy.parameters()}
        kb_parameters = {id(p) for p in algo.policy_kb_prev.parameters()}
        self.assertFalse(policy_parameters & kb_parameters)
        with torch.no_grad():
            dist = algo._get_policy_output(obs, 1)[0]
            _, sampling = algo.policy.get_actions(obs.numpy(), 1)
            np.testing.assert_array_equal(dist.mean.numpy(), sampling['mean'])
            before = algo.policy(obs, 1)[1]['mean'].clone()
        algo.seq_idx = 1
        algo._num_task_seen = 1
        for _ in range(3):
            loss = algo._compress_loss(1, obs)
            algo._policy_kb_optimizer.zero_grad()
            loss.backward()
            algo._policy_kb_optimizer.step()
        after = algo.policy(obs, 1)[1]['mean'].detach()
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(all(p.grad is None for p in algo.policy_kb_prev.parameters()))
        with torch.no_grad():
            for p in algo.policy_kb_prev.parameters():
                p.add_(.1)
        self.assertFalse(torch.equal(before, algo.policy(obs, 1)[1]['mean']))

    def test_deterministic_action_does_not_sample(self):
        algo = make_algo(P_and_C_SAC)
        state = torch.get_rng_state().clone()
        algo.policy.get_deterministic_action(np.ones(3, dtype='float32'), 1)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_dmc_bank_replays_50_resets_and_does_not_touch_training(self):
        bank = make_task_bank('dm_control', 'DMControl-cartpole-balance', 7)
        train, evaluation = BankedEnv(bank), BankedEnv(bank, True)
        try:
            train.reset()
            training_rng = pickle.dumps(train._raw._env.task.random.get_state())
            global_rng = pickle.dumps(np.random.get_state())
            rounds = []
            for _ in range(2):
                evaluation.start_evaluation(50)
                rounds.append([evaluation.reset() for _ in range(50)])
            for first, second in zip(*rounds):
                np.testing.assert_array_equal(first[0], second[0])
                self.assertEqual(first[1], second[1])
            self.assertEqual([r[1]['instance_id'] for r in rounds[0]], list(range(50)))
            self.assertEqual(global_rng, pickle.dumps(np.random.get_state()))
            self.assertEqual(training_rng, pickle.dumps(train._raw._env.task.random.get_state()))
            self.assertNotEqual(bank['reset_seeds'], make_task_bank(
                'dm_control', 'DMControl-cartpole-balance', 8)['reset_seeds'])
        finally:
            train.close()
            evaluation.close()

    def test_real_metaworld_disjoint_banks_replay_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            training, evaluation = build_task_banks(
                'metaworld', ['reach-v2', 'reach-v2'], 7, directory)
            manifest = json.loads((Path(directory) / 'manifest.json').read_text())
            self.assertEqual(len(manifest['banks']), 1)
            bank = pickle.loads((Path(directory) / 'reach-v2.pkl').read_bytes())
            self.assertEqual(len(bank['train']), 50)
            self.assertEqual(len(bank['evaluation']), 50)
            self.assertFalse({t.data for t in bank['train']} &
                             {t.data for t in bank['evaluation']})
            env = evaluation[0]()
            train = training[0]()
            try:
                initial = env.reset()
                env.step(np.zeros(4, dtype='float32'))
                env.start_evaluation(50)
                repeat = env.reset()
                np.testing.assert_array_equal(initial[0], repeat[0])
                self.assertEqual(initial[1], repeat[1])
                observed = {train.reset()[1]['instance_id'] for _ in range(100)}
                self.assertGreater(len(observed), 25)
                second_bank = make_task_bank('metaworld', 'reach-v2', 7)
                self.assertEqual([t.data for t in bank['train']],
                                 [t.data for t in second_bank['train']])
            finally:
                train.close()
                env.close()

    def test_actual_episode_records_include_position_role_seed_and_rng_isolation(self):
        bank = make_task_bank('dm_control', 'DMControl-cartpole-balance', 7)
        envs = [BankedEnv(bank, True) for _ in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            algo = make_algo(P_and_C_SAC, count=3, spec=[env.spec for env in envs],
                eval_env=envs, num_evaluation_episodes=2, max_episode_length_eval=2,
                task_names=[bank['task']] * 3)
            algo.seq_idx = 1
            algo._evaluation_dir = directory
            algo.global_env_step = 30000
            algo._task_env_start_step = 20000
            state = torch.get_rng_state().clone()
            try:
                result = algo._evaluate_policy(2, at_boundary=True)
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                rows = [json.loads(line) for line in
                    (Path(directory) / 'eval_episodes.jsonl').read_text().splitlines()]
                self.assertEqual([r['eval_task_position'] for r in rows], [0, 0, 1, 1])
                self.assertEqual([r['Learner role'] for r in rows],
                    ['pandc_knowledge_base'] * 2 + ['pandc_active_column'] * 2)
                self.assertEqual([r['reset_seed'] for r in rows], bank['reset_seeds'][:2] * 2)
                self.assertEqual([r['bank_role'] for r in rows], ['evaluation'] * 4)
            finally:
                for env in envs:
                    env.close()

    def test_rnd_loads_teacher_artifacts_and_evaluates_only_seen_stages(self):
        names = ['DMControl-cartpole-balance', 'DMControl-cartpole-swingup',
                 'DMControl-cartpole-balance']
        with tempfile.TemporaryDirectory() as directory:
            envs = [BankedEnv(make_task_bank('dm_control', n, 7), True) for n in names]
            algo = make_algo(RND_SAC, count=3, spec=[e.spec for e in envs], eval_env=envs,
                env_seq=names, task_names=names, teacher_root=directory,
                expert_buffer_size=8, replay_buffer_size=16, nepochs_offline=1,
                num_evaluation_episodes=1, max_episode_length_eval=2)
            algo._evaluation_dir = directory
            algo.save_results = Mock()
            for name, env in zip(names, envs):
                path = algo._teacher_model_path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                teacher = TanhGaussianMLPPolicy([env.spec], n_tasks=1,
                    hidden_sizes=(8, 8), no_stats=True)
                torch.save(teacher.state_dict(), str(path))
                path = algo._teacher_rollout_path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('wb') as stream:
                    pickle.dump({'observation': torch.zeros(16,
                        env.spec.observation_space.flat_dim)}, stream)
            trainer = SimpleNamespace(step_itr=0)
            try:
                self.assertTrue(np.isfinite(algo.train(trainer)))
                self.assertEqual(algo.seq_idx, 2)
                self.assertEqual(algo.global_env_step, 0)
                self.assertEqual(algo._distillation_updates, 2)
                rows = [json.loads(line) for line in
                    (Path(directory) / 'eval_episodes.jsonl').read_text().splitlines()]
                self.assertEqual([r['eval_task_position'] for r in rows], [0, 0, 1, 0, 1, 2])
                self.assertEqual([r['Train task position'] for r in rows], [0, 1, 1, 2, 2, 2])
                self.assertTrue(all(r['Learner role'] == 'rnd_offline_student' for r in rows))
            finally:
                for env in envs:
                    env.close()


if __name__ == '__main__':
    unittest.main()
