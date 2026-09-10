import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from args import parse_args
from scripts.extend_mechanism_seeds import (
    available_job, freeze_ready_parent, make_jobs, occupied_gpus, validate_parent)
from scripts.launch_mechanism_pilot import worker


def original_manifest():
    jobs = []
    for kind in ('n1', 'n2', 'n3', 'n4', 'abc_qreset'):
        abc = kind.startswith('abc')
        name = 'mechanism_%s_s1' % kind
        argv = ['python', '-B', '-u', '/frozen/main_garage.py',
                '--env_type', 'metaworld', '--rl_method', 'sac', '--cl_method', 'finetuning', '--seed', '1',
                '--proc_name', name, '--bellman_probe_dir', '/old/records',
                '--train_task_count', '3' if abc else '2', '--steps_per_task', '1000000',
                '--exact_sac_task_budget', 'True', '--wandb', 'True',
                '--num_evaluation_steps', '50000', '--num_evaluation_episodes', '50']
        if abc:
            argv += ['--branch_checkpoint', '/old/s1.pt', '--branch_task_step', '0',
                     '--branch_alpha', '1', '--q_reset', 'True', '--policy_reset', 'False']
        jobs.append(dict(name=name, argv=argv, new_task_count=2, total_new_env_steps=2000000))
    return dict(jobs=jobs)


def parent_payload():
    payload = {key: {} for key in ('qf1', 'qf2', 'target_qf1', 'target_qf2',
                                   'policy_optimizer', 'qf1_optimizer', 'qf2_optimizer')}
    payload.update(seq_idx=0, global_env_step=1000000, critic_optimizer_steps=990500,
                   log_alpha=torch.tensor(0.), policy={
                       'network._output_layers.%d.weight' % i: torch.tensor(float(i))
                       for i in range(6)})
    return payload


class SeedExtensionChecks(unittest.TestCase):
    def test_exact_coverage_budgets_and_parse(self):
        original = original_manifest()
        before = copy.deepcopy(original)
        jobs = make_jobs(original, Path('/new'))
        self.assertEqual(original, before)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(sum(j['total_new_env_steps'] for j in jobs), 26000000)
        self.assertEqual({j['name'] for j in jobs}, {
            'mechanism_%s_s%d' % (kind, seed)
            for kind in ('n1', 'n2', 'n3', 'n4', 'abc_ft', 'abc_qreset') for seed in (2, 3)})
        for job in jobs:
            with patch('sys.argv', job['argv'][3:]):
                args = parse_args()
            self.assertEqual(args.seed, job['seed'])
            self.assertEqual(args.steps_per_task, 1000000)
            self.assertEqual(args.num_evaluation_steps, 50000)
            self.assertTrue(args.wandb)
            self.assertTrue(args.exact_sac_task_budget)
            if 'abc_ft' in job['name']:
                self.assertFalse(args.q_reset)
                self.assertIsNone(args.branch_checkpoint)
                self.assertEqual(args.train_task_count, 3)
            if 'abc_qreset' in job['name']:
                self.assertTrue(args.q_reset)
                self.assertEqual(job['depends_on'], 'mechanism_abc_ft_s%d' % job['seed'])
                self.assertIn('abc_A_s%d.pt' % job['seed'], args.branch_checkpoint)

    def test_ready_dependency_does_not_block_unrelated_jobs(self):
        jobs = make_jobs(original_manifest(), Path('/new'))
        states = [dict(status='queued') for _ in jobs]
        self.assertEqual(available_job(jobs, states, set()), 0)
        states[0]['status'] = states[1]['status'] = 'running'
        self.assertEqual(available_job(jobs, states, set()), 2)
        self.assertEqual(available_job(jobs, states, {'mechanism_abc_qreset_s2'}), 10)

    def test_old_live_jobs_reserve_slots_dead_and_reused_pids_do_not(self):
        old = [dict(pid=10, gpu=0, process_identity='a'),
               dict(pid=11, gpu=7, process_identity='b'),
               dict(pid=12, gpu=7, process_identity='old')]
        with patch('scripts.extend_mechanism_seeds.process_identity',
                   side_effect=lambda pid: {10: 'a', 11: None, 12: 'reused'}[pid]):
            count = occupied_gpus(old, {0: dict(gpu=0), 1: dict(gpu=1)})
        self.assertEqual(count, {0: 2, 1: 1, 2: 0, 3: 0, 7: 0})

    def test_parent_waits_for_valid_archive_then_records_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = make_jobs(original_manifest(), root)[10]
            self.assertFalse(freeze_ready_parent(root, job))
            directory = root / 'records' / job['depends_on'] / 'checkpoints'
            directory.mkdir(parents=True)
            path = directory / 'task_boundary_task0_env1000000_update990500.pt'
            path.write_bytes(b'incomplete archive')
            self.assertFalse(freeze_ready_parent(root, job))
            torch.save(parent_payload(), str(path))
            self.assertTrue(freeze_ready_parent(root, job))
            destination = Path(job['parent_checkpoint'])
            self.assertEqual(path.read_bytes(), destination.read_bytes())
            receipt = json.loads(destination.with_suffix('.json').read_text())
            self.assertEqual(receipt['seed'], 2)
            self.assertEqual(receipt['source_run'], 'mechanism_abc_ft_s2')
            self.assertEqual(len(receipt['sha256']), 64)

    def test_reject_wrong_budget_task_or_head_layout(self):
        for field, value in (('global_env_step', 1500000), ('seq_idx', 1),
                             ('critic_optimizer_steps', 0), ('policy', {})):
            payload = parent_payload()
            payload[field] = value
            with self.assertRaises(ValueError):
                validate_parent(payload, 1000000)

    def test_worker_uses_dispatch_gpu_and_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = dict(jobs=[dict(name='test', gpu=None, argv=['fake-training'])])
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch('scripts.launch_mechanism_pilot.subprocess.Popen') as popen:
                popen.return_value.pid = 12345
                popen.return_value.wait.return_value = 0
                self.assertEqual(worker(root, 0, gpu=3), 0)
                self.assertEqual(popen.call_args[1]['env']['CUDA_VISIBLE_DEVICES'], '3')
            self.assertEqual(json.loads((root / 'manifest.json').read_text()), manifest)
            receipt = json.loads((root / 'runs/test/status.json').read_text())
            self.assertEqual(receipt['gpu'], 3)
            self.assertEqual(receipt['status'], 'completed')

    def test_duplicate_worker_observes_owner_without_starting_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = dict(jobs=[dict(name='test', gpu=None, argv=['fake-training'])])
            (root / 'manifest.json').write_text(json.dumps(manifest))
            directory = root / 'runs/test'
            directory.mkdir(parents=True)
            state = dict(command=['fake-training'], status='running', training_pid=12345)
            receipt = directory / 'status.json'
            receipt.write_text(json.dumps(state))
            def finish(_):
                receipt.write_text(json.dumps(dict(state, status='completed', exit_code=0)))
            with patch('scripts.launch_mechanism_pilot.subprocess.Popen') as popen, \
                    patch('pathlib.Path.read_bytes', return_value=b'fake-training'+bytes([0])), \
                    patch('scripts.launch_mechanism_pilot.time.sleep', side_effect=finish):
                self.assertEqual(worker(root, 0, gpu=2), 0)
                popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
