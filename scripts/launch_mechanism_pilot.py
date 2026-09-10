#!/usr/bin/env python3
"""Freeze and launch four seed-1 FT candidates plus two ABC continuations.

No Git operations. Artifacts must be outside the checkout. GPU placement is
explicit: N1/N2/N3/N4 on 0/1/2/3, ABC Q-reset and Clip-8 on 7. --smoke runs
10k/task with one eval episode and W&B disabled; it is not a paper result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


PARENT = 'bellman_probe_results/sac_transfer_abc_20260907/sac_transfer_abc_s1_1m/checkpoints/task_boundary_task0_step1000000.pt'
PARENT_SHA = '0fc3e440b910411cd50ba3b1089b425fa4c7e36325e323c50286ee06a67053db'
CANDIDATES = [
    ('n1', 'metaworld', ['sweep-into-v2', 'push-wall-v2'], 0),
    ('n2', 'metaworld', ['window-open-v2', 'sweep-into-v2'], 1),
    ('n3', 'dm_control', [('ball_in_cup', 'catch'), ('finger', 'turn_easy')], 2),
    ('n4', 'dm_control', [('cartpole', 'swingup'), ('fish', 'upright')], 3),
]
MW_IDS = {'sweep-into-v2': 46, 'push-wall-v2': 39,
          'window-open-v2': 49, 'window-close-v2': 48}


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def worker(root, index, gpu=None):
    manifest = json.loads((root / 'manifest.json').read_text())
    job = manifest['jobs'][index]
    if gpu is not None:
        # Queue extensions select a free GPU at dispatch; do not mutate commands.
        job['gpu'] = int(gpu)
    workdir = root / 'runs' / job['name']
    try:
        workdir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # A run may have been released ahead of the logical queue. Observe
        # the same owner, never launch a duplicate or overwrite its log.
        receipt_path = workdir / 'status.json'
        for _ in range(100):
            if receipt_path.exists():
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Existing run directory has no launch receipt')
        print('OBSERVING_EXISTING_RUN', job['name'], flush=True)
        while True:
            existing = json.loads(receipt_path.read_text())
            if existing.get('command') != job['argv']:
                raise ValueError('Refusing to reuse a directory for a different command')
            if existing['status'] in ('completed', 'failed'):
                return int(existing['exit_code'])
            pid = existing['training_pid']
            try:
                actual = Path('/proc/%d/cmdline' % pid).read_bytes().split(bytes([0]))
            except FileNotFoundError:
                time.sleep(1)
                final = json.loads(receipt_path.read_text())
                if final['status'] in ('completed', 'failed'):
                    return int(final['exit_code'])
                raise RuntimeError('Existing training exited without a terminal receipt')
            if [x.decode() for x in actual if x] != job['argv']:
                raise RuntimeError('Existing training PID no longer matches its command')
            time.sleep(5)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(job['gpu']), MUJOCO_GL='osmesa',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        LD_LIBRARY_PATH='/home/zqy/.mujoco/mujoco210/bin:/usr/lib/nvidia:' + environment.get('LD_LIBRARY_PATH', ''),
        PYTHONPATH=str(root / 'source'), WANDB_PROJECT='Reset-Distill')
    receipt = dict(name=job['name'], gpu=job['gpu'], status='starting',
                   supervisor_pid=os.getpid(), started_at=time.time(), command=job['argv'])
    with (workdir / 'console.log').open('w') as log:
        process = subprocess.Popen(job['argv'], cwd=str(workdir), env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        receipt.update(status='running', training_pid=process.pid)
        atomic_json(workdir / 'status.json', receipt)
        code = process.wait()
    receipt.update(status='completed' if code == 0 else 'failed',
                   exit_code=code, finished_at=time.time())
    atomic_json(workdir / 'status.json', receipt)
    return code


def prepare(repo, root, smoke=False):
    from dm_control import suite
    if root == repo or repo in root.parents:
        raise ValueError('Keep all artifacts outside the Git checkout')
    if root.exists():
        raise ValueError('Artifact directory already exists; refusing to overwrite/relaunch')
    checkpoint = repo / PARENT
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('ABC parent checksum mismatch')
    root.mkdir(parents=True)
    source = root / 'source'
    source.mkdir()
    for filename in ('main_garage.py', 'args.py', 'utils.py'):
        shutil.copy2(str(repo / filename), str(source / filename))
    for dirname in ('garage', 'scripts'):
        shutil.copytree(str(repo / dirname), str(source / dirname),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.log', 'logs', 'wandb'))
    source_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source.rglob('*.py'))}
    atomic_json(root / 'source_sha256.json', source_hashes)
    steps = 10000 if smoke else 1000000
    jobs = []
    specs = list(CANDIDATES) + [
        ('abc_qreset', 'metaworld', ['sweep-into-v2', 'push-wall-v2', 'window-close-v2'], 7),
        ('abc_clip8', 'metaworld', ['sweep-into-v2', 'push-wall-v2', 'window-close-v2'], 7)]
    for name, domain, tasks, gpu in specs:
        indices = ([MW_IDS[t] for t in tasks] if domain == 'metaworld' else
                   [list(suite.ALL_TASKS).index(tuple(t)) for t in tasks])
        name = ('smoke_' if smoke else 'mechanism_') + name + '_s1'
        argv = [sys.executable, '-B', '-u', str(source / 'main_garage.py'),
                '--env_type', domain, '--rl_method', 'sac', '--cl_method', 'finetuning',
                '--sac_optimizer', 'adam', '--seed', '1', '--device_type', '0',
                '--proc_name', name, '--task_seq_idx', *map(str, indices),
                '--train_task_count', str(len(tasks)), '--steps_per_task', str(steps),
                '--exact_sac_task_budget', 'True',
                '--num_evaluation_steps', str(10000 if smoke else 50000),
                '--num_evaluation_episodes', str(1 if smoke else 50),
                '--wandb', str(not smoke), '--no_stats', str(smoke),
                '--scalar_log_interval', '1000', '--feature_stats_interval', '10000',
                '--hessian_stats_interval', '10000',
                '--bellman_probe', 'True', '--bellman_probe_size', '1024',
                '--bellman_probe_interval', '100000', '--bellman_probe_targets', '8',
                '--bellman_spectral_stats', str(not smoke),
                '--bellman_spectral_fit_lr', '0.0001' if domain == 'dm_control' else '0.0003',
                '--bellman_spectral_task_steps', '10000', '50000', '100000', '500000', '1000000',
                '--bellman_probe_dir', str(root / 'records'),
                '--mechanism_record', str(not smoke)]
        if 'abc_' in name:
            argv += ['--branch_checkpoint', str(checkpoint), '--branch_task_step', '0', '--branch_alpha', '1.0']
        if 'qreset' in name:
            argv += ['--q_reset', 'True', '--policy_reset', 'False']
        if 'clip8' in name:
            argv += ['--sac_singular_clip', 'True', '--singular_clip_min', '0.25',
                     '--singular_clip_max', '8', '--singular_clip_interval', '200000',
                     '--singular_clip_start_task', '1']
        jobs.append(dict(name=name, gpu=gpu, env_type=domain, tasks=tasks,
                         task_indices=indices, seed=1, new_task_count=2,
                         total_new_env_steps=steps*2, argv=argv))
    atomic_json(root / 'manifest.json', dict(status='prepared', smoke=smoke,
        steps_per_task=steps, clock='actual training environment steps, warm-up included',
        dmc_fixed_alpha=0.01, mw_alpha='automatic, initial/reset=1',
        abc_parent=str(checkpoint), abc_parent_sha256=PARENT_SHA,
        abc_parent_clock='legacy 1M optimizer updates; NOT 1M new environment steps',
        jobs=jobs))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', type=int)
    options = parser.parse_args()
    root = options.root.resolve()
    if options.worker is not None:
        sys.exit(worker(root, options.worker))
    repo = Path(__file__).resolve().parents[1]
    jobs = prepare(repo, root, options.smoke)
    if options.launch:
        sessions = []
        for index, job in enumerate(jobs):
            session = 'mech_{}_{}'.format(int(time.time()), index)
            argv = [sys.executable, '-B', str(root / 'source/scripts/launch_mechanism_pilot.py'),
                    '--root', str(root), '--worker', str(index)]
            subprocess.run(['tmux', 'new-session', '-d', '-s', session,
                            ' '.join(shlex.quote(x) for x in argv)], check=True)
            sessions.append(dict(session=session, job=job['name'], gpu=job['gpu']))
        atomic_json(root / 'sessions.json', sessions)
    print(json.dumps(dict(root=str(root), launched=options.launch,
                         jobs=[dict(name=j['name'], gpu=j['gpu']) for j in jobs]), indent=2))


if __name__ == '__main__':
    main()
