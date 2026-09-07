#!/usr/bin/env python3
"""Prepare or launch the 18 FT workers, including one-supervisor-per-GPU mode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    os.replace(str(temporary), str(path))


def job_specs(gpus, run_gpu=None):
    jobs = []
    for pair in range(1, 7):
        for seed in (1, 2, 3):
            gpu = gpus[len(jobs) % len(gpus)]
            jobs.append(dict(run_id='P%d_ft_s%d' % (pair, seed),
                             pair='P%d' % pair, seed=seed, gpu=gpu))
    return [job for job in jobs if run_gpu is None or job['gpu'] == run_gpu]


def prepare_batch(repo, root, args):
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source_snapshot'
    source.mkdir()
    (source / 'scripts').mkdir()
    (root / 'launcher_logs').mkdir()
    (root / 'runs').mkdir()
    shutil.copytree(str(repo / 'garage'), str(source / 'garage'),
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    for name in ('rethink_ft_recorded.py', 'test_rethink_ft_recorded.py', 'launch_rethink_ft_parallel.py'):
        shutil.copy2(str(repo / 'scripts' / name), str(source / 'scripts' / name))
    for path in repo.glob('*.py'):
        shutil.copy2(str(path), str(source / path.name))
    for name in ('PAPER_OUTLINE_20260907.md', 'MOTIVATION_METHOD_THEORY_20260907.md',
                 'RETHINK_FT_RUN_PROTOCOL_20260907.md'):
        if (repo / name).exists():
            shutil.copy2(str(repo / name), str(source / name))
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source.rglob('*')) if path.is_file()}
    write_json(root / 'source_hashes.json', hashes)
    for command, filename in [(['git', 'status', '--short'], 'source_worktree_status.txt'),
                              ([sys.executable, '-m', 'pip', 'freeze'], 'dependencies.txt'),
                              (['nvidia-smi'], 'gpu_at_launch.txt')]:
        result = subprocess.run(command, cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (root / filename).write_bytes(result.stdout)
    assignments = job_specs(args.gpus)
    write_json(root / 'launch_manifest.json', dict(
        status='prepared', concurrency=18, gpu_order=args.gpus,
        assignments=assignments, total_training_interactions=54000000,
        steps_per_task=1500000, warmup_included=10000,
        eval_interval=50000, eval_episodes=50,
        wandb=args.wandb == 'true', wandb_project=args.wandb_project,
        source_snapshot=str(source),
        prepared_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    commands = []
    session_prefix = root.name
    gpu_text = ' '.join(str(gpu) for gpu in args.gpus)
    for gpu in args.gpus:
        commands.append(
            'tmux new-session -d -s {prefix}_gpu{gpu} -c {repo} "'
            'LD_LIBRARY_PATH=/home/zqy/.mujoco/mujoco210/bin:/usr/lib/nvidia '
            '{python} -B -u {launcher} --root {root} --run-gpu {gpu} '
            '--gpus {gpus} --wandb {wandb} --wandb-project {project} '
            '>> {root}/launcher_logs/gpu{gpu}_supervisor.log 2>&1"'.format(
                prefix=session_prefix, gpu=gpu, repo=repo, python=sys.executable,
                launcher=source / 'scripts' / 'launch_rethink_ft_parallel.py',
                root=root, gpus=gpu_text, wandb=args.wandb,
                project=args.wandb_project))
    (root / 'per_gpu_tmux_commands.txt').write_text('\n'.join(commands) + '\n')


def launch_jobs(root, args, selected_jobs, manifest_path):
    source = root / 'source_snapshot'
    if not source.is_dir():
        raise RuntimeError('Batch is not prepared: {}'.format(source))
    jobs = []
    workers = []
    for spec in selected_jobs:
        gpu, run_id = spec['gpu'], spec['run_id']
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(source),
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   LD_LIBRARY_PATH='/home/zqy/.mujoco/mujoco210/bin:/usr/lib/nvidia')
        command = [sys.executable, '-B', '-u', str(source / 'scripts' / 'rethink_ft_recorded.py'),
                   '--pair', spec['pair'], '--seed', str(spec['seed']),
                   '--output', str(root / 'runs' / run_id),
                   '--wandb', args.wandb, '--wandb-project', args.wandb_project]
        log_path = root / 'launcher_logs' / (run_id + '.log')
        with log_path.open('wb') as handle:
            worker = subprocess.Popen(command, cwd=str(source), env=env, stdout=handle,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        workers.append(worker)
        jobs.append(dict(spec, pid=worker.pid, command=command,
                         log=str(log_path), status='launched'))
        print('LAUNCHED', run_id, 'GPU', gpu, 'PID', worker.pid, flush=True)
    manifest = dict(status='running', supervisor_pid=os.getpid(), concurrency=len(jobs), jobs=jobs,
                    gpu_order=args.gpus, run_gpu=args.run_gpu,
                    wandb=args.wandb == 'true', wandb_project=args.wandb_project,
                    source_snapshot=str(source), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write_json(manifest_path, manifest)
    while True:
        living = 0
        for worker, job in zip(workers, jobs):
            code = worker.poll()
            if code is None:
                living += 1
                job['status'] = 'running'
            else:
                job.update(status='completed' if code == 0 else 'failed', exit_code=code)
        manifest['living_workers'] = living
        manifest['last_checked'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        if not living:
            manifest['status'] = 'completed' if all(j['exit_code'] == 0 for j in jobs) else 'finished_with_failures'
        write_json(manifest_path, manifest)
        if not living:
            break
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--wandb', choices=('true', 'false'), default='true')
    parser.add_argument('--wandb-project', default='Reset-Distill')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 6],
                        help='Physical GPU IDs used round-robin (default: 0 1 2 3 6)')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-gpu', type=int,
                        help='Launch only the jobs assigned to this physical GPU')
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error('--gpus must contain unique, non-negative GPU IDs')
    if args.prepare_only and args.run_gpu is not None:
        parser.error('--prepare-only and --run-gpu are mutually exclusive')
    if args.run_gpu is not None and args.run_gpu not in args.gpus:
        parser.error('--run-gpu must be present in --gpus')
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.root).resolve()
    if args.prepare_only:
        prepare_batch(repo, root, args)
        print('PREPARED', root, flush=True)
        return
    if args.run_gpu is not None:
        selected = job_specs(args.gpus, args.run_gpu)
        launch_jobs(root, args, selected,
                    root / ('launch_manifest_gpu%d.json' % args.run_gpu))
        return
    prepare_batch(repo, root, args)
    launch_jobs(root, args, job_specs(args.gpus), root / 'launch_manifest.json')


if __name__ == '__main__':
    main()
