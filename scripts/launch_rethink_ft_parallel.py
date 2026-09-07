#!/usr/bin/env python3
"""Launch all 18 FT workers immediately; never a serial per-GPU job queue."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.root).resolve()
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
    jobs = []
    workers = []
    gpu_order = [0, 1, 2, 3, 4]
    for pair in range(1, 7):
        for seed in (1, 2, 3):
            gpu = gpu_order[len(jobs) % len(gpu_order)]
            run_id = 'P%d_ft_s%d' % (pair, seed)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(source),
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       LD_LIBRARY_PATH='/home/zqy/.mujoco/mujoco210/bin:/usr/lib/nvidia')
            command = [sys.executable, '-B', '-u', str(source / 'scripts' / 'rethink_ft_recorded.py'),
                       '--pair', 'P%d' % pair, '--seed', str(seed), '--output', str(root / 'runs' / run_id)]
            log_path = root / 'launcher_logs' / (run_id + '.log')
            with log_path.open('wb') as handle:
                worker = subprocess.Popen(command, cwd=str(source), env=env, stdout=handle,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            workers.append(worker)
            jobs.append(dict(run_id=run_id, pair='P%d' % pair, seed=seed, gpu=gpu,
                             pid=worker.pid, command=command, log=str(log_path), status='launched'))
            print('LAUNCHED', run_id, 'GPU', gpu, 'PID', worker.pid, flush=True)
    manifest = dict(status='running', supervisor_pid=os.getpid(), concurrency=18, jobs=jobs,
                    total_training_interactions=54000000, steps_per_task=1500000,
                    warmup_included=10000, eval_interval=50000, eval_episodes=50,
                    source_snapshot=str(source), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write_json(root / 'launch_manifest.json', manifest)
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
        write_json(root / 'launch_manifest.json', manifest)
        if not living:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
