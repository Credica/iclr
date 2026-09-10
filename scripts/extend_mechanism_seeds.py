#!/usr/bin/env python3
"""Append seeds 2/3 without interrupting or oversubscribing the first batch.

Eight candidate FT runs, two new full ABC FT runs, and two Q-reset B/C
branches. Each Q-reset depends on its own seed's ABC FT A-exit checkpoint.
All training commands are derived from the first batch's frozen manifest.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import pickle
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

try:
    from .launch_mechanism_pilot import atomic_json, worker
except ImportError:
    from launch_mechanism_pilot import atomic_json, worker

GPUS = (0, 1, 2, 3, 7)
TERMINAL = ('completed', 'failed', 'blocked')


def replace_option(argv, flag, value):
    argv[argv.index(flag)+1] = str(value)


def remove_option(argv, flag):
    if flag in argv:
        offset = argv.index(flag)
        del argv[offset:offset+2]


def make_jobs(original, root):
    templates = {j['name']: j for j in original['jobs']}
    jobs = []
    # Start both missing ABC sources promptly, alongside two DMC replicas.
    order = [('abc_ft', 2), ('abc_ft', 3), ('n3', 2), ('n4', 2),
             ('n1', 2), ('n2', 2), ('n1', 3), ('n2', 3), ('n3', 3), ('n4', 3),
             ('abc_qreset', 2), ('abc_qreset', 3)]
    for kind, seed in order:
        template = 'abc_qreset' if kind.startswith('abc_') else kind
        job = copy.deepcopy(templates['mechanism_%s_s1' % template])
        name = 'mechanism_%s_s%d' % (kind, seed)
        argv = job['argv']
        argv[argv.index('-u')+1] = str(root / 'source/main_garage.py')
        replace_option(argv, '--seed', seed)
        replace_option(argv, '--proc_name', name)
        replace_option(argv, '--bellman_probe_dir', root / 'records')
        job.update(name=name, seed=seed, gpu=None, depends_on=None,
                   priority=0 if kind == 'abc_ft' else 2)
        if kind == 'abc_ft':
            for flag in ('--branch_checkpoint', '--branch_task_step', '--branch_alpha',
                         '--q_reset', '--policy_reset'):
                remove_option(argv, flag)
            job.update(new_task_count=3, total_new_env_steps=3000000)
        elif kind == 'abc_qreset':
            parent_name = 'mechanism_abc_ft_s%d' % seed
            parent = root / 'parents' / ('abc_A_s%d.pt' % seed)
            replace_option(argv, '--branch_checkpoint', parent)
            job.update(depends_on=parent_name, parent_checkpoint=str(parent), priority=1)
        jobs.append(job)
    return jobs


def process_identity(pid):
    """Exclude dead/zombie/reused PIDs when reserving old jobs' GPU slots."""
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def occupied_gpus(external, active):
    counts = {gpu: 0 for gpu in GPUS}
    for job in external:
        identity = process_identity(job['pid'])
        if identity is not None and identity == job['process_identity']:
            counts[job['gpu']] += 1
    for job in active.values():
        counts[job['gpu']] += 1
    return counts


def available_job(jobs, states, parent_ready):
    ready = [i for i, job in enumerate(jobs) if states[i]['status'] == 'queued'
             and (job['depends_on'] is None or job['name'] in parent_ready)]
    return min(ready, key=lambda i: (jobs[i]['priority'], i)) if ready else None


def validate_parent(payload, steps):
    required = ('policy', 'qf1', 'qf2', 'target_qf1', 'target_qf2',
                'policy_optimizer', 'qf1_optimizer', 'qf2_optimizer', 'log_alpha')
    if not all(k in payload for k in required):
        raise ValueError('ABC A checkpoint is missing model/optimizer fields')
    if payload['seq_idx'] != 0 or payload['global_env_step'] != steps:
        raise ValueError('ABC source is not the exact A-exit state')
    if payload.get('critic_optimizer_steps', 0) <= 0:
        raise ValueError('ABC source has no critic updates')
    heads = {key.split('_output_layers.', 1)[1].split('.', 1)[0]
             for key in payload['policy'] if '_output_layers.' in key}
    if heads != {'0', '1', '2', '3', '4', '5'}:
        raise ValueError('ABC source must retain all three policy heads')


def freeze_ready_parent(root, job):
    """Load only a complete, validated A exit; freeze its bytes and provenance."""
    import torch
    source_dir = root / 'records' / job['depends_on'] / 'checkpoints'
    matches = list(source_dir.glob('task_boundary_task0_env1000000_update*.pt'))
    if len(matches) > 1:
        raise ValueError('Ambiguous ABC A parent')
    if not matches:
        return False
    source = matches[0]
    # torch.save is not atomic in the source runner. Its valid archive footer
    # and an unchanged checksum are required before releasing the dependency.
    try:
        before = source.stat()
        payload = torch.load(str(source), map_location='cpu')
    except (EOFError, RuntimeError, OSError, pickle.UnpicklingError, zipfile.BadZipFile):
        return False
    validate_parent(payload, 1000000)
    del payload
    destination = Path(job['parent_checkpoint'])
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    shutil.copyfile(str(source), str(temporary))
    after = source.stat()
    checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or checksum != hashlib.sha256(source.read_bytes()).hexdigest()):
        return False
    temporary.replace(destination)
    atomic_json(destination.with_suffix('.json'), dict(
        seed=job['seed'], source_run=job['depends_on'], source=str(source),
        checkpoint=str(destination), sha256=checksum, task=0,
        source_task_env_steps=1000000,
        limitation='Common A weights/Adam; branch entry reseeds RNG and clears replay, not bitwise FT continuation'))
    return True


def prepare(repo, old_root, root):
    if root.exists() or root == repo or repo in root.parents:
        raise ValueError('Use a new artifact root outside the checkout')
    original = json.loads((old_root / 'manifest.json').read_text())
    if original['smoke'] or original['steps_per_task'] != 1000000:
        raise ValueError('Expected the full 1M/task first-batch manifest')
    expected = json.loads((old_root / 'source_sha256.json').read_text())
    for name, checksum in expected.items():
        if hashlib.sha256((old_root / 'source' / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('First batch frozen source was modified: '+name)
    external = []
    for job in original['jobs']:
        status_path = old_root / 'runs' / job['name'] / 'status.json'
        state = json.loads(status_path.read_text())
        if state['status'] in ('running', 'starting'):
            pid = state['training_pid']
            identity = process_identity(pid)
            if identity is not None:
                command = Path('/proc/%d/cmdline' % pid).read_bytes().split(b'\0')
                if str(old_root / 'source/main_garage.py').encode() not in command:
                    raise ValueError('Old worker PID no longer matches its frozen command')
                external.append(dict(name=job['name'], gpu=job['gpu'], pid=pid,
                                     process_identity=identity, status_file=str(status_path)))
    root.mkdir(parents=True)
    shutil.copytree(str(old_root / 'source'), str(root / 'source'))
    # Only scheduling code is new. SAC/environment/diagnostic code remains
    # byte-identical to the already-running seed-1 snapshot.
    for name in ('extend_mechanism_seeds.py', 'launch_mechanism_pilot.py'):
        shutil.copy2(str(repo / 'scripts' / name), str(root / 'source/scripts' / name))
    atomic_json(root / 'source_sha256.json', {
        str(p.relative_to(root / 'source')): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / 'source').rglob('*.py'))})
    jobs = make_jobs(original, root)
    atomic_json(root / 'manifest.json', dict(
        schema=1, first_batch=str(old_root), steps_per_task=1000000,
        seeds=[2, 3], jobs=jobs, external_jobs=external,
        gpu_pool=list(GPUS), project_slots_per_gpu=2,
        total_new_env_steps=sum(j['total_new_env_steps'] for j in jobs),
        dmc_fixed_alpha=0.01, sac_optimizer='adam',
        note='No ABC seed2/3 parent exists: train each FT A once, then fork its Q-reset B/C. No new Clip.'))
    return jobs


def schedule(root):
    lock = (root / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'queue_state.json').exists():
        raise ValueError('Queue already started; refusing duplicate execution')
    manifest = json.loads((root / 'manifest.json').read_text())
    jobs = manifest['jobs']
    states = [dict(name=j['name'], status='queued', depends_on=j['depends_on']) for j in jobs]
    active, parents = {}, set()
    name_index = {j['name']: i for i, j in enumerate(jobs)}
    while True:
        for i in list(active):
            process = active[i]['process']
            code = process.poll()
            if code is not None:
                receipt = root / 'runs' / jobs[i]['name'] / 'status.json'
                final = json.loads(receipt.read_text()) if receipt.exists() else {}
                states[i].update(status=final.get('status', 'failed'), exit_code=code)
                if states[i]['status'] not in TERMINAL:
                    states[i]['status'] = 'failed'
                print('JOB_EXIT', jobs[i]['name'], code, flush=True)
                del active[i]
        for i, job in enumerate(jobs):
            if not job['depends_on'] or states[i]['status'] != 'queued' or job['name'] in parents:
                continue
            try:
                if freeze_ready_parent(root, job):
                    parents.add(job['name'])
                    print('PARENT_READY', job['name'], job['parent_checkpoint'], flush=True)
                elif states[name_index[job['depends_on']]]['status'] in TERMINAL:
                    states[i].update(status='blocked', reason='Source exited without a valid A checkpoint')
            except ValueError as error:
                states[i].update(status='blocked', reason=str(error))
        counts = occupied_gpus(manifest['external_jobs'], active)
        for gpu in GPUS:
            while counts[gpu] < 2:
                index = available_job(jobs, states, parents)
                if index is None:
                    break
                command = [sys.executable, '-B', str(root / 'source/scripts/extend_mechanism_seeds.py'),
                           '--root', str(root), '--worker', str(index), '--gpu', str(gpu)]
                process = subprocess.Popen(command)
                active[index] = dict(process=process, gpu=gpu)
                states[index].update(status='running', gpu=gpu, supervisor_pid=process.pid,
                                     launched_at=time.time())
                counts[gpu] += 1
                print('JOB_START', jobs[index]['name'], 'gpu', gpu, flush=True)
        finished = all(s['status'] in TERMINAL for s in states)
        atomic_json(root / 'queue_state.json', dict(
            scheduler_pid=os.getpid(), updated_at=time.time(),
            status='finished' if finished else 'running', gpu_load=counts,
            ready_parents=sorted(parents), jobs=states))
        if finished:
            return int(any(s['status'] != 'completed' for s in states))
        time.sleep(5)



def actual_status(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    logical = json.loads((root / 'queue_state.json').read_text())
    states = {s['name']: s for s in logical['jobs']}
    rows = []
    for job in manifest['jobs']:
        receipt = root / 'runs' / job['name'] / 'status.json'
        state = json.loads(receipt.read_text()) if receipt.exists() else states[job['name']]
        rows.append(dict(state, batch='seed23'))
    for job in manifest['external_jobs']:
        receipt = Path(job['status_file'])
        state = json.loads(receipt.read_text())
        rows.append(dict(state, batch='seed1'))
    gpu_load = {gpu: 0 for gpu in GPUS}
    for state in rows:
        if state['status'] == 'running':
            identity = process_identity(state['training_pid'])
            if identity is None:
                state['status'] = 'exited_pending_receipt'
            else:
                gpu_load[state['gpu']] += 1
    return dict(updated_at=time.time(), scheduling_policy='all_nonreset_released',
                gpu_load=gpu_load, jobs=rows)


def watch_status(root):
    while True:
        state = actual_status(root)
        atomic_json(root / 'actual_status.json', state)
        if all(row['status'] in TERMINAL for row in state['jobs']):
            return
        time.sleep(5)


def release_now(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    control = root / 'control'
    control.mkdir(exist_ok=True)
    if (control / 'immediate_launch.json').exists():
        raise ValueError('Immediate release was already submitted')
    # The helper has an idempotent existing-worker observer. The original
    # scheduler remains alive to manage Q-reset parents and all existing jobs.
    sessions = []
    for index, job in enumerate(manifest['jobs']):
        if job['depends_on'] is not None or (root / 'runs' / job['name']).exists():
            continue
        gpu = int(job['name'].split('_n', 1)[1].split('_', 1)[0])-1
        session = 'mechanism_now_%d_%d' % (int(time.time()), index)
        argv = [sys.executable, '-B', str(root / 'source/scripts/extend_mechanism_seeds.py'),
                '--root', str(root), '--worker', str(index), '--gpu', str(gpu)]
        subprocess.run(['tmux', 'new-session', '-d', '-s', session,
                        ' '.join(shlex.quote(x) for x in argv)], check=True)
        sessions.append(dict(name=job['name'], gpu=gpu, session=session))
    atomic_json(control / 'immediate_launch.json', dict(
        reason='User requested all non-reset seeds start immediately',
        jobs=sessions, original_scheduler='unchanged; manages reset dependencies',
        status_view='actual_status.json; logical queue may not yet adopt released jobs'))
    session = 'mechanism_status_%d' % int(time.time())
    argv = [sys.executable, '-B', str(Path(__file__).resolve()),
            '--root', str(root), '--watch-status']
    subprocess.run(['tmux', 'new-session', '-d', '-s', session,
                    ' '.join(shlex.quote(x) for x in argv)], check=True)
    atomic_json(control / 'status_session.json', dict(session=session, command=argv))
    print(json.dumps(dict(released=len(sessions), jobs=sessions), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--first-batch', type=Path)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--schedule', action='store_true')
    parser.add_argument('--release-now', action='store_true')
    parser.add_argument('--watch-status', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--gpu', type=int, choices=GPUS)
    options = parser.parse_args()
    root = options.root.resolve()
    if options.release_now:
        release_now(root)
        return
    if options.watch_status:
        watch_status(root)
        return
    if options.worker is not None:
        if options.gpu is None:
            parser.error('--worker requires --gpu')
        sys.exit(worker(root, options.worker, gpu=options.gpu))
    if options.schedule:
        os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (root / 'queue.log').open('a', buffering=1) as stream:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            sys.exit(schedule(root))
    if options.first_batch is None:
        parser.error('--first-batch is required to prepare the extension')
    repo = Path(__file__).resolve().parents[1]
    jobs = prepare(repo, options.first_batch.resolve(), root)
    if options.launch:
        session = 'mechanism_seed23_%d' % int(time.time())
        command = [sys.executable, '-B', str(root / 'source/scripts/extend_mechanism_seeds.py'),
                   '--root', str(root), '--schedule']
        subprocess.run(['tmux', 'new-session', '-d', '-s', session,
                        ' '.join(shlex.quote(x) for x in command)], check=True)
        atomic_json(root / 'session.json', dict(tmux_session=session, command=command))
    print(json.dumps(dict(root=str(root), runs=len(jobs), launched=options.launch,
                         total_new_env_steps=sum(j['total_new_env_steps'] for j in jobs)), indent=2))


if __name__ == '__main__':
    main()
