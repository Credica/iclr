#!/usr/bin/env python3
"""Prepare named full-sequence Clip runs and optional sequence extensions."""
import argparse
import json
import math
from numbers import Integral
from pathlib import Path
import shlex

try:
    from .generate_baseline_matrix import common_recording_args, TASK_BANK_PROTOCOL
    from .clip_sequences import SEQUENCES, DEFAULT_CLIP_SEQUENCES, canonical_sequence_name
except ImportError:
    from generate_baseline_matrix import common_recording_args, TASK_BANK_PROTOCOL
    from clip_sequences import SEQUENCES, DEFAULT_CLIP_SEQUENCES, canonical_sequence_name


STEPS_PER_TASK = 1500000
SEEDS = (1, 2, 3)
CLIP = dict(lower=0.25, upper=4.0, interval_env_steps=200000,
            start_task_position=1, mode='both', trigger_policy='entry_and_periodic',
            entry_clip=True, periodic_clip=True,
            target_update='entry_hard_sync_periodic_polyak')
MODES = ('both', 'lower', 'upper')
SCHEDULES = ('entry_and_periodic', 'entry_only', 'periodic_only')


def assignments(repo, artifact_root, singular_clip_min=CLIP['lower'],
                singular_clip_max=CLIP['upper'], *,
                singular_clip_mode='both', singular_clip_schedule='entry_and_periodic',
                singular_clip_interval=200000, singular_clip_start_task=1,
                sequences=None, seeds=SEEDS, steps_per_task=STEPS_PER_TASK,
                run_prefix='clip', clip_enabled=True):
    """Use the baseline stream definitions as the single code source of truth."""
    if (not math.isfinite(singular_clip_min) or
            not math.isfinite(singular_clip_max) or
            not 0 < singular_clip_min <= singular_clip_max):
        raise ValueError('Clip bounds must be finite and satisfy 0 < min <= max')
    if singular_clip_mode not in MODES or singular_clip_schedule not in SCHEDULES:
        raise ValueError('Invalid Clip mode or schedule')
    if (not isinstance(singular_clip_interval, Integral) or singular_clip_interval < 1 or
            not isinstance(singular_clip_start_task, Integral) or singular_clip_start_task < 0):
        raise ValueError('Clip interval must be positive and start task non-negative')
    if (not isinstance(steps_per_task, Integral) or steps_per_task < 10000 or
            steps_per_task % 10000):
        raise ValueError('Task budget must be >= 10k and divisible by the 10k evaluation interval')
    sequences = (DEFAULT_CLIP_SEQUENCES if sequences is None else
                 tuple(canonical_sequence_name(s) for s in sequences))
    seeds = tuple(seeds)
    if not sequences or len(set(sequences)) != len(sequences) or any(s not in SEQUENCES for s in sequences):
        raise ValueError('Choose nonempty, distinct known sequences')
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, Integral) or s < 0 for s in seeds):
        raise ValueError('Seeds must be distinct non-negative integers')
    entry = clip_enabled and singular_clip_schedule != 'periodic_only'
    periodic = clip_enabled and singular_clip_schedule != 'entry_only'
    config = dict(CLIP, enabled=clip_enabled, lower=singular_clip_min, upper=singular_clip_max,
                  mode=singular_clip_mode, trigger_policy=singular_clip_schedule,
                  interval_env_steps=singular_clip_interval,
                  start_task_position=singular_clip_start_task,
                  entry_clip=entry, periodic_clip=periodic)
    jobs = []
    for sequence in sequences:
        spec = SEQUENCES[sequence]
        collection_batch = 1000 if spec['env_type'] == 'dm_control' else 500
        if periodic and (singular_clip_interval < 10000 or singular_clip_interval % collection_batch):
            raise ValueError('Clip interval must be >= 10k and divisible by the '
                             '{}-step collection batch for {}'.format(collection_batch, sequence))
        if clip_enabled and singular_clip_start_task >= len(spec['tasks']):
            raise ValueError('Clip start task must precede the end of {}'.format(sequence))
        occurrences = {}
        positions = []
        for position, name in enumerate(spec['tasks']):
            occurrences[name] = occurrences.get(name, 0) + 1
            positions.append(dict(task_position=position, task_name=name,
                                  occurrence=occurrences[name], policy_head=position))
        event_counts = []
        for position in range(len(positions)):
            eligible = clip_enabled and position >= singular_clip_start_task
            # The trainer suppresses outgoing boundary clips when a next
            # task exists. The final task retains a coincident endpoint clip.
            end = steps_per_task if position == len(positions) - 1 else steps_per_task - 1
            event_counts.append(int(eligible and entry) +
                                (end // singular_clip_interval if eligible and periodic else 0))
        for seed in seeds:
            run_id = '{}_{}_s{}'.format(run_prefix, sequence.lower(), seed)
            workdir = artifact_root / 'runs' / run_id
            command = [
                'env', 'MUJOCO_GL=osmesa', 'python', '-B', '-u',
                str(repo / 'main_garage.py'),
                '--env_type', spec['env_type'], '--rl_method', 'sac',
                '--cl_method', 'finetuning', '--sac_singular_clip', str(clip_enabled),
                '--singular_clip_min', str(singular_clip_min),
                '--singular_clip_max', str(singular_clip_max),
                '--singular_clip_mode', singular_clip_mode,
                '--singular_clip_schedule', singular_clip_schedule,
                '--singular_clip_interval', str(singular_clip_interval),
                '--singular_clip_start_task', str(singular_clip_start_task),
                '--task_seq_idx', *map(str, spec['task_indices']),
                '--train_task_count', str(len(positions)),
                '--steps_per_task', str(steps_per_task),
                '--exact_sac_task_budget', 'True',
                '--seed', str(seed), '--device_type', '0', '--proc_name', run_id,
                *common_recording_args(),
                '--bellman_probe', 'True', '--bellman_probe_size', '1024',
                '--bellman_probe_interval', '100000', '--bellman_probe_targets', '8',
                '--bellman_spectral_stats', 'True',
                '--bellman_spectral_task_steps', '10000', '50000', '100000',
                '500000', '1000000', '1500000',
                '--bellman_probe_dir', str(artifact_root / 'records' / run_id),
            ]
            shell = 'mkdir -p {path} && cd {path} && {command}'.format(
                path=shlex.quote(str(workdir)),
                command=' '.join(shlex.quote(str(part)) for part in command))
            jobs.append(dict(run_id=run_id, method='clip' if clip_enabled else 'ft', sequence=sequence,
                             seed=seed, env_type=spec['env_type'],
                             task_indices=list(spec['task_indices']),
                             task_positions=positions, task_count=len(positions),
                             source=spec.get('source'),
                             order_provenance=spec.get('order_provenance', 'main-study protocol'),
                             pair_source=spec.get('pair_source'), pair_target=spec.get('pair_target'),
                             paper_panels=spec.get('paper_panels'),
                             suite_tasks=spec.get('suite_tasks'),
                             total_train_env_steps=len(positions) * steps_per_task,
                             clip=dict(config),
                             expected_clip_events_by_task=list(event_counts),
                             expected_clip_events=sum(event_counts),
                             command=shell))
    return jobs


def add_shared_arguments(parser, default_sequences=None):
    """Parameters shared by the single-configuration and ablation queues."""
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--singular_clip_min', type=float, default=CLIP['lower'],
                        help='Lower singular-value bound (default: %(default)s)')
    parser.add_argument('--singular_clip_max', type=float, default=CLIP['upper'],
                        help='Upper singular-value bound (default: %(default)s)')
    parser.add_argument('--singular_clip_interval', type=int, default=200000,
                        help='Task-local environment steps between periodic clips')
    parser.add_argument('--singular_clip_start_task', type=int, default=1,
                        help='Zero-based first task eligible for clipping')
    parser.add_argument('--sequences', nargs='+', type=canonical_sequence_name,
                        choices=tuple(SEQUENCES), default=default_sequences)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--steps-per-task', type=int, default=STEPS_PER_TASK)


def shared_kwargs(args):
    return dict(singular_clip_min=args.singular_clip_min,
                singular_clip_max=args.singular_clip_max,
                singular_clip_interval=args.singular_clip_interval,
                singular_clip_start_task=args.singular_clip_start_task,
                sequences=args.sequences, seeds=args.seeds, steps_per_task=args.steps_per_task)


def write_manifest(artifact_root, jobs, *, stem='clip_jobs', **metadata):
    manifest_dir = artifact_root / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = manifest_dir / (stem + '.txt')
    manifest_file = manifest_dir / (stem + '.json')
    payload = dict(
        schema_version=2, status='prepared_not_run',
        warmup_included=True, task_bank_protocol=TASK_BANK_PROTOCOL,
        evaluation_protocol='current_every_10k_seen_at_exit_position_keyed',
        sac_optimizer='adam', total_runs=len(jobs),
        total_task_positions=sum(job['task_count'] for job in jobs),
        total_train_env_steps=sum(job['total_train_env_steps'] for job in jobs),
        expected_clip_events=sum(job['expected_clip_events'] for job in jobs),
        jobs=jobs, **metadata)
    jobs_file.write_text('# Clip full-sequence queue; Adam only.\n' +
                         ''.join(job['command'] + '\n' for job in jobs))
    manifest_file.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print('Prepared {} runs ({} task positions, {} env steps).'.format(
        len(jobs), payload['total_task_positions'], payload['total_train_env_steps']))
    print('Jobs:', jobs_file)
    print('Manifest:', manifest_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.add_argument('--singular_clip_mode', choices=MODES, default='both')
    parser.add_argument('--singular_clip_schedule', choices=SCHEDULES, default='entry_and_periodic')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    artifact_root = args.artifact_root.resolve()
    if artifact_root == repo or repo in artifact_root.parents:
        parser.error('Keep experiment artifacts outside the Git checkout')
    try:
        jobs = assignments(repo, artifact_root, **shared_kwargs(args),
                           singular_clip_mode=args.singular_clip_mode,
                           singular_clip_schedule=args.singular_clip_schedule)
    except ValueError as error:
        parser.error(str(error))
    write_manifest(artifact_root, jobs,
        scope='Full Clip sequences; optional extensions; excludes E1/E2 paired branches',
        sequences=args.sequences or list(DEFAULT_CLIP_SEQUENCES), seeds=args.seeds,
        steps_per_task=args.steps_per_task, clip=jobs[0]['clip'])


if __name__ == '__main__':
    main()
