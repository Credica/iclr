#!/usr/bin/env python3
"""Prepare Clip (ours): the five E4 streams, not the E1/E2 pair tests."""
import argparse
import json
from pathlib import Path
import shlex

try:
    from .generate_baseline_matrix import SEQUENCES, common_recording_args
except ImportError:
    from generate_baseline_matrix import SEQUENCES, common_recording_args


STEPS_PER_TASK = 1500000
SEEDS = (1, 2, 3)
CLIP = dict(lower=0.25, upper=4.0, interval_env_steps=200000,
            start_task_position=1, entry_clip=True,
            target_update='entry_hard_sync_periodic_polyak')


def assignments(repo, artifact_root):
    """Use the baseline stream definitions as the single code source of truth."""
    jobs = []
    for sequence, spec in SEQUENCES.items():
        occurrences = {}
        positions = []
        for position, name in enumerate(spec['tasks']):
            occurrences[name] = occurrences.get(name, 0) + 1
            positions.append(dict(task_position=position, task_name=name,
                                  occurrence=occurrences[name], policy_head=position))
        for seed in SEEDS:
            run_id = 'clip_{}_s{}'.format(sequence.lower(), seed)
            workdir = artifact_root / 'runs' / run_id
            command = [
                'env', 'MUJOCO_GL=osmesa', 'python', '-B', '-u',
                str(repo / 'main_garage.py'),
                '--env_type', spec['env_type'], '--rl_method', 'sac',
                '--cl_method', 'finetuning', '--sac_singular_clip', 'True',
                '--singular_clip_min', '0.25', '--singular_clip_max', '4',
                '--singular_clip_interval', '200000',
                '--singular_clip_start_task', '1',
                '--task_seq_idx', *map(str, spec['task_indices']),
                '--train_task_count', str(len(positions)),
                '--steps_per_task', str(STEPS_PER_TASK),
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
            jobs.append(dict(run_id=run_id, method='clip', sequence=sequence,
                             seed=seed, env_type=spec['env_type'],
                             task_indices=list(spec['task_indices']),
                             task_positions=positions, task_count=len(positions),
                             total_train_env_steps=len(positions) * STEPS_PER_TASK,
                             expected_clip_events=8 * (len(positions) - 1),
                             command=shell))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    artifact_root = args.artifact_root.resolve()
    if artifact_root == repo or repo in artifact_root.parents:
        parser.error('Keep experiment artifacts outside the Git checkout')
    jobs = assignments(repo, artifact_root)
    manifest_dir = artifact_root / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = manifest_dir / 'clip_jobs.txt'
    manifest_file = manifest_dir / 'clip_jobs.json'
    jobs_file.write_text('# Clip E4 full sequences; Adam only.\n' +
                         ''.join(job['command'] + '\n' for job in jobs))
    manifest_file.write_text(json.dumps(dict(
        schema_version=1, status='prepared_not_run',
        scope='E4 main sequences; excludes E1/E2 paired branches',
        seeds=SEEDS, steps_per_task=STEPS_PER_TASK, warmup_included=True,
        sac_optimizer='adam', clip=CLIP,
        total_runs=len(jobs),
        total_task_positions=sum(job['task_count'] for job in jobs),
        total_train_env_steps=sum(job['total_train_env_steps'] for job in jobs),
        expected_clip_events=sum(job['expected_clip_events'] for job in jobs),
        jobs=jobs), indent=2) + '\n')
    print('Prepared {} Clip runs (120 task positions, 180M env steps).'.format(len(jobs)))
    print('Jobs:', jobs_file)
    print('Manifest:', manifest_file)


if __name__ == '__main__':
    main()
