#!/usr/bin/env python3
"""Generate one independent, auditable baseline queue for either GPU machine."""
import argparse
import json
from pathlib import Path
import shlex


METHODS = ('ft', 'reset', 'ewc', 'pandc', 'spectral', 'redo', 'rnd')
TASK_BANK_PROTOCOL = 'fixed-task-banks-v1'

SEQUENCES = {
    'F1': {
        'env_type': 'metaworld',
        'task_indices': (6, 33, 48, 35, 29, 34, 8, 49, 25, 13),
        'tasks': ('button-press-v2', 'plate-slide-back-side-v2',
                  'window-close-v2', 'plate-slide-side-v2',
                  'peg-unplug-side-v2', 'plate-slide-back-v2',
                  'coffee-button-v2', 'window-open-v2',
                  'handle-pull-side-v2', 'door-close-v2'),
    },
    'F2': {
        'env_type': 'metaworld',
        'task_indices': (33, 43, 46, 25, 35, 29, 14, 40, 34, 8),
        'tasks': ('plate-slide-back-side-v2', 'soccer-v2', 'sweep-into-v2',
                  'handle-pull-side-v2', 'plate-slide-side-v2',
                  'peg-unplug-side-v2', 'door-lock-v2', 'reach-v2',
                  'plate-slide-back-v2', 'coffee-button-v2'),
    },
    'F3': {
        'env_type': 'metaworld',
        'task_indices': (10, 6, 40, 29, 41, 13, 49, 25, 33, 43),
        'tasks': ('coffee-push-v2', 'button-press-v2', 'reach-v2',
                  'peg-unplug-side-v2', 'reach-wall-v2', 'door-close-v2',
                  'window-open-v2', 'handle-pull-side-v2',
                  'plate-slide-back-side-v2', 'soccer-v2'),
    },
    'D-W6': {
        'env_type': 'dm_control',
        'task_indices': (48, 49, 50, 48, 49, 50),
        'tasks': ('DMControl-walker-stand', 'DMControl-walker-walk',
                  'DMControl-walker-run', 'DMControl-walker-stand',
                  'DMControl-walker-walk', 'DMControl-walker-run'),
    },
    'D-C4': {
        'env_type': 'dm_control',
        'task_indices': (3, 5, 3, 5),
        'tasks': ('DMControl-cartpole-balance', 'DMControl-cartpole-swingup',
                  'DMControl-cartpole-balance', 'DMControl-cartpole-swingup'),
    },
}


METHOD_ARGS = {
    'ft': ('--cl_method', 'finetuning'),
    'reset': ('--cl_method', 'finetuning', '--q_reset', 'True'),
    'ewc': ('--cl_method', 'ewc', '--cl_reg_coef', '1.0'),
    'pandc': ('--cl_method', 'pandc', '--cl_reg_coef', '1.0',
              '--use_pandc_bc', 'False', '--reset_column', 'True',
              '--reset_adaptor', 'True'),
    'spectral': ('--cl_method', 'spectral', '--spectral_actor_coef', '0.0001',
                 '--spectral_critic_coef', '0.0001',
                 '--spectral_power_iterations', '1'),
    'redo': ('--cl_method', 'finetuning', '--ReDo', 'True',
             '--redo_interval', '1000', '--redo_tau', '0.1'),
    'rnd': ('--cl_method', 'rnd', '--cl_reg_coef', '1.0',
            '--rd_teacher_steps', '1500000'),
}


def common_recording_args():
    return [
        '--sac_optimizer', 'adam',
        '--num_evaluation_steps', '10000', '--num_evaluation_episodes', '50',
        '--wandb', 'True', '--no_stats', 'False',
        '--scalar_log_interval', '1000',
        '--feature_stats_interval', '10000',
        '--hessian_stats_interval', '10000',
    ]


def command_text(repo, artifact_root, method, sequence, seed):
    spec = SEQUENCES[sequence]
    run_id = 'baseline_{}_{}_s{}'.format(method, sequence.lower(), seed)
    workdir = artifact_root / 'runs' / run_id
    command = [
        'env', 'MUJOCO_GL=osmesa', 'python', '-B', '-u',
        str(repo / 'main_garage.py'),
        '--env_type', spec['env_type'], '--rl_method', 'sac',
        '--task_seq_idx', *[str(index) for index in spec['task_indices']],
        '--seed', str(seed), '--device_type', '0', '--proc_name', run_id,
        '--steps_per_task', '1500000', '--train_task_count', str(len(spec['tasks'])),
        '--exact_sac_task_budget', 'True',
        *common_recording_args(),
        '--bellman_probe', 'True', '--bellman_probe_size', '1024',
        '--bellman_probe_interval', '100000', '--bellman_probe_targets', '8',
        '--bellman_spectral_stats', 'True',
        '--bellman_spectral_task_steps', '10000', '50000', '100000',
        '500000', '1000000', '1500000',
        '--bellman_probe_dir', str(artifact_root / 'records' / run_id),
        '--rd_teacher_root', str(artifact_root / 'teachers'),
        *METHOD_ARGS[method],
    ]
    launch = ' '.join(shlex.quote(part) for part in command)
    shell = 'mkdir -p {workdir} && cd {workdir} && {launch}'.format(
        workdir=shlex.quote(str(workdir)), launch=launch)
    return run_id, shell


def teacher_stem(env_type, task_name, seed):
    return '{}_sac_{}_1500000_{}'.format(env_type, task_name, seed)


def teacher_prerequisites(repo, artifact_root, selected_jobs):
    teachers = {}
    for job in selected_jobs:
        if job['method'] != 'rnd':
            continue
        spec = SEQUENCES[job['sequence']]
        for task_index, task_name in zip(spec['task_indices'], spec['tasks']):
            key = (spec['env_type'], task_index, task_name, job['seed'])
            if key in teachers:
                continue
            env_type, task_index, task_name, seed = key
            stem = teacher_stem(env_type, task_name, seed)
            teacher_root = artifact_root / 'teachers'
            model_path = (teacher_root / 'models' / 'sac_models' /
                          ('policy_' + stem + '.pt'))
            rollout_path = (teacher_root / 'rollouts' / 'sac_rollouts' /
                            ('rollouts_' + stem + '.pkl'))
            run_id = 'teacher_{}_s{}'.format(
                task_name.lower().replace('_', '-'), seed)
            command = [
                'env', 'MUJOCO_GL=osmesa', 'python', '-B', '-u',
                str(repo / 'main_garage.py'),
                '--env_type', env_type, '--rl_method', 'sac',
                '--task_seq_idx', str(task_index), '--seed', str(seed),
                '--device_type', '0', '--proc_name', run_id,
                '--steps_per_task', '1500000', '--train_task_count', '1',
                '--exact_sac_task_budget', 'True',
                *common_recording_args(),
                '--bellman_probe', 'True', '--bellman_probe_size', '1024',
                '--bellman_probe_interval', '100000',
                '--bellman_probe_targets', '8',
                '--bellman_spectral_stats', 'True',
                '--bellman_spectral_task_steps', '10000', '50000', '100000',
                '500000', '1000000', '1500000',
                '--bellman_probe_dir', str(
                    artifact_root / 'teacher_records' / run_id),
                '--save_single_task_artifacts', 'True',
                '--cl_method', 'finetuning',
            ]
            launch = ' '.join(shlex.quote(part) for part in command)
            # Match the upstream pretrained-model/rollout workflow. Completion
            # receipts are optional provenance, not a new algorithm requirement.
            # Filenames match task/budget/seed, not the full training protocol;
            # users must verify imported caches as documented in README.
            shell = (
                'mkdir -p {root} && cd {root} && '
                'if test -f {model} && test -f {rollout}; '
                'then echo TEACHER_REUSE {stem}; else {launch}; fi'
            ).format(
                root=shlex.quote(str(teacher_root)),
                model=shlex.quote(str(model_path)),
                rollout=shlex.quote(str(rollout_path)),
                stem=shlex.quote(stem), launch=launch)
            teachers[key] = {
                'run_id': run_id,
                'env_type': env_type,
                'task_index': task_index,
                'task': task_name,
                'seed': seed,
                'model_path': str(model_path),
                'rollout_path': str(rollout_path),
                'expected_task_bank_protocol': TASK_BANK_PROTOCOL,
                'cache_match': 'env_task_budget_seed_filenames_and_pair_exists',
                'cache_configuration_check': 'not_automatic_operator_must_verify_imports',
                'command': shell,
            }
    return list(teachers.values())


def assignments(repo, artifact_root):
    jobs = []
    global_job_id = 0
    for method in METHODS:
        for sequence in ('F1', 'F2', 'F3', 'D-W6', 'D-C4'):
            for seed in (1, 2, 3):
                global_job_id += 1
                run_id, command = command_text(
                    repo, artifact_root, method, sequence, seed)
                if method != 'rnd':
                    machine = 1 + (global_job_id - 1) % 2
                    if global_job_id == 1:
                        machine = 2
                elif SEQUENCES[sequence]['env_type'] == 'dm_control':
                    machine = 2
                else:
                    # Keep Meta-World teachers on machine 1 and DMC teachers
                    # on machine 2 to avoid retraining the same prerequisites
                    # on both hosts. One non-R&D run is moved above for 53/52.
                    machine = 1
                jobs.append({
                    'global_job_id': global_job_id,
                    'machine': machine,
                    'run_id': run_id,
                    'method': method,
                    'sequence': sequence,
                    'seed': seed,
                    'env_type': SEQUENCES[sequence]['env_type'],
                    'tasks': list(SEQUENCES[sequence]['tasks']),
                    'task_indices': list(SEQUENCES[sequence]['task_indices']),
                    'wandb': True,
                    'sac_optimizer': 'adam',
                    'bellman_probe': True,
                    'bellman_spectral_stats': True,
                    'requires_teacher_artifacts': method == 'rnd',
                    'command': command,
                })
    assert len(jobs) == 105
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', type=int, choices=(1, 2), required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--jobs-file', type=Path, required=True)
    parser.add_argument('--non-rnd-jobs-file', type=Path,
                        help='Phase 1 queue (default: beside the all-jobs audit file)')
    parser.add_argument('--rnd-jobs-file', type=Path,
                        help='Phase 2 student queue (default: beside the all-jobs audit file)')
    parser.add_argument('--prerequisite-jobs-file', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    artifact_root = args.artifact_root.resolve()
    all_jobs = assignments(repo, artifact_root)
    selected = [job for job in all_jobs
                if job['machine'] == args.machine]
    expected = 53 if args.machine == 1 else 52
    assert len(selected) == expected
    non_rnd_jobs = [job for job in selected if job['method'] != 'rnd']
    rnd_jobs = [job for job in selected if job['method'] == 'rnd']
    prerequisites = teacher_prerequisites(repo, artifact_root, selected)
    non_rnd_file = args.non_rnd_jobs_file or args.jobs_file.with_name(
        'baseline_non_rnd_machine_{}.txt'.format(args.machine))
    rnd_file = args.rnd_jobs_file or args.jobs_file.with_name(
        'baseline_rnd_machine_{}.txt'.format(args.machine))

    outputs = [args.jobs_file, non_rnd_file, rnd_file,
               args.prerequisite_jobs_file, args.manifest]
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error('Queue and manifest output paths must be distinct')
    for path, label, jobs in (
            (args.jobs_file, 'AUDIT ONLY: all baseline commands; use the staged launcher', selected),
            (non_rnd_file, 'PHASE 1: six non-R&D baselines', non_rnd_jobs),
            (args.prerequisite_jobs_file,
             'PHASE 2a: R&D teachers; verify imported cache configuration before reuse', prerequisites),
            (rnd_file, 'PHASE 2b: R&D students; requires all local teachers', rnd_jobs)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# {} (machine {}).\n{}'.format(
            label, args.machine, ''.join(job['command'] + '\n' for job in jobs)))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        'schema_version': 4,
        'status': 'staged_queue_prepared',
        'machine': args.machine,
        'total_global_jobs': 105,
        'machine_jobs': len(selected),
        'phase_job_counts': {'non_rnd': len(non_rnd_jobs),
                             'rnd_teachers': len(prerequisites),
                             'rnd_students': len(rnd_jobs)},
        'methods': list(METHODS),
        'sequences': list(SEQUENCES),
        'sac_optimizer': 'adam',
        'task_bank_protocol': TASK_BANK_PROTOCOL,
        'teacher_cache_policy': 'existing_model_rollout_pair_no_receipt_required',
        'evaluation_protocol': 'current_every_10k_seen_at_exit_position_keyed',
        'excluded_optimizers': ['muon'],
        'recording_cadence_environment_steps': {
            'loss_reward_alpha_speed_zero_ratio': 1000,
            'feature_rank_weight_change': 10000,
            'hessian_rank_and_evaluation': 10000,
            'bellman_probe': 100000,
            'spectral_stats': [10000, 50000, 100000, 500000, 1000000, 1500000],
        },
        'execution_phases': [
            {'id': 'non_rnd', 'jobs_file': str(non_rnd_file),
             'description': 'First finish the six non-R&D baselines on this machine.'},
            {'id': 'rnd_teachers', 'jobs_file': str(args.prerequisite_jobs_file),
             'description': 'Then train/reuse the machine-local R&D teachers and rollouts.'},
            {'id': 'rnd_students', 'jobs_file': str(rnd_file),
             'description': 'After all local teachers succeed, run R&D student distillation.'},
        ],
        'phase_failure_policy': 'stop_before_next_phase_if_current_queue_fails',
        'all_jobs_audit_file': str(args.jobs_file),
        'validation_evidence': [
            'Periodic normalized-activation ReDo, selective Adam-state clearing, and target-Q synchronization are unit-tested.',
            'P&C completes an SAC update on heterogeneous DMC-style observation/action specifications.',
            'R&D maps single-task DMC teacher input slices and destination task heads in unit tests.',
        ],
        'teacher_prerequisites': prerequisites,
        'jobs': selected,
    }, indent=2) + '\n')
    print('GENERATED machine={} non_rnd={} teachers={} rnd={} audit={} manifest={}'.format(
        args.machine, len(non_rnd_jobs), len(prerequisites), len(rnd_jobs),
        args.jobs_file, args.manifest))


if __name__ == '__main__':
    main()
