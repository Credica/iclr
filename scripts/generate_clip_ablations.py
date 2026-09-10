#!/usr/bin/env python3
"""Prepare Clip ablations: clipping sides, entry/periodic schedule, and interval."""
import argparse
import hashlib
import json
from pathlib import Path

try:
    from .generate_clip_matrix import (
        assignments, add_shared_arguments, shared_kwargs, write_manifest)
except ImportError:
    from generate_clip_matrix import (
        assignments, add_shared_arguments, shared_kwargs, write_manifest)


GROUPS = ('sides', 'schedule', 'interval')


def ablation_assignments(repo, artifact_root, *, groups=GROUPS,
                         intervals=(100000, 200000, 400000),
                         include_controls=False, **kwargs):
    groups = tuple(groups)
    if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
        raise ValueError('Choose nonempty, distinct ablation groups: sides, schedule, interval')
    baseline_interval = kwargs.get('singular_clip_interval', 200000)
    variants = []
    if 'sides' in groups:
        variants.extend((name, ['sides'], dict(singular_clip_mode=mode))
                        for name, mode in (('lower_only', 'lower'), ('upper_only', 'upper')))
    if 'schedule' in groups:
        variants.extend((name, ['schedule'], dict(singular_clip_schedule=name))
                        for name in ('entry_only', 'periodic_only'))
    if 'interval' in groups:
        intervals = tuple(intervals)
        if not intervals or len(set(intervals)) != len(intervals):
            raise ValueError('Interval sweep must be nonempty and contain no duplicate values')
        if all(interval == baseline_interval for interval in intervals):
            raise ValueError('Interval sweep needs at least one non-reference interval')
        variants.extend(('interval_{}'.format(interval), ['interval'],
                         dict(singular_clip_interval=interval))
                        for interval in intervals if interval != baseline_interval)

    # Controls share all selected groups and are queued once, not once per
    # table. When omitted, these explicit requirements are still in the JSON.
    controls = []
    if 'sides' in groups or 'schedule' in groups:
        controls.append(('ft', [g for g in groups if g != 'interval'], dict(clip_enabled=False)))
    controls.append(('full_clip', list(groups), {}))

    def jobs_for(variant):
        name, memberships, overrides = variant
        config = dict(kwargs, **overrides)
        signature = hashlib.sha256(json.dumps(
            config, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]
        jobs = assignments(repo, artifact_root,
                           run_prefix='clip_ablation_{}_{}'.format(name, signature), **config)
        for job in jobs:
            job.update(variant=name, ablation_groups=memberships)
        return jobs

    control_jobs = [job for variant in controls for job in jobs_for(variant)]
    jobs = [job for variant in variants for job in jobs_for(variant)]
    if include_controls:
        jobs = control_jobs + jobs
    return jobs, control_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser, default_sequences=['F1', 'D-W6'])
    parser.add_argument('--groups', nargs='+', choices=GROUPS, default=GROUPS)
    parser.add_argument('--clip-intervals', nargs='+', type=int,
                        default=[100000, 200000, 400000],
                        help='Intervals to compare; the reference interval is reused')
    parser.add_argument('--include-controls', action='store_true',
                        help='Also train FT and reference Full Clip once; otherwise reuse matching runs')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    artifact_root = args.artifact_root.resolve()
    if artifact_root == repo or repo in artifact_root.parents:
        parser.error('Keep experiment artifacts outside the Git checkout')
    try:
        jobs, controls = ablation_assignments(
            repo, artifact_root, groups=args.groups, intervals=args.clip_intervals,
            include_controls=args.include_controls, **shared_kwargs(args))
    except ValueError as error:
        parser.error(str(error))
    write_manifest(
        artifact_root, jobs, stem='clip_ablation_jobs',
        scope='Full-sequence method ablations; no rethink/mechanism runs',
        groups=args.groups, sequences=args.sequences, seeds=args.seeds,
        steps_per_task=args.steps_per_task, reference_interval=args.singular_clip_interval,
        requested_intervals=args.clip_intervals if 'interval' in args.groups else [],
        controls_queued=args.include_controls,
        control_reuse='Reuse only runs matching the listed control configuration, '
                      'task banks, code protocol, seed and budget; no automatic cache lookup.',
        required_controls=controls)


if __name__ == '__main__':
    main()
