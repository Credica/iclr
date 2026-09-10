#!/usr/bin/env python3
"""Prepare Clip on the directed two-task transfers in the R&D paper figures."""
import argparse
from pathlib import Path

try:
    from .clip_sequences import PAPER_PAIRS, paper_pair_names
    from .generate_clip_matrix import (
        MODES, SCHEDULES, add_shared_arguments, assignments, shared_kwargs, write_manifest)
except ImportError:
    from clip_sequences import PAPER_PAIRS, paper_pair_names
    from generate_clip_matrix import (
        MODES, SCHEDULES, add_shared_arguments, assignments, shared_kwargs, write_manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.add_argument('--suite', choices=tuple(PAPER_PAIRS), default='dmc')
    parser.add_argument('--pair-directions', choices=['both', 'from_reference', 'to_reference'], default='both')
    parser.add_argument('--sources', nargs='+', help='Filter source task names from the paper task pool')
    parser.add_argument('--targets', nargs='+', help='Filter target task names from the paper task pool')
    parser.add_argument('--singular_clip_mode', choices=MODES, default='both')
    parser.add_argument('--singular_clip_schedule', choices=SCHEDULES, default='entry_and_periodic')
    args = parser.parse_args()
    if args.sequences is not None:
        parser.error('Use --suite/--pair-directions/--sources/--targets to select paper pairs')
    if args.singular_clip_start_task != 1:
        parser.error('Paper transfer queues require unmodified A: --singular_clip_start_task 1')
    repo = Path(__file__).resolve().parents[1]
    root = args.artifact_root.resolve()
    if root == repo or repo in root.parents:
        parser.error('Keep experiment artifacts outside the Git checkout')
    try:
        args.sequences = paper_pair_names(args.suite, args.pair_directions, args.sources, args.targets)
        jobs = assignments(repo, root, **shared_kwargs(args),
                           singular_clip_mode=args.singular_clip_mode,
                           singular_clip_schedule=args.singular_clip_schedule)
    except ValueError as error:
        parser.error(str(error))
    write_manifest(root, jobs, stem='clip_pair_jobs',
                   scope='R&D paper directed two-task transfer pairs; Clip intervention on B',
                   suite=args.suite, pair_directions=args.pair_directions,
                   sequences=args.sequences, seeds=args.seeds, steps_per_task=args.steps_per_task,
                   source_reuse='Each pair trains A then B independently; no incompatible checkpoint cache reuse',
                   protocol='Paper task directions with configurable current-project budget and seeds; '
                            'not a strict reproduction of all original hyperparameters')


if __name__ == '__main__':
    main()
