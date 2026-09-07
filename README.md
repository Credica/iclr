# Rethinking Negative Transfer in Continual Reinforcement Learning

This repository contains the experiment code and research notes for **Rethinking Negative Transfer in Continual Reinforcement Learning: Bellman Demand and Learning Spectra**.

The project studies how an inherited critic can help or hinder adaptation after a task switch. The current paper scope separates the value carried into a new task from the critic's finite-budget learning geometry, and evaluates a task-boundary spectral conditioning intervention, denoted **Clip (ours)**.

## Project documents

- [`PAPER_OUTLINE_20260907.md`](PAPER_OUTLINE_20260907.md): the authoritative paper scope, claims, figures, tables, and recording contract.
- [`EXPERIMENT_PLAN_20260907.md`](EXPERIMENT_PLAN_20260907.md): the authoritative execution checklist for all planned experiments.
- [`MOTIVATION_METHOD_THEORY_20260907.md`](MOTIVATION_METHOD_THEORY_20260907.md): theory draft connecting Bellman demand, finite-budget adaptation, and intervention cost.
- [`FRONTIER_CRL_BELLMAN_SUBSPACE_NOTE.md`](FRONTIER_CRL_BELLMAN_SUBSPACE_NOTE.md): the earliest research note that motivated the project.

When documents disagree, `PAPER_OUTLINE_20260907.md` defines the paper scope and `EXPERIMENT_PLAN_20260907.md` defines the run scope.

## Current experiment code

The first recorded FT batch for the six fixed transfer directions is implemented in:

- `scripts/rethink_ft_recorded.py`
- `scripts/launch_rethink_ft_parallel.py`
- `scripts/test_rethink_ft_recorded.py`

The full main-study matrix is not yet ready to launch. The remaining implementation and validation gates are listed in `EXPERIMENT_PLAN_20260907.md`; a method name or command-line flag must not be treated as evidence that the final protocol is implemented.

## Repository policy

Generated logs, checkpoints, replay buffers, model files, evaluation records, plots, and experiment-result directories are intentionally excluded from version control. This repository is for source code and project-authored Markdown only.

This work is built on the original [Reset-Distill repository](https://github.com/hongjoon0805/Reset-Distill). The upstream remote is retained locally for provenance, but publication should use a separate repository owned by the project authors.
