# Rethinking Negative Transfer in Continual Reinforcement Learning

This repository contains the experiment code and research notes for **Rethinking Negative Transfer in Continual Reinforcement Learning: Bellman Demand and Learning Spectra**.

The project studies how an inherited critic can help or hinder adaptation after a task switch. The current paper scope separates the value carried into a new task from the critic's finite-budget learning geometry, and evaluates a task-boundary spectral conditioning intervention, denoted **Clip (ours)**.

## Project documents

- [`PAPER_OUTLINE_20260907.md`](PAPER_OUTLINE_20260907.md): the authoritative paper scope, claims, figures, tables, and recording contract.
- [`EXPERIMENT_PLAN_20260907.md`](EXPERIMENT_PLAN_20260907.md): the authoritative execution checklist for all planned experiments.
- [`MOTIVATION_METHOD_THEORY_20260907.md`](MOTIVATION_METHOD_THEORY_20260907.md): theory draft connecting Bellman demand, finite-budget adaptation, and intervention cost.
- [`FRONTIER_CRL_BELLMAN_SUBSPACE_NOTE.md`](FRONTIER_CRL_BELLMAN_SUBSPACE_NOTE.md): the earliest research note that motivated the project.

When documents disagree, `PAPER_OUTLINE_20260907.md` defines the paper scope and `EXPERIMENT_PLAN_20260907.md` defines the run scope.

## Environment installation

The current experiments run on Linux x86-64 in a Conda environment with Python 3.7.15 and the PyTorch 1.13.1 CUDA 11.6 build. The repository uses two MuJoCo interfaces:

- `dm-control` uses the official Python package `mujoco==2.3.6`, installed through `requirements.txt`.
- MetaWorld uses the legacy `mujoco-py==2.1.2.14`, which requires a separate MuJoCo 2.1.0 binary installation. The current entry point imports MetaWorld even for DM Control runs, so install both interfaces.

### 1. Install Linux system packages

The following commands target Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential curl patchelf \
    libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
```

### 2. Install the legacy MuJoCo 2.1.0 binaries

Download the official Linux x86-64 release and extract it to the location expected by `mujoco-py`:

```bash
mkdir -p "$HOME/.mujoco"
curl -L \
    https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    -o /tmp/mujoco210-linux-x86_64.tar.gz
tar -xzf /tmp/mujoco210-linux-x86_64.tar.gz -C "$HOME/.mujoco"
test -f "$HOME/.mujoco/mujoco210/bin/libmujoco210.so"
```

MuJoCo 2.1.0 does not require a license key. Export its location in every shell used for this repository:

```bash
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia"
```

To make these variables persistent, add the same two `export` lines to `~/.bashrc`, then open a new terminal or run `source ~/.bashrc`.

### 3. Create the Python environment

From the repository root, run:

```bash
conda create --name reset-distill python=3.7.15 pip=22.3.1 -y
conda activate reset-distill
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
python -m pip install --requirement requirements.txt
```

`requirements.txt` records the packages actually present in the environment used by the running experiments. In particular, MetaWorld is pinned to the exact Git commit installed there. The older `environment.yaml` is retained as a historical full Conda export, but it is not the recommended installation entry point: its `deepmind-lab==1.0` and `metaworld==0.1.0` PyPI entries are no longer independently installable from the current public index.

### 4. Verify the installation

The first `mujoco-py` import compiles a local extension and can take several minutes. Verify the legacy installation with:

```bash
python -c "import mujoco_py; print(mujoco_py.utils.discover_mujoco())"
```

The command should print a path ending in `.mujoco/mujoco210`. Then run a non-rendering DM Control smoke test:

```bash
python -c "from dm_control import suite; env = suite.load('walker', 'walk'); ts = env.reset(); ts = env.step(env.action_spec().generate_value()); print(ts.observation.keys())"
```

Environment creation and non-rendering `walker`/`cartpole` reset-and-step tests have been validated on Linux. Rendering backends such as EGL remain machine-specific. Experiment launch commands will be documented separately after the execution interface is finalized.

MuJoCo references: [MuJoCo 2.1.0 release](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0) and the archived [`mujoco-py` installation guide](https://github.com/openai/mujoco-py#install-mujoco).

## Current experiment code

The first recorded FT batch for the six fixed transfer directions is implemented in:

- `scripts/rethink_ft_recorded.py`
- `scripts/launch_rethink_ft_parallel.py`
- `scripts/test_rethink_ft_recorded.py`

The full main-study matrix is not yet ready to launch. The remaining implementation and validation gates are listed in `EXPERIMENT_PLAN_20260907.md`; a method name or command-line flag must not be treated as evidence that the final protocol is implemented.

## Repository policy

Generated logs, checkpoints, replay buffers, model files, evaluation records, plots, and experiment-result directories are intentionally excluded from version control. This repository is for source code and project-authored Markdown only.

This work is built on the original [Reset-Distill repository](https://github.com/hongjoon0805/Reset-Distill). The upstream remote is retained locally for provenance, but publication should use a separate repository owned by the project authors.
