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

`requirements.txt` records the packages actually present in the environment used by the running experiments. In particular, MetaWorld is pinned to the exact Git commit installed there. Use this documented installation path rather than the historical full Conda export: the latter referenced `deepmind-lab==1.0` and `metaworld==0.1.0` entries that are no longer independently installable from the current public index.

### 4. Verify the installation

The first `mujoco-py` import compiles a local extension and can take several minutes. Verify the legacy installation with:

```bash
python -c "import mujoco_py; print(mujoco_py.utils.discover_mujoco())"
```

The command should print a path ending in `.mujoco/mujoco210`. Then run a non-rendering DM Control smoke test:

```bash
MUJOCO_GL=osmesa python -c "from dm_control import suite; env = suite.load('walker', 'walk'); ts = env.reset(); ts = env.step(env.action_spec().generate_value()); print(ts.observation.keys())"
```

Environment creation and non-rendering `walker`/`cartpole` reset-and-step tests have been validated on Linux with OSMesa. The generated queue commands set `MUJOCO_GL=osmesa`; EGL remains machine-specific and is unnecessary for these non-rendering training runs.

MuJoCo references: [MuJoCo 2.1.0 release](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0) and the archived [`mujoco-py` installation guide](https://github.com/openai/mujoco-py#install-mujoco).

### 5. Configure Weights & Biases

W&B logging is enabled by default. Authenticate once on each experiment
machine; the credential must remain in W&B's local user configuration and must
never be committed to this repository:

```bash
wandb login
```

Use `--wandb false` for an intentionally offline run. Local JSONL/checkpoint
records remain authoritative regardless of W&B availability.

### 6. Queue two workers per GPU

#### Fixed baseline protocol for the two experiment machines

请按本节的固定配置运行，不要把仓库中其他研究入口混入这批 baseline：

| Item | Fixed setting |
|---|---|
| Optimizer | Standard Adam SAC, explicitly `--sac_optimizer adam` |
| Excluded optimizer | **Muon is not used in any teacher or baseline run** |
| Baselines | FT, critic Reset, EWC, P&C, Spectral regularization, ReDo, R&D |
| Task streams | F1, F2, F3, D-W6, D-C4 |
| Seeds | 1, 2, 3 |
| Online budget | 1,500,000 environment interactions per task, replay warm-up included |
| Evaluation | Every 10,000 environment steps, 50 episodes |
| Parallelism | Eight local GPUs per machine, at most two independent runs per GPU |
| Tracking | W&B enabled; stdout/stderr, manifests, probes and checkpoints saved locally |
| Output location | The supplied artifact root outside the Git checkout |

The generated command lines contain `--sac_optimizer adam`; this is intentional
and must not be changed to `muon`. The repository contains experimental Muon
code for unrelated studies, but neither the 105-run baseline matrix nor its R&D
teacher prerequisites use it. This launcher covers the seven baselines only;
Clip (ours) and the separate rethink matrix are not silently added to the queue.

The method-specific settings generated for this batch are:

| Baseline | Fixed command setting |
|---|---|
| FT | `--cl_method finetuning` |
| Reset | FT plus `--q_reset True`; reset both critics and critic Adam state |
| EWC | `--cl_method ewc --cl_reg_coef 1.0` |
| P&C | `--cl_method pandc --cl_reg_coef 1.0 --use_pandc_bc False --reset_column True --reset_adaptor True` |
| SpectralReg | Actor and both online critics, coefficients `1e-4`, one power iteration |
| ReDo | `--ReDo True --redo_interval 1000 --redo_tau 0.1` |
| R&D | `--cl_method rnd --cl_reg_coef 1.0 --rd_teacher_steps 1500000` with an explicit teacher root |

[`scripts/run_two_per_gpu_queue.sh`](scripts/run_two_per_gpu_queue.sh) runs at
most two experiment processes on each local GPU. Its job file contains one
complete shell command per non-empty line. When a process finishes, the next
pending command is launched automatically on the released GPU; failed commands
are recorded and do not stop the rest of the queue.

The two physical machines use separate entry scripts and do not communicate.
Each script generates its own fixed command list and JSON assignment manifest
under the supplied artifact root. Prepare and inspect both manifests before
removing `--prepare-only`:

```bash
# Run this only on machine 1 (53 main-study jobs).
bash scripts/run_baselines_machine_1.sh \
    /data/reset-distill/baselines --prepare-only
bash scripts/run_baselines_machine_1.sh /data/reset-distill/baselines --run

# Run this only on machine 2 (52 main-study jobs).
bash scripts/run_baselines_machine_2.sh \
    /data/reset-distill/baselines --prepare-only
bash scripts/run_baselines_machine_2.sh /data/reset-distill/baselines --run
```

The commands inherit `CUDA_VISIBLE_DEVICES` from the queue. Consequently,
commands using `main_garage.py` must select logical device 0 with
`--device_type 0`, even when the queue assigns them to another physical GPU.
Activate the `reset-distill` environment before starting the queue and keep the
artifact root outside the Git repository. The generated 105-run matrix contains
seven baselines, five sequences, and three seeds. Machine 1 receives 53 runs
(440 task positions) and machine 2 receives 52 runs (400 task positions). The
domain-aware allocation avoids duplicating most R&D teachers across the two
hosts: machine 1 prepares all 48 Meta-World teacher/task/seed artifacts and
machine 2 prepares all 15 DMC artifacts, with no teacher duplicated across
hosts. Each launcher first runs its generated teacher prerequisite queue, reuses
already complete model+rollout pairs, and starts the baseline queue only after
all prerequisites succeed.

All online SAC/teacher commands enable W&B and use actual environment steps:

| Recorded values | Interval |
|---|---:|
| Loss, reward, alpha, training speed, zero ratio | 1,000 environment steps |
| Feature rank and weight change | 10,000 environment steps |
| Hessian rank | 10,000 environment steps, immediately before the matching evaluation |
| Evaluation | 10,000 environment steps, 50 episodes |
| Bellman probe | 100,000 environment steps |
| Full Bellman spectral statistics | Fixed points: 10k, 50k, 100k, 500k, 1M, 1.5M |

Generated results stay under the external artifact root, not in the Git
checkout. R&D's student phase is offline and therefore records distillation
update count/time rather than pretending those updates are environment
interactions. Its launcher first trains or reuses every required single-task
teacher and rollout, then starts distillation only if the prerequisite queue
finishes without failures.

## Clip (ours): E4 full-sequence queue

This is a separate queue from the friends' seven-baseline / 105-run queues and
from the E1/E2 two-task tests. It uses the five streams in the paper outline and
execution plan, with **Adam only**:

| Stream | Task positions per run | Seeds | Runs | Environment steps per run |
|---|---:|---|---:|---:|
| F1 | 10 | 1, 2, 3 | 3 | 15M |
| F2 | 10 | 1, 2, 3 | 3 | 15M |
| F3 | 10 | 1, 2, 3 | 3 | 15M |
| D-W6 | 6 | 1, 2, 3 | 3 | 9M |
| D-C4 | 4 | 1, 2, 3 | 3 | 6M |

Total: **15 runs, 120 task positions, 180M training environment steps**.
The exact order is in [the execution plan](EXPERIMENT_PLAN_20260907.md#61-任务序列).
The Clip generator imports the same stream definitions as the baseline
generator; it does not use the old eight-task `hard/easy` defaults. D-W6 is
stand → walk → run → stand → walk → run; D-C4 is balance → swingup → balance →
swingup. Repeated tasks are separate sequence positions and get separate actor
heads. The manifest records every task name, index, occurrence and head.

### Fixed Clip protocol

- Every task has exactly 1.5M training environment steps, including its 10k
  replay warm-up. MW uses 2×256 networks; DMC uses 2×1024.
- A receives no Clip. B and **every subsequent task**, including revisits,
  receive an immediate entry Clip, before collecting the new warm-up data.
- Periodic Clip uses task-local **environment steps**, at 200k, 400k, 600k,
  800k, 1M, 1.2M and 1.4M. Thus each eligible task has eight events including
  entry; the whole queue expects 840 events. Warm-up and DMC's different UTD
  do not shift these points.
- Both online critics' Linear weight matrices, including the Q output weight,
  are projected to singular values **[0.25, 4]** by default; the bounds are
  configurable using `--singular_clip_min` and `--singular_clip_max`.
  Actor parameters, biases and
  Adam moments are preserved. This is not elementwise or gradient clipping.
- Entry Clip hard-syncs both target critics. Periodic Clip runs after the
  collection's final critic update and uses the ordinary subsequent Polyak
  update, with no extra hard sync. Coincident entry/periodic events are deduplicated.
- Every 10k steps evaluate the current task (50 episodes); at task exit evaluate
  all seen task positions before the next entry intervention, never future heads.
  Revisit metrics are keyed by position. DMC train/eval environments are separate.
- W&B and the recording intervals in the table above are enabled. Clip now
  collects Bellman probe buffers and runs the existing spectral diagnostics.
  `singular_clip_events.jsonl` records actual global/task env clocks, critic
  update counts, trigger, layer changes and target-update policy.

### Prepare or run on your local GPUs

Activate `reset-distill` and configure MuJoCo / W&B as above. For example, using
local GPUs 0 and 1 (replace this list with the GPUs you actually want to use):

```bash
bash scripts/run_clip_main.sh /data/reset-distill/clip-main 0,1 --prepare-only
# Inspect /data/reset-distill/clip-main/manifests/clip_jobs.json first.
bash scripts/run_clip_main.sh /data/reset-distill/clip-main 0,1 --run
```

For the c=16 ablation (`[0.0625, 16]`), supply both bounds when preparing and
running the queue. The generated commands and JSON manifest record these values:

```bash
bash scripts/run_clip_main.sh /data/reset-distill/clip-c16 0,1 --prepare-only \
  --singular_clip_min 0.0625 --singular_clip_max 16
bash scripts/run_clip_main.sh /data/reset-distill/clip-c16 0,1 --run \
  --singular_clip_min 0.0625 --singular_clip_max 16
```

The same two flags work with `python scripts/generate_clip_matrix.py
--artifact-root ...` and with a single `main_garage.py` training command (also
set `--sac_singular_clip True`). Bounds must be finite and satisfy
`0 < min <= max`. Omitting the flags keeps `[0.25, 4]`.

The queue runs at most two complete sequence processes per listed GPU, starts
the next pending run when a slot is freed, and reports failures. `--prepare-only`
does not train. All artifacts stay outside the Git checkout. Use a fresh output
root for a new batch: this launcher does **not** resume partially trained runs
or skip completed main runs on rerun. No teachers are needed for Clip.

The prepared manifest is a launch plan, not evidence that training completed.
These fixes do not retroactively relabel old update-clock Clip runs. Full
checkpoint resumption, instance/reset-seed banks, entry evaluations and the
complete paper data schema remain separate E0 checks; see the execution plan
before treating a batch as final paper evidence.

Validation on 2026-09-08: all 33 repository tests passed in `reset-distill` on
CPU, including the new Clip tests. They cover 500/1000-step collection clocks
with simulated sampling/updates over three exact 1.5M-step tasks, preserved
actor/bias/Adam state, entry hard-sync versus periodic Polyak, the DMC 2×1024
factory configuration, and a real 8k-step cartpole revisit smoke run with small
networks, evaluation, Hessian and Bellman/spectral records. Queue preparation
and dry-run also passed. This is not a full-size V100 or long-run performance test.

```bash
PYTHONPATH=.:scripts python -B -m unittest discover -s scripts -p 'test_*.py'
```

## Current experiment code

The first recorded FT batch for the six fixed transfer directions is implemented in:

- `scripts/rethink_ft_recorded.py`
- `scripts/launch_rethink_ft_parallel.py`
- `scripts/test_rethink_ft_recorded.py`

The full main-study matrix is not yet ready to launch. The remaining implementation and validation gates are listed in `EXPERIMENT_PLAN_20260907.md`; a method name or command-line flag must not be treated as evidence that the final protocol is implemented.

The eight planned main-study methods are FT, critic Reset, EWC, Progress &
Compress (P&C), Spectral regularization, ReDo, Reset & Distill (R&D), and
Clip (ours). P&C replaces the previously planned FAME baseline. The repository also
provides a SAC SpectralReg entry point, `--cl_method spectral`, implementing the
ICLR 2025 k=2 objective on the actor and both online critics with default
coefficients of `1e-4` and one power iteration. For the multi-head actor, it
regularizes the shared layers and current task's mean/log-standard-deviation
heads without modifying inactive task heads. Formal launch commands will be
documented after the unified runner passes the experiment-plan smoke gates.
The `--ReDo True` SAC path recycles neurons every 1k task-local environment
steps with a fixed normalized mean-absolute-activation threshold of 0.1. It
reinitializes dormant incoming parameters, zeros their outgoing connections,
clears the affected Adam moments, synchronizes both target critics, and records
each recycling event. The generated baseline commands fix these values rather
than running a threshold/frequency sweep.

## Repository policy

Generated logs, checkpoints, replay buffers, model files, evaluation records, plots, and experiment-result directories are intentionally excluded from version control. This repository is for source code and project-authored Markdown only.

This work is built on the original [Reset-Distill repository](https://github.com/hongjoon0805/Reset-Distill). The upstream remote is retained locally for provenance, but publication should use a separate repository owned by the project authors.
