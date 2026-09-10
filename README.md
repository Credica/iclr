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
switching from `--prepare-only` to `--run`:

`--run` automatically executes two stages: first all local non-R&D baselines,
then R&D (teacher preparation followed by student distillation). Each queue must
finish successfully before the next queue starts; no second launch is needed.

```bash
# Run this only on machine 1 (44 non-R&D -> 48 teachers -> 9 R&D runs).
bash scripts/run_baselines_machine_1.sh \
    /data/reset-distill/baselines --prepare-only
bash scripts/run_baselines_machine_1.sh /data/reset-distill/baselines --run

# Run this only on machine 2 (46 non-R&D -> 15 teachers -> 6 R&D runs).
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
(440 task positions) and machine 2 receives 52 runs (400 task positions).
All Meta-World R&D runs (F1/F2/F3, seeds 1/2/3) stay on machine 1, and all DMC
R&D runs (D-W6/D-C4, seeds 1/2/3) stay on machine 2. This requires 48 Meta-World
and 15 DMC teacher/task/seed artifacts, with no cross-host teacher duplication.
The automatic queue order is:

| Machine | Phase 1: non-R&D runs | Phase 2a: teachers | Phase 2b: R&D runs |
|---|---:|---:|---:|
| 1 | 44 | 48 Meta-World | 9 Meta-World |
| 2 | 46 | 15 DMC | 6 DMC |

Within phase 1, commands are dispatched in FT → Reset → EWC → P&C → Spectral
regularization → ReDo order. These methods may overlap: whenever a GPU slot
becomes free, the next complete sequence run starts. Each phase uses GPU 0–7
with at most two processes per GPU. Only after all local non-R&D runs succeed
does the launcher prepare teachers; only after all local teacher jobs succeed
does it start students. A failed job is recorded while the remaining jobs in
its queue continue, but prevents entry into the next queue. There are no
automatic retries or background teacher preparation during phase 1.
The two machines do not wait for each other: one may enter R&D while the other
is still finishing non-R&D. This allocation does not guarantee equal wall time.

Generated files under `<ARTIFACT_ROOT>/manifests/` (N is 1 or 2):

- `baseline_non_rnd_machine_N.txt`: phase 1 commands.
- `baseline_prerequisites_machine_N.txt`: phase 2a teacher training/export or reuse.
- `baseline_rnd_machine_N.txt`: phase 2b student distillation.
- `baseline_jobs_machine_N.txt`: the full baseline list **for audit only**; do not
  launch it as one queue, which would bypass the R&D dependency barrier.
- `baseline_jobs_machine_N.json`: all run definitions and the ordered execution
  phases with their queue paths/counts (schema 4).

Queue logs are separated into `queue/machineN/non_rnd`, `teachers`, and `rnd`.
Phase 2a reuses existing model+rollout pairs with matching task/budget/seed
filenames (no completion receipt required). Imported caches must also match the
training configuration; see the R&D compatibility notes below.

All online SAC/teacher commands in the E4 baseline queues enable W&B and use
actual environment steps. The following table does not describe the separate
E1/E2 Rethink protocol, which evaluates every **50,000** environment steps:

| Recorded values | Interval |
|---|---:|
| Loss, reward, alpha, training speed, zero ratio | 1,000 environment steps |
| Feature rank and weight change | 10,000 environment steps |
| Hessian rank | 10,000 environment steps, immediately before the matching evaluation |
| Evaluation | Current task every 10,000 environment steps; all seen positions at task exit; 50 episodes per evaluated position |
| Bellman probe | 100,000 environment steps |
| Full Bellman spectral statistics | Fixed points: 10k, 50k, 100k, 500k, 1M, 1.5M |

Generated results stay under the external artifact root, not in the Git
checkout. R&D's student phase is offline and therefore records distillation
update count/time rather than pretending those updates are environment
interactions. After the non-R&D queue succeeds, the R&D stage trains or reuses
every required single-task teacher and rollout, then starts distillation only
if the teacher prerequisite queue finishes without failures.

### Shared baseline / Clip evaluation and task banks

All seven baselines and Clip now use the same `MTSAC._evaluate_policy` implementation.
There is no future-task evaluation. Task-exit evaluation happens before the next
task's reset/Clip intervention. Revisited tasks retain distinct position/head
keys, `test/<zero-based-position>/<task-name>/`; each curve includes environment
clocks. P&C evaluates past tasks with its knowledge base and the current task
with its active column. R&D evaluates all seen student heads after each offline
stage and records distillation updates separately from online environment steps.

Formal `--exact_sac_task_budget True` SAC runs, including teachers and Clip,
automatically enable `fixed-task-banks-v1`:

- Meta-World: 50 training instances and an independently generated, disjoint
  50-instance evaluation bank per task/seed. Training samples the training bank
  using a private RNG. Each evaluation round restarts the same ordered 50
  held-out instances and reset seeds.
- DMC: independent training/evaluation environments and private reset RNGs;
  the same fixed list of 50 evaluation reset seeds is restarted each round.
- Banks depend on task name and seed, not method, machine, stream, or revisit
  position. One-task teachers and sequence learners therefore share the same
  train/evaluation split. Banks, reset seeds and SHA-256 hashes are saved in
  `task_banks/manifest.json` and task-specific pickle files under the run's
  recording directory. `eval_episodes.jsonl` records individual returns,
  lengths, success when available, positions, heads, learner roles and reset
  identities. Training RNG state is preserved across evaluation.

P&C now uses a frozen previous-knowledge-base lateral source consistently in
sampling, optimization, evaluation, Bellman targets and compression targets.
Compression updates the live knowledge base, not this frozen source; the source
is refreshed only after compression. Its knowledge-base, frozen-source, Fisher
and optimizer states are included in Bellman checkpoints.

R&D follows the upstream single-task expert training, fresh expert rollout, and
sequential student-distillation workflow, including memory from previously seen
tasks. It does **not** replace expert rollouts with the teacher's training replay
buffer. After teacher training, the default export collects 1M observations for
distillation; these extra interactions are separate from the 1.5M online-training
budget. See the [original method, Section 4 and Appendix H](https://arxiv.org/html/2403.05066#S4)
and the [upstream pretrained-model/rollout workflow](https://github.com/hongjoon0805/Reset-Distill#singe-task-experiment).
Our task sequences, 1.5M budget, and shared train/evaluation banks are explicit
experimental settings, not claims of an exact reproduction of the original
paper's experiment configuration. Expert rollouts use the **training** bank;
the independent evaluation bank is never used to train the student.

Teacher artifacts use the original-style layout under `<ARTIFACT_ROOT>/teachers/`:

- Model: `models/sac_models/policy_<env-type>_sac_<task>_1500000_<seed>.pt`.
- Rollout: `rollouts/sac_rollouts/rollouts_<env-type>_sac_<task>_1500000_<seed>.pkl`.
- Both files present: reuse; either missing: run the single-task teacher/export
  prerequisite. `models/sac_models/complete_<teacher-stem>.json` is optional
  provenance written after new exports, **not** a condition for reuse.

Before importing old caches, verify the environment/dependency versions,
observation/action processing, network configuration, Adam training settings,
seed, 1.5M budget (including warm-up), task-instance split, and rollout source.
The queue checks filenames and file presence only; it does **not** validate
these configurations, file integrity, or training completion. Missing metadata
does not by itself mean the teacher must be retrained, but matching filenames
alone do not establish compatibility. Old 3M or single-instance teachers are
not valid same-protocol substitutes. If a compatible cache is in the previously
used `teachers/fixed-task-banks-v1/` directory, verify it and copy the model/rollout
pair into the layout above; the launcher does not search that directory or move
or delete old artifacts automatically.

Use a fresh artifact root for the corrected baseline batch; rerunning a main
queue does not resume training or skip completed baseline runs and can overwrite
their logs. Compatible teacher caches may be imported into that new root.
Both machines still use the same launcher commands shown above; regenerate
prepared queues after updating the code. Use the new ordering for a batch that
has not started.
Do not overwrite manifests for an active batch or rerun completed runs blindly.

Earlier validation after the cache-reuse update in `reset-distill` on CPU: 39 distinct
regression checks passed across the targeted test runs. The eight queue/cache
checks include reuse without a receipt, missing-file branches, task/seed/budget
filename mismatches, and both launchers' prepare/dry-run paths. Environment and
algorithm checks include real 4k-step/two-task DMC runs for all
six online baselines (small networks, Hessian and Bellman records), real MW
bank separation/replay, all 50 DMC evaluation resets, training-bank-only teacher
rollout export, Clip clock/entry checks, and R&D student loading
and seen-task evaluation with tiny synthetic teacher artifacts. Both machine
queues passed prepare/dry-run and all 168 baseline/teacher commands parsed,
including output paths containing spaces. This is **not** a full 1.5M-step
teacher/student experiment or a full-size two-process-per-V100 memory test.

The staged launchers passed all nine queue/cache tests, including automatic
non-R&D → teacher → student order and stop-before-next-phase behavior using a
stub queue, local teacher coverage without cross-host duplication, phase-file
partitions, dry-runs, and parsing all 168 commands (105 baseline + 63 teacher
jobs). Main-run/teacher training commands and machine assignments match the
cache-reuse version; only execution staging changed. No formal training was
started by these checks.

## Rethink pilot: four candidates + ABC continuations (2026-09-10)

The old P1–P6 batch has been stopped at the user's request. Existing logs,
checkpoints and offline analyses are preserved; the old 6×6 plan is cancelled.
See [the outline](PAPER_OUTLINE_20260907.md),
[section-by-section writing framework](PAPER_WRITING_FRAMEWORK_20260910.md),
and [execution plan](EXPERIMENT_PLAN_20260907.md).

The first batch contains the following **six seed-1 runs**, left unchanged.
The seed-2/3 extension below adds another 12 runs with a shared slot limit:

| GPU | Run | Sequence and intervention |
|---|---|---|
| 0 | mechanism_n1_s1 | FT: sweep-into-v2 → push-wall-v2 |
| 1 | mechanism_n2_s1 | FT: window-open-v2 → sweep-into-v2 |
| 2 | mechanism_n3_s1 | FT: ball_in_cup-catch → finger-turn_easy |
| 3 | mechanism_n4_s1 | FT: cartpole-swingup → fish-upright |
| 7 | mechanism_abc_qreset_s1 | Old FT A checkpoint → B push-wall → C window-close; Q-reset at both entries |
| 7 | mechanism_abc_clip8_s1 | Same A checkpoint → B → C; Clip [0.25,8] at both entries + every 200k task-local env steps |

Each **new task uses exactly 1,000,000 training environment steps including
10,000 warm-up steps**. In the first batch, each run trains two new tasks: 12M new interactions
across its six runs, excluding evaluation and smoke tests. The old ABC A parent
was trained for 1M optimizer updates; it is not retrospectively relabelled as
1M environment steps. New ABC environment counters start at B=0; the legacy
parent update counter is retained separately. C continues each branch's own B.
Old ABC FT is a historical reference with a different budget/evaluation
protocol, not a perfectly matched restarted FT control.

All use ordinary **SAC + Adam**, never Muon/PPO. DMC uses the existing factory
settings: fixed **alpha=0.01**, 2×1024 networks, learning rate 1e-4, batch 1024,
nominal UTD 0.25. Meta-World uses 2×256, 3e-4, batch 64, nominal UTD 1 and
automatic alpha (initial/task-boundary alpha=1). Warm-up collection has one
ordinary update batch; record actual update counts instead of inferring them
from environment steps. DMC IDs are resolved against installed suite task names.
The 1M MW budget differs from R&D's paper protocol; these are candidates,
not guaranteed negative-transfer examples. Candidate FT seeds 2/3 are now queued
in the extension. Fresh controls and N1–N4 Q-reset/Clip branches remain unqueued.

Q-reset retains the actor and actor Adam, resets both online critics and their
Adam states, and synchronizes target critics at B/C entry. Clip changes all
Linear weight matrices in both online critics, including output weights;
biases, actor and Adam are retained. Entry Clip hard-syncs targets; periodic
Clip uses ordinary Polyak. Boundary-coincident events are deduplicated.
Exact-budget checkpoint branching now covers all remaining tasks, not only B.

Recording is enabled:

- W&B online, project Reset-Distill; authentication uses the existing local
  login/environment, not a key embedded in this launcher.
- Loss/reward/alpha/speed/zero ratio every 1k environment steps.
- Feature rank/weight change and Hessian rank every 10k environment steps.
- Evaluation every **50k environment steps**, 50 fixed-bank episodes;
  current task during training and all seen positions at exit. The current
  main-based pilot does not record a separate step-0 evaluation; do not invent
  a zero-step score when integrating AUC. Hessian retains its finer 10k cadence.
- Bellman probes/checkpoints every 100k; full spectral stats at
  10k/50k/100k/500k/1M and existing boundary/final events.
- Opt-in mechanism snapshots at initialization/loaded parent, task boundaries
  and key points. At each post-A task's 10k/100k/500k, record 1000 actual
  updates: twin-critic parameter paths, actual minibatches/targets, fixed-input
  outputs/targets, and endpoint model/Adam/RNG states. There are 24 planned
  windows across these six runs (12 candidate windows + 12 ABC windows).
  Reference targets use common fixed Gaussian noise; 128 warm-up transitions
  are reused as anchors, not claimed to be episode-independent.
  This new recorder schema needs its own offline reader; do not blindly pass
  it to a legacy P1–P6 analysis script.

Artifacts are outside Git at
`/home/zqy/plasticity-papers/mechanism_pilot_20260910_v1`.
`manifest.json` contains commands, tasks and the parent hash;
`source/` plus `source_sha256.json` freezes the exact code;
`sessions.json` lists persistent tmux workers;
`runs/<name>/status.json` and `console.log` show status/output;
`records/<name>/` holds task banks, evaluations, checkpoints and mechanism data.
Existing processes on other GPUs, including the existing GPU-7 process,
are left untouched. Do not rerun the launcher against an active artifact root.

To prepare a separate batch, activate the documented reset-distill environment
and run `python -B scripts/launch_mechanism_pilot.py --root /absolute/new/artifact-root`.
Add `--launch` in that same invocation to start the six jobs; the directory
must not already exist. `--smoke --launch` uses 10k/task, one evaluation episode,
W&B off and no heavy/window recording, in its own new directory.
The launcher's GPU assignment is explicit and does not schedule on 4/5/6.

Verification: 13 unit/regression tests passed. All six real-environment smoke
runs completed successfully (20k new env steps per run), including ABC B→C
and both heterogeneous DMC pairs; observed DMC alpha was 0.01. This does not
establish full-run learning performance or long-run numerical stability.

### Seed 2/3 extension

**Latest override:** all non-reset replicas have been released immediately,
without waiting for two-worker slots. Current project load is GPU0=4, GPU1=4,
GPU2=3, GPU3=3, GPU7=2: **16 live training runs**, with only ABC Q-reset s2/s3
still dependency-waiting. Use `actual_status.json` for the merged live view,
not the original logical queue's slot counts. No existing training was stopped.

An additional **12 runs** are registered at
`/home/zqy/plasticity-papers/mechanism_seed23_20260910_v1`:

| Additional runs | Count | New training budget |
|---|---:|---:|
| N1–N4 FT, seeds 2 and 3, A→B | 8 | 16M env steps |
| ABC FT, seeds 2 and 3, A→B→C | 2 | 6M env steps |
| ABC Q-reset, seeds 2 and 3, B→C | 2 | 4M env steps |

All new tasks remain 1M actual env steps including warm-up; evaluation, logging,
DMC alpha=0.01 and Adam match the first batch. The extension adds **26M** steps;
the two new batches together cover **18 runs / 38M** steps. No Clip seeds 2/3
were requested or queued.

There were no existing ABC A checkpoints for seeds 2/3. Each new ABC FT trains
its own A once. After its exact 1M-env A-exit checkpoint is fully written,
validated and copied with a SHA-256 receipt into `parents/abc_A_s<seed>.pt`,
the same seed's Q-reset becomes eligible. FT keeps running B→C independently;
Q-reset never uses the seed-1 parent and never retrains A. Actor/critic weights
and Adam are inherited/reset as specified above, but branch RNG is freshly
seeded and replay starts empty, so this is not bitwise FT trajectory replay.
New ABC A uses 1M env steps, unlike historical seed-1 A's 1M optimizer updates;
keep those provenance differences explicit rather than silently pooling them.

The initial scheduler used two project jobs/GPU; the user explicitly removed
this waiting restriction for non-reset runs. The six pending FT replicas were
released via separate persistent workers on GPUs 0–3. Existing jobs were not
stopped or restarted; GPU7 received no additional load.
ABC FT seeds 2/3 have initial priority; once A parents are ready, Q-reset has
priority over remaining candidate replicas. Dependency waiting consumes no GPU
training process. The original scheduler still handles Q-reset dependencies;
when it reaches an already-released FT job, the worker observes that same run
instead of launching a duplicate or overwriting its outputs. Failed jobs are recorded without automatic retries; an invalid or
missing A checkpoint blocks its Q-reset but not unrelated jobs.

At launch the four free slots receive ABC FT s2 on GPU0, ABC FT s3 on GPU1,
N3 FT s2 on GPU2 and N4 FT s2 on GPU3. Those four initial assignments remain unchanged. The released N1 s2/s3 use
GPU0, N2 s2/s3 GPU1, N3 s3 GPU2, and N4 s3 GPU3. `actual_status.json` merges
per-run receipts and is the authoritative live view after this override.
`queue_state.json` is now only the original coordinator's logical view; it may
show a released FT as queued until it adopts that worker. `queue.log` records
coordinator events, and `control/immediate_launch.json` lists released workers.
Each launched run also has `runs/<name>/status.json` and `console.log`.
Waiting runs do not appear in W&B until they actually start.

The extension copies and verifies the first batch's frozen training code;
only scheduling code differs. The extension's runtime worker helper additionally
has an idempotent observer patch; its hash is recorded in
`control/RUNTIME_CHANGE.md`. The training/environment/diagnostic source is unchanged. Its command is
`python -B scripts/extend_mechanism_seeds.py --first-batch /absolute/first-batch --root /absolute/new-extension --launch`.
Use this only once: do not submit another root for the same seed replicas.
The scheduler rejects a second start in the same extension directory.

The additional 12 runs plan 48 dynamic windows; with the first batch's 24,
there are 72 planned windows (36 candidate FT + 36 ABC). Nine targeted
regression tests passed for command coverage/flags, parent validation,
dependency readiness, old-worker slot reservations and dispatch GPUs, alongside
the existing branch/recorder checks. This is not a completed training result.

The friends' submitted MAIN baseline/teacher queues and full-sequence Clip
generator remain at **1.5M/task, 10k evaluation**, with main Clip [0.25,4].
This pilot does not change their frozen commands or relabel their old curves.

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

### Clip ablations: sides, schedule, and interval

The training entry point and `run_clip_main.sh` now accept these parameters:

| CLI parameter | Default | Meaning |
|---|---|---|
| `--singular_clip_mode` | `both` | `both`, `lower` (floor only), or `upper` (ceiling only) |
| `--singular_clip_schedule` | `entry_and_periodic` | `entry_and_periodic`, `entry_only`, or `periodic_only` |
| `--singular_clip_interval` | `200000` | Task-local environment steps, including warm-up |
| `--singular_clip_min` / `--singular_clip_max` | `0.25` / `4` | Finite positive bounds; keep min <= max even for one-sided modes |
| `--singular_clip_start_task` | `1` | Zero-based first eligible position; `0` also clips task A |

One-sided clipping leaves the inactive tail unconstrained, without using an
approximate very small/large bound. Every mode preserves actor, biases and Adam
moments. Ordinary task-boundary target synchronization remains enabled even in
`periodic_only`; periodic projections use the usual Polyak update. In
`entry_only`, the positive interval setting is recorded but does not trigger clips.
For periodic modes, the exact-budget training entry point requires an interval
>= 10k divisible by the MW/DMC collection batch (500/1000).

The three-group generator is `scripts/generate_clip_ablations.py`, with launcher
`scripts/run_clip_ablations.sh`. Its default matrix is:

| Group (`--groups`) | New variants | Reference |
|---|---|---|
| `sides` | Lower-only, Upper-only | FT and Full Clip |
| `schedule` | Entry-only, Periodic-only | FT and Full Clip |
| `interval` | 100k, 400k, both with entry clipping | Full Clip at 200k |

All variants use the shared full-sequence definitions and recording/evaluation
protocol. By default the matrix uses F1 and D-W6, seeds 1/2/3, and 1.5M steps per
task: **36 new sequence runs, 432M training environment steps**. The generator
omits the existing FT and reference Full Clip runs; reuse them only if their
bounds, interval, start position, budget, seed, task banks and code protocol
match. The manifest explicitly lists required controls, without assuming that
matching artifacts exist. Add `--include-controls` to train each control once
per sequence/seed across all selected groups (48 total runs with defaults).
An interval-only queue requires just Full Clip as its control.

```bash
# Prepare all three groups. This does not start training.
bash scripts/run_clip_ablations.sh /data/reset-distill/clip-ablations 0,1 --prepare-only \
  --groups sides schedule interval --sequences F1 D-W6 --seeds 1 2 3

# Launch the same configuration when ready.
bash scripts/run_clip_ablations.sh /data/reset-distill/clip-ablations 0,1 --run \
  --groups sides schedule interval --sequences F1 D-W6 --seeds 1 2 3

# Interval sensitivity only, on F1: six new runs, reusing the 200k reference.
bash scripts/run_clip_ablations.sh /data/reset-distill/clip-interval-f1 0,1 --prepare-only \
  --groups interval --sequences F1 --seeds 1 2 3 \
  --clip-intervals 100000 200000 400000

# Custom reference bounds and interval. Include controls if no matching runs exist.
bash scripts/run_clip_ablations.sh /data/reset-distill/clip-custom 0,1 --prepare-only \
  --groups sides schedule interval --sequences D-C4 --seeds 1 2 3 \
  --singular_clip_min 0.125 --singular_clip_max 8 \
  --singular_clip_interval 400000 --clip-intervals 200000 400000 800000 \
  --include-controls

# A single manually configured variant through the original queue entry point.
bash scripts/run_clip_main.sh /data/reset-distill/clip-upper-periodic 0,1 --prepare-only \
  --sequences F1 --seeds 1 2 3 --singular_clip_mode upper \
  --singular_clip_schedule periodic_only --singular_clip_interval 400000
```

`--sequences` accepts F1/F2/F3/D-W6/D-C4/Easy/Hard (case-insensitive);
`--seeds` accepts distinct non-negative
integers (formal runs remain 1/2/3). Both queue generators also accept
`--steps-per-task` (default 1500000, >= 10k and divisible by 10k). On
`main_garage.py`, use the existing spelling `--steps_per_task` and explicit
`--exact_sac_task_budget True`. The three-group generator always varies a single
factor from the two-sided, entry-and-periodic reference; use the original
single-configuration launcher for arbitrary combinations.

Outputs are `manifests/clip_ablation_jobs.json` and `.txt`. Each job records its
group, variant, full Clip configuration, and expected event counts by task;
ablation run names include a configuration hash. Reference intervals in
`--clip-intervals` are omitted from new runs. Inter-task endpoint projections
are suppressed before outgoing evaluation; the incoming entry projection runs
only if enabled. The final task has no incoming boundary, so an interval exactly
dividing its length also triggers at its endpoint (e.g. 1.5M for a 100k interval).
Expected counts follow this existing trainer behavior rather than assuming
eight events for every setting. Event JSONL and checkpoints carry the selected
mode, schedule, bounds and interval.

Use a separate artifact root for each batch and pass the same options to
`--prepare-only` and `--run`; launchers regenerate their manifests and do not
resume or skip completed runs. No ablation is automatically launched by editing
the code or preparing a queue.

### Additional Clip streams: Easy and Hard

The optional Easy and Hard streams follow R&D Appendix C and the explicitly
specified task order. They are defined in `scripts/clip_sequences.py`:

- Easy: faucet-open → door-close → button-press-topdown-wall → handle-pull →
  window-close → plate-slide-back-side → handle-press → door-lock.
- Hard: faucet-open → push → sweep → button-press-topdown → window-open →
  sweep-into → button-press-wall → push-wall.

Names are resolved to Meta-World `-v2` tasks. All eight positions have separate
actor heads and shared critics, with the same task banks, environment clock,
Clip options and evaluation as the original five streams. Default main/baseline
queues retain their original scope; select the extra streams explicitly:

```bash
bash scripts/run_clip_main.sh /data/reset-distill/clip-easy-hard 0,1 --prepare-only \
  --sequences Easy Hard --seeds 1 2 3
```

This prepares six runs, 72M training environment steps at 1.5M per task, and
56 Clip events per run with the default 200k schedule. Replace `--prepare-only`
with `--run` using the same arguments to train. The ablation launcher also
accepts `--sequences Easy Hard`. The existing DMC streams are available through
`--sequences D-W6 D-C4`.

Source: [R&D Appendix C](https://arxiv.org/html/2403.05066v3#A3).

### Clip on the paper's two-task DMC transfers

`scripts/run_clip_pairs.sh` follows the directed transfers in
[R&D Figure 14(a), Appendix I](https://arxiv.org/html/2403.05066v3#A9),
using the seven-task pool ball_in_cup-catch, cartpole-balance,
cartpole-swingup, finger-turn_easy, fish-upright, point_mass-easy and reacher-easy.
Each run is exactly A → B, with no Clip on A and the chosen Clip intervention on B.

The reference tasks are ball_in_cup-catch, finger-turn_easy and fish-upright.
`--pair-directions from_reference` selects the figure's left panel (18 directed
pairs), `to_reference` selects the right panel (18), and `both` (default) takes
their union: **30 distinct directed pairs**, because six reference-to-reference
directions appear in both panels. These counts describe figure coverage; the
paper's prose says 18 pairs. Directions are preserved and duplicate runs are
removed explicitly. This does not construct a seven-task long sequence.

```bash
# Cover both panels; A and B each receive 1M environment steps, seeds 1/2/3.
bash scripts/run_clip_pairs.sh /data/reset-distill/clip-dmc-pairs 0,1 --prepare-only \
  --suite dmc --pair-directions both --steps-per-task 1000000 --seeds 1 2 3

# A selected source->target direction. Paper-style hyphens are also accepted.
bash scripts/run_clip_pairs.sh /data/reset-distill/clip-dmc-selected 0,1 --prepare-only \
  --suite dmc --sources ball_in_cup-catch --targets finger-turn_easy \
  --steps-per-task 1000000 --seeds 1 2 3 \
  --singular_clip_min 0.25 --singular_clip_max 4 --singular_clip_interval 200000
```

The first command prepares 90 two-task runs and 180M training environment steps.
Omitting `--steps-per-task` retains the project's 1.5M/task default (270M for the
same 90 runs). A is trained independently for each pair; checkpoints with
different input/action/head layouts are not silently shared. Task banks, 10k
evaluation intervals, 50 evaluation episodes and seeds 1/2/3 follow the current
project protocol, so these runs reproduce the paper's task directions rather
than all original settings. Atari is not routed through this continuous-action
SAC entry point.

Outputs are `manifests/clip_pair_jobs.json` and `.txt`, with explicit source,
target, figure-panel membership and task-index/name mapping. Use a fresh root
and the same options with `--run` to launch. Clip bounds, mode, schedule and
interval use the same CLI options as the main and ablation queues.

The prepared manifest is a launch plan, not evidence that training completed.
These fixes do not retroactively relabel old update-clock Clip runs. Full
checkpoint resumption, entry evaluations and the
complete paper data schema remain separate E0 checks; see the execution plan
before treating a batch as final paper evidence.

Earlier Clip validation on 2026-09-08: all 33 repository tests passed in `reset-distill` on
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

The launch matrix is implemented. The remaining full-size validation and paper
E0 gates are listed in `EXPERIMENT_PLAN_20260907.md`; a prepared queue or passing
small-network test is not evidence of V100 memory capacity or full E0 compliance.

The eight planned main-study methods are FT, critic Reset, EWC, Progress &
Compress (P&C), Spectral regularization, ReDo, Reset & Distill (R&D), and
Clip (ours). P&C replaces the previously planned FAME baseline. The repository also
provides a SAC SpectralReg entry point, `--cl_method spectral`, implementing the
ICLR 2025 k=2 objective on the actor and both online critics with default
coefficients of `1e-4` and one power iteration. For the multi-head actor, it
regularizes the shared layers and current task's mean/log-standard-deviation
heads without modifying inactive task heads. The two-machine baseline launch
commands are documented above.
The `--ReDo True` SAC path recycles neurons every 1k task-local environment
steps with a fixed normalized mean-absolute-activation threshold of 0.1. It
reinitializes dormant incoming parameters, zeros their outgoing connections,
clears the affected Adam moments, synchronizes both target critics, and records
each recycling event. The generated baseline commands fix these values rather
than running a threshold/frequency sweep.

## Repository policy

Generated logs, checkpoints, replay buffers, model files, evaluation records, plots, and experiment-result directories are intentionally excluded from version control. This repository is for source code and project-authored Markdown only.

This work is built on the original [Reset-Distill repository](https://github.com/hongjoon0805/Reset-Distill). The upstream remote is retained locally for provenance, but publication should use a separate repository owned by the project authors.
