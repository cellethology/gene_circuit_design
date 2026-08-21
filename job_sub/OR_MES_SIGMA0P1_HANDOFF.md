# OR MES Noise Sweep Handoff

This branch contains a reproducible Westlake submission for the retrospective
OR task with simulated clonal variation.

## Fixed Experiment

- Dataset file: `job_sub/datasets/1m_base_datasets.yaml`
- 33 structural subsets from the 1M library
- Embedding: `1m_alphagenome_1bp_embeddings_kneedle`
- Query strategy: MES
- Predictor: BoTorch GP
- Initial selection: ProbCover Euclidean
- Score: `or_score`
- Noise: independent Gaussian noise with sigma 0.10 in log10 expression space
- Expression states: basal, 4-OHT, GZV, and dual induction
- Allocations: 24 constructs x 1 replicate, 12 x 2, and 8 x 3
- Assay budget: 24 measurements per round
- Evaluated rounds: 30 (`al_settings.max_rounds=29` after the initial round)
- Seeds: 0 through 29
- Full size: 33 datasets x 3 allocations x 30 seeds = 2,970 tasks

The GP is not given sigma 0.10. For the replicated allocations, its observation
variance is estimated from the simulated replicate measurements accumulated
through the current round. The 24x1 allocation uses the observed-only path.

## 1. Obtain The Branch

For a fresh clone, this checks out the branch directly and does not require
`git switch`:

```bash
git clone \
  --branch codex/or-sigma0p1-mes-30rounds \
  --single-branch \
  https://github.com/cellethology/deepdraw.git \
  deepdraw-or-mes
cd deepdraw-or-mes
```

For an existing clone:

```bash
cd ~/deepdraw
git fetch origin
git checkout -b codex/or-sigma0p1-mes-30rounds \
  origin/codex/or-sigma0p1-mes-30rounds
```

If that local branch already exists, use:

```bash
git checkout codex/or-sigma0p1-mes-30rounds
git pull --ff-only origin codex/or-sigma0p1-mes-30rounds
```

Verify the checkout:

```bash
git branch --show-current
git status --short
```

## 2. Install The Environment

With `uv` available:

```bash
uv sync --python 3.10 --extra cluster
```

Without `uv`:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[cluster]'
```

All commands below deliberately use `.venv/bin/python` so Submitit records the
same interpreter that will run on the compute nodes.

## 3. Choose A Shared Output Directory

Use a path without spaces that the submitting account can write and the analysis
account can later read:

```bash
export OR_SWEEP_ROOT=/storage2/wangzitongLab/share/deepdraw_opt/SHARED_OWNER/or_sigma0p1_mes_30rounds
mkdir -p "$OR_SWEEP_ROOT"
```

Replace `SHARED_OWNER` with the agreed shared location. Do not place outputs
inside the Git checkout.

## 4. Validate Inputs And Preview The Full Submission

This command reads metadata headers and checks every subset and embedding path.
It does not submit work:

```bash
.venv/bin/python job_sub/submit_or_mes_allocations.py \
  --output-root "$OR_SWEEP_ROOT"
```

A new output directory should report:

```text
datasets=33
evaluated_rounds=30
array_parents=99
tasks_to_submit=2970
completed_tasks_skipped=0
active_tasks_skipped=0
submit=False
```

The default Slurm request is one CPU, 30 GB per task, 48 hours, QoS `huge`, and
partitions `intel-sc3,amd-ep2,amd-ep5`. Override these when the account has
different permissions, for example:

```bash
--partitions amd-ep5 --qos huge --mem-per-cpu 30GB --timeout-min 2880
```

## 5. Submit A Three-Task Smoke Test

This submits seed 0 from dataset 0 under all three allocations. The Python
submission process runs on the login node, but all three experiments run as
Slurm compute tasks.

```bash
.venv/bin/python job_sub/submit_or_mes_allocations.py \
  --output-root "$OR_SWEEP_ROOT" \
  --dataset-indices 0 \
  --seeds 0 \
  --submit
```

The three array parent IDs are appended to:

```text
$OR_SWEEP_ROOT/submission_manifest.jsonl
```

Inspect their terminal state with the cluster's Slurm binary:

```bash
SLURM_BIN=/soft/slurm/slurm-25.05.2_installation/bin
tail -n 3 "$OR_SWEEP_ROOT/submission_manifest.jsonl"
$SLURM_BIN/squeue -u "$USER" -n dd_or_mes_24x1,dd_or_mes_12x2,dd_or_mes_8x3
```

After they leave `squeue`, obtain the three parent IDs and check accounting:

```bash
PARENTS=$(
  tail -n 3 "$OR_SWEEP_ROOT/submission_manifest.jsonl" |
  .venv/bin/python -c \
    'import json,sys; print(",".join(json.loads(x)["array_parent_id"] for x in sys.stdin))'
)
$SLURM_BIN/sacct -X -j "$PARENTS" \
  --format=JobIDRaw,JobName,State,Elapsed,ExitCode
```

All three parents must be `COMPLETED` with exit code `0:0`.

## 6. Validate The Smoke Outputs On A Compute Node

The validator checks all scientific invariants, including the 24-assay budget,
replicate counts, four-state noise equation, replicate-averaged OR score,
historical-score calibration, GP variance availability, and round counts.

```bash
$SLURM_BIN/srun \
  --partition=amd-ep5 \
  --qos=huge \
  --cpus-per-task=1 \
  --mem=8G \
  --time=00:30:00 \
  .venv/bin/python job_sub/validate_or_mes_sweep.py \
    --sweep-dir "$OR_SWEEP_ROOT" \
    --expected-runs 3
```

Success prints `status=VALID` and writes:

```text
$OR_SWEEP_ROOT/validation_report.json
```

Do not launch the full sweep if this command reports `INVALID`.

## 7. Submit The Full Sweep

Run the short submission command in `tmux` so an SSH disconnect cannot interrupt
the 99 parent submissions:

```bash
tmux new -s or_mes_submit
.venv/bin/python job_sub/submit_or_mes_allocations.py \
  --output-root "$OR_SWEEP_ROOT" \
  --submit
```

The three completed smoke runs are detected and skipped, so this submits the
remaining 2,967 tasks. Detach with `Ctrl-b`, then `d`.

The scheduler also reads its manifest and skips tasks that are still pending or
running. It is therefore safe to preview the command again while the sweep is
active, but do not use a different output root for a recovery submission.

## 8. Validate And Aggregate The Complete Sweep

Only run final validation after all `dd_or_mes_*` jobs have left `squeue`:

```bash
$SLURM_BIN/srun \
  --partition=amd-ep5 \
  --qos=huge \
  --cpus-per-task=1 \
  --mem=16G \
  --time=04:00:00 \
  .venv/bin/python job_sub/validate_or_mes_sweep.py \
    --sweep-dir "$OR_SWEEP_ROOT" \
    --expected-runs 2970
```

The full validator additionally requires every one of the 2,970 unique
dataset/allocation/seed combinations.

After validation succeeds, aggregate all per-round summaries:

```bash
$SLURM_BIN/srun \
  --partition=amd-ep5 \
  --qos=huge \
  --cpus-per-task=1 \
  --mem=16G \
  --time=02:00:00 \
  .venv/bin/python job_sub/aggregate_summaries.py \
    --sweep-dir "$OR_SWEEP_ROOT" \
    --overwrite
```

This creates one `combined_summaries.by_round.csv` under each allocation
directory. Return the sweep path and `validation_report.json` to the analysis
account.
