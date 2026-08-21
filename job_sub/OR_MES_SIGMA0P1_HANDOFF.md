# OR MES Noise Sweep: Run Instructions

This guide runs the Deepdraw OR experiment with:

- MES selection
- sigma = 0.10 noise in log10 expression space
- 30 evaluated rounds
- 30 random seeds
- allocations `24x1`, `12x2`, and `8x3`
- 33 OR datasets
- 2,970 total runs

The submission commands are lightweight and may run on the login node. The
experiments and validation must run through Slurm on compute nodes.

## 1. Download The Code And Install The Environment

```bash
git clone \
  --branch codex/or-sigma0p1-mes-30rounds \
  --single-branch \
  https://github.com/cellethology/deepdraw.git \
  deepdraw-or-mes

cd deepdraw-or-mes
uv sync --python 3.10 --extra cluster
```

Keep all remaining commands in this `deepdraw-or-mes` directory.

## 2. Choose The Shared Output Directory

```bash
umask 002

export OR_SWEEP_ROOT="/storage2/wangzitongLab/share/deepdraw_opt/jerry/20260821_sigma0p1_mes_or_30rounds_${USER}"
export SLURM_BIN=/soft/slurm/slurm-25.05.2_installation/bin

mkdir -p "$OR_SWEEP_ROOT"
test -w "$OR_SWEEP_ROOT"
echo "$OR_SWEEP_ROOT"
```

Use this same `OR_SWEEP_ROOT` for the smoke test, full run, recovery, and final
validation. Do not create a second output directory for retries.

## 3. Preview The Submission

This checks the input files and prints the planned jobs. It does not submit
anything.

```bash
.venv/bin/python job_sub/submit_or_mes_allocations.py \
  --output-root "$OR_SWEEP_ROOT"
```

For a new output directory, confirm that it reports:

```text
datasets=33
evaluated_rounds=30
array_parents=99
tasks_to_submit=2970
completed_tasks_skipped=0
active_tasks_skipped=0
submit=False
```

Stop and report the error if input validation fails or these counts differ.

## 4. Run A Three-Job Smoke Test

Submit seed 0 from the first dataset for all three allocation schemes:

```bash
.venv/bin/python job_sub/submit_or_mes_allocations.py \
  --output-root "$OR_SWEEP_ROOT" \
  --dataset-indices 0 \
  --seeds 0 \
  --submit
```

Monitor the jobs:

```bash
$SLURM_BIN/squeue -u "$USER" \
  -n dd_or_mes_24x1,dd_or_mes_12x2,dd_or_mes_8x3
```

After all three jobs leave `squeue`, validate their outputs on a compute node:

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

Continue only if the validator prints:

```text
status=VALID
```

## 5. Submit The Full Sweep

Start a `tmux` session so the submission is not interrupted by an SSH
disconnect:

```bash
tmux new -s or_mes_submit
```

Inside `tmux`, run:

```bash
.venv/bin/python job_sub/submit_or_mes_allocations.py \
  --output-root "$OR_SWEEP_ROOT" \
  --submit
```

The scheduler will skip the three completed smoke runs and submit the remaining
2,967 runs. After submission finishes, detach from `tmux` with `Ctrl-b`,
followed by `d`.

## 6. Monitor The Run

Check the queue:

```bash
$SLURM_BIN/squeue -u "$USER" \
  -n dd_or_mes_24x1,dd_or_mes_12x2,dd_or_mes_8x3
```

Count completed outputs:

```bash
find "$OR_SWEEP_ROOT" -name summary.json | wc -l
```

The final count must be `2970`.

To preview missing runs without submitting anything, use:

```bash
.venv/bin/python job_sub/submit_or_mes_allocations.py \
  --output-root "$OR_SWEEP_ROOT"
```

## 7. Validate And Aggregate The Complete Sweep

Wait until all `dd_or_mes_*` jobs have left `squeue`. Then run the full
validator on a compute node:

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

Continue only if it prints `status=VALID`. Then aggregate the outputs:

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

This creates `combined_summaries.by_round.csv` inside each allocation
directory.

## If A Job Fails

1. Wait until the current OR jobs are no longer running or pending.
2. Keep the same `OR_SWEEP_ROOT`.
3. Run the preview command from Step 6 and report its missing-run count.
4. Rerun the full submission command from Step 5.

The scheduler preserves completed outputs and submits only genuinely missing
runs. Do not manually delete completed run directories.

## What To Send Back

After validation and aggregation, send Jerry:

- the value printed by `echo "$OR_SWEEP_ROOT"`
- confirmation that the final validator printed `status=VALID`
- `$OR_SWEEP_ROOT/validation_report.json`
