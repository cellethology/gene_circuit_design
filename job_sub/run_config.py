"""
Minimal wrapper that runs the Hydra multirun entrypoint once per dataset.

Run the script (`python job_sub/run_config.py`) to sweep across the
datasets defined in datasets/datasets.yaml.
"""

import getpass
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import hydra
from omegaconf import OmegaConf

from job_sub.utils.config_utils import (
    DatasetConfig,
    ensure_resolvers,
    load_dataset_configs,
    seed_env_from_datasets,
)
from job_sub.utils.seed_jobs import run_seed_jobs
from job_sub.utils.sweep_utils import collect_user_overrides

_SCRIPT_PATH = Path(__file__).resolve()
_HYDRA_CHILD_ENV = "GENE_CIRCUIT_HYDRA_CHILD"
_DATASET_ENV = "AL_DATASET_NAME"
_METADATA_ENV = "AL_METADATA_PATH"
_EMBED_DIR_ENV = "AL_EMBEDDING_ROOT"
_SUBSET_ENV = "AL_SUBSET_IDS_PATH"
_SQUEUE_WAIT_ENV = "AL_DISABLE_SQUEUE_WAIT"
_SQUEUE_INTERVAL_ENV = "AL_SQUEUE_POLL_SECONDS"
_DEFAULT_POLL_SECONDS = 60.0
# Deprecated but kept for compatibility
_EMBED_MODEL_ENV = "AL_EMBEDDING_MODEL"
ensure_resolvers()
DATASETS: List[DatasetConfig] = load_dataset_configs()
if not DATASETS:
    raise RuntimeError("No datasets configured in job_sub/datasets/datasets.yaml")


seed_env_from_datasets(
    DATASETS,
    hydra_child_env=_HYDRA_CHILD_ENV,
    dataset_env=_DATASET_ENV,
    metadata_env=_METADATA_ENV,
    embedding_env=_EMBED_DIR_ENV,
    subset_env=_SUBSET_ENV,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_one_job(cfg):
    """Run a single experiment with the given configuration (Hydra entrypoint)."""
    print(OmegaConf.to_yaml(cfg.al_settings))
    run_seed_jobs(cfg)


def main():
    """Loop over datasets and launch Hydra multirun for each via subprocess."""
    user_overrides = collect_user_overrides(sys.argv[1:])

    for dataset in DATASETS:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset.name}")
        print(f"{'=' * 80}\n")
        env = os.environ.copy()
        env[_HYDRA_CHILD_ENV] = "1"
        env[_DATASET_ENV] = dataset.name
        env[_METADATA_ENV] = dataset.metadata_path
        env[_EMBED_DIR_ENV] = dataset.embedding_dir
        if dataset.subset_ids_path:
            env[_SUBSET_ENV] = dataset.subset_ids_path
        elif _SUBSET_ENV in env:
            env.pop(_SUBSET_ENV, None)
        cmd = [
            sys.executable,
            str(_SCRIPT_PATH),
            "-m",
            *user_overrides,
        ]
        _run_dataset_sweep(cmd, env, dataset.name)
        _wait_for_slurm_jobs(dataset.name)


def _run_dataset_sweep(cmd: List[str], env: dict, dataset_name: str) -> None:
    """
    Launch a Hydra multirun for a single dataset, suppressing transient SubmitIt errors.
    """
    try:
        subprocess.run(
            cmd,
            check=True,
            env=env,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if "submitit.core.utils.UncompletedJobError" in stderr:
            print(
                f"[WARN] SubmitIt reported unfinished jobs for dataset '{dataset_name}'. Continuing.",
                file=sys.stderr,
            )
            return
        raise


def _wait_for_slurm_jobs(dataset_name: str) -> None:
    """Poll `squeue` until no active jobs remain for the current user."""
    if os.environ.get(_SQUEUE_WAIT_ENV):
        return
    if not shutil.which("squeue"):
        print(
            "[INFO] `squeue` not available; skipping queue wait before next dataset.",
            file=sys.stderr,
        )
        return
    user = (
        os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or os.environ.get("SLURM_JOB_USER")
        or getpass.getuser()
    )
    if not user:
        print(
            "[WARN] Could not determine user for `squeue`; skipping queue wait.",
            file=sys.stderr,
        )
        return
    try:
        poll_seconds = float(
            os.environ.get(_SQUEUE_INTERVAL_ENV, _DEFAULT_POLL_SECONDS)
        )
    except ValueError:
        poll_seconds = _DEFAULT_POLL_SECONDS

    print(
        f"[INFO] Waiting for Slurm jobs to finish before starting the next dataset "
        f"(current: {dataset_name})."
    )
    while True:
        try:
            result = subprocess.run(
                ["squeue", "-h", "-u", user],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"[WARN] Failed to query `squeue` ({exc.stderr.strip()}), "
                "skipping queue wait.",
                file=sys.stderr,
            )
            return
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            print("[INFO] Slurm queue is empty; continuing to next dataset.")
            return
        print(
            f"[INFO] {len(lines)} Slurm job(s) still active. "
            f"Polling again in {poll_seconds:.0f} seconds..."
        )
        time.sleep(max(1.0, poll_seconds))


if __name__ == "__main__":
    if os.environ.get(_HYDRA_CHILD_ENV) == "1":
        # Already inside Hydra child: run once
        run_one_job()
    elif any(flag in sys.argv[1:] for flag in ("-m", "--multirun")):
        # User requested multirun: fan out over data paths via subprocess
        main()
    else:
        # Single run, no wrapper looping and no sweeps
        run_one_job()
