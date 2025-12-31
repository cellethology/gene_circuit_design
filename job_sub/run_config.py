"""
Minimal wrapper that runs the Hydra multirun entrypoint once per dataset.

Run the script (`python job_sub/run_config.py`) to sweep across the
datasets defined in config yaml file.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import OmegaConf

from job_sub.utils.config_utils import (
    load_datasets_or_raise,
    seed_env_from_datasets,
)
from job_sub.utils.seed_jobs import run_seed_jobs
from job_sub.utils.slurm_utils import wait_for_slurm_jobs
from job_sub.utils.sweep_utils import collect_user_overrides

_SCRIPT_PATH = Path(__file__).resolve()
_HYDRA_CHILD_ENV = "GENE_CIRCUIT_HYDRA_CHILD"
_DATASET_ENV = "AL_DATASET_NAME"
_METADATA_ENV = "AL_METADATA_PATH"
_EMBED_DIR_ENV = "AL_EMBEDDING_ROOT"
_SUBSET_ENV = "AL_SUBSET_IDS_PATH"
_CONFIG_PATH = _SCRIPT_PATH.parent / "conf" / "config.yaml"
DATASETS, _ = load_datasets_or_raise(sys.argv[1:], _CONFIG_PATH)


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
        wait_for_slurm_jobs(dataset.name)


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
        if stderr:
            print(
                f"[ERROR] Hydra sweep failed for dataset '{dataset_name}'. stderr:\n{stderr}",
                file=sys.stderr,
            )
        raise


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
