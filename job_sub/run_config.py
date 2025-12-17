"""
Minimal wrapper that runs the Hydra multirun entrypoint once per dataset.

Run the script (`python job_sub/run_config.py`) to sweep across the
datasets defined in datasets/datasets.yaml.
"""

import os
import subprocess
import sys
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
from job_sub.utils.sweep_utils import (
    collect_user_overrides,
)

_SCRIPT_PATH = Path(__file__).resolve()
_HYDRA_CHILD_ENV = "GENE_CIRCUIT_HYDRA_CHILD"
_MULTIRUN_BASE = _SCRIPT_PATH.parent / "multirun"
_DATASET_ENV = "AL_DATASET_NAME"
_METADATA_ENV = "AL_METADATA_PATH"
_EMBED_DIR_ENV = "AL_EMBEDDING_ROOT"
_SUBSET_ENV = "AL_SUBSET_IDS_PATH"
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
        subprocess.run(
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "-m",
                *user_overrides,
            ],
            check=True,
            env=env,
        )


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
