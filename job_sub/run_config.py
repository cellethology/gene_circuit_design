"""
Minimal wrapper that runs the Hydra multirun entrypoint once per dataset.

Run the script (`python job_sub/run_experiments.py`) to sweep across the
paths below.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import OmegaConf

from job_sub.utils.config_utils import ensure_resolvers, load_embedding_paths
from job_sub.utils.seed_jobs import run_seed_jobs
from job_sub.utils.sweep_utils import (
    collect_user_overrides,
    combine_summaries,
    list_sweep_dirs,
)

_SCRIPT_PATH = Path(__file__).resolve()
_HYDRA_CHILD_ENV = "GENE_CIRCUIT_HYDRA_CHILD"
_MULTIRUN_BASE = _SCRIPT_PATH.parent / "multirun"
_EMBED_PATH_ENV = "AL_EMBEDDING_PATH"
_EMBED_NAME_ENV = "AL_EMBEDDING_NAME"
ensure_resolvers()
LIST_OF_EMBEDDING_PATHS: List[str] = load_embedding_paths()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_one_job(cfg):
    """Run a single experiment with the given configuration (Hydra entrypoint)."""
    print(OmegaConf.to_yaml(cfg.al_settings))
    run_seed_jobs(cfg)


def main():
    """Loop over datasets and launch Hydra multirun for each via subprocess."""
    user_overrides = collect_user_overrides(sys.argv[1:])

    for embedding_path in LIST_OF_EMBEDDING_PATHS:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {embedding_path}")
        print(f"{'=' * 80}\n")
        env = os.environ.copy()
        env[_HYDRA_CHILD_ENV] = "1"
        env[_EMBED_PATH_ENV] = embedding_path
        env[_EMBED_NAME_ENV] = Path(embedding_path).stem
        existing_sweeps = list_sweep_dirs(_MULTIRUN_BASE)
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
        new_sweeps = list_sweep_dirs(_MULTIRUN_BASE) - existing_sweeps
        embedding_name = Path(embedding_path).stem
        for sweep_dir in sorted(new_sweeps):
            combine_summaries(sweep_dir, embedding_name=embedding_name)


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
