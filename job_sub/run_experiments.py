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

from experiments.run_experiments_parallelization import run_single_experiment

LIST_OF_EMBEDDING_PATHS: List[str] = [
    "embeddings.npz",
    "embeddings.npz",
]

_SCRIPT_PATH = Path(__file__).resolve()
_HYDRA_CHILD_ENV = "GENE_CIRCUIT_HYDRA_CHILD"


@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def run_experiments(cfg):
    """Run a single experiment with the given configuration (Hydra entrypoint)."""
    print(OmegaConf.to_yaml(cfg.al_settings))
    run_single_experiment(cfg)


def _collect_user_overrides() -> List[str]:
    """Keep any user overrides other than multirun flag or embedding_paths."""
    overrides: List[str] = []
    for arg in sys.argv[1:]:
        if arg in ("-m", "--multirun"):
            continue
        if arg.startswith("embedding_path="):
            continue
        overrides.append(arg)
    return overrides


def main():
    """Loop over datasets and launch Hydra multirun for each via subprocess."""
    user_overrides = _collect_user_overrides()

    for embedding_path in LIST_OF_EMBEDDING_PATHS:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {embedding_path}")
        print(f"{'=' * 80}\n")
        env = os.environ.copy()
        env[_HYDRA_CHILD_ENV] = "1"
        subprocess.run(
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "-m",
                f"embedding_path={embedding_path}",
                *user_overrides,
            ],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    if os.environ.get(_HYDRA_CHILD_ENV) == "1":
        # Already inside Hydra child: run once
        run_experiments()
    elif any(flag in sys.argv[1:] for flag in ("-m", "--multirun")):
        # User requested multirun: fan out over data paths via subprocess
        main()
    else:
        # Single run, no wrapper looping and no sweeps
        run_experiments()
