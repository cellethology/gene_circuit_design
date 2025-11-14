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

LIST_OF_DATA_PATHS: List[str] = [
    "CIS_1-1-1-1.safetensors",
    "CIS_1-1-1-1.safetensors",
]

_SCRIPT_PATH = Path(__file__).resolve()
_HYDRA_CHILD_ENV = "GENE_CIRCUIT_HYDRA_CHILD"


@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def run_experiments(cfg):
    """Run a single experiment with the given configuration (Hydra entrypoint)."""
    print(OmegaConf.to_yaml(cfg.pipeline_params))
    run_single_experiment(data_path=cfg.data_paths, pipeline_param=cfg.pipeline_params)


def main():
    """Loop over datasets and launch Hydra multirun for each via subprocess."""
    for data_path in LIST_OF_DATA_PATHS:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {data_path}")
        print(f"{'=' * 80}\n")
        env = os.environ.copy()
        env[_HYDRA_CHILD_ENV] = "1"
        subprocess.run(
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "-m",
                f"data_paths={data_path}",
            ],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    if os.environ.get(_HYDRA_CHILD_ENV) == "1" or "-m" in sys.argv[1:]:
        run_experiments()
    else:
        main()
