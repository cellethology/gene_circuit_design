"""
Minimal wrapper that runs the Hydra multirun entrypoint once per dataset.

Run the script (`python job_sub/run_experiments.py`) to sweep across the
paths below.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Set

import hydra
import pandas as pd
from omegaconf import OmegaConf

from experiments.active_learning import run_single_experiment

LIST_OF_EMBEDDING_PATHS: List[str] = [
    "embeddings.npz",
]

_SCRIPT_PATH = Path(__file__).resolve()
_HYDRA_CHILD_ENV = "GENE_CIRCUIT_HYDRA_CHILD"
_MULTIRUN_BASE = _SCRIPT_PATH.parent / "multirun"


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


def _list_sweep_dirs() -> Set[Path]:
    """Return all existing sweep directories under the multirun base."""
    if not _MULTIRUN_BASE.exists():
        return set()
    sweeps: Set[Path] = set()
    for date_dir in _MULTIRUN_BASE.iterdir():
        if not date_dir.is_dir():
            continue
        for sweep_dir in date_dir.iterdir():
            if sweep_dir.is_dir():
                sweeps.add(sweep_dir)
    return sweeps


def _combine_summaries(sweep_dir: Path) -> None:
    """Combine all summary.json files inside a sweep directory into one CSV."""
    summary_files = sorted(sweep_dir.rglob("summary.json"))
    if not summary_files:
        return

    rows = []
    for path in summary_files:
        try:
            data = json.loads(path.read_text())
            data["summary_path"] = str(path)
            rows.append(data)
        except json.JSONDecodeError:
            continue

    if not rows:
        return

    df = pd.DataFrame(rows)
    output_csv = sweep_dir / "combined_summaries.csv"
    df.to_csv(output_csv, index=False)
    print(f"Combined {len(rows)} summaries into {output_csv}")


def main():
    """Loop over datasets and launch Hydra multirun for each via subprocess."""
    user_overrides = _collect_user_overrides()

    for embedding_path in LIST_OF_EMBEDDING_PATHS:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {embedding_path}")
        print(f"{'=' * 80}\n")
        env = os.environ.copy()
        env[_HYDRA_CHILD_ENV] = "1"
        existing_sweeps = _list_sweep_dirs()
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
        new_sweeps = _list_sweep_dirs() - existing_sweeps
        for sweep_dir in sorted(new_sweeps):
            _combine_summaries(sweep_dir)


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
