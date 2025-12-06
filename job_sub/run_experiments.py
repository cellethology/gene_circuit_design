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
from typing import List, Optional, Set

import hydra
import pandas as pd
from omegaconf import OmegaConf

from experiments.active_learning import run_single_experiment

_SCRIPT_PATH = Path(__file__).resolve()
_HYDRA_CHILD_ENV = "GENE_CIRCUIT_HYDRA_CHILD"
_MULTIRUN_BASE = _SCRIPT_PATH.parent / "multirun"
_EMBED_PATH_ENV = "AL_EMBEDDING_PATH"
_EMBED_NAME_ENV = "AL_EMBEDDING_NAME"
_PATHS_CONFIG = _SCRIPT_PATH.parent / "conf" / "paths" / "paths.yaml"

_get_resolver_names = getattr(OmegaConf, "get_resolver_names", None)
env_registered = (
    "env" in _get_resolver_names()
    if callable(_get_resolver_names)
    else OmegaConf.has_resolver("env")
)
if not env_registered:
    OmegaConf.register_new_resolver(
        "env",
        lambda key, default=None: os.environ.get(key, default),
        use_cache=False,
    )

OmegaConf.register_new_resolver(
    "path_stem",
    lambda value: Path(value).stem if value else "",
    replace=True,
)


def _load_embedding_paths() -> List[str]:
    """Load embedding paths from the paths config so users edit YAML only."""
    if not _PATHS_CONFIG.exists():
        return []
    cfg = OmegaConf.load(_PATHS_CONFIG)
    root = OmegaConf.create({"paths": cfg})
    OmegaConf.resolve(root)
    paths_cfg = root.paths
    embedding_list = paths_cfg.get("embedding_paths")
    if not embedding_list:
        fallback = paths_cfg.get("embedding_file")
        embedding_list = [fallback] if fallback else []
    return [str(path) for path in embedding_list]


LIST_OF_EMBEDDING_PATHS: List[str] = _load_embedding_paths()


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


def _combine_summaries(sweep_dir: Path, embedding_name: Optional[str] = None) -> None:
    """Combine all summary.json files inside the sweep (per embedding)."""
    search_dir = sweep_dir / embedding_name if embedding_name else sweep_dir
    if not search_dir.exists():
        search_dir = sweep_dir
    summary_files = sorted(search_dir.rglob("summary.json"))
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
    output_csv = search_dir / "combined_summaries.csv"
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
        env[_EMBED_PATH_ENV] = embedding_path
        env[_EMBED_NAME_ENV] = Path(embedding_path).stem
        existing_sweeps = _list_sweep_dirs()
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
        new_sweeps = _list_sweep_dirs() - existing_sweeps
        embedding_name = Path(embedding_path).stem
        for sweep_dir in sorted(new_sweeps):
            _combine_summaries(sweep_dir, embedding_name=embedding_name)


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
