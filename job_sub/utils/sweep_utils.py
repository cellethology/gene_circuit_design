"""Utility helpers for managing Hydra multirun sweeps."""

import json
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd


def collect_user_overrides(argv: List[str]) -> List[str]:
    """Keep any user overrides other than the multirun flag."""
    return [arg for arg in argv if arg not in ("-m", "--multirun")]


def list_sweep_dirs(multirun_base: Path) -> Set[Path]:
    """Return all existing sweep directories under the multirun base."""
    if not multirun_base.exists():
        return set()
    sweeps: Set[Path] = set()
    for date_dir in multirun_base.iterdir():
        if not date_dir.is_dir():
            continue
        for sweep_dir in date_dir.iterdir():
            if sweep_dir.is_dir():
                sweeps.add(sweep_dir)
    return sweeps


def combine_summaries(sweep_dir: Path, dataset_name: Optional[str] = None) -> None:
    """Combine all summary.json files inside the sweep (per dataset)."""
    search_dir = sweep_dir / dataset_name if dataset_name else sweep_dir
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
