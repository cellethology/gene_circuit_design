"""Utility helpers for managing Hydra multirun sweeps."""

from pathlib import Path
from typing import List, Set


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
