"""Helpers for running per-seed experiments in parallel worker processes."""

import os
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from experiments.active_learning import run_one_experiment

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


def run_seed_experiment(raw_cfg: Dict[str, Any]) -> None:
    """Recreate DictConfig and run experiment (used by multiprocessing workers)."""
    cfg = OmegaConf.create(raw_cfg)
    OmegaConf.resolve(cfg)
    run_one_experiment(cfg)
