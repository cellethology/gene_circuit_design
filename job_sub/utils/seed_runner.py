"""Helpers for running per-seed experiments in parallel worker processes."""

from typing import Any, Dict

from omegaconf import OmegaConf

from experiments.active_learning import run_one_experiment
from job_sub.utils.config_utils import ensure_resolvers


def run_seed_experiment(raw_cfg: Dict[str, Any]) -> None:
    """Recreate DictConfig and run experiment (used by multiprocessing workers)."""
    ensure_resolvers()
    cfg = OmegaConf.create(raw_cfg)
    OmegaConf.resolve(cfg)
    run_one_experiment(cfg)
