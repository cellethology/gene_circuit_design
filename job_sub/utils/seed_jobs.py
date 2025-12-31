"""Helpers for expanding a Hydra config into per-seed job configs."""

import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from job_sub.utils.seed_runner import run_seed_experiment


def generate_seed_values(cfg) -> List[int]:
    """Return sequential seeds to run inside a single Hydra job."""
    num_seeds = int(OmegaConf.select(cfg, "num_seeds_per_job", default=1))
    if num_seeds < 1:
        raise ValueError("num_seeds_per_job must be >= 1")
    return list(range(num_seeds))


def materialize_seed_cfgs(cfg) -> List[Dict[str, Any]]:
    """Return serialized configs for each seed to support multiprocessing."""
    base_output_dir = Path(str(cfg.al_settings.output_dir))
    base_cfg = OmegaConf.to_container(cfg, resolve=False)
    overrides = _extract_hydra_overrides()
    seed_cfgs: List[Dict[str, Any]] = []
    for seed in generate_seed_values(cfg):
        seed_output_dir = base_output_dir / f"seed_{seed}"
        seed_cfg = copy.deepcopy(base_cfg)
        al_settings = seed_cfg.setdefault("al_settings", {})
        al_settings["seed"] = seed
        al_settings["output_dir"] = str(seed_output_dir)
        seed_cfg["hydra_overrides"] = list(overrides)
        seed_cfgs.append(seed_cfg)
    return seed_cfgs


def max_seed_workers(cfg, num_tasks: int) -> int:
    """Decide max workers based on config flag and available CPUs."""
    available = os.cpu_count() or 1
    print(f"[seed_jobs] cpu_count={available}")
    if not OmegaConf.select(cfg, "parallelize_seeds", default=True):
        return 1
    return max(1, min(available, num_tasks))


def run_seed_jobs(cfg) -> None:
    """Run all seed configurations, parallelizing when enabled."""
    seed_cfgs = materialize_seed_cfgs(cfg)
    workers = max_seed_workers(cfg, len(seed_cfgs))
    if workers == 1 or len(seed_cfgs) == 1:
        for raw_cfg in seed_cfgs:
            run_seed_experiment(raw_cfg)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_seed_experiment, raw_cfg) for raw_cfg in seed_cfgs
        ]
        for future in as_completed(futures):
            future.result()


def _extract_hydra_overrides() -> List[str]:
    """Return task-level overrides from Hydra runtime if available."""
    if HydraConfig.initialized():
        try:
            return [str(item) for item in HydraConfig.get().overrides.task]
        except Exception:  # pragma: no cover - defensive
            return []
    return []
