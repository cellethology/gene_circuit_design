"""Shared helpers for Hydra/OmegaConf configuration handling."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from omegaconf import OmegaConf

_MODULE_PATH = Path(__file__).resolve()
_DATASETS_FILE = _MODULE_PATH.parents[1] / "datasets.yaml"

_resolvers_registered = False


@dataclass
class DatasetConfig:
    """Container holding dataset-specific paths and models."""

    name: str
    metadata_path: str
    embedding_dir: str


def ensure_resolvers() -> None:
    """Register custom OmegaConf resolvers once."""
    global _resolvers_registered
    if _resolvers_registered:
        return

    get_resolver_names = getattr(OmegaConf, "get_resolver_names", None)
    env_registered = (
        "env" in get_resolver_names()
        if callable(get_resolver_names)
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

    _resolvers_registered = True


def load_dataset_configs() -> List[DatasetConfig]:
    """Load dataset definitions with metadata and available embedding models."""
    ensure_resolvers()
    if not _DATASETS_FILE.exists():
        return []

    cfg = OmegaConf.load(_DATASETS_FILE)
    datasets_cfg = cfg.get("datasets") or []
    dataset_configs: List[DatasetConfig] = []

    for dataset in datasets_cfg:
        name = str(dataset.get("name"))
        metadata_path = str(dataset.get("metadata_path", "")).strip()
        if not metadata_path:
            raise ValueError(f"Dataset '{name}' is missing metadata_path")

        embedding_dir_raw = dataset.get("embedding_dir")
        if not embedding_dir_raw:
            raise ValueError(f"Dataset '{name}' is missing embedding_dir")
        embedding_root = Path(str(embedding_dir_raw)).expanduser()
        embedding_dir_path = (embedding_root / name).resolve()
        if not embedding_dir_path.exists():
            raise ValueError(
                f"Embedding directory '{embedding_dir_path}' for dataset '{name}' does not exist"
            )

        dataset_configs.append(
            DatasetConfig(
                name=name,
                metadata_path=metadata_path,
                embedding_dir=str(embedding_dir_path),
            )
        )

    return dataset_configs


def seed_env_from_datasets(
    datasets: List[DatasetConfig],
    hydra_child_env: str = "GENE_CIRCUIT_HYDRA_CHILD",
    dataset_env: str = "AL_DATASET_NAME",
    metadata_env: str = "AL_METADATA_PATH",
    embedding_env: str = "AL_EMBEDDING_ROOT",
) -> None:
    """Seed dataset env vars when invoked outside the Hydra child."""
    if os.environ.get(hydra_child_env) == "1":
        return
    if os.environ.get(dataset_env):
        return
    if not datasets:
        return
    dataset = datasets[0]
    os.environ.setdefault(dataset_env, dataset.name)
    os.environ.setdefault(metadata_env, dataset.metadata_path)
    os.environ.setdefault(embedding_env, dataset.embedding_dir)
