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
    default_embedding_model: str


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

        default_model_value = str(dataset.get("default_embedding_model", "")).strip()
        default_model = Path(default_model_value).stem if default_model_value else ""
        if not default_model:
            raise ValueError(
                f"Dataset '{name}' is missing default_embedding_model (stem without extension)"
            )
        default_path = embedding_dir_path / f"{default_model}.npz"
        if not default_path.exists():
            raise ValueError(
                f"Default embedding file '{default_path}' not found for dataset '{name}'"
            )

        dataset_configs.append(
            DatasetConfig(
                name=name,
                metadata_path=metadata_path,
                embedding_dir=str(embedding_dir_path),
                default_embedding_model=default_model,
            )
        )

    return dataset_configs
