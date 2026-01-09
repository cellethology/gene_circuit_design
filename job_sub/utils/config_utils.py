"""Shared helpers for Hydra/OmegaConf configuration handling."""

import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

_MODULE_PATH = Path(__file__).resolve()
_JOB_SUB_PATH = _MODULE_PATH.parents[1]
_DEFAULT_DATASETS_FILE = _JOB_SUB_PATH / "datasets" / "datasets.yaml"

_resolvers_registered = False


@dataclass
class DatasetConfig:
    """Container holding dataset-specific paths and models."""

    name: str
    metadata_path: str
    embedding_dir: str
    subset_ids_path: str | None = None


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


def load_dataset_configs(datasets_file: Path | None = None) -> list[DatasetConfig]:
    """Load dataset definitions with metadata and available embedding models."""
    ensure_resolvers()
    datasets_path = Path(str(datasets_file)).expanduser() if datasets_file else None
    if datasets_path is None:
        datasets_path = _DEFAULT_DATASETS_FILE
    elif not datasets_path.is_absolute():
        datasets_path = (_JOB_SUB_PATH / datasets_path).resolve()

    if not datasets_path.exists():
        return []

    cfg = OmegaConf.load(datasets_path)
    datasets_cfg = cfg.get("datasets") or []
    dataset_configs: list[DatasetConfig] = []

    for dataset in datasets_cfg:
        name = str(dataset.get("name"))
        metadata_path = str(dataset.get("metadata_path", "")).strip()
        if not metadata_path:
            raise ValueError(f"Dataset '{name}' is missing metadata_path")

        embedding_dir_raw = dataset.get("embedding_dir")
        if not embedding_dir_raw:
            raise ValueError(f"Dataset '{name}' is missing embedding_dir")
        embedding_dir_path = Path(str(embedding_dir_raw)).expanduser().resolve()
        if not embedding_dir_path.exists():
            raise ValueError(
                f"Embedding directory '{embedding_dir_path}' for dataset '{name}' does not exist"
            )

        subset_ids_path = dataset.get("subset_ids_path")
        subset_ids_str = None
        if subset_ids_path:
            subset_path = Path(str(subset_ids_path)).expanduser()
            if not subset_path.is_absolute():
                subset_path = (datasets_path.parent / subset_path).resolve()
            subset_ids_str = str(subset_path)

        dataset_configs.append(
            DatasetConfig(
                name=name,
                metadata_path=metadata_path,
                embedding_dir=str(embedding_dir_path),
                subset_ids_path=subset_ids_str,
            )
        )

    return dataset_configs


def parse_override_value(argv: list[str], key: str) -> str | None:
    """Extract a Hydra-style override value (key= or +key=) from argv."""
    prefix = f"{key}="
    alt_prefix = f"+{key}="
    for arg in argv:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg.startswith(alt_prefix):
            return arg[len(alt_prefix) :]
    return None


def get_datasets_file_setting(argv: list[str], config_path: Path) -> str | None:
    """Resolve datasets_file from CLI overrides or the base config."""
    override = parse_override_value(argv, "datasets_file")
    if override:
        return override
    if not config_path.exists():
        return None
    cfg = OmegaConf.load(config_path)
    return cfg.get("datasets_file")


def load_datasets_or_raise(
    argv: list[str], config_path: Path
) -> tuple[list[DatasetConfig], str | None]:
    """Load datasets based on config/CLI, raising when no datasets are configured."""
    ensure_resolvers()
    datasets_file_setting = get_datasets_file_setting(argv, config_path)
    datasets = load_dataset_configs(datasets_file_setting)
    if not datasets:
        if datasets_file_setting:
            raise RuntimeError(f"No datasets configured in {datasets_file_setting}")
        raise RuntimeError("No datasets configured in job_sub/datasets/datasets.yaml")
    return datasets, datasets_file_setting


def seed_env_from_datasets(
    datasets: list[DatasetConfig],
    hydra_child_env: str = "GENE_CIRCUIT_HYDRA_CHILD",
    dataset_env: str = "AL_DATASET_NAME",
    metadata_env: str = "AL_METADATA_PATH",
    embedding_env: str = "AL_EMBEDDING_ROOT",
    subset_env: str = "AL_SUBSET_IDS_PATH",
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
    if dataset.subset_ids_path:
        os.environ.setdefault(subset_env, dataset.subset_ids_path)
