"""Shared helpers for Hydra/OmegaConf configuration handling."""

import os
from pathlib import Path
from typing import List

from omegaconf import OmegaConf

_MODULE_PATH = Path(__file__).resolve()
_CONF_DIR = _MODULE_PATH.parent.parent / "conf"
_PATHS_CONFIG = _CONF_DIR / "paths" / "paths.yaml"

_resolvers_registered = False


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


def load_embedding_paths() -> List[str]:
    """Load embedding paths from the shared Hydra config."""
    ensure_resolvers()
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
