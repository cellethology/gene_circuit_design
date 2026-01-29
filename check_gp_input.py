#!/usr/bin/env python3
"""
Sanity checks for GP inputs: embeddings (.npz) + labels (CSV).

This script mirrors the DataLoader alignment logic (ids -> df.iloc[ids])
and reports NaN/inf issues plus basic scale/stability stats.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data:
        raise ValueError(
            f"'embeddings' array not found in {path}. Available keys: {list(data.keys())}"
        )
    if "ids" not in data:
        raise ValueError(
            f"'ids' array not found in {path}. Available keys: {list(data.keys())}"
        )
    embeddings = data["embeddings"]
    sample_ids = data["ids"].astype(np.int32)
    return embeddings, sample_ids


def _load_subset_ids(path: Path | None, dtype: np.dtype) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Subset ids file {path} does not exist.")
    subset_ids: list[int] = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        subset_ids.append(int(text))
    if not subset_ids:
        raise ValueError(f"Subset ids file {path} did not contain any sample ids.")
    return np.asarray(subset_ids, dtype=dtype)


def _apply_subset(
    embeddings: np.ndarray, sample_ids: np.ndarray, subset_ids: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if subset_ids is None:
        return embeddings, sample_ids
    mask = np.isin(sample_ids, subset_ids)
    return embeddings[mask], sample_ids[mask]


def _report_finite(name: str, array: np.ndarray) -> None:
    is_finite = np.isfinite(array)
    num_bad = (~is_finite).sum()
    total = array.size
    print(
        f"{name}: finite={num_bad == 0} bad={num_bad} ({num_bad / max(1, total):.6%})"
    )


def _report_basic_stats(name: str, array: np.ndarray) -> None:
    array = array[np.isfinite(array)]
    if array.size == 0:
        print(f"{name}: no finite values to summarize")
        return
    print(
        f"{name}: min={array.min():.6g} max={array.max():.6g} mean={array.mean():.6g} std={array.std():.6g}"
    )


def _report_embeddings(embeddings: np.ndarray) -> None:
    print(f"embeddings shape: {embeddings.shape}")
    _report_finite("embeddings", embeddings)
    _report_basic_stats("embeddings", embeddings)
    if embeddings.ndim == 2 and embeddings.size > 0:
        row_norms = np.linalg.norm(embeddings, axis=1)
        _report_basic_stats("embeddings row_norm", row_norms)
        col_std = np.std(embeddings, axis=0)
        zero_var = np.sum(col_std == 0)
        print(f"embeddings zero-variance features: {zero_var}/{col_std.size}")


def _report_labels(labels: np.ndarray, log1p_check: bool) -> None:
    print(f"labels shape: {labels.shape}")
    _report_finite("labels", labels)
    _report_basic_stats("labels", labels)
    if log1p_check:
        bad = np.sum(labels <= -1)
        print(f"labels <= -1 (invalid for log1p): {bad}")
    if labels.size > 0:
        unique = np.unique(labels[np.isfinite(labels)])
        if unique.size == 1:
            print("labels variance: 0 (all labels identical)")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data)}")
    return data


def _resolve_env(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("${env:") and value.endswith("}"):
        inner = value[len("${env:") : -1]
        if "," in inner:
            var, default = inner.split(",", 1)
            var = var.strip()
            default = default.strip()
        else:
            var = inner.strip()
            default = None
        env_value = os.environ.get(var)
        if env_value is not None and env_value != "":
            return env_value
        if default in {"null", "None", ""}:
            return None
        return default
    return value


def _pick_dataset(
    datasets_file: Path, dataset_name: str | None
) -> dict[str, str]:
    data = _load_yaml(datasets_file)
    datasets = data.get("datasets", [])
    if not datasets:
        raise ValueError(f"No datasets found in {datasets_file}")
    if dataset_name:
        for entry in datasets:
            if entry.get("name") == dataset_name:
                return entry
        raise ValueError(
            f"Dataset '{dataset_name}' not found in {datasets_file} (available: {len(datasets)})"
        )
    return datasets[0]


def _resolve_defaults(
    config_path: Path,
) -> tuple[Path, Path, str, Path | None]:
    cfg = _load_yaml(config_path)

    datasets_file = cfg.get("datasets_file")
    if not datasets_file:
        raise ValueError("datasets_file missing in config")
    datasets_path = Path("job_sub/datasets") / Path(datasets_file).name
    if not datasets_path.exists():
        datasets_path = Path("job_sub/datasets/datasets.yaml")

    dataset_name = _resolve_env(cfg.get("dataset_name"))
    dataset_entry = _pick_dataset(datasets_path, dataset_name)

    metadata_path = _resolve_env(cfg.get("metadata_path")) or dataset_entry.get(
        "metadata_path"
    )
    if not metadata_path:
        raise ValueError("metadata_path not resolved from config or datasets file")

    embedding_dir = _resolve_env(cfg.get("embedding_dir")) or dataset_entry.get(
        "embedding_dir"
    )
    if not embedding_dir:
        raise ValueError("embedding_dir not resolved from config or datasets file")

    embedding_model = cfg.get("embedding_model")
    embedding_path_raw = cfg.get("embedding_path") or dataset_entry.get(
        "embedding_path"
    )
    embeddings_path: Path | None = None

    if embedding_path_raw and isinstance(embedding_path_raw, str):
        if "${" not in embedding_path_raw:
            embeddings_path = Path(embedding_path_raw)
        else:
            resolved = embedding_path_raw
            if embedding_dir:
                resolved = resolved.replace("${embedding_dir}", str(embedding_dir))
            if embedding_model:
                resolved = resolved.replace("${embedding_model}", str(embedding_model))
            if "${" not in resolved:
                embeddings_path = Path(resolved)

    if embeddings_path is None:
        if not embedding_model:
            raise ValueError("embedding_model missing in config")
        if str(embedding_model).endswith(".npz"):
            embeddings_path = Path(embedding_dir) / str(embedding_model)
        else:
            embeddings_path = Path(embedding_dir) / f"{embedding_model}.npz"

    label_key = cfg.get("al_settings", {}).get("label_key", "and_score")
    subset_ids_path = _resolve_env(cfg.get("subset_ids_path"))
    if subset_ids_path is None:
        subset_ids_path = dataset_entry.get("subset_ids_path")

    return (
        embeddings_path,
        Path(metadata_path),
        str(label_key),
        Path(subset_ids_path) if subset_ids_path else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("job_sub/conf/config.yaml"),
        help="Hydra config to resolve default dataset paths.",
    )
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--label-key")
    parser.add_argument("--subset-ids", type=Path, default=None)
    parser.add_argument(
        "--check-log1p",
        action="store_true",
        help="Warn if labels are invalid for log1p (<= -1).",
    )
    args = parser.parse_args()

    if args.embeddings and args.metadata and args.label_key:
        embeddings_path = args.embeddings
        metadata_path = args.metadata
        label_key = args.label_key
        subset_ids_path = args.subset_ids
    else:
        (
            embeddings_path,
            metadata_path,
            label_key,
            subset_ids_path,
        ) = _resolve_defaults(args.config)
        if args.embeddings:
            embeddings_path = args.embeddings
        if args.metadata:
            metadata_path = args.metadata
        if args.label_key:
            label_key = args.label_key
        if args.subset_ids is not None:
            subset_ids_path = args.subset_ids

    print(f"embeddings path: {embeddings_path}")
    print(f"metadata path: {metadata_path}")
    print(f"label key: {label_key}")
    print(f"subset ids path: {subset_ids_path}")

    embeddings, sample_ids = _load_embeddings(embeddings_path)
    subset_ids = _load_subset_ids(subset_ids_path, sample_ids.dtype)
    embeddings, sample_ids = _apply_subset(embeddings, sample_ids, subset_ids)

    df = pd.read_csv(metadata_path)
    if label_key not in df.columns:
        raise ValueError(
            f"label_key '{label_key}' not found in metadata columns"
        )
    labels = df.iloc[sample_ids][label_key].to_numpy()

    print(f"sample_ids count: {len(sample_ids)}")
    print(f"sample_ids unique: {len(np.unique(sample_ids))}")
    _report_embeddings(embeddings)
    _report_labels(labels, log1p_check=args.check_log1p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
