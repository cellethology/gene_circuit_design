#!/usr/bin/env python3
"""
Add n_top to results.csv files by recomputing top_p counts.

Example:
  python utils/add_n_top.py job_sub/multirun/2025-12-19
  python utils/add_n_top.py job_sub/multirun/2025-12-19 --top-p 0.01 --column-name n_top_1e2 --overwrite
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

DEFAULT_LABEL_KEY = "Fold Change (Induced/Basal)"
DEFAULT_DATASETS_YAML = (
    Path(__file__).resolve().parents[1] / "job_sub" / "datasets" / "datasets.yaml"
)


def _parse_selected_ids(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [item for item in text.split(",") if item.strip()]
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _normalize_id(value: Any) -> str:
    if isinstance(value, int | np.integer):
        return str(int(value))
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(text))
    except ValueError:
        try:
            return str(int(float(text)))
        except ValueError:
            return text


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _resolve_path(raw: Any, base_dir: Path) -> Path | None:
    if raw in (None, "", "null"):
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_subset_ids(path: Path) -> np.ndarray:
    subset_ids = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            subset_ids.append(int(text))
        except ValueError as exc:
            raise ValueError(
                f"Invalid sample id '{text}' in subset file {path}"
            ) from exc
    if not subset_ids:
        raise ValueError(f"Subset ids file {path} did not contain any sample ids.")
    return np.asarray(subset_ids, dtype=np.int64)


def _load_sample_ids(embeddings_path: Path) -> np.ndarray:
    data = np.load(embeddings_path, allow_pickle=True)
    if "ids" not in data:
        raise ValueError(
            f"'ids' array not found in {embeddings_path}. Available keys: {list(data.keys())}"
        )
    return data["ids"].astype(np.int64)


def _load_labels(
    metadata_path: Path, label_key: str, sample_ids: np.ndarray
) -> np.ndarray:
    df = pd.read_csv(metadata_path, usecols=[label_key])
    df = df.iloc[sample_ids]
    return df[label_key].to_numpy()


def _compute_top_id_set(
    embeddings_path: Path,
    metadata_path: Path,
    label_key: str,
    subset_ids_path: Path | None,
    top_p: float,
) -> set[str]:
    sample_ids = _load_sample_ids(embeddings_path)
    if subset_ids_path is not None:
        subset_ids = _load_subset_ids(subset_ids_path)
        mask = np.isin(sample_ids, subset_ids)
        if not np.any(mask):
            raise ValueError(
                "Subset id filtering removed all samples. "
                "Ensure the subset ids match those stored in the embeddings file."
            )
        sample_ids = sample_ids[mask]

    labels = _load_labels(metadata_path, label_key, sample_ids)
    sorted_indices = np.argsort(labels)
    num_top = max(1, int(len(labels) * top_p))
    top_indices = sorted_indices[-num_top:]
    return {_normalize_id(item) for item in sample_ids[top_indices]}


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found: {path}")
    return json.loads(path.read_text())


def _extract_override_value(overrides: list[Any], key: str) -> str | None:
    for entry in overrides:
        text = str(entry).strip()
        if not text:
            continue
        # Strip + prefix used by Hydra for adding new keys
        if text.startswith("+"):
            text = text[1:].strip()
        if "=" in text:
            candidate_key, value = text.split("=", 1)
            if candidate_key.strip() == key:
                return value.strip()
        if text.startswith(f"{key}:"):
            return text.split(":", 1)[1].strip()
    return None


def _resolve_embedding_model(summary: dict[str, Any]) -> str | None:
    model = str(summary.get("embedding_model", "")).strip()
    if model and model.lower() != "none":
        return model
    overrides = summary.get("hydra_overrides") or []
    override_value = _extract_override_value(overrides, "embedding_model")
    if override_value:
        return override_value
    return None


def _load_dataset_map(datasets_yaml_path: Path) -> dict[str, dict[str, Path | None]]:
    if not datasets_yaml_path.exists():
        raise FileNotFoundError(f"Datasets YAML not found: {datasets_yaml_path}")
    payload = _load_yaml(datasets_yaml_path)
    datasets = payload.get("datasets") or []
    if not datasets:
        raise ValueError(f"No datasets found in {datasets_yaml_path}")

    base_dir = datasets_yaml_path.parent
    dataset_map: dict[str, dict[str, Path | None]] = {}
    for entry in datasets:
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"Dataset entry missing name in {datasets_yaml_path}")
        metadata_raw = str(entry.get("metadata_path", "")).strip()
        if not metadata_raw:
            raise ValueError(
                f"Dataset '{name}' missing metadata_path in {datasets_yaml_path}"
            )
        embedding_raw = str(entry.get("embedding_dir", "")).strip()
        if not embedding_raw:
            raise ValueError(
                f"Dataset '{name}' missing embedding_dir in {datasets_yaml_path}"
            )
        subset_raw = entry.get("subset_ids_path")

        dataset_map[name] = {
            "metadata_path": _resolve_path(metadata_raw, base_dir),
            "embedding_dir": _resolve_path(embedding_raw, base_dir),
            "subset_ids_path": _resolve_path(subset_raw, base_dir)
            if subset_raw
            else None,
        }

    return dataset_map


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _update_results_csv(
    results_path: Path,
    top_ids: set[str],
    column_name: str,
    overwrite: bool,
) -> bool:
    rows, fieldnames = _load_rows(results_path)
    if not rows:
        return False
    if "selected_sample_ids" not in fieldnames:
        raise ValueError(f"'selected_sample_ids' column missing in {results_path}")
    if column_name in fieldnames and not overwrite:
        return False

    for row in rows:
        selected_ids = _parse_selected_ids(row.get("selected_sample_ids"))
        normalized = {_normalize_id(item) for item in selected_ids}
        row[column_name] = str(sum(1 for item in normalized if item in top_ids))

    if column_name not in fieldnames:
        fieldnames.append(column_name)
    _write_rows(results_path, rows, fieldnames)
    return True


def _iter_results(root: Path) -> list[Path]:
    return [path for path in root.rglob("results.csv") if path.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add n_top to results.csv files by recomputing top_p."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Directory containing results.csv files (e.g. job_sub/multirun/2025-12-19).",
    )
    parser.add_argument(
        "--datasets-yaml",
        type=Path,
        default=DEFAULT_DATASETS_YAML,
        help="Path to datasets.yaml (default: job_sub/datasets/datasets.yaml).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.01,
        help="Top percentage used to recompute n_top (default: 0.01).",
    )
    parser.add_argument(
        "--column-name",
        type=str,
        default="n_top_1e2",
        help="Column name to write (default: n_top_1e2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the column if it already exists.",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default=DEFAULT_LABEL_KEY,
        help="Label column in the metadata CSV.",
    )
    args = parser.parse_args()

    if not 0.0 < args.top_p <= 1.0:
        raise SystemExit("top_p must be between 0 and 1.")

    root_dir = args.root_dir
    if not root_dir.exists():
        raise SystemExit(f"Root dir not found: {root_dir}")

    dataset_map = _load_dataset_map(args.datasets_yaml)

    results_paths = _iter_results(root_dir)
    if not results_paths:
        raise SystemExit(f"No results.csv found under {root_dir}")

    cache: dict[tuple[Path, Path, Path | None, str, float], set[str]] = {}
    updated = 0
    skipped = 0
    for results_path in tqdm(results_paths, desc="Processing runs", unit="run"):
        run_dir = results_path.parent
        try:
            summary = _load_summary(run_dir / "summary.json")
            dataset_name = str(summary.get("dataset_name", "")).strip()
            if not dataset_name:
                raise ValueError("dataset_name missing in summary.json")
            dataset_spec = dataset_map.get(dataset_name)
            if dataset_spec is None:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in {args.datasets_yaml}"
                )
            embedding_model = _resolve_embedding_model(summary)
            if not embedding_model:
                raise ValueError(
                    f"embedding_model missing for dataset '{dataset_name}'"
                )
            embedding_dir = dataset_spec.get("embedding_dir")
            if embedding_dir is None:
                raise ValueError(f"embedding_dir missing for dataset '{dataset_name}'")
            embedding_file = (
                embedding_model
                if embedding_model.endswith(".npz")
                else f"{embedding_model}.npz"
            )
            embeddings_path = Path(embedding_dir) / embedding_file
            metadata_path = dataset_spec.get("metadata_path")
            if metadata_path is None:
                raise ValueError(f"metadata_path missing for dataset '{dataset_name}'")
            subset_ids_path = dataset_spec.get("subset_ids_path")

            cache_key = (
                embeddings_path,
                Path(metadata_path),
                Path(subset_ids_path) if subset_ids_path else None,
                args.label_key,
                args.top_p,
            )
            top_ids = cache.get(cache_key)
            if top_ids is None:
                top_ids = _compute_top_id_set(
                    embeddings_path=embeddings_path,
                    metadata_path=metadata_path,
                    label_key=args.label_key,
                    subset_ids_path=subset_ids_path,
                    top_p=args.top_p,
                )
                cache[cache_key] = top_ids

            changed = _update_results_csv(
                results_path=results_path,
                top_ids=top_ids,
                column_name=args.column_name,
                overwrite=args.overwrite,
            )
            if changed:
                updated += 1
            else:
                skipped += 1
        except Exception as exc:
            tqdm.write(f"Skipping {results_path}: {exc}")
            skipped += 1

    tqdm.write(f"Updated {updated} runs, skipped {skipped} runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
