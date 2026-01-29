#!/usr/bin/env python3
"""
Sanity checks for GP inputs: embeddings (.npz) + labels (CSV).

This script mirrors the DataLoader alignment logic (ids -> df.iloc[ids])
and reports NaN/inf issues plus basic scale/stability stats.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def _load_subset_ids(path: Path | None) -> np.ndarray | None:
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
    return np.asarray(subset_ids, dtype=np.int32)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--label-key", default="and_score")
    parser.add_argument("--subset-ids", type=Path, default=None)
    parser.add_argument(
        "--check-log1p",
        action="store_true",
        help="Warn if labels are invalid for log1p (<= -1).",
    )
    args = parser.parse_args()

    embeddings, sample_ids = _load_embeddings(args.embeddings)
    subset_ids = _load_subset_ids(args.subset_ids)
    embeddings, sample_ids = _apply_subset(embeddings, sample_ids, subset_ids)

    df = pd.read_csv(args.metadata)
    if args.label_key not in df.columns:
        raise ValueError(
            f"label_key '{args.label_key}' not found in metadata columns"
        )
    labels = df.iloc[sample_ids][args.label_key].to_numpy()

    print(f"sample_ids count: {len(sample_ids)}")
    print(f"sample_ids unique: {len(np.unique(sample_ids))}")
    _report_embeddings(embeddings)
    _report_labels(labels, log1p_check=args.check_log1p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
