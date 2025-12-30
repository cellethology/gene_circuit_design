#!/usr/bin/env python3
"""
Concatenate two embedding NPZ files by matching sample ids.

Each NPZ must contain:
  - embeddings: array with shape (n_samples, n_features)
  - ids: array with shape (n_samples,)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data or "ids" not in data:
        raise ValueError(
            f"{path} must contain 'embeddings' and 'ids' arrays. "
            f"Found keys: {list(data.keys())}"
        )
    embeddings = np.asarray(data["embeddings"])
    ids = np.asarray(data["ids"])
    if ids.ndim != 1:
        raise ValueError(f"{path} ids must be 1D, got shape {ids.shape}")
    if embeddings.shape[0] != ids.shape[0]:
        raise ValueError(
            f"{path} embeddings/ids length mismatch: "
            f"{embeddings.shape[0]} vs {ids.shape[0]}"
        )
    return embeddings, ids


def _ensure_unique(ids: np.ndarray, label: str) -> None:
    unique_count = np.unique(ids).size
    if unique_count != ids.size:
        raise ValueError(f"{label} ids contain duplicates ({ids.size - unique_count}).")


def concat_embeddings(path_a: Path, path_b: Path, output_path: Path) -> None:
    emb_a, ids_a = _load_npz(path_a)
    emb_b, ids_b = _load_npz(path_b)
    _ensure_unique(ids_a, f"{path_a}")
    _ensure_unique(ids_b, f"{path_b}")

    ids_b_set = set(ids_b.tolist())
    mask_a = np.isin(ids_a, ids_b)
    ids_common = ids_a[mask_a]
    if ids_common.size == 0:
        raise ValueError("No overlapping ids found between the two files.")

    index_b = {int(id_val): idx for idx, id_val in enumerate(ids_b)}
    idx_a = np.nonzero(mask_a)[0]
    idx_b = np.array([index_b[int(id_val)] for id_val in ids_common], dtype=int)

    emb_concat = np.concatenate([emb_a[idx_a], emb_b[idx_b]], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, embeddings=emb_concat, ids=ids_common)

    print(f"Saved concatenated embeddings to {output_path}")
    print(f"File A: {path_a} ({ids_a.size} ids, {emb_a.shape[1]} dims)")
    print(f"File B: {path_b} ({ids_b.size} ids, {emb_b.shape[1]} dims)")
    print(f"Overlap: {ids_common.size} ids, output shape: {emb_concat.shape}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate two embedding NPZ files by matching ids."
    )
    parser.add_argument("embedding_a", type=Path, help="Path to first NPZ file.")
    parser.add_argument("embedding_b", type=Path, help="Path to second NPZ file.")
    parser.add_argument("output", type=Path, help="Path for output NPZ file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    concat_embeddings(args.embedding_a, args.embedding_b, args.output)


if __name__ == "__main__":
    main()
