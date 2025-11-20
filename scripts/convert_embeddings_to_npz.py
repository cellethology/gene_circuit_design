#!/usr/bin/env python3
"""
Convert a safetensors embeddings file into a compressed NPZ file.

The resulting NPZ contains:
    - embeddings: numpy array of shape (n_samples, embedding_dim)
    - ids: numpy array of ints [0, 1, ..., n_samples-1]

Usage:
    python scripts/convert_embeddings_to_npz.py path/to/embeddings.safetensors
"""

import argparse
from pathlib import Path

import numpy as np
from safetensors.torch import load_file


def convert_safetensors_to_npz(safetensors_path: Path) -> Path:
    """Convert the provided safetensors file into a compressed NPZ file."""
    tensors = load_file(str(safetensors_path))
    if "embeddings" not in tensors:
        raise ValueError(
            f"'embeddings' tensor not found in {safetensors_path}. "
            f"Available keys: {list(tensors.keys())}"
        )

    embeddings = tensors["embeddings"].float().numpy()
    ids = np.arange(embeddings.shape[0], dtype=np.int64)

    npz_path = safetensors_path.with_suffix(".npz")
    np.savez_compressed(npz_path, embeddings=embeddings, ids=ids)
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "safetensors_path",
        type=Path,
        help="Path to the embeddings safetensors file.",
    )
    args = parser.parse_args()

    if not args.safetensors_path.exists():
        raise FileNotFoundError(f"{args.safetensors_path} does not exist.")

    output_path = convert_safetensors_to_npz(args.safetensors_path)
    print(f"Saved NPZ file to {output_path}")


if __name__ == "__main__":
    main()
