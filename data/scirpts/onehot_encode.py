#!/usr/bin/env python3
"""
One-hot encode DNA sequences and save data to safetensor format.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import save_file


def onehot_encode_sequence(sequence: str, max_length: int) -> np.ndarray:
    """
    One-hot encode a DNA sequence with padding.

    Args:
        sequence: DNA sequence string containing A, T, G, C
        max_length: Maximum sequence length for padding

    Returns:
        One-hot encoded array of shape (max_length, 4)
    """
    # Mapping from nucleotide to index
    nucleotide_map = {"A": 0, "T": 1, "G": 2, "C": 3}

    # Initialize one-hot array with zeros (padding)
    onehot = np.zeros((max_length, 4), dtype=np.float32)

    # Fill in one-hot encoding up to sequence length
    for i, nucleotide in enumerate(sequence):
        if i >= max_length:  # Truncate if sequence too long
            break
        if nucleotide in nucleotide_map:
            onehot[i, nucleotide_map[nucleotide]] = 1.0

    return onehot


def process_csv_to_safetensor(csv_path: str, output_path: str) -> None:
    """
    Process CSV file to one-hot encode sequences and save to safetensor.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output safetensor file
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Extract columns
    variant_ids = df["Variant_ID"].values
    sequences = df["Sequence"].values
    expressions = df["Expression"].values.astype(np.float32)
    log_likelihoods = df["Log_Likelihood"].values.astype(np.float32)

    # Find maximum sequence length
    max_length = max(len(seq) for seq in sequences)
    print(f"Maximum sequence length: {max_length}")

    # One-hot encode sequences
    print("One-hot encoding sequences...")
    onehot_sequences = np.zeros((len(sequences), max_length, 4), dtype=np.float32)

    for i, seq in enumerate(sequences):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(sequences)} sequences")
        onehot_sequences[i] = onehot_encode_sequence(seq, max_length)

    # Flatten to 2D for ML models (samples, features)
    print("Flattening sequences for ML models...")
    flattened_sequences = onehot_sequences.reshape(len(sequences), -1)
    print(f"Final shape: {flattened_sequences.shape}")

    # Prepare data for safetensor
    tensors = {
        "variant_ids": variant_ids.astype(np.int32),
        "embeddings": flattened_sequences,
        "expressions": expressions,
        "log_likelihoods": log_likelihoods,
    }

    # Save to safetensor
    print(f"Saving to {output_path}")
    save_file(tensors, output_path)
    print("Done!")


if __name__ == "__main__":
    csv_path = "/storage2/wangzitongLab/lizelun/project/gene_circuit_design/data/166k_Data/166k_rice/post_embeddings/all_data_with_sequence.csv"
    output_path = "/storage2/wangzitongLab/lizelun/project/gene_circuit_design/data/166k_Data/166k_rice/post_embeddings/onehot_data.safetensors"

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    process_csv_to_safetensor(csv_path, output_path)
