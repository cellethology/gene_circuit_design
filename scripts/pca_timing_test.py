#!/usr/bin/env python3
"""
Quick test version of PCA timing script with sample data
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set number of cores to use
N_CORES = 4
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)


def encode_sequence(sequence):
    """Simple one-hot encoding of DNA sequence."""
    mapping = {"A": 0, "T": 1, "G": 2, "C": 3}
    encoded = []
    for char in sequence:
        encoded.append(mapping.get(char, 4))
    return np.array(encoded)


def main():
    print("Loading data...")
    start_time = time.time()

    # Load the CSV file
    data_path = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/166k_data/166k_rice/post_embeddings/all_data_with_sequence.csv"
    df = pd.read_csv(data_path)

    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    print(f"Dataset shape: {df.shape}")

    # Take a sample for testing
    sample_size = 1000
    df_sample = df.head(sample_size)

    # Extract sequences
    sequences = df_sample["Sequence"].tolist()
    print(f"Number of sequences (sample): {len(sequences)}")

    # Get sequence lengths
    seq_lengths = [len(seq) for seq in sequences]
    max_length = max(seq_lengths)
    min_length = min(seq_lengths)
    print(f"Sequence lengths - Min: {min_length}, Max: {max_length}")

    # Pad sequences to same length
    print("Padding sequences to same length...")
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + "N" * (max_length - len(seq))
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    # Encode sequences
    print("Encoding sequences...")
    encoding_start = time.time()

    encoded_sequences = []
    for i, seq in enumerate(padded_sequences):
        if i % 100 == 0:
            print(f"Processing sequence {i}/{len(padded_sequences)}")
        encoded_sequences.append(encode_sequence(seq))

    # Convert to numpy array
    X = np.array(encoded_sequences)

    encoding_time = time.time() - encoding_start
    print(f"Sequences encoded in {encoding_time:.2f} seconds")
    print(f"Encoded data shape: {X.shape}")

    # Standardize the data
    print("Standardizing data...")
    scaler_start = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_time = time.time() - scaler_start
    print(f"Data standardized in {scaler_time:.2f} seconds")

    # Perform PCA
    print("Performing PCA...")
    pca_start = time.time()

    # Use smaller number of components for test
    n_components = min(100, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    pca_time = time.time() - pca_start
    print(f"PCA completed in {pca_time:.2f} seconds")

    # Print results
    total_time = time.time() - start_time
    print("\n=== RESULTS ===")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"  - Data loading: {start_time:.2f} seconds")
    print(f"  - Sequence encoding: {encoding_time:.2f} seconds")
    print(f"  - Data standardization: {scaler_time:.2f} seconds")
    print(f"  - PCA computation: {pca_time:.2f} seconds")
    print(f"PCA output shape: {X_pca.shape}")
    print(
        f"Explained variance ratio (first 10 components): {pca.explained_variance_ratio_[:10]}"
    )

    # Estimate time for full dataset
    full_size = 121292
    estimated_encoding_time = (encoding_time / sample_size) * full_size
    estimated_total_time = estimated_encoding_time + scaler_time + pca_time
    print(f"\nEstimated time for full dataset ({full_size} sequences):")
    print(f"  - Encoding: {estimated_encoding_time:.2f} seconds")
    print(f"  - Total: {estimated_total_time:.2f} seconds")


if __name__ == "__main__":
    main()
