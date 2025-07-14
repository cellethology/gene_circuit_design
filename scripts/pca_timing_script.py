#!/usr/bin/env python3
"""
PCA Timing Script for all_data_with_sequence.csv
Performs PCA on the sequence data and times the execution on 4 cores.
"""

import os
import time

import numpy as np
import pandas as pd
from safetensors.numpy import save_file
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

# Set number of cores to use
N_CORES = 4
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)


def encode_sequence(sequence):
    """
    Simple one-hot encoding of DNA sequence.
    A=0, T=1, G=2, C=3, other=4
    """
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
    print(f"Columns: {df.columns.tolist()}")

    # Extract sequences
    sequences = df["Sequence"].tolist()
    print(f"Number of sequences: {len(sequences)}")

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
            # Pad with 'N' (which will be encoded as 4)
            padded_seq = seq + "N" * (max_length - len(seq))
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    # Encode sequences
    print("Encoding sequences...")
    encoding_start = time.time()

    encoded_sequences = []
    for i, seq in enumerate(padded_sequences):
        if i % 5000 == 0:
            print(
                f"Processing sequence {i}/{len(padded_sequences)} ({i/len(padded_sequences)*100:.1f}%)"
            )
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

    # Perform Incremental PCA
    print("Performing Incremental PCA...")
    pca_start = time.time()

    # Use 4096 components as requested
    n_components = min(4096, X_scaled.shape[0], X_scaled.shape[1])

    # For IncrementalPCA, batch size must be >= n_components
    batch_size = max(
        5000, n_components + 100
    )  # Ensure batch size is larger than n_components
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # Process in batches for memory efficiency
    n_batches = (X_scaled.shape[0] + batch_size - 1) // batch_size

    print(f"Processing {n_batches} batches of size {batch_size}...")
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_scaled.shape[0])
        batch = X_scaled[start_idx:end_idx]

        if i == 0:
            pca.partial_fit(batch)
        else:
            pca.partial_fit(batch)

        if i % 10 == 0:
            print(f"Processed batch {i+1}/{n_batches}")

    # Transform the data
    X_pca = pca.transform(X_scaled)

    pca_time = time.time() - pca_start
    print(f"Incremental PCA completed in {pca_time:.2f} seconds")

    # Print results
    total_time = time.time() - start_time
    print("\n=== RESULTS ===")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"  - Data loading: {start_time:.2f} seconds")
    print(f"  - Sequence encoding: {encoding_time:.2f} seconds")
    print(f"  - Data standardization: {scaler_time:.2f} seconds")
    print(f"  - Incremental PCA computation: {pca_time:.2f} seconds")
    print(f"PCA output shape: {X_pca.shape}")
    print(
        f"Explained variance ratio (first 10 components): {pca.explained_variance_ratio_[:10]}"
    )
    print(
        f"Cumulative explained variance (first 10 components): {np.cumsum(pca.explained_variance_ratio_[:10])}"
    )

    # Save PCA results
    print("Saving PCA results...")
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_df["Variant_ID"] = df["Variant_ID"]
    pca_df["Expression"] = df["Expression"]
    pca_df["Log_Likelihood"] = df["Log_Likelihood"]

    # Save as CSV
    csv_path = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/pca_results_4cores.csv"
    pca_df.to_csv(csv_path, index=False)
    print(f"PCA results saved to CSV: {csv_path}")

    # Save as safetensors
    safetensors_path = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/pca_results_4cores.safetensors"

    # Prepare tensors for safetensors (need to be numpy arrays)
    tensors = {
        "pca_components": X_pca.astype(np.float32),
        "variant_ids": df["Variant_ID"].values.astype(np.int32),
        "expression": df["Expression"].values.astype(np.float32),
        "log_likelihood": df["Log_Likelihood"].values.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
    }

    save_file(tensors, safetensors_path)
    print(f"PCA results saved to SafeTensors: {safetensors_path}")


if __name__ == "__main__":
    main()
