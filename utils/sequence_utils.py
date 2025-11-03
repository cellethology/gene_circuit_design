"""
Utility functions for DNA sequence processing and analysis.

This module provides functions for one-hot encoding DNA sequences,
loading sequence data, and performing basic sequence analysis tasks.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceModificationMethod(Enum):
    EMBEDDING = "embedding"


def ensure_sequence_modification_method(
    seq_mod_method: "str | SequenceModificationMethod",
) -> SequenceModificationMethod:
    """
    Convert string to SequenceModificationMethod enum if necessary.

    Args:
        seq_mod_method: Either a string or SequenceModificationMethod enum

    Returns:
        SequenceModificationMethod enum

    Raises:
        ValueError: If string is not a valid enum value
    """
    if isinstance(seq_mod_method, str):
        # Only EMBEDDING is supported now
        return SequenceModificationMethod(seq_mod_method)
    return seq_mod_method


def one_hot_encode_single_sequence(sequence: str) -> np.ndarray:
    """
    Convert a DNA sequence to one-hot encoding.

    Args:
        sequence: DNA sequence string (A, T, G, C, N)

    Returns:
        One-hot encoded array of shape (length, 4)

    Raises:
        ValueError: If sequence contains invalid characters
    """
    if not sequence:
        raise ValueError("Sequence cannot be empty")

    # Define nucleotide mapping
    nucleotide_map = {"A": 0, "T": 1, "G": 2, "C": 3}

    # Check for invalid characters
    invalid_chars = set(sequence.upper()) - set(nucleotide_map.keys()) - {"N"}
    if invalid_chars:
        raise ValueError(f"Invalid nucleotides found: {invalid_chars}")

    # Create one-hot encoding
    sequence_upper = sequence.upper()
    one_hot = np.zeros((len(sequence_upper), 4), dtype=np.float32)

    for i, nucleotide in enumerate(sequence_upper):
        if nucleotide != "N":
            one_hot[i, nucleotide_map[nucleotide]] = 1.0

    return one_hot


def one_hot_encode_sequences(
    sequences, seq_mod_method: SequenceModificationMethod
) -> List[np.ndarray]:
    """
    One-hot encode multiple DNA sequences.

    Args:
        sequences: List of DNA sequence strings
        seq_mod_method: Sequence modification method

    Returns:
        List of one-hot encoded arrays

    Raises:
        ValueError: If any sequence is invalid
    """
    if len(sequences) == 0:
        raise ValueError("Sequences cannot be empty")

    # Convert string to enum if necessary (kept for backward compatibility)
    seq_mod_method = ensure_sequence_modification_method(seq_mod_method)

    encoded_sequences = []

    # Handle DNA sequences (list of strings)
    if not isinstance(sequences, list):
        raise ValueError("DNA sequences must be a list of strings")

    for i, sequence in enumerate(sequences):
        try:
            encoded = one_hot_encode_single_sequence(sequence)
            encoded_sequences.append(encoded)
        except ValueError as err:
            raise ValueError(f"Error encoding DNA sequence {i}: {err}") from err

    return encoded_sequences


def flatten_one_hot_sequences(encoded_sequences: List[np.ndarray]) -> np.ndarray:
    """
    Flatten one-hot encoded sequences for machine learning input.

    Args:
        encoded_sequences: List of one-hot encoded sequence arrays

    Returns:
        2D array where each row is a flattened sequence

    Raises:
        ValueError: If sequences have different lengths
    """
    if not encoded_sequences:
        raise ValueError("Encoded sequences list cannot be empty")

    # Check that all sequences have the same length
    lengths = [seq.shape[0] for seq in encoded_sequences]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All sequences must have the same length. Found lengths: {set(lengths)}"
        )

    # Flatten each sequence and stack
    flattened = [seq.flatten() for seq in encoded_sequences]
    return np.array(flattened, dtype=np.float32)


def flatten_one_hot_sequences_with_pca(
    encoded_sequences: List[np.ndarray], n_components: int = 4096
) -> np.ndarray:
    """
    Flatten one-hot encoded sequences and apply PCA for dimensionality reduction.

    Args:
        encoded_sequences: List of one-hot encoded sequence arrays
        n_components: Number of PCA components to keep (default: 4096)

    Returns:
        2D array where each row is a PCA-reduced flattened sequence

    Raises:
        ValueError: If sequences have different lengths or n_components is invalid
    """
    if encoded_sequences is None or len(encoded_sequences) == 0:
        raise ValueError("Encoded sequences list cannot be empty")

    # Check that all sequences have the same length
    lengths = [seq.shape[0] for seq in encoded_sequences]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All sequences must have the same length. Found lengths: {set(lengths)}"
        )

    # Flatten each sequence and stack
    flattened = [seq.flatten() for seq in encoded_sequences]
    flattened_array = np.array(flattened, dtype=np.float32)

    # Check if PCA is needed
    original_dim = flattened_array.shape[1]
    if original_dim <= n_components:
        logger.info(
            f"Original dimension ({original_dim}) is <= n_components ({n_components}). "
            f"Skipping PCA and returning original data."
        )
        return flattened_array

    # Apply PCA
    logger.info(
        f"Applying PCA: reducing from {original_dim} to {n_components} dimensions"
    )
    pca = PCA(n_components=n_components, random_state=42)
    reduced_sequences = pca.fit_transform(flattened_array)

    # Log variance explained
    variance_explained = pca.explained_variance_ratio_.sum()
    logger.info(
        f"PCA completed. Variance explained: {variance_explained:.4f} "
        f"({variance_explained * 100:.2f}%)"
    )

    return reduced_sequences.astype(np.float32)


def load_sequence_data(
    file_path: str,
    seq_mod_method: SequenceModificationMethod = SequenceModificationMethod.EMBEDDING,
    max_length: Optional[int] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Load sequence and target data from CSV file.

    Supports two formats:
    1. Expression data: columns ['Sequence', 'Expression'] (and optional others)
    2. Log likelihood data: columns ['seqs', 'scores']

    Args:
        file_path: Path to CSV file
        trim_sequences: Whether to trim sequences to same length
        max_length: Maximum sequence length (uses minimum if None)

    Returns:
        Tuple of (sequences, target_values)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading data from {file_path}")

    # Convert string to enum if necessary
    seq_mod_method = ensure_sequence_modification_method(seq_mod_method)

    try:
        # Initialize variables
        sequences = None
        targets = None
        data_type = None

        # Load CSV and infer format. Only EMBEDDING mode is supported now.
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV with columns: {list(df.columns)}")
        # Determine data format and extract sequences and targets
        if "seqs" in df.columns and "scores" in df.columns:
            # Log likelihood format
            sequences = df["seqs"].tolist()
            targets = df["scores"].values
            data_type = "log likelihood"
            logger.info("Detected log likelihood data format (seqs/scores)")

        elif "Sequence" in df.columns and "Expression" in df.columns:
            # Expression format
            sequences = df["Sequence"].tolist()
            targets = df["Expression"].values
            data_type = "expression"
            logger.info("Detected expression data format (Sequence/Expression)")

        else:
            raise ValueError(
                "CSV must contain either ('seqs', 'scores') or ('Sequence', 'Expression') columns. "
                f"Found columns: {list(df.columns)}"
            )

        # Verify data was loaded
        if sequences is None or targets is None:
            raise ValueError("Failed to load sequences and targets from file")

        logger.info(f"Loaded {len(sequences)} sequences with {data_type} targets")

        # Trimming/Padding has been removed. Sequences are returned as-is.

        return sequences, targets.astype(np.float32)

    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}") from e


def load_log_likelihood_data(
    file_path: str, trim_sequences: bool = True
) -> Tuple[List[str], np.ndarray]:
    """
    Load sequence and log likelihood data from CSV file.

    Expected format: CSV with columns 'seqs' and 'scores'

    Args:
        file_path: Path to CSV file with log likelihood data
        trim_sequences: Whether to trim sequences to same length

    Returns:
        Tuple of (sequences, log_likelihood_scores)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading log likelihood data from {file_path}")

    try:
        df = pd.read_csv(file_path, sep="\t")  # Try tab-separated first
        if "seqs" not in df.columns or "scores" not in df.columns:
            # Try comma-separated
            df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ["seqs", "scores"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. Found: {list(df.columns)}"
            )

        sequences = df["seqs"].tolist()
        log_likelihoods = df["scores"].values

        logger.info(f"Loaded {len(sequences)} sequences with log likelihood scores")

        # Trimming has been removed. Sequences are returned as-is.

        return sequences, log_likelihoods.astype(np.float32)

    except Exception as e:
        raise ValueError(
            f"Error loading log likelihood data from {file_path}: {e}"
        ) from e


def calculate_sequence_statistics(sequences) -> Dict[str, Any]:
    """
    Calculate basic statistics about DNA sequences.

    Args:
        sequences: List of DNA sequences

    Returns:
        Dictionary with sequence statistics

    Raises:
        ValueError: If sequences is empty
    """
    # Handle DNA sequences
    if not sequences:
        raise ValueError("Sequences list cannot be empty")
    lengths = [len(seq) for seq in sequences]
    count = len(sequences)

    stats = {
        "count": count,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }

    # Calculate nucleotide composition for DNA sequences
    all_nucleotides = "".join(sequences).upper()
    total_nucleotides = len(all_nucleotides)

    if total_nucleotides > 0:
        nucleotide_counts = {
            "A": all_nucleotides.count("A"),
            "T": all_nucleotides.count("T"),
            "G": all_nucleotides.count("G"),
            "C": all_nucleotides.count("C"),
        }

        nucleotide_frequencies = {
            f"{nuc}_frequency": count / total_nucleotides
            for nuc, count in nucleotide_counts.items()
        }

        stats.update(nucleotide_counts)
        stats.update(nucleotide_frequencies)

    return stats


def validate_sequences(sequences: List[str]) -> List[str]:
    """
    Validate and clean DNA sequences.

    Args:
        sequences: List of DNA sequences

    Returns:
        List of validated sequences

    Raises:
        ValueError: If any sequence contains invalid characters
    """
    if not sequences:
        raise ValueError("Sequences list cannot be empty")

    valid_nucleotides = set("ATGC")
    validated_sequences = []

    for i, seq in enumerate(sequences):
        if not seq:
            raise ValueError(f"Sequence {i} is empty")

        # Convert to uppercase
        seq_upper = seq.upper()

        # Check for invalid characters
        invalid_chars = set(seq_upper) - valid_nucleotides
        if invalid_chars:
            raise ValueError(
                f"Sequence {i} contains invalid characters: {invalid_chars}"
            )

        validated_sequences.append(seq_upper)

    return validated_sequences
