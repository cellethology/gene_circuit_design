"""
Utility functions for DNA sequence processing and analysis.

This module provides functions for one-hot encoding DNA sequences,
loading sequence data, and performing basic sequence analysis tasks.
"""

import logging
from typing import List

import numpy as np
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def one_hot_encode_sequences(sequences: List[str]) -> List[np.ndarray]:
    """
    One-hot encode multiple DNA sequences.

    Args:
        sequences: List of DNA sequence strings

    Returns:
        List of one-hot encoded arrays

    Raises:
        ValueError: If any sequence is invalid
    """
    if len(sequences) == 0:
        raise ValueError("Sequences cannot be empty")

    encoded_sequences = []

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
