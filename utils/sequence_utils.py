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
import torch
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceModificationMethod(Enum):
    TRIM = "trim"
    PAD = "pad"
    EMBEDDING = "embedding"
    CAR = "car"


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
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

def one_hot_encode_motif_sequence(motif_sequence: np.ndarray, num_motifs: int = 18) -> np.ndarray:
    """
    Convert a CAR motif sequence to one-hot encoding.

    Args:
        motif_sequence: 1D numpy array of motif indices (e.g., [13, 13, 17, 0, 0])
        num_motifs: Total number of unique motifs (default 18 for 0-17 range)

    Returns:
        One-hot encoded array of shape (sequence_length, num_motifs)
    """
    if motif_sequence.ndim != 1:
        raise ValueError(f"Expected 1D motif sequence, got shape {motif_sequence.shape}")

    # Convert to PyTorch tensor for one-hot encoding
    motif_tensor = torch.tensor(motif_sequence, dtype=torch.long)
    one_hot = F.one_hot(motif_tensor, num_classes=num_motifs)

    return one_hot.numpy().astype(np.float32)

def one_hot_encode_sequences(sequences, seq_mod_method: str) -> List[np.ndarray]:
    """
    One-hot encode multiple sequences (DNA sequences or motif sequences).

    Args:
        sequences: List of DNA sequence strings or numpy array of motif sequences
        seq_mod_method: Sequence modification method (determines encoding type)

    Returns:
        List of one-hot encoded arrays

    Raises:
        ValueError: If any sequence is invalid
    """
    if len(sequences) == 0:
        raise ValueError("Sequences cannot be empty")

    encoded_sequences = []

    if seq_mod_method == SequenceModificationMethod.CAR.value:
        # Handle motif sequences (numpy array)
        if not isinstance(sequences, np.ndarray):
            raise ValueError("CAR sequences must be a numpy array")

        for i, motif_seq in enumerate(sequences):
            try:
                encoded = one_hot_encode_motif_sequence(motif_seq)
                encoded_sequences.append(encoded)
            except ValueError as err:
                raise ValueError(f"Error encoding motif sequence {i}: {err}") from err
    else:
        # Handle DNA sequences (list of strings)
        if not isinstance(sequences, list):
            raise ValueError("DNA sequences must be a list of strings")

        for i, sequence in enumerate(sequences):
            try:
                encoded = one_hot_encode_sequence(sequence)
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

def load_multiple_car_files(file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine multiple CAR T cell data files.

    Args:
        file_paths: List of CSV file paths containing CAR T cell data

    Returns:
        Tuple of (combined_motif_sequences, combined_targets)

    Raises:
        FileNotFoundError: If any file doesn't exist
        ValueError: If required columns are missing or data formats differ
    """
    all_sequences = []
    all_targets = []

    for file_path in file_paths:
        sequences, targets = load_sequence_data(file_path, seq_mod_method=SequenceModificationMethod.CAR.value)
        all_sequences.append(sequences)
        all_targets.append(targets)

    # Combine all data
    combined_sequences = np.vstack(all_sequences)
    combined_targets = np.concatenate(all_targets)

    logger.info(f"Combined {len(file_paths)} CAR data files: {combined_sequences.shape[0]} total samples")
    return combined_sequences, combined_targets



def load_sequence_data(
    file_path: str,
    seq_mod_method: str = "trim",
    max_length: Optional[int] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Load sequence and target data from CSV file.

    Supports three formats:
    1. Expression data: columns ['Sequence', 'Expression'] (and optional others)
    2. Log likelihood data: columns ['seqs', 'scores']
    3. CAR motif data: columns ['motif i', 'motif j', ...] and 'Cytotoxicity (Nalm 6 Survival)'

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

    # NOTE: rewrite this function to handle CAR-based sequence modifications with grace
    try:
        if seq_mod_method == SequenceModificationMethod.TRIM or seq_mod_method == SequenceModificationMethod.PAD:
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
        elif seq_mod_method == SequenceModificationMethod.CAR.value:
            # Load and preprocess CAR data
            df = pd.read_csv(file_path, encoding='unicode_escape', sep=',')
            logger.info(f"Loaded CAR CSV with columns: {list(df.columns)}")

            # Drop unnecessary columns if they exist
            columns_to_drop = ['AA Seq', 'Parts', 'NaiveCM', 'Initial CAR T Cell Number']
            df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)

            # Validate required columns
            motif_cols = [col for col in df.columns if col.startswith('motif')]
            if not motif_cols:
                raise ValueError("No motif columns found in CAR data")
            if 'Cytotoxicity (Nalm 6 Survival)' not in df.columns:
                raise ValueError("Missing required column: Cytotoxicity (Nalm 6 Survival)")

            # Extract motif sequences and targets
            sequences = df[motif_cols].values
            targets = df['Cytotoxicity (Nalm 6 Survival)'].values

            data_type = "cytotoxicity"
            logger.info(f"Loaded CAR-T cell data with {len(motif_cols)} motif columns and {len(sequences)} samples")
            logger.info(f"Motif value range: {sequences.min()} to {sequences.max()}")

        logger.info(f"Loaded {len(sequences)} sequences with {data_type} targets")

        # Trim sequences if requested
        if seq_mod_method == SequenceModificationMethod.TRIM:
            sequences = trim_sequences_to_length(sequences, max_length)
            logger.info(f"Trimmed sequences to length {len(sequences[0])}")
        elif seq_mod_method == SequenceModificationMethod.PAD:
            sequences = pad_sequences_to_length(sequences, max_length)
            logger.info(f"Padded sequences to length {len(sequences[0])}")
        elif seq_mod_method == SequenceModificationMethod.CAR.value:
            # No length modification needed for motif sequences
            logger.info(f"CAR motif sequences: {sequences.shape}")
        else:
            raise ValueError(f"Invalid sequence modification method: {seq_mod_method}")

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

        # Trim sequences if requested
        if trim_sequences:
            sequences = trim_sequences_to_length(sequences)
            logger.info(f"Trimmed sequences to length {len(sequences[0])}")

        return sequences, log_likelihoods.astype(np.float32)

    except Exception as e:
        raise ValueError(f"Error loading log likelihood data from {file_path}: {e}") from e


def pad_sequences_to_length(
    sequences: List[str], max_length: Optional[int] = None
) -> List[str]:
    """
    Pad all sequences to the same length by adding N characters.

    Args:
        sequences: List of DNA sequences
        max_length: Maximum length to pad to (uses maximum sequence length if None)

    Returns:
        List of padded sequences

    Raises:
        ValueError: If sequences list is empty
    """
    if not sequences:
        raise ValueError("Sequences list cannot be empty")

    # Determine target length
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    # Pad sequences with N
    padded_sequences = [
        seq + "N" * (max_length - len(seq)) for seq in sequences
    ]

    return padded_sequences


def trim_sequences_to_length(
    sequences: List[str], max_length: Optional[int] = None
) -> List[str]:
    """
    Trim all sequences to the same length.

    Args:
        sequences: List of DNA sequences
        max_length: Maximum length to trim to (uses minimum length if None)

    Returns:
        List of trimmed sequences

    Raises:
        ValueError: If sequences list is empty
    """
    if not sequences:
        raise ValueError("Sequences list cannot be empty")

    # Determine target length
    if max_length is None:
        target_length = min(len(seq) for seq in sequences)
    else:
        target_length = min(max_length, min(len(seq) for seq in sequences))

    logger.info(f"Trimming {len(sequences)} sequences to length {target_length}")

    # Trim sequences
    trimmed_sequences = [seq[:target_length] for seq in sequences]

    return trimmed_sequences


def calculate_sequence_statistics(sequences: List[str]) -> Dict[str, Any]:
    """
    Calculate basic statistics about DNA sequences.

    Args:
        sequences: List of DNA sequences

    Returns:
        Dictionary with sequence statistics

    Raises:
        ValueError: If sequences list is empty
    """
    if not sequences:
        raise ValueError("Sequences list cannot be empty")

    lengths = [len(seq) for seq in sequences]

    stats = {
        "count": len(sequences),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }

    # Calculate nucleotide composition
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
