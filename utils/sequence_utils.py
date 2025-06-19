"""
Utility functions for DNA sequence processing and analysis.

This module provides functions for one-hot encoding DNA sequences,
loading sequence data, and performing basic sequence analysis tasks.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """
    Convert a DNA sequence to one-hot encoding.
    
    Args:
        sequence: DNA sequence string (A, T, G, C)
        
    Returns:
        One-hot encoded array of shape (length, 4)
        
    Raises:
        ValueError: If sequence contains invalid characters
    """
    if not sequence:
        raise ValueError("Sequence cannot be empty")
    
    # Define nucleotide mapping
    nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    
    # Check for invalid characters
    invalid_chars = set(sequence.upper()) - set(nucleotide_map.keys())
    if invalid_chars:
        raise ValueError(f"Invalid nucleotides found: {invalid_chars}")
    
    # Create one-hot encoding
    sequence_upper = sequence.upper()
    one_hot = np.zeros((len(sequence_upper), 4), dtype=np.float32)
    
    for i, nucleotide in enumerate(sequence_upper):
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
    if not sequences:
        raise ValueError("Sequences list cannot be empty")
    
    encoded_sequences = []
    for i, sequence in enumerate(sequences):
        try:
            encoded = one_hot_encode_sequence(sequence)
            encoded_sequences.append(encoded)
        except ValueError as e:
            raise ValueError(f"Error encoding sequence {i}: {e}")
    
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
        raise ValueError(f"All sequences must have the same length. Found lengths: {set(lengths)}")
    
    # Flatten each sequence and stack
    flattened = [seq.flatten() for seq in encoded_sequences]
    return np.array(flattened, dtype=np.float32)


def load_sequence_data(
    file_path: str, 
    trim_sequences: bool = True, 
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
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV with columns: {list(df.columns)}")
        
        # Determine data format and extract sequences and targets
        if 'seqs' in df.columns and 'scores' in df.columns:
            # Log likelihood format
            sequences = df['seqs'].tolist()
            targets = df['scores'].values
            data_type = "log likelihood"
            logger.info("Detected log likelihood data format (seqs/scores)")
            
        elif 'Sequence' in df.columns and 'Expression' in df.columns:
            # Expression format
            sequences = df['Sequence'].tolist()
            targets = df['Expression'].values
            data_type = "expression"
            logger.info("Detected expression data format (Sequence/Expression)")
            
        else:
            raise ValueError(
                "CSV must contain either ('seqs', 'scores') or ('Sequence', 'Expression') columns. "
                f"Found columns: {list(df.columns)}"
            )

        logger.info(f"Loaded {len(sequences)} sequences with {data_type} targets")

        # Trim sequences if requested
        if trim_sequences:
            sequences = trim_sequences_to_length(sequences, max_length)
            logger.info(f"Trimmed sequences to length {len(sequences[0])}")

        return sequences, targets.astype(np.float32)

    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")


def load_log_likelihood_data(file_path: str, trim_sequences: bool = True) -> Tuple[List[str], np.ndarray]:
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
        df = pd.read_csv(file_path, sep='\t')  # Try tab-separated first
        if 'seqs' not in df.columns or 'scores' not in df.columns:
            # Try comma-separated
            df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ['seqs', 'scores']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. Found: {list(df.columns)}")

        sequences = df['seqs'].tolist()
        log_likelihoods = df['scores'].values

        logger.info(f"Loaded {len(sequences)} sequences with log likelihood scores")

        # Trim sequences if requested
        if trim_sequences:
            sequences = trim_sequences_to_length(sequences)
            logger.info(f"Trimmed sequences to length {len(sequences[0])}")

        return sequences, log_likelihoods.astype(np.float32)

    except Exception as e:
        raise ValueError(f"Error loading log likelihood data from {file_path}: {e}")


def trim_sequences_to_length(
    sequences: List[str], 
    max_length: Optional[int] = None
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
        'count': len(sequences),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths)
    }

    # Calculate nucleotide composition
    all_nucleotides = ''.join(sequences).upper()
    total_nucleotides = len(all_nucleotides)

    if total_nucleotides > 0:
        nucleotide_counts = {
            'A': all_nucleotides.count('A'),
            'T': all_nucleotides.count('T'),
            'G': all_nucleotides.count('G'),
            'C': all_nucleotides.count('C')
        }

        nucleotide_frequencies = {
            f'{nuc}_frequency': count / total_nucleotides 
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

    valid_nucleotides = set('ATGC')
    validated_sequences = []

    for i, seq in enumerate(sequences):
        if not seq:
            raise ValueError(f"Sequence {i} is empty")

        # Convert to uppercase
        seq_upper = seq.upper()

        # Check for invalid characters
        invalid_chars = set(seq_upper) - valid_nucleotides
        if invalid_chars:
            raise ValueError(f"Sequence {i} contains invalid characters: {invalid_chars}")

        validated_sequences.append(seq_upper)

    return validated_sequences