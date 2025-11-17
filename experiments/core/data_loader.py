"""
Data loading utilities for active learning experiments.

This module handles loading data from various formats (safetensors, CSV)
and preparing train/test/unlabeled splits.
"""

import logging
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from safetensors.torch import load_file

from experiments.core.query_strategies import QueryStrategyBase
from utils.config_loader import SelectionStrategy
from utils.sequence_utils import (
    SequenceModificationMethod,
    ensure_sequence_modification_method,
    load_sequence_data,
)

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """
    Container for loaded dataset with all necessary components.

    Attributes:
        sequences: List of sequence strings or identifiers
        sequence_labels: Array of target label values (e.g., expression)
        log_likelihoods: Array of log likelihood values (may contain NaN)
        embeddings: Pre-computed embeddings (None if not available)
        variant_ids: Variant IDs if available (None otherwise)
    """

    sequences: List[str]
    sequence_labels: np.ndarray
    log_likelihoods: np.ndarray
    embeddings: Optional[np.ndarray]
    variant_ids: Optional[np.ndarray]

    def __post_init__(self) -> None:
        """Validate dataset after initialization."""
        if len(self.sequences) != len(self.sequence_labels):
            raise ValueError(
                f"Sequences ({len(self.sequences)}) and expressions "
                f"({len(self.sequence_labels)}) must have the same length"
            )
        if len(self.sequences) != len(self.log_likelihoods):
            raise ValueError(
                f"Sequences ({len(self.sequences)}) and log_likelihoods "
                f"({len(self.log_likelihoods)}) must have the same length"
            )


@dataclass
class DataSplit:
    """
    Container for train/test/unlabeled data splits.

    Attributes:
        train_indices: Indices for training set
        test_indices: Indices for test set
        unlabeled_indices: Indices for unlabeled pool
    """

    train_indices: List[int]
    test_indices: List[int]
    unlabeled_indices: List[int]


class DataLoader:
    """
    Handles loading and preprocessing of data for active learning experiments.

    Supports multiple data formats:
    - Safetensors files (with embeddings)
    - CSV files (with sequences)
    """

    def __init__(
        self,
        data_path: str,
        seq_mod_method: SequenceModificationMethod = SequenceModificationMethod.EMBEDDING,
        normalize_input_output: bool = True,
        target_val_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the data loader.

        Args:
            data_path: Path to data file (safetensors or CSV)
            seq_mod_method: Sequence modification method for CSV files
            normalize_input_output: Whether to normalize expressions and embeddings
            target_val_key: Optional key for target values in safetensors files
        """
        self.data_path = data_path
        self.seq_mod_method = ensure_sequence_modification_method(seq_mod_method)
        self.normalize_input_output = normalize_input_output
        self.target_val_key = target_val_key
        self.dataset: Optional[Dataset] = None

    def load(self) -> Dataset:
        """
        Load data from file and return Dataset object.

        Returns:
            Dataset object containing all loaded data

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid or missing required fields
        """
        logger.info(f"Loading data from {self.data_path}")

        if self.data_path.endswith(".safetensors"):
            self.dataset = self._load_safetensors()
        else:
            self.dataset = self._load_csv()

        if self.normalize_input_output:
            self._normalize_data()

        logger.info(
            f"Loaded dataset with {len(self.dataset.sequences)} sequences. "
            f"Embeddings shape: {self.dataset.embeddings.shape if self.dataset.embeddings is not None else 'N/A'}"
        )

        return self.dataset

    def _load_safetensors(self) -> Dataset:
        """
        Load data from safetensors file.

        Returns:
            Dataset object with loaded data

        Raises:
            ValueError: If safetensors format is invalid
        """
        tensors = load_file(self.data_path)

        # Handle different safetensors formats
        if "embeddings" in tensors:
            return self._load_embeddings_format(tensors)
        elif "pca_components" in tensors:
            return self._load_pca_format(tensors)
        else:
            raise ValueError(
                f"Unknown safetensors format. Available keys: {list(tensors.keys())}"
            )

    def _load_embeddings_format(self, tensors: dict) -> Dataset:
        """
        Load data from standard embeddings format.

        Args:
            tensors: Dictionary of loaded tensors

        Returns:
            Dataset object
        """
        embeddings = tensors["embeddings"].float().numpy()

        # Handle expression data
        if self.target_val_key and self.target_val_key in tensors:
            sequence_labels = tensors[self.target_val_key].float().numpy()
        elif "expressions" in tensors:
            sequence_labels = tensors["expressions"].float().numpy()
        elif "expression" in tensors:
            sequence_labels = tensors["expression"].float().numpy()
        else:
            raise ValueError(
                f"No expression data found. Available keys: {list(tensors.keys())}"
            )

        # Handle log likelihood
        if "log_likelihoods" in tensors:
            log_likelihoods = tensors["log_likelihoods"].float().numpy()
        elif "log_likelihood" in tensors:
            log_likelihoods = tensors["log_likelihood"].float().numpy()
        else:
            log_likelihoods = np.full(len(sequence_labels), np.nan)
            logger.warning(
                "No log likelihood data found. LOG_LIKELIHOOD strategy will be skipped."
            )

        # Handle variant IDs
        variant_ids = (
            tensors["variant_ids"].numpy() if "variant_ids" in tensors else None
        )

        # Generate dummy sequences for embeddings
        num_sequences = len(sequence_labels)
        sequences = [f"embedding_{i}" for i in range(num_sequences)]

        return Dataset(
            sequences=sequences,
            sequence_labels=sequence_labels,
            log_likelihoods=log_likelihoods,
            embeddings=embeddings,
            variant_ids=variant_ids,
        )

    def _load_pca_format(self, tensors: dict) -> Dataset:
        """
        Load data from PCA components format.

        Args:
            tensors: Dictionary of loaded tensors

        Returns:
            Dataset object
        """
        embeddings = tensors["pca_components"].float().numpy()
        sequence_labels = tensors["expression"].float().numpy()
        log_likelihoods = tensors["log_likelihood"].float().numpy()

        # Handle variant IDs
        if "variant_ids" in tensors:
            variant_ids = tensors["variant_ids"].numpy()
            sequences = [f"variant_{int(vid)}" for vid in variant_ids]
        else:
            variant_ids = None
            sequences = [f"pca_seq_{i}" for i in range(len(sequence_labels))]

        return Dataset(
            sequences=sequences,
            sequence_labels=sequence_labels,
            log_likelihoods=log_likelihoods,
            embeddings=embeddings,
            variant_ids=variant_ids,
        )

    def _load_csv(self) -> Dataset:
        """
        Load data from CSV file.

        Returns:
            Dataset object
        """
        logger.info(f"Using sequence modification method: {self.seq_mod_method}")

        df = pd.read_csv(self.data_path, encoding="latin-1")

        if "Log_Likelihood" in df.columns:
            # Combined dataset with log likelihood
            sequences = df["Sequence"].tolist()
            sequence_labels = df["Expression"].values
            log_likelihoods = df["Log_Likelihood"].values
            logger.info(
                f"Loaded combined dataset with {len(sequences)} sequences "
                "including log likelihood data"
            )
        else:
            # Original expression-only dataset
            sequences, sequence_labels = load_sequence_data(
                self.data_path, seq_mod_method=self.seq_mod_method
            )
            log_likelihoods = np.full(len(sequences), np.nan)
            logger.info(
                f"Loaded expression-only dataset with {len(sequences)} sequences"
            )

        return Dataset(
            sequences=sequences,
            sequence_labels=sequence_labels,
            log_likelihoods=log_likelihoods,
            embeddings=None,  # Will be computed via one-hot encoding
            variant_ids=None,
        )

    def _normalize_data(self) -> None:
        """Normalize labels and embeddings in-place."""
        if self.dataset is None:
            raise ValueError("Dataset must be loaded before normalization")

        # Normalize expressions
        mean_expr = self.dataset.sequence_labels.mean(axis=0, keepdims=True)
        std_expr = self.dataset.sequence_labels.std(axis=0, keepdims=True) + 1e-30
        self.dataset.sequence_labels = (
            self.dataset.sequence_labels - mean_expr
        ) / std_expr

        # Normalize embeddings if available
        if self.dataset.embeddings is not None:
            mean_emb = self.dataset.embeddings.mean(axis=0, keepdims=True)
            std_emb = self.dataset.embeddings.std(axis=0, keepdims=True) + 1e-30
            self.dataset.embeddings = (self.dataset.embeddings - mean_emb) / std_emb

        logger.info(
            f"Normalized labels -> mean={mean_expr.mean():.4f}, "
            f"std={std_expr.mean():.4f}"
        )

    def create_data_split(
        self,
        dataset: Dataset,
        initial_sample_size: int,
        test_size: int,
        no_test: bool,
        query_strategy: QueryStrategyBase,
        random_seed: int,
        encode_sequences_fn,
    ) -> DataSplit:
        """
        Create train/test/unlabeled splits from dataset.

        Args:
            dataset: Loaded dataset
            initial_sample_size: Size of initial training set
            test_size: Size of test set (ignored if no_test=True)
            no_test: Whether to skip test set
            query_strategy: Strategy for initial selection
            random_seed: Random seed for reproducibility
            encode_sequences_fn: Function to encode sequences for K-means

        Returns:
            DataSplit object with train/test/unlabeled indices
        """
        total_samples = len(dataset.sequences)

        # Adjust test size
        if no_test:
            test_size = 0
            test_indices: List[int] = []
        else:
            test_indices = []

        # Create indices for all samples
        all_indices = list(range(total_samples))
        random.Random(random_seed).shuffle(all_indices)

        # Reserve test set
        if test_size > 0:
            test_indices = all_indices[:test_size]
            remaining_indices = all_indices[test_size:]
        else:
            remaining_indices = all_indices

        # Initial training set selection
        if query_strategy in [
            SelectionStrategy.KMEANS_HIGH_EXPRESSION,
            SelectionStrategy.KMEANS_RANDOM,
        ]:
            # Use K-means clustering for initial selection
            train_indices = self._select_initial_batch_kmeans(
                dataset=dataset,
                initial_sample_size=initial_sample_size,
                random_seed=random_seed,
                encode_sequences_fn=encode_sequences_fn,
            )
            unlabeled_indices = [
                idx for idx in remaining_indices if idx not in train_indices
            ]
        else:
            # Use random selection for initial training set
            train_indices = remaining_indices[:initial_sample_size]
            unlabeled_indices = remaining_indices[initial_sample_size:]

        logger.info(
            f"Data split - Train: {len(train_indices)}, "
            f"Test: {len(test_indices)}, "
            f"Unlabeled: {len(unlabeled_indices)}"
        )

        return DataSplit(
            train_indices=train_indices,
            test_indices=test_indices,
            unlabeled_indices=unlabeled_indices,
        )

    def _select_initial_batch_kmeans(
        self,
        dataset: Dataset,
        initial_sample_size: int,
        random_seed: int,
        encode_sequences_fn,
    ) -> List[int]:
        """
        Select initial batch using K-means clustering.

        Args:
            dataset: Loaded dataset
            initial_sample_size: Number of clusters/initial samples
            random_seed: Random seed for K-means
            encode_sequences_fn: Function to encode sequences

        Returns:
            List of selected indices
        """
        from experiments.util import select_initial_batch_kmeans_from_features

        # Encode all sequences
        all_indices = list(range(len(dataset.sequences)))
        X_all = encode_sequences_fn(all_indices)

        selected_indices = select_initial_batch_kmeans_from_features(
            X_all=X_all,
            initial_sample_size=initial_sample_size,
            random_seed=random_seed,
        )

        selected_labels = dataset.sequence_labels[selected_indices]
        logger.info(
            f"KMEANS_CLUSTERING: Selected {len(selected_indices)} sequences "
            f"from {initial_sample_size} clusters with actual labels: "
            f"[{', '.join(f'{expr:.1f}' for expr in selected_labels)}]"
        )

        return selected_indices
