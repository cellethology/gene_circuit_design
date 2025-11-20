"""
Data loading utilities for active learning experiments.

This loader pairs an embeddings file with a CSV containing
labels (and optional sample identifiers), ensuring row alignment between the two.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """
    Container for loaded dataset with all necessary components.

    Attributes:
        sample_ids: Stable identifiers aligned with embeddings/labels
        labels: Array of target label values (e.g., expression)
        embeddings: Pre-computed embeddings (required)
    """

    sample_ids: List[str]
    labels: np.ndarray
    embeddings: np.ndarray

    def __post_init__(self) -> None:
        """Validate dataset after initialization."""
        if len(self.sample_ids) != len(self.labels):
            raise ValueError(
                f"Sample IDs ({len(self.sample_ids)}) and labels "
                f"({len(self.labels)}) must have the same length"
            )
        if len(self.sample_ids) != len(self.embeddings):
            raise ValueError(
                f"Sample IDs ({len(self.sample_ids)}) and embeddings "
                f"({len(self.embeddings)}) must have the same length"
            )


class DataLoader:
    """
    Load embeddings from NPZ and labels from a paired CSV.
    """

    def __init__(
        self,
        embeddings_path: str,
        metadata_path: str,
        target_val_key: str,
    ) -> None:
        """
        Initialize the data loader.

        Args:
            embeddings_path: Path to safetensors file containing embeddings.
            metadata_path: CSV with labels aligned to embeddings.
            target_val_key: Column in the CSV to use as the training target.
        """
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.target_val_key = target_val_key
        self.dataset: Optional[Dataset] = None

    def load(self) -> Dataset:
        """
        Load paired embeddings/metadata and return a Dataset.
        """
        logger.info(
            f"Loading embeddings from {self.embeddings_path} "
            f"and metadata from {self.metadata_path}"
        )

        embeddings, sample_ids = self._load_embeddings()
        labels = self._load_metadata(sample_ids)

        self.dataset = Dataset(
            sample_ids=sample_ids,
            labels=labels,
            embeddings=embeddings,
        )

        logger.info(
            f"Loaded dataset with {len(self.dataset.sample_ids)} samples. "
            f"Embeddings shape: {self.dataset.embeddings.shape}"
        )

        return self.dataset

    def _load_embeddings(self) -> np.ndarray:
        data = np.load(self.embeddings_path)
        if "embeddings" not in data:
            raise ValueError(
                f"'embeddings' array not found in {self.embeddings_path}. "
                f"Available keys: {list(data.keys())}"
            )
        embeddings = data["embeddings"]
        sample_ids = data["ids"].astype(str).tolist() if "ids" in data else None
        return embeddings, sample_ids

    def _load_metadata(self, sample_ids: List[int]) -> np.ndarray:
        df = pd.read_csv(self.metadata_path)

        # IMPORTANT: sample_ids is row index of csv, so we can use it to index the dataframe
        df = df.iloc[sample_ids]
        labels = df[self.target_val_key].to_numpy()
        return labels
