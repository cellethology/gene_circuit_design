"""
Initial selection strategies for choosing the seed batch.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from core.data_loader import Dataset

logger = logging.getLogger(__name__)


class InitialSelectionStrategy(ABC):
    """Interface for selecting the initial labeled pool."""

    def __init__(
        self, name: Optional[str] = None, starting_batch_size: int = 8
    ) -> None:
        self.name = name or self.__class__.__name__
        self.starting_batch_size = starting_batch_size

    @abstractmethod
    def select(
        self,
        dataset: Dataset,
    ) -> List[int]:
        """Return indices for the initial training pool."""


class RandomInitialSelection(InitialSelectionStrategy):
    """Randomly sample the initial training points."""

    def __init__(self, seed: int, starting_batch_size: int) -> None:
        super().__init__("RANDOM", starting_batch_size=starting_batch_size)
        self.seed = seed

    def select(
        self,
        dataset: Dataset,
    ) -> List[int]:
        rng = np.random.default_rng(self.seed)
        return rng.choice(
            len(dataset.sample_ids), self.starting_batch_size, replace=False
        ).tolist()


class KMeansInitialSelection(InitialSelectionStrategy):
    """Select initial samples using K-means clustering on sequence features."""

    def __init__(self, seed: int, starting_batch_size: int) -> None:
        super().__init__("KMEANS", starting_batch_size=starting_batch_size)
        self.seed = seed

    def select(
        self,
        dataset: Dataset,
    ) -> List[int]:
        selected = self.kmeans_initial_selection(embeddings=dataset.embeddings)

        labels = dataset.labels[selected]
        logger.info(
            "KMEANS_INITIAL: selected %d sequences. Labels: [%s]",
            len(selected),
            ", ".join(f"{val:.3f}" for val in labels),
        )

        return selected

    def kmeans_initial_selection(
        self,
        embeddings: np.ndarray,
    ) -> List[int]:
        kmeans = KMeans(
            n_clusters=self.starting_batch_size, random_state=self.seed
        ).fit(embeddings)
        closest_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, embeddings
        )
        return closest_indices.tolist()
