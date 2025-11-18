"""
Initial selection strategies for choosing the seed batch.
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional

from experiments.core.data_loader import Dataset

logger = logging.getLogger(__name__)


class InitialSelectionStrategy(ABC):
    """Interface for selecting the initial labeled pool."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def select(
        self,
        dataset: Dataset,
        initial_sample_size: int,
    ) -> List[int]:
        """Return indices for the initial training pool."""


class RandomInitialSelection(InitialSelectionStrategy):
    """Randomly sample the initial training points."""

    def __init__(self, seed: int) -> None:
        super().__init__("RANDOM_INITIAL")
        self.seed = seed

    def select(
        self,
        dataset: Dataset,
        initial_sample_size: int,
    ) -> List[int]:
        total = len(dataset.sample_ids)
        if initial_sample_size >= total:
            return list(range(total))
        rng = random.Random(self.seed)
        return rng.sample(range(total), initial_sample_size)


class KMeansInitialSelection(InitialSelectionStrategy):
    """Select initial samples using K-means clustering on sequence features."""

    def __init__(self, seed: int) -> None:
        super().__init__("KMEANS_INITIAL")
        self.seed = seed

    def select(
        self,
        dataset: Dataset,
        initial_sample_size: int,
    ) -> List[int]:
        total = len(dataset.sample_ids)
        if initial_sample_size >= total:
            return list(range(total))

        from experiments.util import select_initial_batch_kmeans_from_features

        all_indices = list(range(len(dataset.sample_ids)))
        embeddings = dataset.embeddings[all_indices, :]

        selected = select_initial_batch_kmeans_from_features(
            X_all=embeddings,
            initial_sample_size=initial_sample_size,
            seed=self.seed,
        )

        labels = dataset.labels[selected]
        logger.info(
            "KMEANS_INITIAL: selected %d sequences. Labels: [%s]",
            len(selected),
            ", ".join(f"{val:.3f}" for val in labels),
        )

        return selected
