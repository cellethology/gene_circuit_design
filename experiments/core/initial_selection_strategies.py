"""
Initial selection strategies for choosing the seed batch.
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np

from experiments.core.data_loader import Dataset

logger = logging.getLogger(__name__)


EncodeFn = Callable[[List[int]], np.ndarray]


class InitialSelectionStrategy(ABC):
    """Interface for selecting the initial labeled pool."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def select(
        self,
        dataset: Dataset,
        initial_sample_size: int,
        encode_sequences_fn: EncodeFn,
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
        encode_sequences_fn: EncodeFn,
    ) -> List[int]:
        total = len(dataset.sequences)
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
        encode_sequences_fn: EncodeFn,
    ) -> List[int]:
        if initial_sample_size >= len(dataset.sequences):
            return list(range(len(dataset.sequences)))

        from experiments.util import select_initial_batch_kmeans_from_features

        all_indices = list(range(len(dataset.sequences)))
        features = encode_sequences_fn(all_indices)

        selected = select_initial_batch_kmeans_from_features(
            X_all=features,
            initial_sample_size=initial_sample_size,
            seed=self.seed,
        )

        labels = dataset.sequence_labels[selected]
        logger.info(
            "KMEANS_INITIAL: selected %d sequences. Labels: [%s]",
            len(selected),
            ", ".join(f"{val:.3f}" for val in labels),
        )

        return selected
