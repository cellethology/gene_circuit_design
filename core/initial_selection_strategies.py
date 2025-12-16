"""
Initial selection strategies for choosing the seed batch.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors

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


class CoreSetInitialSelection(InitialSelectionStrategy):
    """Select initial samples via k-center greedy (core-set) coverage."""

    def __init__(
        self,
        seed: int,
        starting_batch_size: int,
        metric: str = "cosine",
        density_neighbors: int = 32,
    ) -> None:
        super().__init__("CORESET", starting_batch_size=starting_batch_size)
        self.seed = seed
        self.metric = metric
        self.density_neighbors = max(1, density_neighbors)
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        dataset: Dataset,
    ) -> List[int]:
        selected = self.k_center_greedy(dataset.embeddings)

        labels = dataset.labels[selected]
        logger.info(
            "%s_INITIAL: selected %d sequences. Labels: [%s]",
            self.name.upper(),
            len(selected),
            ", ".join(f"{val:.3f}" for val in labels),
        )

        return selected

    def k_center_greedy(self, embeddings: np.ndarray) -> List[int]:
        num_samples = embeddings.shape[0]
        if num_samples == 0 or self.starting_batch_size == 0:
            return []

        target = min(self.starting_batch_size, num_samples)
        density_scores = self._estimate_density_scores(embeddings)
        first_idx = self._select_initial_index(num_samples, density_scores)
        selected = [first_idx]

        min_distances = pairwise_distances(
            embeddings, embeddings[[first_idx]], metric=self.metric
        ).reshape(-1)

        while len(selected) < target:
            farthest_idx = int(np.argmax(min_distances))
            selected.append(farthest_idx)
            if len(selected) == target:
                break
            new_distances = pairwise_distances(
                embeddings, embeddings[[farthest_idx]], metric=self.metric
            ).reshape(-1)
            min_distances = np.minimum(min_distances, new_distances)

        return selected

    def _estimate_density_scores(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """Estimate local density via average distance to k nearest neighbors."""
        num_samples = embeddings.shape[0]
        if num_samples <= 1 or self.density_neighbors <= 1:
            return None
        n_neighbors = min(self.density_neighbors, num_samples)
        try:
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
            nn.fit(embeddings)
            distances, _ = nn.kneighbors(embeddings)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Failed to compute density scores for CORE-SET (metric=%s): %s",
                self.metric,
                exc,
            )
            return None

        if distances.shape[1] > 1:
            local_density = np.mean(distances[:, 1:], axis=1)
        else:
            local_density = distances[:, 0]
        return local_density

    def _select_initial_index(
        self, num_samples: int, density_scores: Optional[np.ndarray]
    ) -> int:
        if density_scores is None or not np.all(np.isfinite(density_scores)):
            return int(self._rng.integers(0, num_samples))
        return int(np.argmin(density_scores))


class DensityWeightedCoreSetInitialSelection(CoreSetInitialSelection):
    """K-center greedy that upweights dense regions when selecting new points."""

    def __init__(
        self,
        seed: int,
        starting_batch_size: int,
        metric: str = "cosine",
        density_neighbors: int = 32,
        density_scale: float = 1.0,
    ) -> None:
        super().__init__(
            seed=seed,
            starting_batch_size=starting_batch_size,
            metric=metric,
            density_neighbors=density_neighbors,
        )
        self.name = "CORESET_DENSITY"
        self.density_scale = max(0.0, density_scale)

    def k_center_greedy(self, embeddings: np.ndarray) -> List[int]:
        num_samples = embeddings.shape[0]
        if num_samples == 0 or self.starting_batch_size == 0:
            return []

        target = min(self.starting_batch_size, num_samples)
        density_scores = self._estimate_density_scores(embeddings)
        weights = self._build_density_weights(density_scores, num_samples)
        first_idx = self._select_initial_index(num_samples, density_scores)
        selected = [first_idx]

        min_distances = pairwise_distances(
            embeddings, embeddings[[first_idx]], metric=self.metric
        ).reshape(-1)

        while len(selected) < target:
            weighted_scores = min_distances * weights
            farthest_idx = int(np.argmax(weighted_scores))
            selected.append(farthest_idx)
            if len(selected) == target:
                break
            new_distances = pairwise_distances(
                embeddings, embeddings[[farthest_idx]], metric=self.metric
            ).reshape(-1)
            min_distances = np.minimum(min_distances, new_distances)

        return selected

    def _build_density_weights(
        self, density_scores: Optional[np.ndarray], num_samples: int
    ) -> np.ndarray:
        if density_scores is None or self.density_scale == 0.0:
            return np.ones(num_samples)

        eps = 1e-8
        inv_density = 1.0 / (density_scores + eps)
        max_val = np.max(inv_density)
        if max_val > 0:
            inv_density = inv_density / max_val
        return 1.0 + self.density_scale * inv_density


# TODO: implement a strategy where we first do hierarchical clustering on the embeddings, then select the centroids of the clusters as the initial samples. (may need to use minibatchkmeans since we have so many points)
