"""
Tests for initial selection strategies.
"""

import numpy as np

from core.data_loader import Dataset
from core.initial_selection_strategies import (
    CoreSetInitialSelection,
    DensityWeightedCoreSetInitialSelection,
    KMeansInitialSelection,
    RandomInitialSelection,
)


def _create_dataset(n_samples: int, embedding_dim: int = 4) -> Dataset:
    sample_ids = [f"sample_{i}" for i in range(n_samples)]
    labels = np.linspace(0, 1, n_samples)
    embeddings = np.random.randn(n_samples, embedding_dim)
    return Dataset(sample_ids=sample_ids, labels=labels, embeddings=embeddings)


def _dataset_from_embeddings(embeddings) -> Dataset:
    embeddings = np.asarray(embeddings, dtype=float)
    sample_ids = [f"sample_{i}" for i in range(len(embeddings))]
    labels = np.linspace(0, 1, len(embeddings))
    return Dataset(sample_ids=sample_ids, labels=labels, embeddings=embeddings)


def test_random_initial_selection_basic():
    dataset = _create_dataset(20)
    strategy = RandomInitialSelection(seed=123, starting_batch_size=5)

    indices = strategy.select(dataset=dataset)

    assert len(indices) == 5
    assert len(set(indices)) == 5


def test_kmeans_initial_selection_returns_expected_count():
    dataset = _create_dataset(15, embedding_dim=3)
    strategy = KMeansInitialSelection(seed=42, starting_batch_size=6)

    indices = strategy.select(dataset=dataset)

    assert len(indices) == 6
    assert len(set(indices)) == 6


def test_core_set_initial_selection_prefers_dense_seed():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.05, 0.0],
            [0.0, 0.05],
            [5.0, 5.0],
            [6.0, 6.0],
        ],
        dtype=float,
    )
    dataset = _dataset_from_embeddings(embeddings)
    # Use a batch size equal to the number of unique clusters to avoid duplicates
    strategy = CoreSetInitialSelection(
        seed=0, starting_batch_size=2, density_neighbors=3
    )

    indices = strategy.select(dataset=dataset)

    density_scores = strategy._estimate_density_scores(dataset.embeddings)
    assert density_scores is not None
    expected_first = int(np.argmin(density_scores))

    assert indices[0] == expected_first
    assert len(indices) == 2
    assert len(set(indices)) == 2


def test_density_weighted_core_set_applies_weights(monkeypatch):
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [0.9, 0.0],
        ],
        dtype=float,
    )
    dataset = _dataset_from_embeddings(embeddings)
    strategy = DensityWeightedCoreSetInitialSelection(
        seed=0, starting_batch_size=2, density_scale=1.0
    )

    def fake_density_scores(self, _embeddings):
        return np.array([0.0, 1.0, 2.0], dtype=float)

    def fake_density_weights(self, _scores, num_samples):
        assert num_samples == 3
        return np.array([1.0, 50.0, 1.0], dtype=float)

    monkeypatch.setattr(
        DensityWeightedCoreSetInitialSelection,
        "_estimate_density_scores",
        fake_density_scores,
    )
    monkeypatch.setattr(
        DensityWeightedCoreSetInitialSelection,
        "_build_density_weights",
        fake_density_weights,
    )

    indices = strategy.select(dataset=dataset)

    assert indices == [0, 1]
