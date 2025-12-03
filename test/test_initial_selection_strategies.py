"""
Tests for initial selection strategies.
"""

import numpy as np

from experiments.core.data_loader import Dataset
from experiments.core.initial_selection_strategies import (
    KMeansInitialSelection,
    RandomInitialSelection,
)


def _create_dataset(n_samples: int, embedding_dim: int = 4) -> Dataset:
    sample_ids = [f"sample_{i}" for i in range(n_samples)]
    labels = np.linspace(0, 1, n_samples)
    embeddings = np.random.randn(n_samples, embedding_dim)
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
