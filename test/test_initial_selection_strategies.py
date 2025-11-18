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
    sequences = [f"seq_{i}" for i in range(n_samples)]
    labels = np.linspace(0, 1, n_samples)
    log_likelihoods = np.zeros(n_samples)
    embeddings = np.random.randn(n_samples, embedding_dim)
    return Dataset(
        sequences=sequences,
        sequence_labels=labels,
        log_likelihoods=log_likelihoods,
        embeddings=embeddings,
        variant_ids=None,
    )


def test_random_initial_selection_basic():
    dataset = _create_dataset(20)
    strategy = RandomInitialSelection()

    indices = strategy.select(
        dataset=dataset,
        initial_sample_size=5,
        random_seed=123,
        encode_sequences_fn=lambda idxs: dataset.embeddings[idxs],
    )

    assert len(indices) == 5
    assert len(set(indices)) == 5


def test_kmeans_initial_selection_returns_expected_count():
    dataset = _create_dataset(15, embedding_dim=3)
    strategy = KMeansInitialSelection()

    indices = strategy.select(
        dataset=dataset,
        initial_sample_size=6,
        random_seed=42,
        encode_sequences_fn=lambda idxs: dataset.embeddings[idxs],
    )

    assert len(indices) == 6
    assert len(set(indices)) == 6
