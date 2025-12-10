"""
Unit tests for query strategy implementations.
"""

from dataclasses import dataclass

import numpy as np

from core.query_strategies import (
    PredStdHybrid,
    Random,
    TopLogLikelihood,
    TopPredictions,
)


@dataclass
class DummyDataset:
    embeddings: np.ndarray
    labels: np.ndarray


class DummyTrainer:
    def __init__(self, outputs: np.ndarray, stds: np.ndarray = None) -> None:
        self._outputs = outputs
        self._stds = stds if stds is not None else np.zeros_like(outputs)

    def predict(self, _, return_std: bool = False):
        if return_std:
            return self._outputs, self._stds
        return self._outputs


class DummyExperiment:
    def __init__(
        self,
        unlabeled_indices,
        batch_size,
        embeddings=None,
        labels=None,
        trainer=None,
        log_likelihoods=None,
    ):
        self.unlabeled_indices = unlabeled_indices
        self.batch_size = batch_size
        self.dataset = DummyDataset(
            embeddings=embeddings
            if embeddings is not None
            else np.zeros((len(unlabeled_indices), 1)),
            labels=labels if labels is not None else np.zeros(len(unlabeled_indices)),
        )
        self.trainer = trainer or DummyTrainer(
            np.zeros(len(unlabeled_indices), dtype=float)
        )
        self.all_log_likelihoods = log_likelihoods


class TestRandomStrategy:
    def test_random_sampling_is_reproducible(self):
        exp = DummyExperiment(unlabeled_indices=list(range(10)), batch_size=4)
        strategy = Random(seed=42)
        first = strategy.select(exp)
        second = strategy.select(exp)

        assert first == second
        assert len(first) == 4
        assert len(set(first)) == 4


class TestTopPredictionsStrategy:
    def test_top_predictions_selects_highest_scores(self):
        unlabeled = [0, 1, 2, 3]
        embeddings = np.eye(4)
        trainer = DummyTrainer(np.array([0.1, 0.9, 0.2, 0.8]))
        exp = DummyExperiment(
            unlabeled_indices=unlabeled,
            batch_size=2,
            embeddings=embeddings,
            trainer=trainer,
        )

        selected = TopPredictions().select(exp)
        assert sorted(selected) == [1, 3]

    def test_top_predictions_returns_all_when_pool_small(self):
        unlabeled = [0, 1]
        trainer = DummyTrainer(np.array([0.5, 0.4]))
        exp = DummyExperiment(
            unlabeled_indices=unlabeled, batch_size=5, trainer=trainer
        )

        assert TopPredictions().select(exp) == unlabeled


class TestTopLogLikelihoodStrategy:
    def test_returns_empty_when_no_log_likelihoods(self, caplog):
        exp = DummyExperiment(
            unlabeled_indices=[0, 1], batch_size=2, log_likelihoods=None
        )
        selected = TopLogLikelihood().select(exp)
        assert selected == []
        assert "No log likelihood data available" in caplog.text

    def test_returns_empty_when_no_valid_entries(self, caplog):
        log_likelihoods = np.array([np.nan, np.nan, np.nan, 0.3])
        exp = DummyExperiment(
            unlabeled_indices=[0, 1, 2],
            batch_size=2,
            log_likelihoods=log_likelihoods,
        )
        selected = TopLogLikelihood().select(exp)
        assert selected == []
        assert "No valid log likelihood values" in caplog.text

    def test_selects_highest_log_likelihood(self):
        unlabeled = [0, 1, 2, 3]
        log_likelihoods = np.array([0.1, np.nan, 0.9, 0.5])
        labels = np.array([10.0, 11.0, 12.0, 13.0])
        exp = DummyExperiment(
            unlabeled_indices=unlabeled,
            batch_size=2,
            labels=labels,
            log_likelihoods=log_likelihoods,
        )

        selected = TopLogLikelihood().select(exp)
        assert sorted(selected) == [2, 3]


class TestPredStdHybridStrategy:
    def test_pred_std_hybrid_selects_based_on_weighted_score(self):
        unlabeled = [0, 1, 2, 3]
        embeddings = np.eye(4)
        # High pred, low std -> score = 0.75 * 0.9 + 0.25 * 0.1 = 0.7
        # Low pred, high std -> score = 0.75 * 0.1 + 0.25 * 0.9 = 0.3
        preds = np.array([0.1, 0.9, 0.2, 0.8])
        stds = np.array([0.9, 0.1, 0.8, 0.2])
        trainer = DummyTrainer(preds, stds)
        exp = DummyExperiment(
            unlabeled_indices=unlabeled,
            batch_size=2,
            embeddings=embeddings,
            trainer=trainer,
        )

        # With alpha=0.75, should prefer high predictions
        selected = PredStdHybrid(alpha=0.75).select(exp)
        assert sorted(selected) == [1, 3]

    def test_pred_std_hybrid_returns_all_when_pool_small(self):
        unlabeled = [0, 1]
        preds = np.array([0.5, 0.4])
        stds = np.array([0.1, 0.2])
        trainer = DummyTrainer(preds, stds)
        exp = DummyExperiment(
            unlabeled_indices=unlabeled, batch_size=5, trainer=trainer
        )

        assert PredStdHybrid(alpha=0.75).select(exp) == unlabeled
