"""
Custom metrics calculation for active learning experiments.
"""

import logging
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates and tracks custom metrics for active learning experiments.
    """

    def __init__(self, labels: np.ndarray) -> None:
        """
        Initialize the metrics calculator.

        Args:
            labels: Array of all target values in the dataset
        """
        self.labels = labels
        self.cumulative_metrics: List[Dict[str, float]] = []

    def compute_metrics_for_round(
        self,
        selected_indices: np.ndarray,
        train_indices: np.ndarray,
        train_predictions: np.ndarray | None,
        pool_indices: np.ndarray,
        pool_predictions: np.ndarray | None,
        top_p: float = 0.1,
    ) -> Dict[str, float]:
        """
        Compute metrics for a single round.

        Args:
            selected_indices: Indices of selected sequences
            train_indices: Indices used to fit the model
            train_predictions: Model predictions for the training set
            pool_indices: Indices remaining in the unlabeled pool
            pool_predictions: Model predictions for the pool set
            top_p: Percentage of top labels to consider
        """
        # Number of selected samples within the top performers
        n_top = self.n_selected_in_top(
            selected_indices=selected_indices,
            top_p=top_p,
        )

        # Best value ground truth
        best_value_true = np.max(self.labels[selected_indices])
        normalized_true_values = best_value_true / np.max(self.labels)

        train_spearman = self._compute_spearman(
            indices=train_indices,
            predictions=train_predictions,
        )
        pool_spearman = self._compute_spearman(
            indices=pool_indices,
            predictions=pool_predictions,
        )
        if not np.isnan(train_spearman):
            pool_text = f"{pool_spearman:.3f}" if not np.isnan(pool_spearman) else "n/a"
            logger.info(
                "Train Spearman: %.3f, Pool Spearman: %s",
                train_spearman,
                pool_text,
            )

        return {
            "n_top": n_top,
            "best_true": self._round_metric(best_value_true),
            "normalized_true": self._round_metric(normalized_true_values),
            "train_spearman": self._round_metric(train_spearman),
            "pool_spearman": self._round_metric(pool_spearman),
        }

    def n_selected_in_top(
        self, selected_indices: np.ndarray, top_p: float = 0.01
    ) -> int:
        """
        Calculate how many selected samples fall into the top performers.

        Args:
            selected_indices: Indices of selected sequences
            top_p: Percentage of top labels to consider
        """
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")

        num_top = max(1, int(len(self.labels) * top_p))
        top_labels = np.argsort(self.labels)[-num_top:]
        return int(len(np.intersect1d(selected_indices, top_labels)))

    def _compute_spearman(
        self, indices: np.ndarray, predictions: np.ndarray | None
    ) -> float:
        if predictions is None or len(indices) < 2:
            return float("nan")
        labels = np.asarray(self.labels)[indices]
        preds = np.asarray(predictions)
        corr = spearmanr(labels, preds).correlation
        if corr is None:
            return float("nan")
        return float(corr)

    def _round_metric(self, value: float, digits: int = 6) -> float:
        if np.isnan(value):
            return value
        return round(float(value), digits)
