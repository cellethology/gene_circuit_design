"""
Custom metrics calculation for active learning experiments.
"""

import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

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

        train_rmse, train_r2 = self._compute_performance_metrics(
            indices=train_indices,
            predictions=train_predictions,
        )
        pool_rmse, pool_r2 = self._compute_performance_metrics(
            indices=pool_indices,
            predictions=pool_predictions,
        )
        if not np.isnan(train_r2):
            pool_text = f"{pool_r2:.3f}" if not np.isnan(pool_r2) else "n/a"
            logger.info("Train R²: %.3f, Pool R²: %s", train_r2, pool_text)

        return {
            "n_top": n_top,
            "best_true": self._round_metric(best_value_true),
            "normalized_true": self._round_metric(normalized_true_values),
            "train_rmse": self._round_metric(train_rmse),
            "train_r2": self._round_metric(train_r2),
            "pool_rmse": self._round_metric(pool_rmse),
            "pool_r2": self._round_metric(pool_r2),
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

    def _compute_performance_metrics(
        self, indices: np.ndarray, predictions: np.ndarray | None
    ) -> tuple[float, float]:
        if predictions is None or len(indices) == 0:
            return float("nan"), float("nan")
        labels = np.asarray(self.labels)[indices]
        preds = np.asarray(predictions)
        rmse = root_mean_squared_error(labels, preds)
        r2 = r2_score(labels, preds)
        return float(rmse), float(r2)

    def _round_metric(self, value: float, digits: int = 6) -> float:
        if np.isnan(value):
            return value
        return round(float(value), digits)
