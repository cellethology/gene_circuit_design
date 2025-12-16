"""
Custom metrics calculation for active learning experiments.
"""

import logging
from typing import Dict, List

import numpy as np

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
        predictions: np.ndarray,
        top_p: float = 0.1,
    ) -> Dict[str, float]:
        """
        Compute metrics for a single round.

        Args:
            selected_indices: Indices of selected sequences
            predictions: Model predictions for selected indices
            top_p: Percentage of top labels to consider
        """
        # Number of selected samples within the top performers
        n_selected_in_top = self.n_selected_in_top(
            selected_indices=selected_indices,
            top_p=top_p,
        )

        # Best value predictions
        best_value_pred = np.max(predictions)
        normalized_pred_values = best_value_pred / np.max(self.labels)

        # Best value ground truth
        best_value_true = np.max(self.labels[selected_indices])
        normalized_true_values = best_value_true / np.max(self.labels)

        return {
            "n_selected_in_top": n_selected_in_top,
            "best_pred": best_value_pred,
            "normalized_pred": normalized_pred_values,
            "best_true": best_value_true,
            "normalized_true": normalized_true_values,
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
