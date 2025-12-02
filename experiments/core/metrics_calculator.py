"""
Custom metrics calculation for active learning experiments.
"""

import logging
from typing import Dict, List

import numpy as np

from utils.metrics import (
    proportion_of_selected_indices_in_top_labels,
)

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

    def calculate_round_metrics(
        self,
        selected_indices: np.ndarray,
        predictions: np.ndarray,
        top_percentage: float = 0.1,
    ) -> Dict[str, float]:
        """
        Calculate metrics for a single round.

        Args:
            selected_indices: Indices of selected sequences
            predictions: Model predictions for selected indices
            top_percentage: Percentage of top labels to consider
        """
        # Proportion of selected indices in top labels
        top_proportion = proportion_of_selected_indices_in_top_labels(
            selected_indices, self.labels, top_percentage=top_percentage
        )

        # Best value predictions
        best_value_pred = np.max(predictions)
        normalized_pred_values = best_value_pred / np.max(self.labels)

        # Best value ground truth
        best_value_true = np.max(self.labels[selected_indices])
        normalized_true_values = best_value_true / np.max(self.labels)

        return {
            "top_proportion": top_proportion,
            "best_pred": best_value_pred,
            "normalized_pred": normalized_pred_values,
            "best_true": best_value_true,
            "normalized_true": normalized_true_values,
        }

    def update_cumulative(self, round_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Update cumulative metrics based on current round metrics.

        Args:
            round_metrics: Metrics from current round

        Returns:
            Dictionary with cumulative metrics added
        """
        if len(self.cumulative_metrics) > 0:
            # Calculate cumulative metrics
            prev_metrics = self.cumulative_metrics[-1]

            cumulative_metrics = {
                "top_proportion_cumulative": (
                    prev_metrics["top_proportion_cumulative"]
                    + round_metrics["top_proportion"]
                ),
                "best_pred_cumulative": max(
                    prev_metrics["best_pred_cumulative"],
                    round_metrics["best_pred"],
                ),
                "normalized_pred_cumulative": max(
                    prev_metrics["normalized_pred_cumulative"],
                    round_metrics["normalized_pred"],
                ),
                "best_true_cumulative": max(
                    prev_metrics["best_true_cumulative"],
                    round_metrics["best_true"],
                ),
                "normalized_true_cumulative": max(
                    prev_metrics["normalized_true_cumulative"],
                    round_metrics["normalized_true"],
                ),
            }
        else:
            # First round: cumulative equals current
            cumulative_metrics = {
                "top_proportion_cumulative": round_metrics["top_proportion"],
                "best_pred_cumulative": round_metrics["best_pred"],
                "normalized_pred_cumulative": round_metrics["normalized_pred"],
                "best_true_cumulative": round_metrics["best_true"],
                "normalized_true_cumulative": round_metrics["normalized_true"],
            }

        # Combine round and cumulative metrics
        combined_metrics = {**round_metrics, **cumulative_metrics}
        self.cumulative_metrics.append(combined_metrics)

        return combined_metrics

    def get_all_metrics(self) -> List[Dict[str, float]]:
        """
        Get all calculated metrics.

        Returns:
            List of metric dictionaries for each round
        """
        return self.cumulative_metrics.copy()
