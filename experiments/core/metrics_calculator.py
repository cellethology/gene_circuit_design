"""
Custom metrics calculation for active learning experiments.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from utils.config_loader import SelectionStrategy
from utils.metrics import (
    get_best_value_metric,
    normalized_to_best_val_metric,
    top_10_ratio_intersected_indices_metric,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates and tracks custom metrics for active learning experiments.
    """

    def __init__(self, all_expressions: np.ndarray) -> None:
        """
        Initialize the metrics calculator.

        Args:
            all_expressions: Array of all expression values in the dataset
        """
        self.all_expressions = all_expressions
        self.cumulative_metrics: List[Dict[str, float]] = []

    def calculate_round_metrics(
        self,
        selected_indices: List[int],
        selection_strategy: SelectionStrategy,
        predictions: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate metrics for a single round.

        Args:
            selected_indices: Indices of selected sequences
            selection_strategy: Current selection strategy
            predictions: Model predictions for selected indices (optional)

        Returns:
            Dictionary with round metrics
        """
        # Top 10 ratio intersection
        top_10_ratio_intersection = top_10_ratio_intersected_indices_metric(
            selected_indices, self.all_expressions
        )

        # Get true values
        y_true = self.all_expressions[selected_indices]
        best_value_true = get_best_value_metric(y_true)
        normalized_predictions_true = normalized_to_best_val_metric(
            y_true, self.all_expressions
        )

        # Get prediction metrics
        if selection_strategy == SelectionStrategy.LOG_LIKELIHOOD:
            # For LOG_LIKELIHOOD, use true values to maintain model independence
            best_value_pred = best_value_true
            normalized_predictions_pred = normalized_predictions_true
        else:
            # For other strategies, use model predictions if available
            if predictions is not None:
                best_value_pred = get_best_value_metric(predictions)
                normalized_predictions_pred = normalized_to_best_val_metric(
                    predictions, self.all_expressions
                )
            else:
                # Fallback to true values if predictions not available
                best_value_pred = best_value_true
                normalized_predictions_pred = normalized_predictions_true

        return {
            "top_10_ratio_intersected_indices": top_10_ratio_intersection,
            "best_value_predictions_values": best_value_pred,
            "normalized_predictions_predictions_values": normalized_predictions_pred,
            "best_value_ground_truth_values": best_value_true,
            "normalized_predictions_ground_truth_values": normalized_predictions_true,
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
                "top_10_ratio_intersected_indices_cumulative": (
                    prev_metrics["top_10_ratio_intersected_indices_cumulative"]
                    + round_metrics["top_10_ratio_intersected_indices"]
                ),
                "best_value_predictions_values_cumulative": max(
                    prev_metrics["best_value_predictions_values_cumulative"],
                    round_metrics["best_value_predictions_values"],
                ),
                "normalized_predictions_predictions_values_cumulative": max(
                    prev_metrics[
                        "normalized_predictions_predictions_values_cumulative"
                    ],
                    round_metrics["normalized_predictions_predictions_values"],
                ),
                "best_value_ground_truth_values_cumulative": max(
                    prev_metrics["best_value_ground_truth_values_cumulative"],
                    round_metrics["best_value_ground_truth_values"],
                ),
                "normalized_predictions_ground_truth_values_cumulative": max(
                    prev_metrics[
                        "normalized_predictions_ground_truth_values_cumulative"
                    ],
                    round_metrics["normalized_predictions_ground_truth_values"],
                ),
            }
        else:
            # First round: cumulative equals current
            cumulative_metrics = {
                "top_10_ratio_intersected_indices_cumulative": round_metrics[
                    "top_10_ratio_intersected_indices"
                ],
                "best_value_predictions_values_cumulative": round_metrics[
                    "best_value_predictions_values"
                ],
                "normalized_predictions_predictions_values_cumulative": round_metrics[
                    "normalized_predictions_predictions_values"
                ],
                "best_value_ground_truth_values_cumulative": round_metrics[
                    "best_value_ground_truth_values"
                ],
                "normalized_predictions_ground_truth_values_cumulative": round_metrics[
                    "normalized_predictions_ground_truth_values"
                ],
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
