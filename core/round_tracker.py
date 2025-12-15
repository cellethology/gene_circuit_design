"""
Variant tracking utilities for active learning experiments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default metric columns to summarize (can be overridden per call).
DEFAULT_SUMMARY_METRICS = [
    "normalized_true",
    "normalized_pred",
    "n_selected_in_top",
]
# Aggregation rules (and descriptions) keyed by per-round metric names.
SUMMARY_METRIC_RULES = {
    # Average of the cumulative best normalized label observed so far.
    "normalized_true": (
        "cumulative_max_mean",
        "Best normalized ground truth per round.",
    ),
    # Average of the cumulative best predictions observed so far.
    "normalized_pred": ("cumulative_max_mean", "Best normalized prediction per round."),
    # Fractional area under curve of how many selected samples fall in the top bucket.
    "n_selected_in_top": (
        "normalized_cumulative_sum",
        "Selected samples in top bracket.",
    ),
    # The raw best ground-truth value (used in some tests); defaults to cumulative max.
    "best_true": ("cumulative_max_mean", "Best absolute ground truth per round."),
    "best_pred": ("cumulative_max_mean", "Best absolute prediction per round."),
}


class RoundTracker:
    """
    Tracks selected samples across active learning rounds.
    """

    def __init__(self, sample_ids: np.ndarray) -> None:
        """
        Initialize the round tracker.

        Args:
            sample_ids: Identifiers for each sample in the dataset
        """
        self.sample_ids = sample_ids
        self.rounds: List[Dict[str, Any]] = []
        self.round_num = 0

    def track_round(
        self,
        selected_indices: List[int],
        metrics: Dict[str, float],
    ) -> None:
        """
        Track samples selected in a round.

        Args:
            selected_indices: Indices of selected samples
            metrics: Metrics for the round
        """
        train_size = (
            self.rounds[-1]["train_size"] + len(self.rounds[-1]["selected_sample_ids"])
            if self.rounds
            else 0
        )
        unlabeled_pool_size = len(self.sample_ids) - train_size - len(selected_indices)
        selected_ids = [int(self.sample_ids[idx]) for idx in selected_indices]
        self.rounds.append(
            {
                "round": self.round_num,
                "train_size": train_size,
                "unlabeled_pool_size": unlabeled_pool_size,
                "selected_sample_ids": selected_ids,
                **metrics,
            }
        )
        self.round_num += 1

    def compute_summary_metrics(
        self, metric_columns: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute summary metrics defined by `metric_columns`.
        The default list captures the common Hydra sweep outputs and describes the aggregation
        strategy for each metric via `SUMMARY_METRIC_RULES`.
        """
        if not self.rounds:
            raise ValueError(
                "Cannot compute summary metrics: no rounds have been tracked yet"
            )

        cumulative_selected = np.sum(
            np.cumsum(
                np.array([len(round["selected_sample_ids"]) for round in self.rounds])
            )
        )
        metric_columns = metric_columns or DEFAULT_SUMMARY_METRICS
        summary_values: Dict[str, float] = {}

        for metric_column in metric_columns:
            if metric_column not in self.rounds[0]:
                raise ValueError(f"Metric column {metric_column} not found in rounds")

            values = np.array([round[metric_column] for round in self.rounds])
            rule, _ = SUMMARY_METRIC_RULES.get(
                metric_column, ("cumulative_max_mean", "")
            )
            if rule == "normalized_cumulative_sum":
                cumulative_sum = np.cumsum(values)
                summary_values[metric_column] = (
                    float(np.sum(cumulative_sum)) / cumulative_selected
                    if cumulative_selected > 0
                    else 0.0
                )
            elif rule == "cumulative_max_mean":
                cumulative_max_per_round = np.maximum.accumulate(values)
                summary_values[metric_column] = float(
                    np.sum(cumulative_max_per_round)
                ) / len(self.rounds)
            elif rule == "max_overall":
                summary_values[metric_column] = float(np.max(values))
            else:
                raise ValueError(
                    f"Unknown summary metric rule '{rule}' for {metric_column}"
                )
        return summary_values

    def save_to_csv(self, output_path: Path) -> None:
        """
        Save rounds to CSV file.

        Args:
            output_path: Path to save rounds
        """
        df = pd.DataFrame(self.rounds)
        df.to_csv(output_path, index=False)
        logger.info(f"Rounds saved to {output_path}")
