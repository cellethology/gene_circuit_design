"""
Variant tracking utilities for active learning experiments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


SUMMARY_METRIC_RULES = {
    "auc_true": ("max_accumulate", "normalized_true"),
    "auc_pred": ("max_accumulate", "normalized_pred"),
    "avg_top": ("top_mean", "n_selected_in_top"),
    "overall_true": ("max_overall", "normalized_true"),
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

    def compute_summary_metrics(self) -> Dict[str, float]:
        """
        Compute summary metrics defined by `SUMMARY_METRIC_RULES`.
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

        summary_values: Dict[str, float] = {}

        for metric_name, (rule, metric_column) in SUMMARY_METRIC_RULES.items():
            if metric_column not in self.rounds[0]:
                raise ValueError(f"Metric column {metric_name} not found in rounds")

            values = np.array([round[metric_column] for round in self.rounds])
            if rule == "top_mean":
                cumulative_sum = np.cumsum(values)
                summary_values[metric_name] = (
                    float(np.sum(cumulative_sum)) / cumulative_selected
                    if cumulative_selected > 0
                    else 0.0
                )
            elif rule == "max_accumulate":
                cumulative_max_per_round = np.maximum.accumulate(values)
                summary_values[metric_name] = float(
                    np.sum(cumulative_max_per_round)
                ) / len(self.rounds)
            elif rule == "max_overall":
                summary_values[metric_name] = float(np.max(values))
            else:
                raise ValueError(
                    f"Unknown summary metric rule '{rule}' for {metric_name}"
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
