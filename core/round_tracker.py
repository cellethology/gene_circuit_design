"""
Variant tracking utilities for active learning experiments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

    def compute_auc(self, metric_columns: List[str]) -> Dict[str, float]:
        """
        Compute the AUC across all rounds.
        The AUC is computed by summing up the maximum value of the metric column up to that round.
        """
        if not self.rounds:
            raise ValueError("Cannot compute AUC: no rounds have been tracked yet")

        aucs = {}
        cumulative_selected = np.sum(
            np.cumsum(
                np.array([len(round["selected_sample_ids"]) for round in self.rounds])
            )
        )

        for metric_column in metric_columns:
            if metric_column not in self.rounds[0].keys():
                raise ValueError(f"Metric column {metric_column} not found in rounds")

            values = np.array([round[metric_column] for round in self.rounds])
            if metric_column == "n_selected_in_top":
                cumulative_sum = np.cumsum(values)
                aucs[metric_column] = (
                    float(np.sum(cumulative_sum)) / cumulative_selected
                    if cumulative_selected > 0
                    else 0.0
                )
            else:
                cumulative_max_per_round = np.maximum.accumulate(values)
                aucs[metric_column] = float(np.sum(cumulative_max_per_round)) / len(
                    self.rounds
                )
        return aucs

    def save_to_csv(self, output_path: Path) -> None:
        """
        Save rounds to CSV file.

        Args:
            output_path: Path to save rounds
        """
        df = pd.DataFrame(self.rounds)
        df.to_csv(output_path, index=False)
        logger.info(f"Rounds saved to {output_path}")
