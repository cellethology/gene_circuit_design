"""
Variant tracking utilities for active learning experiments.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class VariantTracker:
    """
    Tracks selected variants across active learning rounds.
    """

    def __init__(
        self,
        sample_ids: List[str],
        all_labels: np.ndarray,
    ) -> None:
        """
        Initialize the variant tracker.

        Args:
            sample_ids: Identifiers for each sample in the dataset
            all_labels: Array of all label values
        """
        self.sample_ids = sample_ids
        self.all_labels = all_labels
        self.selected_variants: List[Dict[str, any]] = []

    def track_round(
        self,
        round_num: int,
        selected_indices: List[int],
    ) -> None:
        """
        Track variants selected in a round.

        Args:
            round_num: Current round number
            selected_indices: Indices of selected variants
        """
        for idx in selected_indices:
            variant_info = {
                "round": round_num,
                "variant_index": idx,
                "expression": float(self.all_labels[idx]),
                "sample_id": self.sample_ids[idx],
            }

            self.selected_variants.append(variant_info)

    def get_all_variants(self) -> List[Dict[str, any]]:
        """
        Get all tracked variants.

        Returns:
            List of variant dictionaries
        """
        return self.selected_variants.copy()
