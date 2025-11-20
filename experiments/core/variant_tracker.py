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
        all_expressions: np.ndarray,
    ) -> None:
        """
        Initialize the variant tracker.

        Args:
            sample_ids: Identifiers for each sample in the dataset
            all_expressions: Array of all expression values
        """
        self.sample_ids = sample_ids
        self.all_expressions = all_expressions
        self.selected_variants: List[Dict[str, any]] = []

    def track_round(
        self,
        round_num: int,
        selected_indices: List[int],
        strategy: str,
        seed: int,
    ) -> None:
        """
        Track variants selected in a round.

        Args:
            round_num: Current round number
            selected_indices: Indices of selected variants
            strategy: Selection strategy name
            seed: Random seed used
        """
        for idx in selected_indices:
            variant_info = {
                "round": round_num,
                "strategy": strategy,
                "seed": seed,
                "variant_index": idx,
                "expression": float(self.all_expressions[idx]),
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
