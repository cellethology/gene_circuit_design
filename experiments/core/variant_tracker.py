"""
Variant tracking utilities for active learning experiments.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VariantTracker:
    """
    Tracks selected variants across active learning rounds.
    """

    def __init__(
        self,
        all_expressions: np.ndarray,
        all_log_likelihoods: np.ndarray,
        all_sequences: List[str],
        variant_ids: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the variant tracker.

        Args:
            all_expressions: Array of all expression values
            all_log_likelihoods: Array of all log likelihood values
            all_sequences: List of all sequences
            variant_ids: Optional array of variant IDs
        """
        self.all_expressions = all_expressions
        self.all_log_likelihoods = all_log_likelihoods
        self.all_sequences = all_sequences
        self.variant_ids = variant_ids
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
                "log_likelihood": (
                    float(self.all_log_likelihoods[idx])
                    if not np.isnan(self.all_log_likelihoods[idx])
                    else None
                ),
            }

            # Add variant ID if available
            if self.variant_ids is not None:
                variant_info["variant_id"] = (
                    int(self.variant_ids[idx])
                    if not np.isnan(self.variant_ids[idx])
                    else None
                )
            else:
                variant_info["variant_id"] = f"variant_{idx}"

            # Add sequence (truncated for readability)
            if idx < len(self.all_sequences):
                sequence = str(self.all_sequences[idx])
                variant_info["sequence"] = (
                    sequence[:50] + "..." if len(sequence) > 50 else sequence
                )
            else:
                variant_info["sequence"] = f"seq_{idx}"

            self.selected_variants.append(variant_info)

    def get_all_variants(self) -> List[Dict[str, any]]:
        """
        Get all tracked variants.

        Returns:
            List of variant dictionaries
        """
        return self.selected_variants.copy()
