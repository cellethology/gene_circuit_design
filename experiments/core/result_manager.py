"""
Result saving utilities for active learning experiments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class ResultManager:
    """
    Handles saving experiment results to files.
    """

    def __init__(
        self,
        initial_sample_size: int,
        batch_size: int,
    ) -> None:
        """
        Initialize the result manager.

        Args:
            strategy: Selection strategy name
            predictor_name: Name of predictor
            seed: Random seed
            initial_sample_size: Initial training set size
            batch_size: Batch size for each round
        """
        self.initial_sample_size = initial_sample_size
        self.batch_size = batch_size

    def save_results(
        self,
        output_path: Path,
        results: List[Dict[str, Any]] = None,
        custom_metrics: List[Dict[str, Any]] = None,
        selected_variants: List[Dict[str, Any]] = None,
    ) -> None:
        """
        Save all experiment results to CSV files.

        Args:
            output_path: Base path for output files
            results: List of result dictionaries for each round
            custom_metrics: List of custom metric dictionaries
            selected_variants: List of selected variant dictionaries
        """
        # Save main results
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

        # Save custom metrics
        if custom_metrics:
            custom_metrics_path = output_path.with_name(
                output_path.stem + "_custom_metrics.csv"
            )
            custom_metrics_df = pd.DataFrame(custom_metrics)

            # Add metadata columns
            custom_metrics_df["round"] = range(1, len(custom_metrics) + 1)

            # Calculate train_size for each round
            train_sizes = []
            for i in range(len(custom_metrics)):
                if i == 0:
                    train_size = self.initial_sample_size
                else:
                    train_size = self.initial_sample_size + (i * self.batch_size)
                train_sizes.append(train_size)
            custom_metrics_df["train_size"] = train_sizes

            # Reorder columns
            metadata_cols = [
                "round",
                "train_size",
            ]
            cols = metadata_cols + [
                col for col in custom_metrics_df.columns if col not in metadata_cols
            ]
            custom_metrics_df = custom_metrics_df[cols]

            custom_metrics_df.to_csv(custom_metrics_path, index=False)
            logger.info(f"Custom metrics saved to {custom_metrics_path}")

        # Save selected variants
        if selected_variants:
            selected_variants_path = output_path.with_name(
                output_path.stem + "_selected_variants.csv"
            )
            selected_variants_df = pd.DataFrame(selected_variants)

            # Reorder columns
            metadata_cols = [
                "round",
                "variant_index",
                "sample_id",
                "expression",
            ]
            cols = metadata_cols + [
                col for col in selected_variants_df.columns if col not in metadata_cols
            ]
            selected_variants_df = selected_variants_df[cols]

            selected_variants_df.to_csv(selected_variants_path, index=False)
            logger.info(f"Selected variants saved to {selected_variants_path}")
