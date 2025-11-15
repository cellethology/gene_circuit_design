"""
Selection strategy implementations using the Strategy pattern.

This module provides a clean, extensible way to implement different
selection strategies for active learning experiments.
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class QueryStrategyBase(ABC):
    """
    Abstract base class for selection strategies.

    Each concrete strategy implements the `select` method to choose
    the next batch of sequences based on its specific criteria.
    """

    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def select(self, experiment: Any, round_idx: int) -> List[int]:
        """
        Select the next batch of sequences.

        Returns:
            List of selected indices from unlabeled_indices
        """
        pass

    def _log_round(
        self,
        round_idx: int,
        strategy_name: str,
        selected_indices: List[int],
        extra_info: Optional[str] = None,
    ) -> None:
        """
        Log information about the round.

        Args:
            round_idx: Round index
            strategy_name: Name of the strategy for logging
            selected_indices: Indices that were selected
            extra_info: Extra information to include in the log message
        """
        log_msg = (
            f"Round {round_idx}: Selected {len(selected_indices)} sequences "
            f"with strategy: {strategy_name}"
        )
        if extra_info:
            log_msg += f" {extra_info}"
        logger.info(log_msg)


class Random(QueryStrategyBase):
    """Strategy that selects sequences randomly."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.seed = seed

    def select(self, experiment: Any, round_idx: int) -> List[int]:
        batch_size = min(
            experiment.batch_size, len(experiment.data_split.unlabeled_indices)
        )
        selected_indices = random.Random(self.seed).sample(
            experiment.data_split.unlabeled_indices, batch_size
        )
        self._log_round(round_idx, "RANDOM", selected_indices)
        return selected_indices


class TopKPredictions(QueryStrategyBase):
    """Strategy that selects sequences with highest k predicted expression values."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def select(self, experiment: Any, round_idx: int) -> List[int]:
        # Predict on all unlabeled sequences
        X_unlabeled = experiment._encode_sequences(
            experiment.data_split.unlabeled_indices
        )
        predictions = experiment.model.predict(X_unlabeled)

        # Select indices with highest predicted values
        sorted_indices = np.argsort(predictions)[::-1]  # Descending order
        batch_size = min(
            experiment.batch_size, len(experiment.data_split.unlabeled_indices)
        )
        selected_local_indices = sorted_indices[:batch_size]

        # Convert to global indices
        selected_indices = [
            experiment.data_split.unlabeled_indices[i] for i in selected_local_indices
        ]

        # Log selection info
        selected_predictions = predictions[selected_local_indices]
        extra_info = (
            f"with predicted expressions: "
            f"[{', '.join(f'{pred:.1f}' for pred in selected_predictions)}]"
        )
        self._log_round(round_idx, "HIGH_EXPRESSION", selected_indices, extra_info)

        return selected_indices


class TopZeroShot(QueryStrategyBase):
    """Strategy that selects sequences with highest zero-shot log likelihood values."""

    def select(self, experiment: Any, round_idx: int) -> List[int]:
        if np.all(np.isnan(experiment.dataset.log_likelihoods)):
            logger.warning("No log likelihood data available.")
            return []
        # Get log likelihood values for unlabeled sequences
        unlabeled_log_likelihoods = experiment.dataset.log_likelihoods[
            experiment.data_split.unlabeled_indices
        ]

        # Filter out NaN values
        valid_mask = ~np.isnan(unlabeled_log_likelihoods)
        valid_unlabeled_indices = [
            experiment.data_split.unlabeled_indices[i]
            for i in range(len(experiment.data_split.unlabeled_indices))
            if valid_mask[i]
        ]
        valid_log_likelihoods = unlabeled_log_likelihoods[valid_mask]

        if len(valid_unlabeled_indices) == 0:
            logger.warning("No valid log likelihood values for unlabeled sequences. ")
            return []

        # Select indices with highest log likelihood values
        sorted_indices = np.argsort(valid_log_likelihoods)[::-1]
        batch_size = min(experiment.batch_size, len(valid_unlabeled_indices))
        selected_local_indices = sorted_indices[:batch_size]

        selected_indices = [
            experiment.data_split.unlabeled_indices[i] for i in selected_local_indices
        ]

        selected_log_likelihoods = valid_log_likelihoods[selected_local_indices]
        selected_values = experiment.dataset.sequence_labels[selected_indices]
        extra_info = (
            f"with log likelihoods: "
            f"[{', '.join(f'{ll:.4f}' for ll in selected_log_likelihoods)}] "
            f"and actual labels: [{', '.join(f'{expr:.1f}' for expr in selected_values)}]"
        )
        self._log_round(round_idx, "LOG_LIKELIHOOD", selected_indices, extra_info)

        return selected_indices
