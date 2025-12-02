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

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

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
        selected_indices: List[int],
        extra_info: Optional[str] = None,
    ) -> None:
        """
        Log information about the round.

        Args:
            round_idx: Round index
            selected_indices: Indices that were selected
            extra_info: Extra information to include in the log message
        """
        log_msg = f"Round {round_idx} - selected indices: {selected_indices}"
        if extra_info:
            log_msg += f" {extra_info}"
        logger.info(log_msg)


class Random(QueryStrategyBase):
    """Strategy that selects sequences randomly."""

    def __init__(self, seed: int) -> None:
        super().__init__("RANDOM")
        self.seed = seed

    def select(self, experiment: Any, round_idx: int) -> List[int]:
        unlabeled_pool = experiment.unlabeled_indices
        batch_size = min(experiment.batch_size, len(unlabeled_pool))
        selected_indices = random.Random(self.seed).sample(unlabeled_pool, batch_size)
        self._log_round(round_idx, selected_indices)
        return selected_indices


class TopPredictions(QueryStrategyBase):
    """Selects sequences with highest k predicted expression values."""

    def __init__(self) -> None:
        super().__init__("TOP_K_PRED")

    def select(self, experiment: Any, round_idx: int) -> List[int]:
        unlabeled = experiment.unlabeled_indices
        if len(unlabeled) < experiment.batch_size:
            return unlabeled

        preds = experiment.predictor.predict(
            experiment.dataset.embeddings[unlabeled, :]
        )
        k = experiment.batch_size

        # get indices of top k predictions (descending)
        top_k_local = np.argpartition(-preds, k - 1)[:k]

        # map to original indices
        selected_indices = [unlabeled[i] for i in top_k_local]
        self._log_round(round_idx, selected_indices)

        return selected_indices


class TopLogLikelihood(QueryStrategyBase):
    """Selects samples with highest zero-shot log likelihood values."""

    def __init__(self) -> None:
        super().__init__("TOP_LOG_LIKELIHOOD")

    def select(self, experiment: Any, round_idx: int) -> List[int]:
        log_likelihoods = experiment.all_log_likelihoods
        if log_likelihoods is None or np.all(np.isnan(log_likelihoods)):
            logger.warning("No log likelihood data available.")
            return []
        # Get log likelihood values for unlabeled sequences
        unlabeled_log_likelihoods = log_likelihoods[experiment.unlabeled_indices]

        # Filter out NaN values
        valid_mask = ~np.isnan(unlabeled_log_likelihoods)
        valid_unlabeled_indices = [
            experiment.unlabeled_indices[i]
            for i in range(len(experiment.unlabeled_indices))
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
            experiment.unlabeled_indices[i] for i in selected_local_indices
        ]

        selected_log_likelihoods = valid_log_likelihoods[selected_local_indices]
        selected_values = experiment.dataset.labels[selected_indices]
        extra_info = (
            f"Log likelihoods: "
            f"[{', '.join(f'{ll:.4f}' for ll in selected_log_likelihoods)}] "
            f"True values: [{', '.join(f'{expr:.1f}' for expr in selected_values)}]"
        )
        self._log_round(round_idx, selected_indices, extra_info)

        return selected_indices
