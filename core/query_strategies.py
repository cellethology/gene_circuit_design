"""
Query strategy implementations.

This module provides a clean, extensible way to implement different
selection strategies for active learning experiments.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class QueryStrategyBase(ABC):
    """
    Abstract base class for query strategies.

    Each concrete strategy implements the `_select_batch` method to choose
    the next batch of samples based on its specific criteria.
    """

    requires_model: bool = True

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def select(self, experiment: Any) -> List[int]:
        """
        Select the next batch of samples (template method).

        This method handles bounds checking and delegates to _select_batch()
        for the actual selection logic.

        Returns:
            List of selected indices from unlabeled_indices
        """
        unlabeled_pool = experiment.unlabeled_indices
        batch_size = experiment.batch_size

        # Early return if pool is smaller than batch size
        if len(unlabeled_pool) < batch_size:
            return unlabeled_pool

        # Delegate to strategy-specific selection logic
        selected_indices = self._select_batch(experiment, unlabeled_pool, batch_size)
        self._log_round(selected_indices)
        return selected_indices

    @abstractmethod
    def _select_batch(
        self, experiment: Any, unlabeled_pool: List[int], batch_size: int
    ) -> List[int]:
        """
        Strategy-specific batch selection logic.

        Args:
            experiment: The active learning experiment
            unlabeled_pool: List of unlabeled indices (guaranteed to have length >= batch_size)
            batch_size: Number of samples to select (guaranteed to be <= len(unlabeled_pool))

        Returns:
            List of selected indices from unlabeled_pool
        """
        pass

    def _log_round(
        self,
        selected_indices: List[int],
        extra_info: Optional[str] = None,
    ) -> None:
        """
        Log information about the round.

        Args:
            selected_indices: Indices that were selected
            extra_info: Extra information to include in the log message
        """
        log_msg = f"Selected indices: {selected_indices}"
        if extra_info:
            log_msg += f" {extra_info}"
        logger.info(log_msg)


class Random(QueryStrategyBase):
    """Strategy that selects samples randomly."""

    def __init__(self, seed: int) -> None:
        super().__init__("RANDOM")
        self.seed = seed
        self.requires_model = False

    def _select_batch(
        self, experiment: Any, unlabeled_pool: List[int], batch_size: int
    ) -> List[int]:
        rng = np.random.default_rng(self.seed)
        selected_indices = rng.choice(
            unlabeled_pool, batch_size, replace=False
        ).tolist()
        return selected_indices


class TopPredictions(QueryStrategyBase):
    """Selects samples with highest k predicted label values."""

    def __init__(self) -> None:
        super().__init__("TOP_K_PRED")

    def _select_batch(
        self, experiment: Any, unlabeled_pool: List[int], batch_size: int
    ) -> List[int]:
        preds = experiment.trainer.predict(
            experiment.dataset.embeddings[unlabeled_pool, :]
        )

        # get indices of top k predictions (descending)
        top_k_local = np.argpartition(-preds, batch_size - 1)[:batch_size]

        # map to original indices
        selected_indices = [unlabeled_pool[i] for i in top_k_local]
        return selected_indices


class PredStdHybrid(QueryStrategyBase):
    """Selects samples with highest combined prediction and uncertainty."""

    def __init__(self, alpha: float) -> None:
        super().__init__("PRED_STD_HYBRID")
        self.alpha = alpha

    def _select_batch(
        self, experiment: Any, unlabeled_pool: List[int], batch_size: int
    ) -> List[int]:
        # get weighted sum of prediction and standard deviation of prediction
        preds, stds = experiment.trainer.predict(
            experiment.dataset.embeddings[unlabeled_pool, :],
            return_std=True,
        )
        weights = np.array([self.alpha, 1 - self.alpha])
        weighted_preds = preds * weights[0] + stds * weights[1]
        top_k_local = np.argpartition(-weighted_preds, batch_size - 1)[:batch_size]
        selected_indices = [unlabeled_pool[i] for i in top_k_local]
        return selected_indices


class TopLogLikelihood(QueryStrategyBase):
    """Selects samples with highest log likelihood values."""

    def __init__(self) -> None:
        super().__init__("TOP_LOG_LIKELIHOOD")

    def select(self, experiment: Any) -> List[int]:
        """
        Override select() to handle special case of missing/invalid log likelihoods.
        """
        log_likelihoods = experiment.all_log_likelihoods
        if log_likelihoods is None or np.all(np.isnan(log_likelihoods)):
            logger.warning("No log likelihood data available.")
            return []
        # Get log likelihood values for unlabeled samples
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
            logger.warning("No valid log likelihood values for unlabeled samples. ")
            return []

        # Early return if valid pool is smaller than batch size
        if len(valid_unlabeled_indices) < experiment.batch_size:
            selected_indices = valid_unlabeled_indices
            # Get log likelihoods for selected indices
            selected_log_likelihoods = log_likelihoods[selected_indices]
        else:
            selected_indices = self._select_batch(
                experiment, valid_unlabeled_indices, experiment.batch_size
            )
            # Get log likelihoods for selected indices
            selected_log_likelihoods = log_likelihoods[selected_indices]
        selected_values = experiment.dataset.labels[selected_indices]
        extra_info = (
            f"Log likelihoods: "
            f"[{', '.join(f'{ll:.4f}' for ll in selected_log_likelihoods)}] "
            f"True values: [{', '.join(f'{expr:.1f}' for expr in selected_values)}]"
        )
        self._log_round(selected_indices, extra_info)

        return selected_indices

    def _select_batch(
        self, experiment: Any, unlabeled_pool: List[int], batch_size: int
    ) -> List[int]:
        """Select batch based on log likelihood values."""
        # Note: unlabeled_pool here is already filtered to valid indices
        # We need to get log likelihoods for these indices
        log_likelihoods = experiment.all_log_likelihoods[unlabeled_pool]
        # Select indices with highest log likelihood values
        sorted_indices = np.argsort(log_likelihoods)[::-1]
        selected_local_indices = sorted_indices[:batch_size]
        selected_indices = [unlabeled_pool[i] for i in selected_local_indices]
        return selected_indices
