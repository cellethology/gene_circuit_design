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

    Each concrete strategy implements the `select` method to choose
    the next batch of samples based on its specific criteria.
    """

    requires_model: bool = True

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def select(self, experiment: Any) -> List[int]:
        """
        Select the next batch of samples.

        Returns:
            List of selected indices from unlabeled_indices
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

    def select(self, experiment: Any) -> List[int]:
        unlabeled_pool = experiment.unlabeled_indices
        if len(unlabeled_pool) < experiment.batch_size:
            return unlabeled_pool

        rng = np.random.default_rng(self.seed)
        selected_indices = rng.choice(
            unlabeled_pool, experiment.batch_size, replace=False
        ).tolist()
        self._log_round(selected_indices)
        return selected_indices


class TopPredictions(QueryStrategyBase):
    """Selects samples with highest k predicted label values."""

    def __init__(self) -> None:
        super().__init__("TOP_K_PRED")

    def select(self, experiment: Any) -> List[int]:
        k = experiment.batch_size
        unlabeled = experiment.unlabeled_indices
        if len(unlabeled) < k:
            return unlabeled

        preds = experiment.trainer.predict(experiment.dataset.embeddings[unlabeled, :])

        # get indices of top k predictions (descending)
        top_k_local = np.argpartition(-preds, k - 1)[:k]

        # map to original indices
        selected_indices = [unlabeled[i] for i in top_k_local]
        self._log_round(selected_indices)

        return selected_indices


class CombinedPredictionUncertainty(QueryStrategyBase):
    """Selects samples with highest combined prediction uncertainty."""

    def __init__(self, alpha: float) -> None:
        super().__init__("TOP_K_PRED_AND_UNCERTAINTY")
        self.alpha = alpha

    def select(self, experiment: Any) -> List[int]:
        # get weighted sum of prediction and standard deviation of prediction
        preds, stds = experiment.trainer.predict(
            experiment.dataset.embeddings[experiment.unlabeled_indices, :],
            return_std=True,
        )
        weights = np.array([self.alpha, 1 - self.alpha])
        weighted_preds = preds * weights[0] + stds * weights[1]
        top_k_local = np.argpartition(-weighted_preds, experiment.batch_size - 1)[
            : experiment.batch_size
        ]
        selected_indices = [experiment.unlabeled_indices[i] for i in top_k_local]
        self._log_round(selected_indices)

        return selected_indices


class TopLogLikelihood(QueryStrategyBase):
    """Selects samples with highest log likelihood values."""

    def __init__(self) -> None:
        super().__init__("TOP_LOG_LIKELIHOOD")

    def select(self, experiment: Any) -> List[int]:
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

        # Select indices with highest log likelihood values
        sorted_indices = np.argsort(valid_log_likelihoods)[::-1]
        batch_size = min(experiment.batch_size, len(valid_unlabeled_indices))
        selected_local_indices = sorted_indices[:batch_size]

        selected_indices = [valid_unlabeled_indices[i] for i in selected_local_indices]

        selected_log_likelihoods = valid_log_likelihoods[selected_local_indices]
        selected_values = experiment.dataset.labels[selected_indices]
        extra_info = (
            f"Log likelihoods: "
            f"[{', '.join(f'{ll:.4f}' for ll in selected_log_likelihoods)}] "
            f"True values: [{', '.join(f'{expr:.1f}' for expr in selected_values)}]"
        )
        self._log_round(selected_indices, extra_info)

        return selected_indices
