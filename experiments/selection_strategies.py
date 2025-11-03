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

from utils.config_loader import SelectionStrategy

logger = logging.getLogger(__name__)


class SelectionStrategyBase(ABC):
    """
    Abstract base class for selection strategies.

    Each concrete strategy implements the `select` method to choose
    the next batch of sequences based on its specific criteria.
    """

    def __init__(
        self,
        batch_size: int,
        unlabeled_indices: List[int],
        all_expressions: np.ndarray,
        experiment: Any,  # ActiveLearningExperiment - avoid circular import
    ) -> None:
        """
        Initialize the selection strategy.

        Args:
            batch_size: Number of sequences to select
            unlabeled_indices: List of indices available for selection
            all_expressions: Array of expression values for all sequences
            experiment: Reference to the experiment object for accessing
                       model, embeddings, log_likelihoods, etc.
        """
        self.batch_size = batch_size
        self.unlabeled_indices = unlabeled_indices
        self.all_expressions = all_expressions
        self.experiment = experiment

    @abstractmethod
    def select(self) -> List[int]:
        """
        Select the next batch of sequences.

        Returns:
            List of selected indices from unlabeled_indices
        """
        pass

    def _log_selection(
        self,
        selected_indices: List[int],
        strategy_name: str,
        extra_info: Optional[str] = None,
    ) -> None:
        """
        Log information about the selected sequences.

        Args:
            selected_indices: Indices that were selected
            strategy_name: Name of the strategy for logging
            extra_info: Additional information to include in log message
        """
        selected_expressions = self.all_expressions[selected_indices]
        log_msg = (
            f"{strategy_name}: Selected {len(selected_indices)} sequences "
            f"with actual expressions: "
            f"[{', '.join(f'{expr:.1f}' for expr in selected_expressions)}]"
        )
        if extra_info:
            log_msg += f" {extra_info}"
        logger.info(log_msg)


class RandomSelectionStrategy(SelectionStrategyBase):
    """Strategy that selects sequences randomly."""

    def select(self) -> List[int]:
        """
        Select next batch randomly.

        Returns:
            List of indices for next batch
        """
        batch_size = min(self.batch_size, len(self.unlabeled_indices))
        selected_indices = random.sample(self.unlabeled_indices, batch_size)
        self._log_selection(selected_indices, "RANDOM")
        return selected_indices


class HighExpressionSelectionStrategy(SelectionStrategyBase):
    """Strategy that selects sequences with highest predicted expression values."""

    def select(self) -> List[int]:
        """
        Select next batch using active learning (highest predicted values).

        Returns:
            List of indices for next batch
        """
        # Predict on all unlabeled sequences
        X_unlabeled = self.experiment._encode_sequences(self.unlabeled_indices)
        predictions = self.experiment.model.predict(X_unlabeled)

        # Select indices with highest predicted values
        sorted_indices = np.argsort(predictions)[::-1]  # Descending order
        batch_size = min(self.batch_size, len(self.unlabeled_indices))
        selected_local_indices = sorted_indices[:batch_size]

        # Convert to global indices
        selected_indices = [self.unlabeled_indices[i] for i in selected_local_indices]

        # Log selection info
        selected_predictions = predictions[selected_local_indices]
        extra_info = (
            f"with predicted expressions: "
            f"[{', '.join(f'{pred:.1f}' for pred in selected_predictions)}]"
        )
        self._log_selection(selected_indices, "HIGH_EXPRESSION", extra_info)

        return selected_indices


class LogLikelihoodSelectionStrategy(SelectionStrategyBase):
    """Strategy that selects sequences with highest log likelihood values."""

    def __init__(
        self,
        batch_size: int,
        unlabeled_indices: List[int],
        all_expressions: np.ndarray,
        experiment: Any,
        all_log_likelihoods: np.ndarray,
    ) -> None:
        """
        Initialize the log likelihood selection strategy.

        Args:
            batch_size: Number of sequences to select
            unlabeled_indices: List of indices available for selection
            all_expressions: Array of expression values for all sequences
            experiment: Reference to the experiment object
            all_log_likelihoods: Array of log likelihood values for all sequences
        """
        super().__init__(batch_size, unlabeled_indices, all_expressions, experiment)
        self.all_log_likelihoods = all_log_likelihoods

    def select(self) -> List[int]:
        """
        Select next batch using log likelihood (highest log likelihood values).

        Returns:
            List of indices for next batch
        """
        # Check if log likelihood data is available
        if np.all(np.isnan(self.all_log_likelihoods)):
            logger.warning(
                "No log likelihood data available. Falling back to random selection."
            )
            random_strategy = RandomSelectionStrategy(
                self.batch_size,
                self.unlabeled_indices,
                self.all_expressions,
                self.experiment,
            )
            return random_strategy.select()

        # Get log likelihood values for unlabeled sequences
        unlabeled_log_likelihoods = self.all_log_likelihoods[self.unlabeled_indices]

        # Filter out NaN values
        valid_mask = ~np.isnan(unlabeled_log_likelihoods)
        valid_unlabeled_indices = [
            self.unlabeled_indices[i]
            for i in range(len(self.unlabeled_indices))
            if valid_mask[i]
        ]
        valid_log_likelihoods = unlabeled_log_likelihoods[valid_mask]

        if len(valid_unlabeled_indices) == 0:
            logger.warning(
                "No valid log likelihood values for unlabeled sequences. "
                "Falling back to random selection."
            )
            random_strategy = RandomSelectionStrategy(
                self.batch_size,
                self.unlabeled_indices,
                self.all_expressions,
                self.experiment,
            )
            return random_strategy.select()

        # Select indices with highest log likelihood values (less negative = higher probability)
        # TODO: could add a bit of stochasticity to the selection process if log_likelihoods are similar
        sorted_indices = np.argsort(valid_log_likelihoods)[
            ::-1
        ]  # Descending order (highest first)
        batch_size = min(self.batch_size, len(valid_unlabeled_indices))
        selected_local_indices = sorted_indices[:batch_size]

        # Convert to global indices
        selected_indices = [valid_unlabeled_indices[i] for i in selected_local_indices]

        # Log selection info
        selected_log_likelihoods = valid_log_likelihoods[selected_local_indices]
        selected_expressions = self.all_expressions[selected_indices]
        extra_info = (
            f"with log likelihoods: "
            f"[{', '.join(f'{ll:.4f}' for ll in selected_log_likelihoods)}] "
            f"and actual expressions: [{', '.join(f'{expr:.1f}' for expr in selected_expressions)}]"
        )
        self._log_selection(selected_indices, "LOG_LIKELIHOOD", extra_info)

        return selected_indices


def create_selection_strategy(
    strategy: SelectionStrategy,
    batch_size: int,
    unlabeled_indices: List[int],
    all_expressions: np.ndarray,
    experiment: Any,
    all_log_likelihoods: Optional[np.ndarray] = None,
) -> SelectionStrategyBase:
    """
    Factory function to create the appropriate selection strategy.

    Args:
        strategy: The selection strategy enum value
        batch_size: Number of sequences to select
        unlabeled_indices: List of indices available for selection
        all_expressions: Array of expression values for all sequences
        experiment: Reference to the experiment object
        all_log_likelihoods: Array of log likelihood values (required for LOG_LIKELIHOOD strategy)

    Returns:
        Instance of the appropriate selection strategy

    Raises:
        ValueError: If strategy is not recognized or required data is missing
    """
    # Handle composite strategies (KMEANS_*)
    if strategy == SelectionStrategy.KMEANS_HIGH_EXPRESSION:
        # After initial K-means selection, use high expression selection
        strategy = SelectionStrategy.HIGH_EXPRESSION
    elif strategy == SelectionStrategy.KMEANS_RANDOM:
        # After initial K-means selection, use random selection
        strategy = SelectionStrategy.RANDOM

    if strategy == SelectionStrategy.RANDOM:
        return RandomSelectionStrategy(
            batch_size, unlabeled_indices, all_expressions, experiment
        )
    elif strategy == SelectionStrategy.HIGH_EXPRESSION:
        return HighExpressionSelectionStrategy(
            batch_size, unlabeled_indices, all_expressions, experiment
        )
    elif strategy == SelectionStrategy.LOG_LIKELIHOOD:
        if all_log_likelihoods is None:
            raise ValueError(
                "all_log_likelihoods must be provided for LOG_LIKELIHOOD strategy"
            )
        return LogLikelihoodSelectionStrategy(
            batch_size,
            unlabeled_indices,
            all_expressions,
            experiment,
            all_log_likelihoods,
        )
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
