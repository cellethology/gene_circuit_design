"""
ActiveLearningExperiment class.

This class orchestrates the active learning loop using composed components.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import RegressorMixin

from experiments.core.data_loader import DataLoader, Dataset
from experiments.core.initial_selection_strategies import (
    InitialSelectionStrategy,
)
from experiments.core.metrics_calculator import MetricsCalculator
from experiments.core.predictor_trainer import PredictorTrainer
from experiments.core.query_strategies import QueryStrategyBase
from experiments.core.round_tracker import RoundTracker

logger = logging.getLogger(__name__)


class ActiveLearningExperiment:
    """
    Active learning experiment for sequence-expression prediction.

    This class orchestrates the active learning loop using composed components
    for data loading, model training, metrics calculation, and result management.
    """

    def __init__(
        self,
        embeddings_path: str,
        metadata_path: str,
        initial_selection_strategy: InitialSelectionStrategy,
        query_strategy: QueryStrategyBase,
        predictor: RegressorMixin,
        batch_size: int = 8,
        random_seed: int = 42,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        starting_batch_size: Optional[int] = None,
        label_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the active learning experiment.

        Args:
            embeddings_path: Path to NPZ file containing embeddings (and optional IDs)
            metadata_csv_path: CSV file with labels aligned to embeddings
            query_strategy: Strategy for selecting next samples
            predictor: Regression model to fit during the loop
            batch_size: Number of samples to select in each round\
            random_seed: Random seed for reproducibility
            normalize_features: Whether to re-normalize features each round
            normalize_labels: Whether to re-normalize labels each round
            label_key: Column name in the metadata CSV containing target values
            starting_batch_size: Number of samples to sample initially. If None, will be set to batch_size.
        """
        # Store configuration
        if label_key is None:
            raise ValueError("label_key must be provided for metadata loading.")

        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.query_strategy = query_strategy
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.label_key = label_key
        self.initial_selection_strategy = initial_selection_strategy
        self.normalize_features = normalize_features
        self.normalize_labels = normalize_labels
        if starting_batch_size is None:
            self.starting_batch_size = self.batch_size
        else:
            self.starting_batch_size = starting_batch_size

        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Initialize components
        self.data_loader = DataLoader(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            label_key=label_key,
        )

        # Load data
        self.dataset: Dataset = self.data_loader.load()

        # Initialize training pool
        self.train_indices = self._select_starting_batch()

        # Initialize trainer
        self.trainer = PredictorTrainer(
            predictor,
            normalize_features=self.normalize_features,
            normalize_labels=self.normalize_labels,
        )

        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(self.dataset.labels)

        # Initialize round tracker
        self.round_tracker = RoundTracker(
            sample_ids=self.dataset.sample_ids,
        )

        logger.info(f"Start experiment with {query_strategy.name}, seed={random_seed}")
        logger.info(
            f"Query strategy random state={getattr(query_strategy, 'random_state', None)}, Predictor random state={getattr(predictor, 'random_state', None)}"
        )

    def _select_starting_batch(self) -> Tuple[List[int], List[int]]:
        """
        Sample the initial training pool using the configured strategy.

        Returns:
            List of indices for the initial training pool
        """
        total_samples = len(self.dataset.sample_ids)
        if self.starting_batch_size >= total_samples:
            logger.warning(
                "starting_batch_size >= total samples. Using all samples for training."
            )
            all_indices = list(range(total_samples))
            return all_indices

        selected_indices = self.initial_selection_strategy.select(dataset=self.dataset)

        logger.info(
            f"Initialized training pool with {len(selected_indices)} samples via "
            f"{self.initial_selection_strategy.name}, leaving "
            f"{total_samples - len(selected_indices)} unlabeled samples."
        )

        return selected_indices

    def _select_next_batch(self) -> List[int]:
        """
        Select next batch of samples based on the configured strategy.

        Returns:
            List of indices for next batch of samples
        """
        return self.query_strategy.select(self)

    def _evaluate_and_track(
        self, indices: List[int], top_p: float = 0.1
    ) -> Dict[str, float]:
        """
        Evaluate and track the performance of the model on the given indices.

        Args:
            indices: Indices of the samples to evaluate
            top_p: Percentage of top labels to consider
        Returns:
            Metrics for the round
        """
        predictions = self.trainer.predict(self.dataset.embeddings[indices, :])
        round_metrics = self.metrics_calculator.compute_metrics_for_round(
            selected_indices=indices,
            predictions=predictions,
            top_p=top_p,
        )
        self.round_tracker.track_round(selected_indices=indices, metrics=round_metrics)
        return round_metrics

    def run_experiment(
        self, max_rounds: int = 30, top_p: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Run the active learning experiment.

        Args:
            max_rounds: Maximum number of active learning rounds
            top_p: Percentage of top labels to consider
        Returns:
            List of results for each round
        """
        logger.info(f"Starting active learning run with {max_rounds} max rounds")

        for round_num in range(max_rounds):
            logger.info(f"\n--- Round {round_num} ---")

            # Train model
            X_train = self.dataset.embeddings[self.train_indices, :]
            y_train = self.dataset.labels[self.train_indices]
            self.trainer.train(X_train=X_train, y_train=y_train)
            if round_num == 0:
                self._evaluate_and_track(indices=self.train_indices, top_p=top_p)

            # Select next batch
            next_batch = self._select_next_batch()

            # Track selected variants
            self._evaluate_and_track(indices=next_batch, top_p=top_p)

            # Update training set
            self.train_indices.extend(next_batch)

            # Stop if all samples have been selected
            if len(self.train_indices) == len(self.dataset.sample_ids):
                logger.info("All samples have been selected. Stopping.")
                break

    def save_results(self, output_path: Path) -> None:
        """
        Save experiment results to CSV files.

        Args:
            output_path: Path to save results
        """
        self.round_tracker.save_to_csv(output_path=output_path)

    @property
    def unlabeled_indices(self) -> List[int]:
        """Compute the remaining unlabeled indices."""
        return [
            idx
            for idx in range(len(self.dataset.sample_ids))
            if idx not in set(self.train_indices)
        ]
