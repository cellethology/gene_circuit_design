"""
Refactored ActiveLearningExperiment class using composition.

This module provides a cleaner, more maintainable implementation
by breaking down responsibilities into separate components.
"""

import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.core.data_loader import DataLoader, Dataset, DataSplit
from experiments.core.metrics_calculator import MetricsCalculator
from experiments.core.model_trainer import ModelTrainer
from experiments.core.result_manager import ResultManager
from experiments.core.variant_tracker import VariantTracker
from experiments.selection_strategies import create_selection_strategy
from experiments.util import encode_sequences as util_encode_sequences
from utils.config_loader import SelectionStrategy
from utils.model_loader import RegressionModelType, return_model
from utils.sequence_utils import (
    SequenceModificationMethod,
    ensure_sequence_modification_method,
)

logger = logging.getLogger(__name__)


class ActiveLearningExperiment:
    """
    Active learning experiment for sequence-expression prediction.

    This class orchestrates the active learning loop using composed components
    for data loading, model training, metrics calculation, and result management.
    """

    def __init__(
        self,
        data_path: str,
        selection_strategy: SelectionStrategy = SelectionStrategy.HIGH_EXPRESSION,
        regression_model: RegressionModelType = RegressionModelType.LINEAR,
        initial_sample_size: int = 8,
        batch_size: int = 8,
        test_size: int = 50,
        random_seed: int = 42,
        seq_mod_method: SequenceModificationMethod = SequenceModificationMethod.EMBEDDING,
        no_test: bool = True,
        normalize_input_output: bool = True,
        use_pca: bool = False,
        pca_components: int = 4096,
        target_val_key: str = None,
    ) -> None:
        """
        Initialize the active learning experiment.

        Args:
            data_path: Path to CSV or safetensors file with sequence and expression data
            selection_strategy: Strategy for selecting next sequences
            regression_model: Type of regression model to use
            initial_sample_size: Number of sequences to start with
            batch_size: Number of sequences to select in each round
            test_size: Number of sequences reserved for testing
            random_seed: Random seed for reproducibility
            seq_mod_method: Sequence modification method for encoding
            no_test: Whether to use the test set
            normalize_input_output: Whether to normalize expressions and embeddings
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components if use_pca=True
            target_val_key: Optional key for target values in safetensors files
        """
        # Store configuration
        self.data_path = data_path
        self.selection_strategy = selection_strategy
        self.initial_sample_size = initial_sample_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.seq_mod_method = ensure_sequence_modification_method(seq_mod_method)
        self.no_test = no_test
        self.normalize_input_output = normalize_input_output
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.target_val_key = target_val_key

        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Initialize components
        self.data_loader = DataLoader(
            data_path=data_path,
            seq_mod_method=self.seq_mod_method,
            normalize_input_output=normalize_input_output,
            target_val_key=target_val_key,
        )

        # Load data
        self.dataset: Dataset = self.data_loader.load()

        # Create data splits
        self.data_split: DataSplit = self.data_loader.create_data_split(
            dataset=self.dataset,
            initial_sample_size=initial_sample_size,
            test_size=test_size,
            no_test=no_test,
            selection_strategy=selection_strategy,
            random_seed=random_seed,
            encode_sequences_fn=self._encode_sequences,
        )

        # Initialize model and trainer
        self.model = return_model(regression_model, random_state=random_seed)
        self.model_trainer = ModelTrainer(self.model)

        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(self.dataset.expressions)

        # Initialize variant tracker
        self.variant_tracker = VariantTracker(
            all_expressions=self.dataset.expressions,
            all_log_likelihoods=self.dataset.log_likelihoods,
            all_sequences=self.dataset.sequences,
            variant_ids=self.dataset.variant_ids,
        )

        # Initialize result manager
        self.result_manager = ResultManager(
            strategy=selection_strategy.value,
            seq_mod_method=self.seq_mod_method.value,
            regression_model_name=self.model.__class__.__name__,
            seed=random_seed,
            initial_sample_size=initial_sample_size,
            batch_size=batch_size,
        )

        # Results storage
        self.results: List[Dict[str, Any]] = []

        logger.info(
            f"Experiment initialized with {selection_strategy.value} selection strategy (seed={random_seed})"
        )

    def _encode_sequences(self, indices: List[int]) -> np.ndarray:
        """
        Encode sequences at given indices.

        Args:
            indices: List of sequence indices to encode

        Returns:
            Encoded sequences as numpy array
        """
        return util_encode_sequences(
            indices=indices,
            all_sequences=self.dataset.sequences,
            embeddings=self.dataset.embeddings,
            seq_mod_method=self.seq_mod_method,
            use_pca=self.use_pca,
            pca_components=self.pca_components,
        )

    def _select_next_batch(self) -> List[int]:
        """
        Select next batch of sequences based on the configured strategy.

        Returns:
            List of indices for next batch
        """
        if len(self.data_split.unlabeled_indices) == 0:
            return []

        # Create and use the appropriate selection strategy
        strategy = create_selection_strategy(
            strategy=self.selection_strategy,
            batch_size=self.batch_size,
            unlabeled_indices=self.data_split.unlabeled_indices,
            all_expressions=self.dataset.expressions,
            experiment=self,
            all_log_likelihoods=self.dataset.log_likelihoods,
        )

        return strategy.select()

    def run_experiment(self, max_rounds: int = 30) -> List[Dict[str, Any]]:
        """
        Run the active learning experiment.

        Args:
            max_rounds: Maximum number of active learning rounds

        Returns:
            List of results for each round
        """
        logger.info(
            f"Starting {self.selection_strategy.value} learning experiment with {max_rounds} max rounds"
        )

        for round_num in range(max_rounds):
            logger.info(
                f"\n--- Round {round_num + 1} ({self.selection_strategy.value.upper()}) ---"
            )

            # Train model
            X_train = self._encode_sequences(self.data_split.train_indices)
            y_train = self.dataset.expressions[self.data_split.train_indices]
            self.model_trainer.train(
                X_train=X_train,
                y_train=y_train,
                train_indices=self.data_split.train_indices,
            )

            # Evaluate on test set
            if not self.no_test and len(self.data_split.test_indices) > 0:
                X_test = self._encode_sequences(self.data_split.test_indices)
                y_test = self.dataset.expressions[self.data_split.test_indices]
                metrics = self.model_trainer.evaluate(
                    X_test=X_test, y_test=y_test, no_test=self.no_test
                )
            else:
                metrics = {}

            # Store results
            round_results = {
                "round": round_num + 1,
                "strategy": self.selection_strategy.value,
                "seq_mod_method": self.seq_mod_method.value,
                "seed": self.random_seed,
                "train_size": len(self.data_split.train_indices),
                "unlabeled_size": len(self.data_split.unlabeled_indices),
                **metrics,
            }
            self.results.append(round_results)

            # First round: evaluate custom metrics on initial training set
            if round_num == 0:
                X_train_encoded = self._encode_sequences(self.data_split.train_indices)
                predictions = self.model_trainer.predict(X_train_encoded)
                round_metrics = self.metrics_calculator.calculate_round_metrics(
                    selected_indices=self.data_split.train_indices,
                    selection_strategy=self.selection_strategy,
                    predictions=predictions,
                )
                self.metrics_calculator.update_cumulative(round_metrics)

                # Track initial training set
                self.variant_tracker.track_round(
                    round_num=0,
                    selected_indices=self.data_split.train_indices,
                    strategy=self.selection_strategy.value,
                    seed=self.random_seed,
                )

            # Check stopping criteria
            if len(self.data_split.unlabeled_indices) == 0:
                logger.info("No more unlabeled data available. Stopping.")
                break

            # Select next batch
            next_batch = self._select_next_batch()
            if not next_batch:
                logger.info("No more sequences to select. Stopping.")
                break

            # Track selected variants
            self.variant_tracker.track_round(
                round_num=round_num + 1,
                selected_indices=next_batch,
                strategy=self.selection_strategy.value,
                seed=self.random_seed,
            )

            # Evaluate custom metrics
            X_next_batch = self._encode_sequences(next_batch)
            predictions = self.model_trainer.predict(X_next_batch)
            round_metrics = self.metrics_calculator.calculate_round_metrics(
                selected_indices=next_batch,
                selection_strategy=self.selection_strategy,
                predictions=predictions,
            )
            self.metrics_calculator.update_cumulative(round_metrics)

            # Update training set
            self.data_split.train_indices.extend(next_batch)
            self.data_split.unlabeled_indices = [
                idx
                for idx in self.data_split.unlabeled_indices
                if idx not in next_batch
            ]

            logger.info(f"Added {len(next_batch)} sequences to training set")

        logger.info(
            f"{self.selection_strategy.value.capitalize()} experiment completed!"
        )
        return self.results

    def save_results(self, output_path: str) -> None:
        """
        Save experiment results to CSV files.

        Args:
            output_path: Path to save results CSV
        """
        self.result_manager.save_results(
            output_path=output_path,
            results=self.results,
            custom_metrics=self.metrics_calculator.get_all_metrics(),
            selected_variants=self.variant_tracker.get_all_variants(),
        )

    def get_final_performance(self) -> Dict[str, float]:
        """
        Get final performance metrics.

        Returns:
            Dictionary with final performance metrics
        """
        if not self.results:
            return {}

        return {
            k: v
            for k, v in self.results[-1].items()
            if k not in ["round", "strategy", "seed", "train_size", "unlabeled_size"]
        }

    # Expose properties for backward compatibility
    @property
    def all_sequences(self) -> List[str]:
        """Get all sequences (for backward compatibility)."""
        return self.dataset.sequences

    @property
    def all_expressions(self) -> np.ndarray:
        """Get all expressions (for backward compatibility)."""
        return self.dataset.expressions

    @property
    def all_log_likelihoods(self) -> np.ndarray:
        """Get all log likelihoods (for backward compatibility)."""
        return self.dataset.log_likelihoods

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Get embeddings (for backward compatibility)."""
        return self.dataset.embeddings

    @property
    def train_indices(self) -> List[int]:
        """Get training indices (for backward compatibility)."""
        return self.data_split.train_indices

    @property
    def test_indices(self) -> List[int]:
        """Get test indices (for backward compatibility)."""
        return self.data_split.test_indices

    @property
    def unlabeled_indices(self) -> List[int]:
        """Get unlabeled indices (for backward compatibility)."""
        return self.data_split.unlabeled_indices

    @property
    def custom_metrics(self) -> List[Dict[str, Any]]:
        """Get custom metrics (for backward compatibility)."""
        return self.metrics_calculator.get_all_metrics()

    @property
    def selected_variants(self) -> List[Dict[str, Any]]:
        """Get selected variants (for backward compatibility)."""
        return self.variant_tracker.get_all_variants()

    @property
    def variant_ids(self) -> Optional[np.ndarray]:
        """Get variant IDs (for backward compatibility)."""
        return self.dataset.variant_ids
