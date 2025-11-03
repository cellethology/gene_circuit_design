"""
Active Learning Loop for DNA Sequence-Expression Prediction.

This script implements an active learning approach to predict gene expression
from DNA sequences using linear regression with one-hot encoded features.
"""

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from safetensors.torch import load_file
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, root_mean_squared_error
from tqdm import tqdm

from utils.config_loader import SelectionStrategy
from utils.metrics import (
    get_best_value_metric,
    normalized_to_best_val_metric,
    top_10_ratio_intersected_indices_metric,
)
from utils.model_loader import RegressionModelType, return_model
from utils.sequence_utils import (
    SequenceModificationMethod,
    ensure_sequence_modification_method,
    flatten_one_hot_sequences,
    load_sequence_data,
    one_hot_encode_sequences,
)

# Create a file handler
log_path = Path("logs") / "experiment.log"
log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
file_handler = logging.FileHandler(
    log_path, mode="w"
)  # 'a' to append instead of overwrite
file_handler.setLevel(logging.INFO)

# Optional: set formatter for file logs
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger().addHandler(file_handler)
logger = logging.getLogger(__name__)


class ActiveLearningExperiment:
    """
    Active learning experiment for sequence-expression prediction.

    This class manages the active learning loop, starting with a small random
    sample and iteratively selecting the most promising sequences for labeling
    using different selection strategies.
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
    ) -> None:
        """
        Initialize the active learning experiment.

        Args:
            data_path: Path to CSV file with sequence and expression data
            selection_strategy: Strategy for selecting next sequences
            initial_sample_size: Number of sequences to start with
            batch_size: Number of sequences to select in each round
            test_size: Number of sequences reserved for testing
            random_seed: Random seed for reproducibility
            trim_sequences: Whether to trim sequences to same length
            no_test: Whether to use the test set
        """
        self.data_path = data_path
        self.selection_strategy = selection_strategy
        self.initial_sample_size = initial_sample_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_seed = random_seed
        # Convert string to enum if necessary
        self.seq_mod_method = ensure_sequence_modification_method(seq_mod_method)
        self.no_test = no_test
        self.normalize_input_output = normalize_input_output
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Data containers
        self.all_sequences = []  # Can be List[str] for DNA or np.ndarray for motifs
        self.all_expressions: np.ndarray = np.array([])
        self.all_log_likelihoods: np.ndarray = np.array(
            []
        )  # Add log likelihood storage
        self.embeddings: np.ndarray = None  # Add embeddings storage
        self.train_indices: List[int] = []
        self.test_indices: List[int] = []
        self.unlabeled_indices: List[int] = []

        # Model and results
        self.model = return_model(regression_model, random_state=random_seed)
        self.results: List[Dict[str, Any]] = []
        self.custom_metrics: List[Dict[str, Any]] = []

        # Load and prepare data
        self._load_and_prepare_data(seq_mod_method=self.seq_mod_method)

        logger.info(
            f"Experiment initialized with {self.selection_strategy.value} selection strategy (seed={random_seed})"
        )

    def _load_and_prepare_data(
        self,
        seq_mod_method: SequenceModificationMethod = SequenceModificationMethod.EMBEDDING,
    ) -> None:
        """Load data and create train/test/unlabeled splits."""
        logger.info(f"Loading data from {self.data_path}")

        # Check if this is a safetensors file or CSV file
        if self.data_path.endswith(".safetensors"):
            logger.info("Loading from safetensors file")
            # Load from safetensors file
            tensors = load_file(self.data_path)
            self.embeddings = (
                tensors["embeddings"].float().numpy()
            )  # Convert to float32 then numpy
            self.all_expressions = (
                tensors["expressions"].float().numpy()
            )  # Convert to float32 then numpy
            self.all_log_likelihoods = (
                tensors["log_likelihoods"].float().numpy()
            )  # Convert to float32 then numpy
            sequences = tensors["sequences"]

            # Convert sequences to list of strings
            if hasattr(sequences, "numpy"):
                # If it's a torch tensor, convert to numpy first
                sequences_np = sequences.numpy()
            else:
                sequences_np = sequences

            # Handle different sequence formats
            if sequences_np.dtype.kind in ["S", "U"]:  # String or Unicode
                self.all_sequences = [
                    seq.decode("utf-8") if isinstance(seq, bytes) else str(seq)
                    for seq in sequences_np
                ]
            elif sequences_np.dtype == object:  # Object array (might contain strings)
                self.all_sequences = [str(seq) for seq in sequences_np]
            else:
                # Try to convert each element to string
                self.all_sequences = [str(seq) for seq in sequences_np]

            if self.normalize_input_output:
                self.all_expressions = (
                    self.all_expressions
                    - self.all_expressions.mean(axis=0, keepdims=True)
                ) / (self.all_expressions.std(axis=0, keepdims=True) + 1e-30)
                self.embeddings = (
                    self.embeddings - self.embeddings.mean(axis=0, keepdims=True)
                ) / (self.embeddings.std(axis=0, keepdims=True) + 1e-30)
                logger.info(
                    f"Normalized expression data with mean {self.all_expressions.mean()} and std {self.all_expressions.std()}"
                )

            logger.info(
                f"Loaded safetensors dataset with {len(self.all_sequences)} sequences"
            )
            logger.info(f"Embeddings shape: {self.embeddings.shape}")
        else:
            # Load from CSV file
            logger.info(f"Using sequence modification method: {seq_mod_method}")

            # Load standard CSV data
            df = pd.read_csv(self.data_path, encoding="latin-1")

            if "Log_Likelihood" in df.columns:
                # Combined dataset with log likelihood
                sequences = df["Sequence"].tolist()
                self.all_sequences = sequences
                self.all_expressions = df["Expression"].values
                self.all_log_likelihoods = df["Log_Likelihood"].values
                logger.info(
                    f"Loaded combined dataset with {len(self.all_sequences)} sequences including log likelihood data"
                )
            else:
                # Original expression-only dataset
                self.all_sequences, self.all_expressions = load_sequence_data(
                    self.data_path, seq_mod_method=seq_mod_method
                )
                self.all_log_likelihoods = np.full(
                    len(self.all_sequences), np.nan
                )  # Fill with NaN if no log likelihood
                logger.info(
                    f"Loaded expression-only dataset with {len(self.all_sequences)} sequences"
                )

            # For CSV files, embeddings will be None (will be computed via one-hot encoding)
            self.embeddings = None

        total_samples = len(self.all_sequences)

        # Log sequence statistics
        # NOTE: comment out to clean clutter in logs
        # NOTE: base on sequence type, calculate statistics
        # stats = calculate_sequence_statistics(self.all_sequences)
        # logger.info(f"Sequence statistics: {stats}")

        # Log log likelihood statistics if available
        if not np.all(np.isnan(self.all_log_likelihoods)):
            ll_stats = pd.Series(self.all_log_likelihoods).describe()
            logger.info(f"Log likelihood statistics:\n{ll_stats}")

        # If no test set is used, set test size to 0 and test indices to empty list
        # NOTE: NO SPLIT ON DIFFERENT SEEDS
        if self.no_test:
            self.test_size = 0
            self.test_indices = []

        # Create indices for all samples
        all_indices = list(range(total_samples))
        # NOTE: DEBUG CODE: FIX SEED FOR SHUFFLING
        if self.selection_strategy == SelectionStrategy.LOG_LIKELIHOOD:
            random.Random(0).shuffle(all_indices)
        else:
            random.shuffle(all_indices)

        # Reserve test set
        self.test_indices = all_indices[: self.test_size]
        remaining_indices = all_indices[self.test_size :]

        # Initial training set selection
        if self.selection_strategy in [
            SelectionStrategy.KMEANS_HIGH_EXPRESSION,
            SelectionStrategy.KMEANS_RANDOM,
        ]:
            # Use K-means clustering for initial selection
            self.train_indices = self._select_initial_batch_kmeans_clustering()
            # Remove selected indices from remaining indices
            self.unlabeled_indices = [
                idx for idx in remaining_indices if idx not in self.train_indices
            ]
        else:
            # Use random selection for initial training set
            self.train_indices = remaining_indices[: self.initial_sample_size]
            self.unlabeled_indices = remaining_indices[self.initial_sample_size :]

        logger.info(
            f"Data split - Train: {len(self.train_indices)}, "
            f"Test: {len(self.test_indices)}, "
            f"Unlabeled: {len(self.unlabeled_indices)}"
        )

    def _encode_sequences(self, indices: List[int]) -> np.ndarray:
        """
        Encode sequences at given indices using pre-computed embeddings or one-hot encoding.

        Args:
            indices: List of sequence indices to encode

        Returns:
            Encoded sequences (either pre-computed embeddings or flattened one-hot encoded sequences)
        """
        if self.embeddings is not None:
            # Use pre-computed embeddings from safetensors file
            return self.embeddings[indices]
        else:
            # Fall back to one-hot encoding for CSV files
            # For DNA data, all_sequences is a list of strings
            sequences = [self.all_sequences[i] for i in indices]

            encoded = one_hot_encode_sequences(sequences, self.seq_mod_method)
            return flatten_one_hot_sequences(encoded)
            # # Flatten first, then apply PCA to reduce dimensionality, keeping 90% of variance
            # flattened = flatten_one_hot_sequences(encoded)
            # pca = PCA(n_components=0.9)  # 0.9 = 90% variance explained
            # reduced = pca.fit_transform(flattened)
            # logger.info(f"PCA reduced dimensions from {flattened.shape[1]} to {reduced.shape[1]} (90% variance explained)")
            # return reduced

    def _train_model(self) -> None:
        """Train the linear regression model on current training data."""
        logger.info(f"Training model with {len(self.train_indices)} samples")

        X_train = self._encode_sequences(self.train_indices)
        y_train = self.all_expressions[self.train_indices]

        self.model.fit(X_train, y_train)

        # Log training performance
        train_pred = self.model.predict(X_train)
        train_mse = root_mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)

        logger.info(f"Training RMSE: {train_mse:.2f}, R²: {train_r2:.3f}")

    def _evaluate_on_test_set(self) -> Dict[str, float]:
        """
        Evaluate model performance on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        if self.no_test:
            return {}

        X_test = self._encode_sequences(self.test_indices)
        y_test = self.all_expressions[self.test_indices]

        y_pred = self.model.predict(X_test)

        # Calculate metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(y_test, y_pred)
        spearman_corr, spearman_p = spearmanr(y_test, y_pred)

        metrics = {
            "rmse": rmse,
            "r2": r2,
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
        }

        logger.info(
            f"Test metrics - RMSE: {rmse:.2f}, R²: {r2:.3f}, "
            f"Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}"
        )

        return metrics

    def _select_next_batch_active(self) -> List[int]:
        """
        Select next batch using active learning (highest predicted values).

        Returns:
            List of indices for next batch
        """
        # Predict on all unlabeled sequences
        X_unlabeled = self._encode_sequences(self.unlabeled_indices)
        predictions = self.model.predict(X_unlabeled)

        # Select indices with highest predicted values
        sorted_indices = np.argsort(predictions)[::-1]  # Descending order
        batch_size = min(self.batch_size, len(self.unlabeled_indices))
        selected_local_indices = sorted_indices[:batch_size]

        # Convert to global indices
        selected_indices = [self.unlabeled_indices[i] for i in selected_local_indices]

        # Log selection info
        selected_predictions = predictions[selected_local_indices]
        logger.info(
            f"HIGH_EXPRESSION: Selected {len(selected_indices)} sequences with predicted expressions: "
            f"[{', '.join(f'{pred:.1f}' for pred in selected_predictions)}]"
        )

        return selected_indices

    def _select_next_batch_random(self) -> List[int]:
        """
        Select next batch randomly.

        Returns:
            List of indices for next batch
        """
        batch_size = min(self.batch_size, len(self.unlabeled_indices))
        selected_indices = random.sample(self.unlabeled_indices, batch_size)

        # Log selection info
        selected_expressions = self.all_expressions[selected_indices]
        logger.info(
            f"RANDOM: Selected {len(selected_indices)} sequences with actual expressions: "
            f"[{', '.join(f'{expr:.1f}' for expr in selected_expressions)}]"
        )

        return selected_indices

    def _select_next_batch_log_likelihood(self) -> List[int]:
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
            return self._select_next_batch_random()

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
                "No valid log likelihood values for unlabeled sequences. Falling back to random selection."
            )
            return self._select_next_batch_random()

        # Select indices with highest log likelihood values (less negative = higher probability)
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
        logger.info(
            f"LOG_LIKELIHOOD: Selected {len(selected_indices)} sequences with log likelihoods: "
            f"[{', '.join(f'{ll:.4f}' for ll in selected_log_likelihoods)}] "
            f"and actual expressions: [{', '.join(f'{expr:.1f}' for expr in selected_expressions)}]"
        )

        return selected_indices

    def _select_initial_batch_kmeans_clustering(self) -> List[int]:
        """
        Select initial batch using K-means clustering on the whole dataset.

        Steps:
        1. Use K-means clustering on the whole dataset
        2. Set the centroid number to be the same as the initial sample size
        3. Select the data point in each cluster which is closest to that cluster's centroid

        Returns:
            List of indices for initial batch
        """
        # Encode all sequences to get feature representations
        all_indices = list(range(len(self.all_sequences)))
        X_all = self._encode_sequences(all_indices)

        # Apply K-means clustering with k = initial_sample_size
        kmeans = KMeans(
            n_clusters=self.initial_sample_size,
            random_state=self.random_seed,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(X_all)
        cluster_centers = kmeans.cluster_centers_

        selected_indices = []

        # For each cluster, find the point closest to the cluster centroid
        for cluster_id in range(self.initial_sample_size):
            # Get indices of points in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                # If cluster is empty (shouldn't happen with proper k-means), skip
                continue

            # Get feature vectors for points in this cluster
            cluster_points = X_all[cluster_indices]
            cluster_center = cluster_centers[cluster_id]

            # Calculate distances from each point in cluster to cluster centroid
            distances_to_center = np.linalg.norm(
                cluster_points - cluster_center, axis=1
            )

            # Find the point in this cluster closest to cluster centroid
            closest_idx_in_cluster = np.argmin(distances_to_center)
            closest_global_idx = cluster_indices[closest_idx_in_cluster]

            selected_indices.append(closest_global_idx)

        # Log selection info
        selected_expressions = self.all_expressions[selected_indices]
        logger.info(
            f"KMEANS_CLUSTERING: Selected {len(selected_indices)} sequences "
            f"from {self.initial_sample_size} clusters with actual expressions: "
            f"[{', '.join(f'{expr:.1f}' for expr in selected_expressions)}]"
        )

        return selected_indices

    def _select_next_batch(self) -> List[int]:
        """
        Select next batch of sequences based on the configured strategy.

        Returns:
            List of indices for next batch
        """
        if len(self.unlabeled_indices) == 0:
            return []

        # TODO: make this into a function
        if self.selection_strategy == SelectionStrategy.HIGH_EXPRESSION:
            return self._select_next_batch_active()
        elif self.selection_strategy == SelectionStrategy.RANDOM:
            return self._select_next_batch_random()
        elif self.selection_strategy == SelectionStrategy.LOG_LIKELIHOOD:
            return self._select_next_batch_log_likelihood()
        elif self.selection_strategy == SelectionStrategy.KMEANS_HIGH_EXPRESSION:
            # After initial K-means selection, use high expression selection for subsequent batches
            return self._select_next_batch_active()
        elif self.selection_strategy == SelectionStrategy.KMEANS_RANDOM:
            # After initial K-means selection, use random selection for subsequent batches
            return self._select_next_batch_random()
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def _intermediate_evaluation_custom_metrics(self, next_batch: List[int]) -> None:
        """
        Evaluate the model on the test set and store the results.
        """
        # Get predictions for the next batch
        X_next_batch = self._encode_sequences(next_batch)
        y_pred = self.model.predict(X_next_batch)

        # Custom metrics
        top_10_ratio_intersection_pred = top_10_ratio_intersected_indices_metric(
            next_batch, self.all_expressions
        )
        best_value_pred = get_best_value_metric(y_pred)
        normalized_predictions_pred = normalized_to_best_val_metric(
            y_pred, self.all_expressions
        )

        # Get true values for the next batch
        y_true = self.all_expressions[next_batch]
        # top_10_ratio_intersection_true = top_10_ratio_intersected_indices_metric(next_batch, self.all_expressions)
        best_value_true = get_best_value_metric(y_true)
        normalized_predictions_true = normalized_to_best_val_metric(
            y_true, self.all_expressions
        )

        if len(self.custom_metrics) > 0:
            # Calculate cumulative metrics by adding current value to previous cumulative value
            top_10_ratio_intersection_pred_cumulative = (
                self.custom_metrics[-1]["top_10_ratio_intersected_indices_cumulative"]
                + top_10_ratio_intersection_pred
            )
            best_value_pred_cumulative = max(
                self.custom_metrics[-1]["best_value_predictions_values_cumulative"],
                best_value_pred,
            )
            normalized_predictions_pred_cumulative = max(
                self.custom_metrics[-1][
                    "normalized_predictions_predictions_values_cumulative"
                ],
                normalized_predictions_pred,
            )
            best_value_true_cumulative = max(
                self.custom_metrics[-1]["best_value_ground_truth_values_cumulative"],
                best_value_true,
            )
            normalized_predictions_true_cumulative = max(
                self.custom_metrics[-1][
                    "normalized_predictions_ground_truth_values_cumulative"
                ],
                normalized_predictions_true,
            )
        else:
            # For first round, cumulative equals current value
            top_10_ratio_intersection_pred_cumulative = top_10_ratio_intersection_pred
            best_value_pred_cumulative = best_value_pred
            normalized_predictions_pred_cumulative = normalized_predictions_pred
            best_value_true_cumulative = best_value_true
            normalized_predictions_true_cumulative = normalized_predictions_true

        # Store custom metrics
        self.custom_metrics.append(
            {
                "top_10_ratio_intersected_indices": top_10_ratio_intersection_pred,
                "top_10_ratio_intersected_indices_cumulative": top_10_ratio_intersection_pred_cumulative,
                "best_value_predictions_values": best_value_pred,
                "normalized_predictions_predictions_values": normalized_predictions_pred,
                "best_value_ground_truth_values": best_value_true,
                "normalized_predictions_ground_truth_values": normalized_predictions_true,
                "best_value_predictions_values_cumulative": best_value_pred_cumulative,
                "normalized_predictions_predictions_values_cumulative": normalized_predictions_pred_cumulative,
                "best_value_ground_truth_values_cumulative": best_value_true_cumulative,
                "normalized_predictions_ground_truth_values_cumulative": normalized_predictions_true_cumulative,
            }
        )

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
            self._train_model()

            # Evaluate on test set
            metrics = self._evaluate_on_test_set()

            # Store results
            round_results = {
                "round": round_num + 1,
                "strategy": self.selection_strategy.value,
                "seq_mod_method": self.seq_mod_method,
                "seed": self.random_seed,
                "train_size": len(self.train_indices),
                "unlabeled_size": len(self.unlabeled_indices),
                **metrics,
            }
            self.results.append(round_results)

            # first round custom_evaluation
            # NOTE: double check
            if round_num == 0:
                self._intermediate_evaluation_custom_metrics(self.train_indices)

            # Check stopping criteria
            if len(self.unlabeled_indices) == 0:
                logger.info("No more unlabeled data available. Stopping.")
                break

            # Select next batch
            next_batch = self._select_next_batch()
            if not next_batch:
                logger.info("No more sequences to select. Stopping.")
                break

            # Evaluate custom metrics
            self._intermediate_evaluation_custom_metrics(next_batch)
            logger.info(f"Custom metrics: {self.custom_metrics} evaluated")

            # Update training set
            self.train_indices.extend(next_batch)
            self.unlabeled_indices = [
                idx for idx in self.unlabeled_indices if idx not in next_batch
            ]

            logger.info(f"Added {len(next_batch)} sequences to training set")

        logger.info(
            f"{self.selection_strategy.value.capitalize()} experiment completed!"
        )
        return self.results

    def save_results(self, output_path: str) -> None:
        """
        Save experiment results to CSV file.

        Args:
            output_path: Path to save results CSV
        """
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        # Save custom metrics if available
        if self.custom_metrics:
            custom_metrics_path = output_path.replace(".csv", "_custom_metrics.csv")
            custom_metrics_df = pd.DataFrame(self.custom_metrics)

            # Add metadata columns to match with main results
            custom_metrics_df["strategy"] = self.selection_strategy.value
            custom_metrics_df["seq_mod_method"] = self.seq_mod_method
            custom_metrics_df["regression_model"] = self.model.__class__.__name__
            custom_metrics_df["seed"] = self.random_seed
            custom_metrics_df["round"] = range(1, len(self.custom_metrics) + 1)

            # Calculate correct train_size for each round
            # Round 0: initial_sample_size (8)
            # Round 1: initial_sample_size + batch_size (16)
            # Round 2: initial_sample_size + 2*batch_size (24)
            # etc.
            train_sizes = []
            for i in range(len(self.custom_metrics)):
                if i == 0:
                    # First custom metric is for initial training set
                    train_size = self.initial_sample_size
                else:
                    # Subsequent metrics are after adding each batch
                    train_size = self.initial_sample_size + (i * self.batch_size)
                train_sizes.append(train_size)
            custom_metrics_df["train_size"] = train_sizes

            # Reorder columns to put metadata first
            metadata_cols = [
                "round",
                "strategy",
                "seed",
                "train_size",
                "regression_model",
                "seq_mod_method",
            ]
            cols = metadata_cols + [
                col for col in custom_metrics_df.columns if col not in metadata_cols
            ]
            custom_metrics_df = custom_metrics_df[cols]

            custom_metrics_df.to_csv(custom_metrics_path, index=False)
            logger.info(f"Custom metrics saved to {custom_metrics_path}")

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


def run_controlled_experiment(
    data_path: str,
    strategies: List[SelectionStrategy],
    regression_models: List[RegressionModelType],
    seq_mod_methods: List[SequenceModificationMethod],
    seeds: List[int],
    initial_sample_size: int = 8,
    batch_size: int = 8,
    test_size: int = 50,
    no_test: bool = True,
    max_rounds: int = 20,
    output_dir: str = "results",
    normalize_input_output: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run controlled experiments comparing different selection strategies with multiple seeds.

    Args:
        data_path: Path to CSV file with sequence and expression data
        strategies: List of selection strategies to compare
        seq_mod_methods: List of sequence modification methods to compare
        seeds: List of random seeds for multiple replicates
        initial_sample_size: Number of sequences to start with
        batch_size: Number of sequences to select in each round
        test_size: Number of sequences reserved for testing
        no_test: Whether to use the test set
        max_rounds: Maximum number of rounds per experiment
        output_dir: Directory to save results

    Returns:
        Dictionary mapping strategy names to their results across all seeds
    """
    all_results = {}
    all_custom_metrics = {}  # Add storage for custom metrics
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Running controlled experiment with strategies: {[s.value for s in strategies]}"
    )
    logger.info(f"Using {len(seeds)} different seeds: {seeds}")

    # Initialize results storage
    for strategy in strategies:
        all_results[strategy.value] = {}
        all_custom_metrics[strategy.value] = {}
        for seq_mod_method in seq_mod_methods:
            all_results[strategy.value][seq_mod_method.value] = {}
            all_custom_metrics[strategy.value][seq_mod_method.value] = {}
            for regression_model in regression_models:
                all_results[strategy.value][seq_mod_method.value][
                    regression_model.value
                ] = []
                all_custom_metrics[strategy.value][seq_mod_method.value][
                    regression_model.value
                ] = []

    # Create progress bar for total experiments
    total_experiments = (
        len(strategies) * len(seq_mod_methods) * len(seeds) * len(regression_models)
    )
    with tqdm(total=total_experiments, desc="Running experiments") as pbar:
        for strategy in strategies:
            for seq_mod_method in seq_mod_methods:
                for regression_model in regression_models:
                    logger.info(f"\n{'=' * 60}")
                    logger.info(
                        f"Running {strategy.value.upper()} strategy with regressor {regression_model} with {len(seeds)} seeds and {seq_mod_method.value.upper()} sequence modification method"
                    )
                    logger.info(f"{'=' * 60}")

                    for seed_idx, seed in enumerate(seeds):
                        logger.info(
                            f"\n--- {strategy.value.upper()} Strategy - Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---"
                        )

                        # Create experiment with specific strategy and seed
                        experiment = ActiveLearningExperiment(
                            data_path=data_path,
                            selection_strategy=strategy,
                            regression_model=regression_model,
                            initial_sample_size=initial_sample_size,
                            batch_size=batch_size,
                            test_size=test_size,
                            random_seed=seed,
                            seq_mod_method=seq_mod_method.value,
                            no_test=no_test,
                            normalize_input_output=normalize_input_output,
                        )

                        # Run experiment
                        results = experiment.run_experiment(max_rounds=max_rounds)
                        all_results[strategy.value][seq_mod_method.value][
                            regression_model.value
                        ].extend(results)

                        # Collect custom metrics
                        if experiment.custom_metrics:
                            # Add metadata to custom metrics
                            for i, metrics in enumerate(experiment.custom_metrics):
                                # Calculate train_size for this round
                                # Custom metrics are collected after selecting next batch, so train_size = initial + (round * batch_size)
                                train_size_for_round = initial_sample_size + (
                                    i * batch_size
                                )

                                metrics_with_metadata = {
                                    "round": i,
                                    "strategy": strategy.value,
                                    "seq_mod_method": seq_mod_method.value,
                                    "regression_model": regression_model.value,
                                    "seed": seed,
                                    "train_size": train_size_for_round,
                                    **metrics,
                                }
                                all_custom_metrics[strategy.value][
                                    seq_mod_method.value
                                ][regression_model.value].append(metrics_with_metadata)

                        # Save individual seed results (this now also saves custom metrics)
                        seed_output_path = (
                            output_path
                            / f"{strategy.value}_{seq_mod_method.value}_{regression_model.value}_seed_{seed}_results.csv"
                        )
                        experiment.save_results(str(seed_output_path))

                        # Log final performance for this seed
                        final_performance = experiment.get_final_performance()
                        logger.info(
                            f"Seed {seed} final performance - "
                            f"Pearson: {final_performance.get('pearson_correlation', 0):.4f}, "
                            f"Spearman: {final_performance.get('spearman_correlation', 0):.4f}"
                        )

                        pbar.update(1)

    # Save combined results for each strategy (Pearson and Spearman correlation, R2, RMSE)
    for strategy in all_results.keys():
        for seq_mod_method in all_results[strategy].keys():
            for regression_model in all_results[strategy][seq_mod_method].keys():
                results = all_results[strategy][seq_mod_method][regression_model]
                if results:  # Only save if there are results
                    strategy_df = pd.DataFrame(results)
                    strategy_output_path = (
                        output_path
                        / f"{strategy}_{seq_mod_method}_{regression_model}_all_seeds_results.csv"
                    )
                    strategy_df.to_csv(strategy_output_path, index=False)
                    logger.info(
                        f"Combined {strategy} {seq_mod_method} {regression_model} results saved to {strategy_output_path}"
                    )
    # Save combined custom metrics for each strategy (top_10_ratio_intersected_indices, best_value_predictions_values, normalized_predictions_predictions_values, best_value_ground_truth_values, normalized_predictions_ground_truth_values)
    for strategy in all_custom_metrics.keys():
        for seq_mod_method in all_custom_metrics[strategy].keys():
            for regression_model in all_custom_metrics[strategy][seq_mod_method].keys():
                custom_metrics = all_custom_metrics[strategy][seq_mod_method][
                    regression_model
                ]
                if custom_metrics:  # Only save if custom metrics exist
                    custom_metrics_df = pd.DataFrame(custom_metrics)
                    custom_metrics_output_path = (
                        output_path
                        / f"{strategy}_{seq_mod_method}_{regression_model}_all_seeds_custom_metrics.csv"
                    )
                    custom_metrics_df.to_csv(custom_metrics_output_path, index=False)
                    logger.info(
                        f"Combined {strategy} {seq_mod_method} {regression_model} custom metrics saved to {custom_metrics_output_path}"
                    )

    # Create overall combined results file
    combined_results = []
    for strategy in all_results.keys():
        for seq_mod_method in all_results[strategy].keys():
            for regression_model in all_results[strategy][seq_mod_method].keys():
                results = all_results[strategy][seq_mod_method][regression_model]
                # Add seq_mod_method and regression_model to each result dictionary (in case they're missing)
                for result_dict in results:
                    result_dict_with_metadata = (
                        result_dict.copy()
                    )  # Create a copy to avoid modifying original
                    result_dict_with_metadata["seq_mod_method"] = seq_mod_method
                    result_dict_with_metadata["regression_model"] = regression_model
                    combined_results.append(result_dict_with_metadata)

    combined_df = pd.DataFrame(combined_results)
    combined_output_path = output_path / "combined_all_results.csv"
    combined_df.to_csv(combined_output_path, index=False)
    logger.info(f"All combined results saved to {combined_output_path}")

    # Create overall combined custom metrics file
    combined_custom_metrics = []
    for strategy in all_custom_metrics.keys():
        for seq_mod_method in all_custom_metrics[strategy].keys():
            for regression_model in all_custom_metrics[strategy][seq_mod_method].keys():
                custom_metrics = all_custom_metrics[strategy][seq_mod_method][
                    regression_model
                ]
                # Add seq_mod_method and regression_model to each custom metric dictionary
                for metric_dict in custom_metrics:
                    metric_dict_with_metadata = (
                        metric_dict.copy()
                    )  # Create a copy to avoid modifying original
                    metric_dict_with_metadata["seq_mod_method"] = seq_mod_method
                    metric_dict_with_metadata["regression_model"] = regression_model
                    combined_custom_metrics.append(metric_dict_with_metadata)

    if combined_custom_metrics:  # Only save if there are custom metrics
        combined_custom_metrics_df = pd.DataFrame(combined_custom_metrics)
        combined_custom_metrics_output_path = (
            output_path / "combined_all_custom_metrics.csv"
        )
        combined_custom_metrics_df.to_csv(
            combined_custom_metrics_output_path, index=False
        )
        logger.info(
            f"All combined custom metrics saved to {combined_custom_metrics_output_path}"
        )

    # Ensure combined results are created even if experiments were interrupted
    create_combined_results_from_files(output_path)

    return all_results


def create_combined_results_from_files(output_path: Path) -> None:
    """
    Create combined results files from individual experiment files.
    This is useful when experiments are interrupted but individual files exist.
    """
    import re

    # Find all individual results files
    results_files = list(output_path.glob("*_results.csv"))
    custom_metrics_files = list(output_path.glob("*_custom_metrics.csv"))

    if not results_files:
        logger.warning("No individual results files found to combine")
        return

    # Combine results files
    all_results = []
    for file_path in results_files:
        filename = file_path.stem
        # Parse filename: strategy_seqmod_regressor_seed_X_results
        # Handle complex regressor names like "KNN_regression" or "linear_regresion"
        pattern = r"([^_]+)_([^_]+)_(.+)_seed_(\d+)_results"
        match = re.match(pattern, filename)

        if not match:
            logger.warning(f"Could not parse filename {filename}")
            continue

        strategy, seq_mod_method, regression_model, seed = match.groups()
        seed = int(seed)

        try:
            df = pd.read_csv(file_path)
            # Add metadata columns if missing
            df["strategy"] = strategy
            df["seq_mod_method"] = seq_mod_method
            df["regression_model"] = regression_model
            df["seed"] = seed
            all_results.append(df)
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            continue

    if all_results:
        combined_df = pd.DataFrame(pd.concat(all_results, ignore_index=True))
        combined_output_path = output_path / "combined_all_results.csv"
        combined_df.to_csv(combined_output_path, index=False)
        logger.info(
            f"Combined results from {len(results_files)} files saved to {combined_output_path}"
        )

    # Combine custom metrics files
    all_custom_metrics = []
    for file_path in custom_metrics_files:
        filename = file_path.stem.replace("_custom_metrics", "")
        pattern = r"([^_]+)_([^_]+)_(.+)_seed_(\d+)"
        match = re.match(pattern, filename)

        if not match:
            logger.warning(f"Could not parse custom metrics filename {filename}")
            continue

        strategy, seq_mod_method, regression_model, seed = match.groups()
        seed = int(seed)

        try:
            df = pd.read_csv(file_path)
            # Add metadata columns if missing
            if "strategy" not in df.columns:
                df["strategy"] = strategy
            if "seq_mod_method" not in df.columns:
                df["seq_mod_method"] = seq_mod_method
            if "regression_model" not in df.columns:
                df["regression_model"] = regression_model
            if "seed" not in df.columns:
                df["seed"] = seed
            all_custom_metrics.append(df)
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            continue

    if all_custom_metrics:
        combined_custom_df = pd.DataFrame(
            pd.concat(all_custom_metrics, ignore_index=True)
        )
        combined_custom_output_path = output_path / "combined_all_custom_metrics.csv"
        combined_custom_df.to_csv(combined_custom_output_path, index=False)
        logger.info(
            f"Combined custom metrics from {len(custom_metrics_files)} files saved to {combined_custom_output_path}"
        )


def analyze_multi_seed_results(
    results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze results across multiple seeds for each strategy.

    Args:
        results: Dictionary mapping strategy names to their results

    Returns:
        Dictionary with aggregated statistics for each strategy
    """
    # dict within dict
    analysis = defaultdict(lambda: defaultdict(dict))

    for strategy, strategy_results in results.items():
        for seq_mod_method, seq_mod_method_results in strategy_results.items():
            for (
                regression_model,
                regression_model_results,
            ) in seq_mod_method_results.items():
                if not regression_model_results:
                    continue

                # Convert to DataFrame for easier analysis
                df = pd.DataFrame(regression_model_results)

                # Group by round to get statistics across seeds
                round_stats = (
                    df.groupby("round")
                    .agg(
                        {
                            "pearson_correlation": ["mean", "std", "min", "max"],
                            "spearman_correlation": ["mean", "std", "min", "max"],
                            "r2": ["mean", "std", "min", "max"],
                            "rmse": ["mean", "std", "min", "max"],
                            "train_size": "first",  # Should be same across seeds for same round
                        }
                    )
                    .round(4)
                )

                # Final round statistics
                final_round = df["round"].max()
                final_results = df[df["round"] == final_round]

                final_stats = {
                    "final_pearson_mean": final_results["pearson_correlation"].mean(),
                    "final_pearson_std": final_results["pearson_correlation"].std(),
                    "final_spearman_mean": final_results["spearman_correlation"].mean(),
                    "final_spearman_std": final_results["spearman_correlation"].std(),
                    "final_r2_mean": final_results["r2"].mean(),
                    "final_r2_std": final_results["r2"].std(),
                    "final_rmse_mean": final_results["rmse"].mean(),
                    "final_rmse_std": final_results["rmse"].std(),
                    "final_train_size": final_results["train_size"].iloc[0],
                    "n_seeds": len(final_results),
                    "n_rounds": final_round,
                }

                analysis[strategy][seq_mod_method][regression_model] = {
                    "round_statistics": round_stats,
                    "final_statistics": final_stats,
                }

    return analysis


def compare_strategies_performance(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Compare and log performance across different strategies with statistical analysis.

    Args:
        results: Dictionary mapping strategy names to their results
    """
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-SEED STRATEGY COMPARISON")
    logger.info("=" * 80)

    analysis = analyze_multi_seed_results(results)

    for strategy, seq_mod_methods in analysis.items():
        for seq_mod_method, regression_models in seq_mod_methods.items():
            for (
                regression_model,
                regression_model_analysis,
            ) in regression_models.items():
                final_stats = regression_model_analysis["final_statistics"]

                logger.info(
                    f"\n{strategy.upper()} {seq_mod_method.upper()} {regression_model.upper()} Strategy (across {final_stats['n_seeds']} seeds):"
                )
                logger.info(f"  Final Training Size: {final_stats['final_train_size']}")
                logger.info(
                    f"  Final Pearson Correlation: {final_stats['final_pearson_mean']:.4f} ± {final_stats['final_pearson_std']:.4f}"
                )
                logger.info(
                    f"  Final Spearman Correlation: {final_stats['final_spearman_mean']:.4f} ± {final_stats['final_spearman_std']:.4f}"
                )
                logger.info(
                    f"  Final R²: {final_stats['final_r2_mean']:.4f} ± {final_stats['final_r2_std']:.4f}"
                )
                logger.info(
                    f"  Final RMSE: {final_stats['final_rmse_mean']:.2f} ± {final_stats['final_rmse_std']:.2f}"
                )

    # Statistical comparison between strategies - now simplified to just log a summary
    logger.info("\nSTATISTICAL COMPARISON SUMMARY:")
    # TODO: Come back and double check this calculation - the len() logic doesn't make sense
    # depends on whether seq_mod_method.values() is returning a list of lists or just a list\
    # that would make the difference
    total_configs = sum(
        len(regression_models)
        for seq_mod_methods in analysis.values()
        for regression_models in seq_mod_methods.values()
    )
    logger.info(f"Total configurations analyzed: {total_configs}")
    logger.info(
        "Detailed comparisons available in the saved CSV files for further analysis."
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run active learning experiments for DNA sequence-expression prediction"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/experiment_configs.yaml",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="Name of specific experiment to run from config file",
    )

    parser.add_argument(
        "--list-experiments",
        "-l",
        action="store_true",
        help="List all available experiments in config file",
    )

    parser.add_argument(
        "--run-all", action="store_true", help="Run all experiments in config file"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    return parser


def main() -> None:
    """Main function to run controlled active learning experiments with multiple seeds."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Import config loader (avoid circular import)
    try:
        from utils.config_loader import (
            list_available_experiments,
            run_experiment_from_config,
        )
    except ImportError:
        logger.error(
            "Could not import config_loader. Make sure PyYAML is installed: pip install pyyaml"
        )
        return

    # Handle list experiments
    if args.list_experiments:
        print("Available experiments:")
        experiments = list_available_experiments(args.config)
        for exp in experiments:
            print(f"  - {exp}")
        return

    # Handle run all experiments
    if args.run_all:
        experiments = list_available_experiments(args.config)
        logger.info(f"Running all {len(experiments)} experiments...")

        for exp_name in experiments:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running experiment: {exp_name}")
            logger.info(f"{'=' * 60}")

            try:
                if args.dry_run:
                    run_experiment_from_config(exp_name, args.config, dry_run=True)
                else:
                    results = run_experiment_from_config(exp_name, args.config)
                    logger.info(f"Completed experiment: {exp_name}")
            except Exception as e:
                logger.error(f"Error running experiment {exp_name}: {e}")
                continue

        logger.info("All experiments completed!")
        return

    # Handle single experiment
    if args.experiment:
        try:
            if args.dry_run:
                run_experiment_from_config(args.experiment, args.config, dry_run=True)
            else:
                results = run_experiment_from_config(args.experiment, args.config)
                logger.info(f"Experiment {args.experiment} completed successfully!")
                logger.info(f"Results: {results}")
        except Exception as e:
            logger.error(f"Error running experiment {args.experiment}: {e}")
        return

    # Fallback to hardcoded config if no arguments provided
    logger.warning("No experiment specified. Using hardcoded config...")
    logger.warning(
        "Use --help to see available options or --list-experiments to see available configs"
    )

    config = {
        "data_path": "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/384_Data/embeddings/384_rice/post_embedding/combined_sequence_data_rank_0_ori_log_likelihood.safetensors",
        "strategies": [
            SelectionStrategy.HIGH_EXPRESSION,
            SelectionStrategy.RANDOM,
            SelectionStrategy.LOG_LIKELIHOOD,
        ],
        "regression_models": [
            RegressionModelType.KNN,
            RegressionModelType.LINEAR,
            RegressionModelType.RANDOM_FOREST,
        ],
        "seq_mod_methods": [SequenceModificationMethod.EMBEDDING],
        "seeds": [42, 123, 456, 789, 999],
        "initial_sample_size": 8,
        "batch_size": 8,
        "test_size": 30,
        "max_rounds": 2,
        "normalize_input_output": False,
        "output_dir": "results_all_strategies_ori_log_likelihood_embeddings_no_test_normalization",
        "no_test": True,
    }

    logger.info("Starting Multi-Seed Controlled Active Learning Experiments")
    logger.info(f"Configuration: {config}")

    # Run controlled experiments
    all_results = run_controlled_experiment(**config)
    if not config["no_test"]:
        # Analyze and compare performance across strategies
        # NOTE: only compare performance if test set is used
        compare_strategies_performance(all_results)

        # Save summary analysis
        analysis = analyze_multi_seed_results(all_results)

        # Create summary DataFrame
        summary_data = []
        for strategy, strategy_analysis in analysis.items():
            for seq_mod_method, seq_mod_method_analysis in strategy_analysis.items():
                for (
                    regression_model,
                    regression_model_analysis,
                ) in seq_mod_method_analysis.items():
                    final_stats = regression_model_analysis["final_statistics"]
                    summary_data.append(
                        {
                            "strategy": strategy,
                            "seq_mod_method": seq_mod_method,
                            "regression_model": regression_model,
                            "n_seeds": final_stats["n_seeds"],
                            "final_pearson_mean": final_stats["final_pearson_mean"],
                            "final_pearson_std": final_stats["final_pearson_std"],
                            "final_spearman_mean": final_stats["final_spearman_mean"],
                            "final_spearman_std": final_stats["final_spearman_std"],
                            "final_r2_mean": final_stats["final_r2_mean"],
                            "final_r2_std": final_stats["final_r2_std"],
                            "final_rmse_mean": final_stats["final_rmse_mean"],
                            "final_rmse_std": final_stats["final_rmse_std"],
                            "final_train_size": final_stats["final_train_size"],
                        }
                    )

        summary_df = pd.DataFrame(summary_data)
        summary_path = Path(config["output_dir"]) / "summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary statistics saved to {summary_path}")

        logger.info("\nAll multi-seed experiments completed successfully!")


if __name__ == "__main__":
    main()
