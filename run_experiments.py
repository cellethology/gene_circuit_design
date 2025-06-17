"""
Active Learning Loop for DNA Sequence-Expression Prediction.

This script implements an active learning approach to predict gene expression
from DNA sequences using linear regression with one-hot encoded features.
"""

import logging  # noqa: I001
import random
from typing import List, Dict, Any
from pathlib import Path
from enum import Enum
from tqdm import tqdm  # noqa: I001

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from utils.sequence_utils import (
    load_sequence_data,
    one_hot_encode_sequences,
    flatten_one_hot_sequences,
    calculate_sequence_statistics,
    trim_sequences_to_length
)

from utils.metrics import (
    top_10_ratio_intersection_metric,
    get_best_value_metric,
    normalized_to_best_val_metric
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelectionStrategy(str, Enum):
    """Enumeration of available selection strategies."""
    HIGH_EXPRESSION = "highExpression"  # Select sequences with highest predicted expression
    RANDOM = "random"  # Select sequences randomly
    LOG_LIKELIHOOD = "log_likelihood"  # Select sequences with highest log likelihood
    UNCERTAINTY = "uncertainty"  # Select sequences with highest prediction uncertainty (future extension)


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
        initial_sample_size: int = 8,
        batch_size: int = 8,
        test_size: int = 50,
        random_seed: int = 42,
        trim_sequences: bool = True
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
        """
        self.data_path = data_path
        self.selection_strategy = selection_strategy
        self.initial_sample_size = initial_sample_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.trim_sequences = trim_sequences

        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Data containers
        self.all_sequences: List[str] = []
        self.all_expressions: np.ndarray = np.array([])
        self.all_log_likelihoods: np.ndarray = np.array([])  # Add log likelihood storage
        self.train_indices: List[int] = []
        self.test_indices: List[int] = []
        self.unlabeled_indices: List[int] = []

        # Model and results
        self.model = LinearRegression()
        self.results: List[Dict[str, Any]] = []
        self.custom_metrics: List[Dict[str, Any]] = []

        # Load and prepare data
        self._load_and_prepare_data(trim_sequences=self.trim_sequences)

        logger.info(f"Experiment initialized with {self.selection_strategy.value} selection strategy (seed={random_seed})")

    def _load_and_prepare_data(self, trim_sequences: bool = True) -> None:
        """Load data and create train/test/unlabeled splits."""
        logger.info(f"Loading data from {self.data_path}")

        # Check if this is the combined dataset with log likelihood or the original expression-only dataset
        df = pd.read_csv(self.data_path)

        if 'Log_Likelihood' in df.columns:
            # Combined dataset with log likelihood
            sequences = df['Sequence'].tolist()
            if trim_sequences:
                sequences = trim_sequences_to_length(sequences)
                logger.info(f"Trimmed sequences to length {len(sequences[0])}")
            self.all_sequences = sequences
            self.all_expressions = df['Expression'].values
            self.all_log_likelihoods = df['Log_Likelihood'].values
            logger.info(f"Loaded combined dataset with {len(self.all_sequences)} sequences including log likelihood data")
        else:
            # Original expression-only dataset
            self.all_sequences, self.all_expressions = load_sequence_data(self.data_path, trim_sequences=trim_sequences)
            self.all_log_likelihoods = np.full(len(self.all_sequences), np.nan)  # Fill with NaN if no log likelihood
            logger.info(f"Loaded expression-only dataset with {len(self.all_sequences)} sequences")

        total_samples = len(self.all_sequences)

        # Log sequence statistics
        stats = calculate_sequence_statistics(self.all_sequences)
        logger.info(f"Sequence statistics: {stats}")

        # Log log likelihood statistics if available
        if not np.all(np.isnan(self.all_log_likelihoods)):
            ll_stats = pd.Series(self.all_log_likelihoods).describe()
            logger.info(f"Log likelihood statistics:\n{ll_stats}")

        # Create indices for all samples
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)

        # Reserve test set
        self.test_indices = all_indices[:self.test_size]
        remaining_indices = all_indices[self.test_size:]

        # Initial training set
        self.train_indices = remaining_indices[:self.initial_sample_size]
        self.unlabeled_indices = remaining_indices[self.initial_sample_size:]

        logger.info(f"Data split - Train: {len(self.train_indices)}, "
                   f"Test: {len(self.test_indices)}, "
                   f"Unlabeled: {len(self.unlabeled_indices)}")

    def _encode_sequences(self, indices: List[int]) -> np.ndarray:
        """
        One-hot encode sequences at given indices.

        Args:
            indices: List of sequence indices to encode

        Returns:
            Flattened one-hot encoded sequences
        """
        sequences = [self.all_sequences[i] for i in indices]
        encoded = one_hot_encode_sequences(sequences)
        return flatten_one_hot_sequences(encoded)

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
            'rmse': rmse,
            'r2': r2,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p
        }

        logger.info(f"Test metrics - RMSE: {rmse:.2f}, R²: {r2:.3f}, "
                   f"Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")

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
        logger.info(f"HIGH_EXPRESSION: Selected {len(selected_indices)} sequences with predicted expressions: "
                   f"[{', '.join(f'{pred:.1f}' for pred in selected_predictions)}]")

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
        logger.info(f"RANDOM: Selected {len(selected_indices)} sequences with actual expressions: "
                   f"[{', '.join(f'{expr:.1f}' for expr in selected_expressions)}]")

        return selected_indices

    def _select_next_batch_log_likelihood(self) -> List[int]:
        """
        Select next batch using log likelihood (highest log likelihood values).

        Returns:
            List of indices for next batch
        """
        # Check if log likelihood data is available
        if np.all(np.isnan(self.all_log_likelihoods)):
            logger.warning("No log likelihood data available. Falling back to random selection.")
            return self._select_next_batch_random()

        # Get log likelihood values for unlabeled sequences
        unlabeled_log_likelihoods = self.all_log_likelihoods[self.unlabeled_indices]

        # Filter out NaN values
        valid_mask = ~np.isnan(unlabeled_log_likelihoods)
        valid_unlabeled_indices = [self.unlabeled_indices[i] for i in range(len(self.unlabeled_indices)) if valid_mask[i]]
        valid_log_likelihoods = unlabeled_log_likelihoods[valid_mask]

        if len(valid_unlabeled_indices) == 0:
            logger.warning("No valid log likelihood values for unlabeled sequences. Falling back to random selection.")
            return self._select_next_batch_random()

        # Select indices with highest log likelihood values (less negative = higher probability)
        sorted_indices = np.argsort(valid_log_likelihoods)[::-1]  # Descending order (highest first)
        batch_size = min(self.batch_size, len(valid_unlabeled_indices))
        selected_local_indices = sorted_indices[:batch_size]

        # Convert to global indices
        selected_indices = [valid_unlabeled_indices[i] for i in selected_local_indices]

        # Log selection info
        selected_log_likelihoods = valid_log_likelihoods[selected_local_indices]
        selected_expressions = self.all_expressions[selected_indices]
        logger.info(f"LOG_LIKELIHOOD: Selected {len(selected_indices)} sequences with log likelihoods: "
                   f"[{', '.join(f'{ll:.4f}' for ll in selected_log_likelihoods)}] "
                   f"and actual expressions: [{', '.join(f'{expr:.1f}' for expr in selected_expressions)}]")

        return selected_indices

    def _select_next_batch(self) -> List[int]:
        """
        Select next batch of sequences based on the configured strategy.

        Returns:
            List of indices for next batch
        """
        if len(self.unlabeled_indices) == 0:
            return []

        if self.selection_strategy == SelectionStrategy.HIGH_EXPRESSION:
            return self._select_next_batch_active()
        elif self.selection_strategy == SelectionStrategy.RANDOM:
            return self._select_next_batch_random()
        elif self.selection_strategy == SelectionStrategy.LOG_LIKELIHOOD:
            return self._select_next_batch_log_likelihood()
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
        top_10_ratio_intersection_pred = top_10_ratio_intersection_metric(y_pred, self.all_expressions)
        best_value_pred = get_best_value_metric(y_pred)
        normalized_predictions_pred = normalized_to_best_val_metric(y_pred, self.all_expressions)

        # Get true values for the next batch
        y_true = self.all_expressions[next_batch]
        top_10_ratio_intersection_true = top_10_ratio_intersection_metric(y_true, self.all_expressions)
        best_value_true = get_best_value_metric(y_true)
        normalized_predictions_true = normalized_to_best_val_metric(y_true, self.all_expressions)

        # Store custom metrics
        self.custom_metrics.append({
            'top_10_ratio_intersection_pred': top_10_ratio_intersection_pred,
            'best_value_pred': best_value_pred,
            'normalized_predictions_pred': normalized_predictions_pred,
            'top_10_ratio_intersection_true': top_10_ratio_intersection_true,
            'best_value_true': best_value_true,
            'normalized_predictions_true': normalized_predictions_true
        })

    def run_experiment(self, max_rounds: int = 20) -> List[Dict[str, Any]]:
        """
        Run the active learning experiment.

        Args:
            max_rounds: Maximum number of active learning rounds

        Returns:
            List of results for each round
        """
        logger.info(f"Starting {self.selection_strategy.value} learning experiment with {max_rounds} max rounds")

        for round_num in range(max_rounds):
            logger.info(f"\n--- Round {round_num + 1} ({self.selection_strategy.value.upper()}) ---")

            # Train model
            self._train_model()

            # Evaluate on test set
            metrics = self._evaluate_on_test_set()

            # Store results
            round_results = {
                'round': round_num + 1,
                'strategy': self.selection_strategy.value,
                'seed': self.random_seed,
                'train_size': len(self.train_indices),
                'unlabeled_size': len(self.unlabeled_indices),
                **metrics
            }
            self.results.append(round_results)

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
            self.unlabeled_indices = [idx for idx in self.unlabeled_indices
                                    if idx not in next_batch]

            logger.info(f"Added {len(next_batch)} sequences to training set")

        logger.info(f"{self.selection_strategy.value.capitalize()} experiment completed!")
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
            custom_metrics_path = output_path.replace('.csv', '_custom_metrics.csv')
            custom_metrics_df = pd.DataFrame(self.custom_metrics)

            # Add metadata columns to match with main results
            custom_metrics_df['strategy'] = self.selection_strategy.value
            custom_metrics_df['seed'] = self.random_seed
            custom_metrics_df['round'] = range(1, len(self.custom_metrics) + 1)

            # Reorder columns to put metadata first
            cols = ['round', 'strategy', 'seed'] + [col for col in custom_metrics_df.columns if col not in ['round', 'strategy', 'seed']]
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

        return {k: v for k, v in self.results[-1].items()
                if k not in ['round', 'strategy', 'seed', 'train_size', 'unlabeled_size']}


def run_controlled_experiment(
    data_path: str,
    strategies: List[SelectionStrategy],
    seeds: List[int],
    initial_sample_size: int = 8,
    batch_size: int = 8,
    test_size: int = 50,
    max_rounds: int = 20,
    output_dir: str = "results"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run controlled experiments comparing different selection strategies with multiple seeds.

    Args:
        data_path: Path to CSV file with sequence and expression data
        strategies: List of selection strategies to compare
        seeds: List of random seeds for multiple replicates
        initial_sample_size: Number of sequences to start with
        batch_size: Number of sequences to select in each round
        test_size: Number of sequences reserved for testing
        max_rounds: Maximum number of rounds per experiment
        output_dir: Directory to save results

    Returns:
        Dictionary mapping strategy names to their results across all seeds
    """
    all_results = {}
    all_custom_metrics = {}  # Add storage for custom metrics
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running controlled experiment with strategies: {[s.value for s in strategies]}")
    logger.info(f"Using {len(seeds)} different seeds: {seeds}")

    # Initialize results storage
    for strategy in strategies:
        all_results[strategy.value] = []
        all_custom_metrics[strategy.value] = []  # Initialize custom metrics storage

    # Create progress bar for total experiments
    total_experiments = len(strategies) * len(seeds)
    with tqdm(total=total_experiments, desc="Running experiments") as pbar:

        for strategy in strategies:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {strategy.value.upper()} strategy with {len(seeds)} seeds")
            logger.info(f"{'='*60}")

            for seed_idx, seed in enumerate(seeds):
                logger.info(f"\n--- {strategy.value.upper()} Strategy - Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")

                # Create experiment with specific strategy and seed
                experiment = ActiveLearningExperiment(
                    data_path=data_path,
                    selection_strategy=strategy,
                    initial_sample_size=initial_sample_size,
                    batch_size=batch_size,
                    test_size=test_size,
                    random_seed=seed
                )

                # Run experiment
                results = experiment.run_experiment(max_rounds=max_rounds)
                all_results[strategy.value].extend(results)

                # Collect custom metrics
                if experiment.custom_metrics:
                    # Add metadata to custom metrics
                    for i, metrics in enumerate(experiment.custom_metrics):
                        metrics_with_metadata = {
                            'round': i + 1,
                            'strategy': strategy.value,
                            'seed': seed,
                            **metrics
                        }
                        all_custom_metrics[strategy.value].append(metrics_with_metadata)

                # Save individual seed results (this now also saves custom metrics)
                seed_output_path = output_path / f"{strategy.value}_seed_{seed}_results.csv"
                experiment.save_results(str(seed_output_path))

                # Log final performance for this seed
                final_performance = experiment.get_final_performance()
                logger.info(f"Seed {seed} final performance - "
                           f"Pearson: {final_performance.get('pearson_correlation', 0):.4f}, "
                           f"Spearman: {final_performance.get('spearman_correlation', 0):.4f}")

                pbar.update(1)

    # Save combined results for each strategy
    for strategy, results in all_results.items():
        strategy_df = pd.DataFrame(results)
        strategy_output_path = output_path / f"{strategy}_all_seeds_results.csv"
        strategy_df.to_csv(strategy_output_path, index=False)
        logger.info(f"Combined {strategy} results saved to {strategy_output_path}")

    # Save combined custom metrics for each strategy
    for strategy, custom_metrics in all_custom_metrics.items():
        if custom_metrics:  # Only save if custom metrics exist
            custom_metrics_df = pd.DataFrame(custom_metrics)
            custom_metrics_output_path = output_path / f"{strategy}_all_seeds_custom_metrics.csv"
            custom_metrics_df.to_csv(custom_metrics_output_path, index=False)
            logger.info(f"Combined {strategy} custom metrics saved to {custom_metrics_output_path}")

    # Create overall combined results file
    combined_results = []
    for strategy, results in all_results.items():
        combined_results.extend(results)

    combined_df = pd.DataFrame(combined_results)
    combined_output_path = output_path / "combined_all_results.csv"
    combined_df.to_csv(combined_output_path, index=False)
    logger.info(f"All combined results saved to {combined_output_path}")

    # Create overall combined custom metrics file
    combined_custom_metrics = []
    for strategy, custom_metrics in all_custom_metrics.items():
        combined_custom_metrics.extend(custom_metrics)

    if combined_custom_metrics:  # Only save if there are custom metrics
        combined_custom_metrics_df = pd.DataFrame(combined_custom_metrics)
        combined_custom_metrics_output_path = output_path / "combined_all_custom_metrics.csv"
        combined_custom_metrics_df.to_csv(combined_custom_metrics_output_path, index=False)
        logger.info(f"All combined custom metrics saved to {combined_custom_metrics_output_path}")

    return all_results


def analyze_multi_seed_results(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze results across multiple seeds for each strategy.

    Args:
        results: Dictionary mapping strategy names to their results

    Returns:
        Dictionary with aggregated statistics for each strategy
    """
    analysis = {}

    for strategy, strategy_results in results.items():
        if not strategy_results:
            continue

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(strategy_results)

        # Group by round to get statistics across seeds
        round_stats = df.groupby('round').agg({
            'pearson_correlation': ['mean', 'std', 'min', 'max'],
            'spearman_correlation': ['mean', 'std', 'min', 'max'],
            'r2': ['mean', 'std', 'min', 'max'],
            'rmse': ['mean', 'std', 'min', 'max'],
            'train_size': 'first'  # Should be same across seeds for same round
        }).round(4)

        # Final round statistics
        final_round = df['round'].max()
        final_results = df[df['round'] == final_round]

        final_stats = {
            'final_pearson_mean': final_results['pearson_correlation'].mean(),
            'final_pearson_std': final_results['pearson_correlation'].std(),
            'final_spearman_mean': final_results['spearman_correlation'].mean(),
            'final_spearman_std': final_results['spearman_correlation'].std(),
            'final_r2_mean': final_results['r2'].mean(),
            'final_r2_std': final_results['r2'].std(),
            'final_rmse_mean': final_results['rmse'].mean(),
            'final_rmse_std': final_results['rmse'].std(),
            'final_train_size': final_results['train_size'].iloc[0],
            'n_seeds': len(final_results),
            'n_rounds': final_round
        }

        analysis[strategy] = {
            'round_statistics': round_stats,
            'final_statistics': final_stats
        }

    return analysis


def compare_strategies_performance(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Compare and log performance across different strategies with statistical analysis.

    Args:
        results: Dictionary mapping strategy names to their results
    """
    logger.info("\n" + "="*80)
    logger.info("MULTI-SEED STRATEGY COMPARISON")
    logger.info("="*80)

    analysis = analyze_multi_seed_results(results)

    for strategy, strategy_analysis in analysis.items():
        final_stats = strategy_analysis['final_statistics']

        logger.info(f"\n{strategy.upper()} Strategy (across {final_stats['n_seeds']} seeds):")
        logger.info(f"  Final Training Size: {final_stats['final_train_size']}")
        logger.info(f"  Final Pearson Correlation: {final_stats['final_pearson_mean']:.4f} ± {final_stats['final_pearson_std']:.4f}")
        logger.info(f"  Final Spearman Correlation: {final_stats['final_spearman_mean']:.4f} ± {final_stats['final_spearman_std']:.4f}")
        logger.info(f"  Final R²: {final_stats['final_r2_mean']:.4f} ± {final_stats['final_r2_std']:.4f}")
        logger.info(f"  Final RMSE: {final_stats['final_rmse_mean']:.2f} ± {final_stats['final_rmse_std']:.2f}")

    # Statistical comparison between strategies
    if len(analysis) == 2:
        strategies = list(analysis.keys())
        strategy1, strategy2 = strategies[0], strategies[1]

        stats1 = analysis[strategy1]['final_statistics']
        stats2 = analysis[strategy2]['final_statistics']

        logger.info(f"\nSTATISTICAL COMPARISON:")
        logger.info(f"Pearson Correlation Difference ({strategy1} - {strategy2}): "
                   f"{stats1['final_pearson_mean'] - stats2['final_pearson_mean']:.4f}")
        logger.info(f"Spearman Correlation Difference ({strategy1} - {strategy2}): "
                   f"{stats1['final_spearman_mean'] - stats2['final_spearman_mean']:.4f}")


def main() -> None:
    """Main function to run controlled active learning experiments with multiple seeds."""
    # Configuration
    config = {
        'data_path': '/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/384_Data/combined_sequence_data.csv',
        'strategies': [SelectionStrategy.HIGH_EXPRESSION, SelectionStrategy.RANDOM, SelectionStrategy.LOG_LIKELIHOOD],
        'seeds': [42, 123, 456, 789, 999],  # 5 different seeds
        'initial_sample_size': 8,
        'batch_size': 8,
        'test_size': 50,
        'max_rounds': 20,
        'output_dir': 'results_all_strategies'
    }

    logger.info("Starting Multi-Seed Controlled Active Learning Experiments with Log Likelihood Strategy")
    logger.info(f"Configuration: {config}")

    # Run controlled experiments
    all_results = run_controlled_experiment(**config)

    # Analyze and compare performance across strategies
    compare_strategies_performance(all_results)

    # Save summary analysis
    analysis = analyze_multi_seed_results(all_results)

    # Create summary DataFrame
    summary_data = []
    for strategy, strategy_analysis in analysis.items():
        final_stats = strategy_analysis['final_statistics']
        summary_data.append({
            'strategy': strategy,
            'n_seeds': final_stats['n_seeds'],
            'final_pearson_mean': final_stats['final_pearson_mean'],
            'final_pearson_std': final_stats['final_pearson_std'],
            'final_spearman_mean': final_stats['final_spearman_mean'],
            'final_spearman_std': final_stats['final_spearman_std'],
            'final_r2_mean': final_stats['final_r2_mean'],
            'final_r2_std': final_stats['final_r2_std'],
            'final_rmse_mean': final_stats['final_rmse_mean'],
            'final_rmse_std': final_stats['final_rmse_std'],
            'final_train_size': final_stats['final_train_size']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(config['output_dir']) / "summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary statistics saved to {summary_path}")

    logger.info("\nAll multi-seed experiments completed successfully!")


if __name__ == "__main__":
    main()
