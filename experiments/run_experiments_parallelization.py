"""
Active Learning Loop for DNA Sequence-Expression Prediction.

This script implements an active learning approach to predict gene expression
from DNA sequences using linear regression with one-hot encoded features.
"""

import argparse
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.core.experiment import ActiveLearningExperiment
from utils.config_loader import SelectionStrategy
from utils.model_loader import RegressionModelType
from utils.sequence_utils import SequenceModificationMethod

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


def run_single_experiment(
    experiment_config: Dict[str, Any],
) -> Tuple[str, str, str, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run a single experiment with given configuration.

    Args:
        experiment_config: Dictionary containing all experiment parameters

    Returns:
        Tuple of (strategy, seq_mod_method, regression_model, seed, results, custom_metrics)
    """
    # Extract parameters from config
    data_path = experiment_config["data_path"]
    strategy = experiment_config["strategy"]
    regression_model = experiment_config["regression_model"]
    seq_mod_method = experiment_config["seq_mod_method"]
    seed = experiment_config["seed"]
    initial_sample_size = experiment_config["initial_sample_size"]
    batch_size = experiment_config["batch_size"]
    test_size = experiment_config["test_size"]
    no_test = experiment_config["no_test"]
    max_rounds = experiment_config["max_rounds"]
    normalize_input_output = experiment_config["normalize_input_output"]
    output_dir = experiment_config["output_dir"]
    target_val_key = experiment_config["target_val_key"]
    use_pca = experiment_config.get("use_pca", False)
    pca_components = experiment_config.get("pca_components", 4096)

    # Create experiment
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
        use_pca=use_pca,
        pca_components=pca_components,
        target_val_key=target_val_key,
    )

    # Run experiment
    results = experiment.run_experiment(max_rounds=max_rounds)

    # Process custom metrics
    custom_metrics = []
    if experiment.custom_metrics:
        for i, metrics in enumerate(experiment.custom_metrics):
            train_size_for_round = initial_sample_size + (i * batch_size)
            metrics_with_metadata = {
                "round": i,
                "strategy": strategy.value,
                "seq_mod_method": seq_mod_method.value,
                "regression_model": regression_model.value,
                "seed": seed,
                "train_size": train_size_for_round,
                **metrics,
            }
            custom_metrics.append(metrics_with_metadata)

    # Save individual results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seed_output_path = (
        output_path
        / f"{strategy.value}_{seq_mod_method.value}_{regression_model.value}_seed_{seed}_results.csv"
    )
    experiment.save_results(str(seed_output_path))

    # Log final performance
    final_performance = experiment.get_final_performance()
    logger.info(
        f"Seed {seed} final performance - "
        f"Pearson: {final_performance.get('pearson_correlation', 0):.4f}, "
        f"Spearman: {final_performance.get('spearman_correlation', 0):.4f}"
    )

    return (
        strategy.value,
        seq_mod_method.value,
        regression_model.value,
        seed,
        results,
        custom_metrics,
    )


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
    max_workers: int = None,
    use_pca: bool = False,
    pca_components: int = 4096,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run controlled experiments comparing different selection strategies with multiple seeds using parallel processing.

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
        max_workers: Maximum number of parallel workers (None for auto-detection)

    Returns:
        Dictionary mapping strategy names to their results across all seeds
    """
    all_results = {}
    all_custom_metrics = {}  # Add storage for custom metrics
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if LOG_LIKELIHOOD strategy is viable by checking data availability
    if SelectionStrategy.LOG_LIKELIHOOD in strategies:
        logger.info("Checking log likelihood data availability...")
        try:
            # Create temporary experiment to check log likelihood data
            temp_exp = ActiveLearningExperiment(
                data_path=data_path,
                selection_strategy=SelectionStrategy.RANDOM,  # Use random for temp check
                initial_sample_size=initial_sample_size,
                batch_size=batch_size,
                test_size=test_size,
                random_seed=42,
                no_test=no_test,
                normalize_input_output=normalize_input_output,
                use_pca=use_pca,
                pca_components=pca_components,
            )

            # Check if log likelihood data is available and not all NaN
            if (
                hasattr(temp_exp, "all_log_likelihoods")
                and len(temp_exp.all_log_likelihoods) > 0
            ):
                if np.all(np.isnan(temp_exp.all_log_likelihoods)):
                    logger.warning(
                        "LOG_LIKELIHOOD strategy requested but all log likelihood values are NaN. Skipping LOG_LIKELIHOOD strategy."
                    )
                    strategies = [
                        s for s in strategies if s != SelectionStrategy.LOG_LIKELIHOOD
                    ]
                else:
                    logger.info(
                        "Log likelihood data is available. LOG_LIKELIHOOD strategy will be used."
                    )
            else:
                logger.warning(
                    "LOG_LIKELIHOOD strategy requested but no log likelihood data found. Skipping LOG_LIKELIHOOD strategy."
                )
                strategies = [
                    s for s in strategies if s != SelectionStrategy.LOG_LIKELIHOOD
                ]

        except Exception as e:
            logger.error(
                f"Error checking log likelihood data: {e}. Skipping LOG_LIKELIHOOD strategy."
            )
            strategies = [
                s for s in strategies if s != SelectionStrategy.LOG_LIKELIHOOD
            ]

    logger.info(
        f"Running controlled experiment with strategies: {[s.value for s in strategies]}"
    )
    logger.info(f"Using {len(seeds)} different seeds: {seeds}")
    logger.info(f"Using parallel processing with max_workers: {max_workers}")

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

    # Create experiment configurations for parallel processing
    experiment_configs = []
    for strategy in strategies:
        for seq_mod_method in seq_mod_methods:
            for regression_model in regression_models:
                for seed in seeds:
                    config = {
                        "data_path": data_path,
                        "strategy": strategy,
                        "seq_mod_method": seq_mod_method,
                        "regression_model": regression_model,
                        "seed": seed,
                        "initial_sample_size": initial_sample_size,
                        "batch_size": batch_size,
                        "test_size": test_size,
                        "no_test": no_test,
                        "max_rounds": max_rounds,
                        "normalize_input_output": normalize_input_output,
                        "output_dir": output_dir,
                        "use_pca": use_pca,
                        "pca_components": pca_components,
                    }
                    experiment_configs.append(config)

    # Run experiments in parallel
    total_experiments = len(experiment_configs)
    logger.info(f"Running {total_experiments} experiments in parallel...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_config = {
            executor.submit(run_single_experiment, config): config
            for config in experiment_configs
        }

        # Process completed experiments with progress bar
        with tqdm(total=total_experiments, desc="Running experiments") as pbar:
            for future in as_completed(future_to_config):
                try:
                    (
                        strategy_name,
                        seq_mod_method_name,
                        regression_model_name,
                        seed,
                        results,
                        custom_metrics,
                    ) = future.result()

                    # Store results
                    all_results[strategy_name][seq_mod_method_name][
                        regression_model_name
                    ].extend(results)

                    # Store custom metrics
                    if custom_metrics:
                        all_custom_metrics[strategy_name][seq_mod_method_name][
                            regression_model_name
                        ].extend(custom_metrics)

                    logger.info(
                        f"Completed experiment: {strategy_name}_{seq_mod_method_name}_{regression_model_name}_seed_{seed}"
                    )

                except Exception as e:
                    config = future_to_config[future]
                    logger.error(
                        f"Experiment failed: {config['strategy'].value}_{config['seq_mod_method'].value}_{config['regression_model'].value}_seed_{config['seed']} - {e}"
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

    # Find all individual results files (exclude combined files)
    results_files = [
        f
        for f in output_path.glob("*_results.csv")
        if "_all_seeds_" not in f.name and "combined_all_" not in f.name
    ]
    custom_metrics_files = [
        f
        for f in output_path.glob("*_custom_metrics.csv")
        if "_all_seeds_" not in f.name and "combined_all_" not in f.name
    ]

    if not results_files:
        logger.warning("No individual results files found to combine")
        return

    # Combine results files
    all_results = []
    for file_path in results_files:
        filename = file_path.stem
        # Parse filename: strategy_seqmod_regressor_seed_X_results
        # Handle complex regressor names like "KNN_regression" or "linear_regresion"
        pattern = r"([^_]+)_([^_]+)_(.+?)_seed_(\d+)_results"
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
        pattern = r"([^_]+)_([^_]+)_(.+?)_seed_(\d+)"
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

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: auto-detect)",
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
            run_experiment_from_config_parallel,
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
                    run_experiment_from_config_parallel(
                        exp_name, args.config, dry_run=True
                    )
                else:
                    results = run_experiment_from_config_parallel(
                        exp_name,
                        args.config,
                        max_workers=getattr(args, "max_workers", None),
                    )
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
                run_experiment_from_config_parallel(
                    args.experiment, args.config, dry_run=True
                )
            else:
                results = run_experiment_from_config_parallel(
                    args.experiment,
                    args.config,
                    max_workers=getattr(args, "max_workers", None),
                )
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

    # Run controlled experiments with max_workers parameter
    config["max_workers"] = getattr(args, "max_workers", None)
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


def parallelization():
    """Example parallelization function - kept for reference but not used in main flow."""
    # This function is kept for reference but not used in the main flow
    pass


if __name__ == "__main__":
    main()
