"""
Active Learning Loop for DNA Sequence-Expression Prediction.

This script implements an active learning approach to predict gene expression
from DNA sequences using linear regression with one-hot encoded features.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from experiments.core.experiment import ActiveLearningExperiment

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
    data_path: str,
    pipeline_param: Dict[str, Any],
) -> Tuple[str, str, str, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run a single experiment with given configuration.

    Args:
        experiment_config: Dictionary containing all experiment parameters

    Returns:
        Tuple of (strategy, seq_mod_method, regression_model, seed, results, custom_metrics)
    """
    from utils.config_loader import SelectionStrategy
    from utils.model_loader import RegressionModelType
    from utils.sequence_utils import SequenceModificationMethod

    # Extract parameters from config and convert strings to enums if needed
    strategy = pipeline_param["strategy"]
    if isinstance(strategy, str):
        strategy = SelectionStrategy(strategy)

    regression_model = pipeline_param["regression_model"]
    if isinstance(regression_model, str):
        regression_model = RegressionModelType(regression_model)

    seq_mod_method = pipeline_param["seq_mod_method"]
    if isinstance(seq_mod_method, str):
        seq_mod_method = SequenceModificationMethod(seq_mod_method)

    seed = pipeline_param["seed"]
    initial_sample_size = pipeline_param["initial_sample_size"]
    batch_size = pipeline_param["batch_size"]
    test_size = pipeline_param["test_size"]
    no_test = pipeline_param["no_test"]
    max_rounds = pipeline_param["max_rounds"]
    normalize_input_output = pipeline_param["normalize_input_output"]
    output_dir = pipeline_param["output_dir"]
    target_val_key = pipeline_param["target_val_key"]
    use_pca = pipeline_param.get("use_pca", False)
    pca_components = pipeline_param.get("pca_components", 4096)

    # Create experiment
    experiment = ActiveLearningExperiment(
        data_path=data_path,
        selection_strategy=strategy,
        regression_model=regression_model,
        initial_sample_size=initial_sample_size,
        batch_size=batch_size,
        test_size=test_size,
        random_seed=seed,
        seq_mod_method=seq_mod_method,
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
                "strategy": strategy,
                "seq_mod_method": seq_mod_method,
                "regression_model": regression_model,
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
        / f"{strategy}_{seq_mod_method}_{regression_model}_seed_{seed}_results.csv"
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
        strategy,
        seq_mod_method,
        regression_model,
        seed,
        results,
        custom_metrics,
    )


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
