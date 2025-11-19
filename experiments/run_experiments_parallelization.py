"""
Active Learning Loop for DNA Sequence-Expression Prediction.

This script implements an active learning approach to predict gene expression
from DNA sequences using linear regression with one-hot encoded features.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from hydra.utils import instantiate

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
    cfg: Dict[str, Any],
) -> Tuple[str, str, str, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run a single experiment with given configuration.

    Args:
        cfg: Dictionary containing all experiment parameters

    Returns:
        Tuple of (strategy, seq_mod_method, regression_model, seed, results, custom_metrics)
    """

    data_path = cfg.data_paths
    predictor = cfg.predictor
    query_strategy = cfg.query_strategy
    al_settings = cfg.al_settings

    from utils.sequence_utils import SequenceModificationMethod

    # Instantiate query strategy and predictor
    query_strategy = instantiate(query_strategy)
    predictor = instantiate(predictor)
    predictor_name = predictor.__class__.__name__

    # Instantiate sequence modification method
    seq_mod_method = al_settings["seq_mod_method"]
    if isinstance(seq_mod_method, str):
        seq_mod_method = SequenceModificationMethod(seq_mod_method)

    seed = al_settings["seed"]
    initial_sample_size = al_settings["initial_sample_size"]
    batch_size = al_settings["batch_size"]
    test_size = al_settings["test_size"]
    no_test = al_settings["no_test"]
    max_rounds = al_settings["max_rounds"]
    normalize_input_output = al_settings["normalize_input_output"]
    output_dir = al_settings["output_dir"]
    target_val_key = al_settings["target_val_key"]
    use_pca = al_settings.get("use_pca", False)
    pca_components = al_settings.get("pca_components", 4096)

    # Create experiment
    experiment = ActiveLearningExperiment(
        data_path=data_path,
        query_strategy=query_strategy,
        predictor=predictor,
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
                "strategy": query_strategy,
                "seq_mod_method": seq_mod_method,
                "predictor": predictor_name,
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
        / f"{query_strategy.name}_{seq_mod_method}_{predictor_name}_seed_{seed}_results.csv"
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
        query_strategy,
        seq_mod_method,
        predictor_name,
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
        pattern = r"([^_]+)_([^_]+)_([^_]+)_seed_(\d+)_results"
        match = re.match(pattern, filename)

        if not match:
            logger.warning(f"Could not parse filename {filename}")
            continue

        strategy, seq_mod_method, predictor, seed = match.groups()
        seed = int(seed)

        try:
            df = pd.read_csv(file_path)
            # Add metadata columns if missing
            df["strategy"] = strategy
            df["seq_mod_method"] = seq_mod_method
            df["predictor"] = predictor
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
        pattern = r"([^_]+)_([^_]+)_([^_]+)_seed_(\d+)"
        match = re.match(pattern, filename)

        if not match:
            logger.warning(f"Could not parse custom metrics filename {filename}")
            continue

        strategy, seq_mod_method, predictor, seed = match.groups()
        seed = int(seed)

        try:
            df = pd.read_csv(file_path)
            # Add metadata columns if missing
            if "strategy" not in df.columns:
                df["strategy"] = strategy
            if "seq_mod_method" not in df.columns:
                df["seq_mod_method"] = seq_mod_method
            if "predictor" not in df.columns:
                df["predictor"] = predictor
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
