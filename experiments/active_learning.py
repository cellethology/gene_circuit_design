"""
Active Learning Loop for genetic circuit design.

This script implements an active learning approach to design circuit with specific function.
"""

import json
from pathlib import Path
from typing import Any, Dict

from hydra.utils import instantiate

from experiments.core.experiment import ActiveLearningExperiment


def run_single_experiment(
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a single experiment with given configuration.

    Args:
        cfg: Dictionary containing all experiment parameters

    Returns:
        Dictionary containing the results summary
    """

    # Instantiate components
    predictor = instantiate(cfg.predictor)
    query_strategy = instantiate(cfg.query_strategy)
    initial_selection_strategy = instantiate(cfg.initial_selection_strategy)
    seed = cfg.seed

    # Extract active learning settings
    embeddings_path = cfg.embedding_path
    metadata_path = cfg.metadata_path
    al_settings = cfg.al_settings
    batch_size = al_settings.get("batch_size", 8)
    starting_batch_size = al_settings.get("starting_batch_size", batch_size)
    max_rounds = al_settings.get("max_rounds", 30)
    normalize_features = al_settings.get("normalize_features", True)
    normalize_labels = al_settings.get("normalize_labels", True)
    output_dir = al_settings.get("output_dir", None)
    label_key = al_settings.get("label_key", None)

    # Create experiment
    experiment = ActiveLearningExperiment(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        query_strategy=query_strategy,
        predictor=predictor,
        starting_batch_size=starting_batch_size,
        batch_size=batch_size,
        random_seed=seed,
        normalize_features=normalize_features,
        normalize_labels=normalize_labels,
        label_key=label_key,
        initial_selection_strategy=initial_selection_strategy,
    )

    # Resolve and create output directory
    if output_dir is None:
        raise ValueError("al_settings.output_dir must be provided in the config.")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Run experiment
    experiment.run_experiment(max_rounds=max_rounds)

    # Save individual results
    experiment.save_results(output_path=output_dir_path / "results.csv")

    # Compute AUC
    aucs = experiment.round_tracker.compute_auc(
        metric_columns=["normalized_true", "normalized_pred", "top_proportion"]
    )

    # Summarize results
    summary = {
        "embedding_path": embeddings_path,
        "metadata_path": metadata_path,
        "query_strategy": query_strategy.name,
        "predictor": predictor.__class__.__name__,
        "initial_selection": initial_selection_strategy.name,
        "normalize_features": normalize_features,
        "normalize_labels": normalize_labels,
        "seed": seed,
        "auc_normalized_true": aucs["normalized_true"],
        "auc_normalized_pred": aucs["normalized_pred"],
        "auc_top_proportion": aucs["top_proportion"],
    }

    # Persist summary for downstream aggregation
    summary_path = output_dir_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary
