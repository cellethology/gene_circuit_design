"""
Active Learning Loop for genetic circuit design.

This script implements an active learning approach to design circuit with specific function.
"""

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

    predictor = instantiate(cfg.predictor)
    query_strategy = instantiate(cfg.query_strategy)
    initial_selection_strategy = instantiate(cfg.initial_selection_strategy)
    seed = cfg.seed

    predictor_name = predictor.__class__.__name__

    embeddings_path = cfg.embedding_path
    metadata_path = cfg.metadata_path

    al_settings = cfg.al_settings
    batch_size = al_settings.get("batch_size", 8)
    initial_sample_size = al_settings.get("initial_sample_size", batch_size)
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
        initial_sample_size=initial_sample_size,
        batch_size=batch_size,
        random_seed=seed,
        normalize_features=normalize_features,
        normalize_labels=normalize_labels,
        label_key=label_key,
        initial_selection_strategy=initial_selection_strategy,
    )

    # Run experiment
    experiment.run_experiment(max_rounds=max_rounds)

    # Save individual results
    experiment.save_results(output_path=Path(output_dir) / "results.csv")

    # Compute AUC
    auc = experiment.round_tracker.compute_auc(metric_column="normalized_true")

    # Summarize results
    results_summary = {
        "strategy": query_strategy,
        "predictor": predictor_name,
        "seed": seed,
        "auc": auc,
    }

    return results_summary
