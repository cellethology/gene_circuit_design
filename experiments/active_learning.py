"""
Active Learning Loop for genetic circuit design.

This script implements an active learning approach to design circuit with specific function.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hydra.utils import instantiate
from omegaconf import ListConfig, OmegaConf

from experiments.core.experiment import ActiveLearningExperiment


def run_one_experiment(
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
    feature_transforms = make_steps(cfg.feature_transforms.steps)
    target_transforms = make_steps(cfg.target_transforms.steps)

    # Extract active learning settings
    embeddings_path = cfg.embedding_path
    dataset_name = getattr(cfg, "dataset_name", "")
    embedding_model_name = getattr(cfg, "embedding_model", "")
    metadata_path = cfg.metadata_path
    al_settings = cfg.al_settings
    batch_size = al_settings.get("batch_size", 8)
    starting_batch_size = al_settings.get("starting_batch_size", batch_size)
    max_rounds = al_settings.get("max_rounds", 30)
    output_dir = al_settings.get("output_dir", None)
    label_key = al_settings.get("label_key", None)
    seed = al_settings.get("seed", 0)

    # Check embedding path matches embedding model
    if not embeddings_path.endswith(f"{embedding_model_name}.npz"):
        raise ValueError(
            f"Embedding path {embeddings_path} does not match embedding model {embedding_model_name}"
        )

    # Create experiment
    experiment = ActiveLearningExperiment(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        query_strategy=query_strategy,
        predictor=predictor,
        starting_batch_size=starting_batch_size,
        batch_size=batch_size,
        random_seed=seed,
        feature_transforms=feature_transforms,
        target_transforms=target_transforms,
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
    feature_transforms_names = (
        [item[0] for item in feature_transforms] if feature_transforms else []
    )
    target_transforms_names = (
        [item[0] for item in target_transforms] if target_transforms else []
    )
    summary = {
        "dataset_name": dataset_name,
        "embedding_model": embedding_model_name,
        "query_strategy": query_strategy.name,
        "predictor": predictor.__class__.__name__,
        "initial_selection": initial_selection_strategy.name,
        "feature_transforms": feature_transforms_names,
        "target_transforms": target_transforms_names,
        "seed": seed,
        "auc_normalized_true": aucs["normalized_true"],
        "auc_normalized_pred": aucs["normalized_pred"],
        "auc_top_proportion": aucs["top_proportion"],
    }

    # Persist summary for downstream aggregation
    summary_path = output_dir_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def make_steps(steps_cfg: ListConfig) -> List[Tuple[str, Any]]:
    """
    Make a list of (name, transformer) steps from a list of step configurations.

    Args:
        steps_cfg: List of step configurations

    Returns:
        List of (name, transformer) steps
    """
    steps: List[Tuple[str, Any]] = []
    for step_cfg in steps_cfg:
        step_dict = OmegaConf.to_container(step_cfg, resolve=True)
        name = step_dict.pop("id")
        transformer = instantiate(step_dict)
        steps.append((name, transformer))
    return steps
