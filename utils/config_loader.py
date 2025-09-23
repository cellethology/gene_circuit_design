"""
Configuration loading utilities for active learning experiments.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils.model_loader import RegressionModelType
from utils.sequence_utils import SequenceModificationMethod


class SelectionStrategy(str, Enum):
    """Enumeration of available selection strategies."""

    HIGH_EXPRESSION = (
        "highExpression"  # Select sequences with highest predicted expression
    )
    RANDOM = "random"  # Select sequences randomly
    LOG_LIKELIHOOD = "log_likelihood"  # Select sequences with highest log likelihood
    UNCERTAINTY = "uncertainty"  # Select sequences with highest prediction uncertainty (future extension)


def load_experiment_config(
    config_file: str = "configs/experiment_configs.yaml",
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file.

    Args:
        config_file: Path to the YAML configuration file
        experiment_name: Specific experiment to load. If None, returns all experiments.

    Returns:
        Dictionary containing experiment configuration(s)

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If experiment_name doesn't exist in config
        ValueError: If config format is invalid
    """
    config_path = Path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    if "experiments" not in config_data:
        raise ValueError("Configuration file must contain 'experiments' section")

    experiments = config_data["experiments"]

    if experiment_name is None:
        return experiments

    if experiment_name not in experiments:
        available = list(experiments.keys())
        raise KeyError(
            f"Experiment '{experiment_name}' not found. Available: {available}"
        )

    return experiments[experiment_name]


def convert_config_to_enums(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert string values in config to appropriate enum types.

    Args:
        config: Raw configuration dictionary

    Returns:
        Configuration with enums converted
    """
    # Convert strategies from strings to enums
    if "strategies" in config:
        strategy_map = {
            "HIGH_EXPRESSION": SelectionStrategy.HIGH_EXPRESSION,
            "RANDOM": SelectionStrategy.RANDOM,
            "LOG_LIKELIHOOD": SelectionStrategy.LOG_LIKELIHOOD,
            "UNCERTAINTY": SelectionStrategy.UNCERTAINTY,
        }
        config["strategies"] = [
            strategy_map[strategy] for strategy in config["strategies"]
        ]

    # Convert sequence modification methods from strings to enums
    if "seq_mod_methods" in config:
        seq_mod_map = {
            "TRIM": SequenceModificationMethod.TRIM,
            "PAD": SequenceModificationMethod.PAD,
            "EMBEDDING": SequenceModificationMethod.EMBEDDING,
            "CAR": SequenceModificationMethod.CAR,
        }
        config["seq_mod_methods"] = [
            seq_mod_map[method] for method in config["seq_mod_methods"]
        ]

    # Convert regression models from strings to enums (with default if missing)
    if "regression_models" in config:
        regression_model_map = {
            "LINEAR": RegressionModelType.LINEAR,
            "KNN": RegressionModelType.KNN,
            "RANDOM_FOREST": RegressionModelType.RANDOM_FOREST,
            "XGBOOST": RegressionModelType.XGBOOST,
            "MLP": RegressionModelType.MLP
        }
        config["regression_models"] = [
            regression_model_map[model] for model in config["regression_models"]
        ]
    else:
        # Default to all three models if not specified
        config["regression_models"] = [
            RegressionModelType.LINEAR,
            RegressionModelType.KNN,
            RegressionModelType.RANDOM_FOREST,
            RegressionModelType.XGBOOST,
        ]

    return config


def list_available_experiments(
    config_file: str = "configs/experiment_configs.yaml"
) -> List[str]:
    """
    List all available experiment configurations.

    Args:
        config_file: Path to the YAML configuration file

    Returns:
        List of experiment names
    """
    try:
        experiments = load_experiment_config(config_file)
        return list(experiments.keys())
    except Exception as e:
        print(f"Error loading config: {e}")
        return []


def get_experiment_config(
    experiment_name: str, config_file: str = "configs/experiment_configs.yaml"
) -> Dict[str, Any]:
    """
    Get a complete experiment configuration ready for use.

    Args:
        experiment_name: Name of the experiment to load
        config_file: Path to the YAML configuration file

    Returns:
        Complete experiment configuration with enums converted
    """
    # Load the raw config
    raw_config = load_experiment_config(config_file, experiment_name)

    # Convert strings to enums
    config = convert_config_to_enums(raw_config)

    return config


def run_experiment_from_config(
    experiment_name: str,
    config_file: str = "configs/experiment_configs.yaml",
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run an experiment from configuration.

    Args:
        experiment_name: Name of the experiment to run
        config_file: Path to the YAML configuration file
        dry_run: If True, only print what would be run without executing

    Returns:
        Experiment results if executed, None if dry_run
    """
    from experiments.run_experiments_parallelization import (
        run_controlled_experiment,
    )

    # Get the configuration
    config = get_experiment_config(experiment_name, config_file)

    if dry_run:
        print(f"Would run experiment '{experiment_name}' with config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        return None

    print(f"Running experiment: {experiment_name}")
    print(f"Configuration: {config}")
    # Run the experiment
    results = run_controlled_experiment(**config)

    return results


def run_experiment_from_config_parallel(
    experiment_name: str,
    config_file: str = "configs/experiment_configs.yaml",
    dry_run: bool = False,
    max_workers: int = None,
) -> Optional[Dict[str, Any]]:
    """
    Run an experiment from configuration using parallelization.

    Args:
        experiment_name: Name of the experiment to run
        config_file: Path to the YAML configuration file
        dry_run: If True, only print what would be run without executing
        max_workers: Maximum number of parallel workers (None for auto-detect)

    Returns:
        Experiment results if executed, None if dry_run
    """
    from experiments.run_experiments_parallelization import (
        run_controlled_experiment,
    )

    # Get the configuration
    config = get_experiment_config(experiment_name, config_file)

    # Handle cores_per_process configuration
    if "cores_per_process" in config:
        cores_per_process = config["cores_per_process"]
        total_cores = os.cpu_count()
        calculated_max_workers = total_cores // cores_per_process

        if total_cores % cores_per_process != 0:
            print(
                f"Warning: Total cores ({total_cores}) is not divisible by cores_per_process ({cores_per_process})"
            )
            print(
                f"Using {calculated_max_workers} workers, leaving {total_cores % cores_per_process} cores unused"
            )

        config["max_workers"] = calculated_max_workers
        print(
            f"Auto-calculated max_workers: {calculated_max_workers} (total cores: {total_cores}, cores per process: {cores_per_process})"
        )

        # Remove cores_per_process from config as it's not needed by run_controlled_experiment
        del config["cores_per_process"]
    else:
        # Use provided max_workers or None for auto-detect
        config["max_workers"] = max_workers

    if dry_run:
        print(f"Would run experiment '{experiment_name}' with config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        return None

    print(f"Running experiment: {experiment_name}")
    print(f"Configuration: {config}")

    # Run the experiment with parallelization
    results = run_controlled_experiment(**config)

    return results


def create_custom_config(
    name: str,
    data_path: str,
    strategies: List[str],
    seq_mod_methods: List[str],
    output_dir: str,
    seeds: Optional[List[int]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create a custom experiment configuration.

    Args:
        name: Name for the experiment
        data_path: Path to the data file
        strategies: List of strategy names
        seq_mod_methods: List of sequence modification method names
        output_dir: Output directory for results
        seeds: List of random seeds
        **kwargs: Additional configuration parameters

    Returns:
        Configuration dictionary
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 999]

    config = {
        "data_path": data_path,
        "strategies": strategies,
        "seq_mod_methods": seq_mod_methods,
        "output_dir": output_dir,
        "seeds": seeds,
        "initial_sample_size": kwargs.get("initial_sample_size", 8),
        "batch_size": kwargs.get("batch_size", 8),
        "test_size": kwargs.get("test_size", 30),
        "max_rounds": kwargs.get("max_rounds", 20),
        "normalize_input_output": kwargs.get("normalize_input_output", False),
        "no_test": kwargs.get("no_test", True),
    }

    # Add any additional parameters
    config.update(kwargs)

    return config


if __name__ == "__main__":
    # Example usage
    print("Available experiments:")
    experiments = list_available_experiments()
    for exp in experiments:
        print(f"  - {exp}")

    if experiments:
        print(f"\nExample config for '{experiments[0]}':")
        config = get_experiment_config(experiments[0])
        for key, value in config.items():
            print(f"  {key}: {value}")
