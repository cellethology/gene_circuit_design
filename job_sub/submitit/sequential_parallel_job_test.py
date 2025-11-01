"""
Sequential Parallel Job Submission Script

This script provides functionality to run multiple experiment configurations on a Slurm cluster
using submitit. It allows for running experiments with different parameter combinations in parallel,
with each combination submitted as a separate Slurm job.

Usage:
    python sequential_parallel_job_test.py --config-files config1.yaml config2.yaml --experiment-names exp1 exp2

    Or import the run_slurm_experiments function in your own script:
    from job_sub.submitit.sequential_parallel_job_test import run_slurm_experiments
    run_slurm_experiments(config_files=["config1.yaml"], experiment_names=["exp1"])
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import submitit
import yaml

from experiments.run_experiments_parallelization import run_single_experiment
from utils.config_loader import get_experiment_config
from utils.plotting import create_combined_results_from_files

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_all_experiments_from_config(
    config_path: str = "configs/166k_regulators_auto_gen.yaml",
) -> None:
    """Run SLURM experiments for all experiments defined in a YAML config."""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Load YAML
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Extract experiment names under the top-level 'experiments' key
    experiments = list(config.get("experiments", {}).keys())

    if not experiments:
        print(f"No experiments found in {config_file}")
        return

    print(f"Running {len(experiments)} experiments from {config_file}...")

    for exp_name in experiments:
        print(f"→ Submitting {exp_name} ...")
        run_slurm_experiments(
            config_files=[str(config_file)],
            experiment_names=[exp_name],
        )


def run_slurm_experiments(
    config_files: List[str],
    experiment_names: List[str],
    slurm_params: Optional[Dict[str, Any]] = None,
    executor_folder: str = "logs_experiments",
) -> None:
    """
    Run multiple experiment configurations on a Slurm cluster using submitit.

    Args:
        config_files: List of configuration file paths
        experiment_names: List of experiment names to run from the config files
                         (must match the length of config_files)
        slurm_params: Optional dictionary of Slurm parameters to override defaults
        executor_folder: Folder to store submitit logs
    """
    # Validate inputs
    if len(config_files) != len(experiment_names):
        raise ValueError("config_files and experiment_names must have the same length")

    # Validate config files exist
    for config_file in config_files:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # Set up default Slurm parameters
    # NOTE: suits for onehot no PCA (update on the readme.md)
    default_slurm_params = {
        "timeout_min": 240,
        "slurm_partition": "wzt_20250411,intel-sc3",
        "slurm_cpus_per_task": 8,
        "slurm_mail_user": "lizelun@westlake.edu.cn",
        "slurm_mail_type": "BEGIN,END,FAIL",
        "slurm_qos": "huge",
        "slurm_mem_per_cpu": "8GB",
    }

    # Override defaults with provided parameters
    if slurm_params:
        default_slurm_params.update(slurm_params)

    # Initialize the executor
    executor = submitit.AutoExecutor(folder=executor_folder)
    executor.update_parameters(**default_slurm_params)

    all_experiment_params = []
    output_dirs = []

    # Process each configuration
    for config_file, experiment_name in zip(config_files, experiment_names):
        logger.info(f"Processing experiment: {experiment_name} from {config_file}")

        try:
            # Load the configuration
            config = get_experiment_config(
                experiment_name=experiment_name, config_file=config_file
            )

            # Validate required parameters
            required_params = [
                "initial_sample_size",
                "data_path",
                "target_val_key",
                "batch_size",
                "test_size",
                "no_test",
                "max_rounds",
                "output_dir",
                "normalize_input_output",
                "strategies",
                "regression_models",
                "seq_mod_methods",
                "seeds",
            ]

            for param in required_params:
                if param not in config:
                    raise ValueError(
                        f"Required parameter '{param}' not found in experiment '{experiment_name}'"
                    )

            # Extract common parameters
            initial_sample_size = config["initial_sample_size"]
            target_val_key = config["target_val_key"]
            data_path = config["data_path"]
            batch_size = config["batch_size"]
            test_size = config["test_size"]
            no_test = config["no_test"]
            max_rounds = config["max_rounds"]
            output_dir = config["output_dir"]
            normalize_input_output = config["normalize_input_output"]

            # Store output directory for later use
            output_dirs.append(output_dir)

            # Generate parameter combinations
            for strategy in config["strategies"]:
                for regression_model in config["regression_models"]:
                    for seq_mod_method in config["seq_mod_methods"]:
                        for seed in config["seeds"]:
                            all_experiment_params.append(
                                {
                                    "strategy": strategy,
                                    "regression_model": regression_model,
                                    "target_val_key": target_val_key,
                                    "seq_mod_method": seq_mod_method,
                                    "seed": seed,
                                    "data_path": data_path,
                                    "initial_sample_size": initial_sample_size,
                                    "batch_size": batch_size,
                                    "test_size": test_size,
                                    "no_test": no_test,
                                    "max_rounds": max_rounds,
                                    "output_dir": output_dir,
                                    "normalize_input_output": normalize_input_output,
                                }
                            )
        except Exception as e:
            logger.error(f"Error processing experiment {experiment_name}: {str(e)}")
            raise

    # Submit all jobs
    logger.info(f"Submitting {len(all_experiment_params)} jobs to Slurm")
    jobs = executor.map_array(run_single_experiment, all_experiment_params)

    # Wait for all jobs to complete
    logger.info("Waiting for jobs to complete...")
    results = [job.result() for job in jobs]
    logger.info("All jobs completed")

    # Combine results for each output directory
    for output_dir in set(output_dirs):
        logger.info(f"Combining results in {output_dir}")
        create_combined_results_from_files(output_path=Path(output_dir))


def parse_arguments():
    """
    Parse command line arguments for running experiments on Slurm.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run experiments on Slurm using submitit"
    )

    # Required arguments
    parser.add_argument(
        "--config-files",
        type=str,
        nargs="+",
        required=True,
        help="List of configuration file paths",
    )
    parser.add_argument(
        "--experiment-names",
        type=str,
        nargs="+",
        required=True,
        help="List of experiment names to run from the config files",
    )

    # Optional Slurm parameters
    parser.add_argument(
        "--timeout-min", type=int, default=240, help="Timeout in minutes for each job"
    )
    parser.add_argument(
        "--slurm-partition",
        type=str,
        default="wzt_20250411,intel-fat",
        help="Slurm partition to use",
    )
    parser.add_argument(
        "--slurm-cpus-per-task", type=int, default=8, help="Number of CPUs per task"
    )
    parser.add_argument(
        "--slurm-mem-per-cpu", type=str, default="8GB", help="Memory per CPU"
    )
    parser.add_argument(
        "--slurm-mail-user",
        type=str,
        default="lizelun@westlake.edu.cn",
        help="Email address for job notifications",
    )
    parser.add_argument(
        "--slurm-mail-type",
        type=str,
        default="BEGIN,END,FAIL",
        help="When to send email notifications",
    )
    parser.add_argument(
        "--slurm-qos", type=str, default="huge", help="Slurm QOS to use"
    )
    parser.add_argument(
        "--executor-folder",
        type=str,
        default="logs_experiments",
        help="Folder to store submitit logs",
    )

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    # Check if arguments are provided
    import sys

    if len(sys.argv) > 2:
        # Parse command line arguments
        args = parse_arguments()

        # Extract Slurm parameters
        slurm_params = {
            "timeout_min": args.timeout_min,
            "slurm_partition": args.slurm_partition,
            "slurm_cpus_per_task": args.slurm_cpus_per_task,
            "slurm_mem_per_cpu": args.slurm_mem_per_cpu,
            "slurm_mail_user": args.slurm_mail_user,
            "slurm_mail_type": args.slurm_mail_type,
            "slurm_qos": args.slurm_qos,
        }

        # Run experiments with provided arguments
        run_slurm_experiments(
            config_files=args.config_files,
            experiment_names=args.experiment_names,
            slurm_params=slurm_params,
            executor_folder=args.executor_folder,
        )
    else:
        args = parse_arguments()
        config_files = args.config_files
        run_all_experiments_from_config(
            config_path=config_files,
        )
