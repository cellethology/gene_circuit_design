#!/usr/bin/env python3
"""
Master script to run all experiments in the gene circuit design project.

This script automatically runs all experiment configurations and handles
parallelization and result organization.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/run_all_experiments.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_all_experiment_configs() -> List[Path]:
    """Get all experiment configuration files."""
    config_dir = Path("configs")
    return list(config_dir.glob("*.yaml"))


def list_experiments_in_config(config_path: Path) -> List[str]:
    """List all experiments in a configuration file."""
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "experiments/run_experiments_parallelization.py",
                "--config",
                str(config_path),
                "--list-experiments",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the output to extract experiment names
        experiments = []
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if line.strip().startswith("- "):
                experiments.append(line.strip()[2:])  # Remove '- ' prefix

        return experiments
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to list experiments in {config_path}: {e}")
        return []


def run_single_experiment(
    config_path: Path, experiment_name: str, max_workers: Optional[int] = None
) -> bool:
    """Run a single experiment configuration."""
    logger.info(f"Running experiment: {experiment_name} from {config_path}")

    cmd = [
        "uv",
        "run",
        "python",
        "experiments/run_experiments_parallelization.py",
        "--config",
        str(config_path),
        "--experiment",
        experiment_name,
    ]

    if max_workers:
        cmd.extend(["--max-workers", str(max_workers)])

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"Successfully completed experiment: {experiment_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run experiment {experiment_name}: {e}")
        return False


def run_all_experiments_in_config(
    config_path: Path, max_workers: Optional[int] = None
) -> bool:
    """Run all experiments in a configuration file."""
    logger.info(f"Running all experiments in {config_path}")

    cmd = [
        "uv",
        "run",
        "python",
        "experiments/run_experiments_parallelization.py",
        "--config",
        str(config_path),
        "--run-all",
    ]

    if max_workers:
        cmd.extend(["--max-workers", str(max_workers)])

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"Successfully completed all experiments in {config_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run experiments in {config_path}: {e}")
        return False


def run_pca_experiments(max_workers: Optional[int] = None) -> bool:
    """Run PCA-specific experiments."""
    logger.info("Running PCA experiments")

    cmd = [
        "uv",
        "run",
        "python",
        "experiments/run_experiments_parallelization.py",
        "--config",
        "scripts/test_pca_config.yaml",
        "--experiment",
        "test_pca_experiment",
    ]

    if max_workers:
        cmd.extend(["--max-workers", str(max_workers)])

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("Successfully completed PCA experiments")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run PCA experiments: {e}")
        return False


def generate_plots():
    """Generate plots for all results."""
    logger.info("Generating plots...")

    plot_scripts = [
        "plotting/visualize_all_results.py",
        "plotting/plot_regressor_comparison.py",
    ]

    for script in plot_scripts:
        if Path(script).exists():
            try:
                subprocess.run(["uv", "run", "python", script], check=True)
                logger.info(f"Successfully generated plots from {script}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to generate plots from {script}: {e}")


def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(
        description="Run all gene circuit design experiments"
    )
    parser.add_argument(
        "--config", type=str, help="Run only experiments from specific config file"
    )
    parser.add_argument("--experiment", type=str, help="Run only specific experiment")
    parser.add_argument(
        "--max-workers", type=int, help="Maximum number of parallel workers"
    )
    parser.add_argument("--skip-pca", action="store_true", help="Skip PCA experiments")
    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip plot generation"
    )
    parser.add_argument(
        "--list-all", action="store_true", help="List all available experiments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    logger.info("Starting gene circuit design experiment suite")

    # List all experiments if requested
    if args.list_all:
        config_files = get_all_experiment_configs()
        logger.info("Available experiments:")
        for config_path in config_files:
            logger.info(f"\nConfig: {config_path}")
            experiments = list_experiments_in_config(config_path)
            for exp in experiments:
                logger.info(f"  - {exp}")

        # Also list PCA experiments
        logger.info("\nPCA experiments (scripts/test_pca_config.yaml):")
        pca_experiments = list_experiments_in_config(
            Path("scripts/test_pca_config.yaml")
        )
        for exp in pca_experiments:
            logger.info(f"  - {exp}")
        return

    success_count = 0
    total_count = 0

    # Run specific config and experiment if provided
    if args.config and args.experiment:
        config_path = Path(args.config)
        if config_path.exists():
            total_count = 1
            if args.dry_run:
                logger.info(f"Would run: {args.experiment} from {config_path}")
            else:
                if run_single_experiment(
                    config_path, args.experiment, args.max_workers
                ):
                    success_count += 1
        else:
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

    # Run all experiments from specific config
    elif args.config:
        config_path = Path(args.config)
        if config_path.exists():
            total_count = 1
            if args.dry_run:
                logger.info(f"Would run all experiments from {config_path}")
            else:
                if run_all_experiments_in_config(config_path, args.max_workers):
                    success_count += 1
        else:
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

    # Run all experiments from all configs
    else:
        config_files = get_all_experiment_configs()
        total_count = len(config_files)

        if args.dry_run:
            logger.info("Would run all experiments from all config files:")
            for config_path in config_files:
                logger.info(f"  - {config_path}")
            if not args.skip_pca:
                logger.info("  - PCA experiments")
        else:
            for config_path in config_files:
                if run_all_experiments_in_config(config_path, args.max_workers):
                    success_count += 1

    # Run PCA experiments unless skipped
    if not args.skip_pca and not args.config:
        total_count += 1
        if args.dry_run:
            logger.info("Would run PCA experiments")
        else:
            if run_pca_experiments(args.max_workers):
                success_count += 1

    # Generate plots unless skipped
    if not args.skip_plots and not args.dry_run:
        generate_plots()

    if not args.dry_run:
        logger.info(
            f"\nExperiment suite completed: {success_count}/{total_count} successful"
        )

        if success_count == total_count:
            logger.info("All experiments completed successfully!")
        else:
            logger.warning(f"{total_count - success_count} experiments failed")
            sys.exit(1)
    else:
        logger.info("Dry run completed - no experiments were actually executed")


if __name__ == "__main__":
    #     The run_all_experiments.py script provides:

    #   - Run all experiments: ./run_all_experiments.py
    #   - Run specific config: ./run_all_experiments.py --config
    #   configs/experiment_configs.yaml
    #   - Run specific experiment: ./run_all_experiments.py --config
    #   configs/experiment_configs.yaml --experiment test_experiment
    #   - List all experiments: ./run_all_experiments.py --list-all
    #   - Dry run: ./run_all_experiments.py --dry-run
    #   - Skip PCA/plots: ./run_all_experiments.py --skip-pca --skip-plots
    #   - Parallel control: ./run_all_experiments.py --max-workers 4
    main()
