#!/usr/bin/env python3
"""
Combined script to run experiments and generate plots.

This script runs experiments and automatically generates plots for the results.
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
        logging.FileHandler("logs/run_and_plot.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def extract_output_dir_from_config(
    config_path: str, experiment_name: Optional[str] = None
) -> Optional[str]:
    """Extract output directory from config file or experiment configuration."""
    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # If specific experiment is requested, look for it in experiments section
        if experiment_name and "experiments" in config:
            experiments = config["experiments"]
            if experiment_name in experiments:
                return experiments[experiment_name].get("output_dir")

        # If no specific experiment, try to find a default output_dir
        # or use the first experiment's output_dir as fallback
        if "output_dir" in config:
            return config["output_dir"]
        elif "experiments" in config and config["experiments"]:
            # Return the first experiment's output_dir as fallback
            first_exp = next(iter(config["experiments"].values()))
            return first_exp.get("output_dir", "results")

        return "results"  # Final fallback

    except Exception as e:
        logger.warning(f"Could not extract output_dir from config: {e}")
        return None


def run_experiment(
    config_path: str,
    experiment_name: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """Run a single experiment or all experiments in a config.

    Returns:
        Tuple of (success, output_dir) where output_dir is the directory where results were saved
    """
    logger.info(f"Running experiments from config: {config_path}")

    cmd = [
        "python",
        "experiments/run_experiments_parallelization.py",
        "--config",
        config_path,
    ]

    if experiment_name:
        cmd.extend(["--experiment", experiment_name])
        logger.info(f"Running specific experiment: {experiment_name}")
    else:
        logger.info("Running all experiments in config")

    if max_workers:
        cmd.extend(["--max-workers", str(max_workers)])

    try:
        # Run with real-time output instead of capturing
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        logger.info("Experiments completed successfully")

        # Extract output directory from the experiment output
        output_dir = extract_output_dir_from_config(config_path, experiment_name)

        return True, output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiments failed: {e}")
        return False, None


def generate_plots_for_results(results_dir: str) -> bool:
    """Generate plots for a specific results directory."""
    logger.info(f"Generating plots for results in: {results_dir}")

    # Check if results directory exists and has data
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning(f"Results directory does not exist: {results_dir}")
        return False

    # Check for combined results files
    combined_results = results_path / "combined_all_results.csv"
    combined_custom_metrics = results_path / "combined_all_custom_metrics.csv"

    if not combined_results.exists():
        logger.warning(f"No combined results found in {results_dir}")
        return False

    # Run visualization script
    try:
        # Change to project root and run plotting scripts
        import os

        original_cwd = os.getcwd()

        subprocess.run(
            ["python", "plotting/visualize_all_results.py", "-f", results_path],
            check=True,
            cwd=original_cwd,
        )
        logger.info("Visualization plots generated successfully")

        # Also run regressor comparison
        subprocess.run(
            ["python", "plotting/plot_regressor_comparison.py"],
            check=True,
            cwd=original_cwd,
        )
        logger.info("Regressor comparison plots generated successfully")

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Plot generation failed: {e}")
        return False


def create_summary_report(results_dirs: List[str]) -> None:
    """Create a summary report of all experiments."""
    logger.info("Creating summary report")

    summary_data = []
    for results_dir in results_dirs:
        results_path = Path(results_dir)
        if not results_path.exists():
            continue

        combined_results = results_path / "combined_all_results.csv"
        if combined_results.exists():
            try:
                import pandas as pd

                df = pd.read_csv(combined_results)
                summary_data.append(
                    {
                        "experiment": results_path.name,
                        "total_runs": len(df),
                        "strategies": df["strategy"].nunique()
                        if "strategy" in df.columns
                        else 0,
                        "seeds": df["seed"].nunique() if "seed" in df.columns else 0,
                        "regressors": df["regression_model"].nunique()
                        if "regression_model" in df.columns
                        else 0,
                        "max_rounds": df["round"].max() if "round" in df.columns else 0,
                    }
                )
            except Exception as e:
                logger.warning(f"Could not process {results_dir}: {e}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = Path("results/experiment_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary report saved to {summary_path}")

        # Print summary to console
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        for _, row in summary_df.iterrows():
            print(f"Experiment: {row['experiment']}")
            print(f"  Total runs: {row['total_runs']}")
            print(f"  Strategies: {row['strategies']}")
            print(f"  Seeds: {row['seeds']}")
            print(f"  Regressors: {row['regressors']}")
            print(f"  Max rounds: {row['max_rounds']}")
            print()


def find_all_result_dirs() -> List[str]:
    """Find all result directories."""
    results_root = Path("results")
    if not results_root.exists():
        return []

    result_dirs = []
    for item in results_root.rglob("*"):
        if item.is_dir() and (item / "combined_all_results.csv").exists():
            result_dirs.append(str(item))

    return result_dirs


def main():
    """Main function."""
    # python3 run_and_plot.py --config configs/config.yaml --experiment "experiment_name"
    parser = argparse.ArgumentParser(description="Run experiments and generate plots")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--experiment", type=str, help="Specific experiment to run")
    parser.add_argument(
        "--max-workers", type=int, help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip experiments, only generate plots",
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip plots, only run experiments"
    )

    args = parser.parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    logger.info("Starting combined experiment and plotting pipeline")

    success = True
    experiment_output_dir = None

    # Run experiments unless skipped
    if not args.skip_experiments:
        success, experiment_output_dir = run_experiment(
            args.config, args.experiment, args.max_workers
        )
        if not success:
            logger.error("Experiments failed, stopping")
            sys.exit(1)

    # Generate plots unless skipped
    if not args.skip_plots:
        if experiment_output_dir:
            # Plot only the directory from the experiment that was just run
            logger.info(
                f"Plotting results from recent experiment: {experiment_output_dir}"
            )
            plot_success = generate_plots_for_results(experiment_output_dir)
        else:
            # Fallback: Find and plot all result directories
            result_dirs = find_all_result_dirs()
            logger.info(f"Found {len(result_dirs)} result directories")

            plot_success = True
            for results_dir in result_dirs:
                if not generate_plots_for_results(results_dir):
                    plot_success = False

        if plot_success:
            logger.info("All plots generated successfully")
        else:
            logger.warning("Some plots failed to generate")

    # Create summary report - only for recent experiment or all if no recent experiment
    if experiment_output_dir:
        # Only create summary for the recent experiment
        if Path(experiment_output_dir).exists():
            create_summary_report([experiment_output_dir])
    else:
        # Fallback: create summary for all result directories
        result_dirs = find_all_result_dirs()
        if result_dirs:
            create_summary_report(result_dirs)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
