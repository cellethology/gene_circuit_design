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


def run_experiment(
    config_path: str,
    experiment_name: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> bool:
    """Run a single experiment or all experiments in a config."""
    logger.info(f"Running experiments from config: {config_path}")

    cmd = ["python", "run_all_experiments.py", "--config", config_path]

    if experiment_name:
        cmd.extend(["--experiment", experiment_name])
        logger.info(f"Running specific experiment: {experiment_name}")
    else:
        logger.info("Running all experiments in config")

    if max_workers:
        cmd.extend(["--max-workers", str(max_workers)])

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("Experiments completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiments failed: {e}")
        return False


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
            ["python", "plotting/visualize_all_results.py"],
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
    parser.add_argument(
        "--results-dir", type=str, help="Specific results directory to plot"
    )

    args = parser.parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    logger.info("Starting combined experiment and plotting pipeline")

    success = True

    # Run experiments unless skipped
    if not args.skip_experiments:
        success = run_experiment(args.config, args.experiment, args.max_workers)
        if not success:
            logger.error("Experiments failed, stopping")
            sys.exit(1)

    # Generate plots unless skipped
    if not args.skip_plots:
        if args.results_dir:
            # Plot specific results directory
            plot_success = generate_plots_for_results(args.results_dir)
        else:
            # Find and plot all result directories
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

    # Create summary report
    result_dirs = find_all_result_dirs()
    if result_dirs:
        create_summary_report(result_dirs)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
