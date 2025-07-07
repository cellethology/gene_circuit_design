#!/usr/bin/env python3
"""
Comprehensive visualization script for all active learning experiment results.

This script automatically discovers all results folders and generates visualizations
for both standard metrics and custom metrics using the plotting utilities.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple


from utils.plotting import (
    plot_active_learning_metrics,
    plot_top10_ratio_metrics,
    plot_value_metrics,
    plot_regressor_comparison,
    q1,
    q3,
    sem,
    STATEGY_LABELS,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define consistent color scheme for strategies
STRATEGY_COLORS = {
    "highExpression": "#1f77b4",  # Blue
    "log_likelihood": "#ff7f0e",  # Orange
    "random": "#2ca02c",  # Green
}


def discover_results_folders(base_path: str = ".") -> List[Path]:
    """
    Discover all results folders in the base path.

    Args:
        base_path: Base directory to search for results folders

    Returns:
        List of Path objects for results folders
    """
    base_path = Path(base_path)
    results_folders = []

    # Look for folders that start with "results" at the top level
    for folder in base_path.glob("results*"):
        if folder.is_dir():
            results_folders.append(folder)

    # Also look inside the results/ directory for nested results folders
    results_dir = base_path / "results"
    if results_dir.exists() and results_dir.is_dir():
        for folder in results_dir.glob("results*"):
            if folder.is_dir():
                results_folders.append(folder)

    return sorted(results_folders)


def check_folder_contents(folder_path: Path) -> Tuple[bool, bool, bool]:
    """
    Check what type of results files are available in a folder.

    Args:
        folder_path: Path to the results folder

    Returns:
        Tuple of (has_standard_metrics, has_custom_metrics, has_regressor_data)
    """
    standard_file = folder_path / "combined_all_results.csv"
    custom_file = folder_path / "combined_all_custom_metrics.csv"

    has_standard = standard_file.exists()
    has_custom = custom_file.exists()

    # Check if the combined results file contains regression_model column
    has_regressor_data = False
    if has_standard:
        try:
            import pandas as pd

            df = pd.read_csv(standard_file, nrows=1)  # Read just the header
            has_regressor_data = "regression_model" in df.columns
        except Exception:
            has_regressor_data = False

    return has_standard, has_custom, has_regressor_data


def visualize_folder(
    folder_path: Path, output_dir: Path = None, show_plots: bool = False, plot_type: str = "mean"
) -> None:
    """
    Create visualizations for a single results folder.

    Args:
        folder_path: Path to the results folder
        output_dir: Optional output directory for plots. If None, saves in plots/ folder
        show_plots: Whether to display plots interactively
    """
    if output_dir is None:
        output_dir = Path("plots")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_standard, has_custom, has_regressor_data = check_folder_contents(folder_path)
    logger.info(f"Processing folder: {folder_path}")
    logger.info(f"  Standard metrics: {'✓' if has_standard else '✗'}")
    logger.info(f"  Custom metrics: {'✓' if has_custom else '✗'}")
    logger.info(f"  Regressor comparison data: {'✓' if has_regressor_data else '✗'}")

    # Plot standard metrics if available
    if has_standard:
        try:
            logger.info("  Generating standard metrics plot...")
            plot_active_learning_metrics(
                results_folder_path=str(folder_path),
                save_path=str(output_dir / f"{folder_path.name}_standard_metrics.pdf"),
                show_plot=show_plots,
            )
            logger.info("  ✓ Standard metrics plot created")
        except Exception as e:
            logger.error(f"  ✗ Error creating standard metrics plot: {e}")

    # Plot custom metrics if available
    if has_custom:
        try:
            logger.info("  Generating top 10 ratio metrics plot...")
            plot_top10_ratio_metrics(
                results_folder_path=str(folder_path),
                save_path=str(
                    output_dir / f"{folder_path.name}_top10_ratio_metrics.pdf"
                ),
                show_plot=show_plots,
            )
            logger.info("  ✓ Top 10 ratio metrics plot created")
        except Exception as e:
            logger.error(f"  ✗ Error creating top 10 ratio metrics plot: {e}")

        try:
            logger.info("  Generating value metrics plot...")
            plot_value_metrics(
                results_folder_path=str(folder_path),
                save_path=str(output_dir / f"{folder_path.name}_value_metrics.pdf"),
                show_plot=show_plots,
            )
            logger.info("  ✓ Value metrics plot created")
        except Exception as e:
            logger.error(f"  ✗ Error creating value metrics plot: {e}")

    # Plot individual regressor value metrics if available
    if has_regressor_data and has_custom:
        try:
            plot_regressor_comparison(
                results_folder_path=str(folder_path),
                save_path=str(output_dir / f"{folder_path.name}_regressor_comparison.pdf"),
                show_plot=show_plots,
                plot_type=plot_type,
                strategy="highExpression",
            )
            logger.info("  ✓ Regressor comparison plot created")
        except Exception as e:
            logger.error(f"  ✗ Error creating regressor comparison plot: {e}")

        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            custom_file = folder_path / "combined_all_custom_metrics.csv"
            df = pd.read_csv(custom_file)

            # Get unique regressors
            if "regression_model" in df.columns:
                regressors = df["regression_model"].unique()

                # Define the value metrics to plot
                value_metrics = [
                    "best_value_predictions_values",
                    "best_value_predictions_values_cumulative",
                    "normalized_predictions_predictions_values",
                    "normalized_predictions_predictions_values_cumulative",
                    "best_value_ground_truth_values",
                    "best_value_ground_truth_values_cumulative",
                    "normalized_predictions_ground_truth_values",
                    "normalized_predictions_ground_truth_values_cumulative",
                ]

                # Filter to only available metrics
                available_value_metrics = [m for m in value_metrics if m in df.columns]

                for regressor in regressors:
                    logger.info(f"  Generating value metrics plot for {regressor}...")

                    # Filter data for this specific regressor
                    regressor_data = df[df["regression_model"] == regressor]

                    if len(regressor_data) == 0:
                        logger.warning(f"  No data found for {regressor}")
                        continue

                    try:
                        # Create custom plot for this regressor
                        n_metrics = len(available_value_metrics)
                        n_cols = min(4, n_metrics)
                        n_rows = (n_metrics + n_cols - 1) // n_cols

                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(35, 12))

                        # Handle different subplot configurations
                        if n_metrics == 1:
                            axes = [axes]
                        elif n_rows == 1:
                            axes = axes if isinstance(axes, list) else [axes]
                        else:
                            axes = axes.flatten()

                        for i, metric in enumerate(available_value_metrics):
                            if i >= len(axes):
                                break
                            ax = axes[i]

                            # Plot each strategy for this regressor
                            for strategy in regressor_data["strategy"].unique():
                                strategy_data = regressor_data[
                                    regressor_data["strategy"] == strategy
                                ]

                                if len(strategy_data) == 0:
                                    continue

                                # Get consistent color for this strategy
                                color = STRATEGY_COLORS.get(
                                    strategy, "#1f77b4"
                                )  # Default to blue if not found

                                # Calculate mean and std for each train_size
                                stats = (
                                    strategy_data.groupby("train_size")[metric]
                                    .agg(["mean", "std", "median", q1, q3, sem])
                                    .reset_index()
                                )
                                if plot_type == "median": # Plot median line
                                    # Plot median line
                                    ax.plot(
                                        stats["train_size"],
                                        stats["median"],
                                        marker="o",
                                        label=STATEGY_LABELS[strategy],
                                        color=color,
                                    )
                                    ax.fill_between(
                                        stats["train_size"],
                                        stats["q1"],
                                        stats["q3"],
                                        alpha=0.2,
                                        color=color,
                                    )
                                elif plot_type == "sem":
                                    # Plot mean line
                                    ax.plot(
                                        stats["train_size"],
                                        stats["mean"],
                                        marker="o",
                                        label=STATEGY_LABELS[strategy],
                                        color=color,
                                    )
                                    # Add fill between mean ± sem
                                    ax.fill_between(
                                        stats["train_size"],
                                        stats["mean"] - stats["sem"],
                                        stats["mean"] + stats["sem"],
                                        alpha=0.2,
                                        color=color,
                                    )
                                else:
                                    # Plot mean line
                                    ax.plot(
                                        stats["train_size"],
                                        stats["mean"],
                                        marker="o",
                                        label=STATEGY_LABELS[strategy],
                                        color=color,
                                    )
                                    # Add fill between mean ± std
                                    ax.fill_between(
                                        stats["train_size"],
                                        stats["mean"] - stats["std"],
                                        stats["mean"] + stats["std"],
                                        alpha=0.2,
                                        color=color,
                                    )

                            # Format the plot
                            ax.set_title(
                                f'{metric.replace("_", " ").title()} ({plot_type.title()})', fontsize=12
                            )
                            ax.set_xlabel("Train Size", fontsize=10)
                            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
                            ax.legend(loc='upper left', fontsize=10)
                            # ax.grid(True, alpha=0.3)

                        # Hide unused subplots
                        for i in range(len(available_value_metrics), len(axes)):
                            axes[i].set_visible(False)

                        # Set overall title
                        fig.suptitle(
                            f"{regressor} - Value Metrics vs Train Size", fontsize=16
                        )
                        plt.tight_layout()

                        # Save plot
                        save_path = (
                            output_dir
                            / f"{folder_path.name}_{regressor}_value_metrics.pdf"
                        )
                        plt.savefig(save_path, dpi=300, bbox_inches="tight")
                        print(f"Plot saved to: {save_path}")

                        if not show_plots:
                            plt.close()
                        else:
                            plt.show()

                        logger.info(f"  ✓ {regressor} value metrics plot created")

                    except Exception as e:
                        logger.error(
                            f"  ✗ Error creating {regressor} value metrics plot: {e}"
                        )

        except Exception as e:
            logger.error(f"  ✗ Error processing regressor data: {e}")

    if not has_standard and not has_custom:
        logger.warning(f"  No suitable data files found in {folder_path}")


def create_summary_report(
    results_folders: List[Path], output_file: str = "results_summary.md"
) -> None:
    """
    Create a markdown summary report of all results folders.

    Args:
        results_folders: List of results folder paths
        output_file: Output markdown file name
    """
    with open(output_file, "w") as f:
        f.write("# Active Learning Experiments Results Summary\n\n")
        f.write(f"Found {len(results_folders)} results folders:\n\n")

        for folder in results_folders:
            has_standard, has_custom, has_regressor_data = check_folder_contents(folder)

            f.write(f"## {folder.name}\n\n")
            f.write(f"- **Path**: `{folder}`\n")
            f.write(
                f"- **Standard metrics**: {'✅ Available' if has_standard else '❌ Not found'}\n"
            )
            f.write(
                f"- **Custom metrics**: {'✅ Available' if has_custom else '❌ Not found'}\n"
            )
            f.write(
                f"- **Regressor comparison**: {'✅ Available' if has_regressor_data else '❌ Not found'}\n"
            )

            # List generated plots
            plots = []
            if has_standard:
                plots.append(f"![Standard Metrics]({folder.name}_standard_metrics.pdf)")
            if has_custom:
                plots.append(
                    f"![Top 10 Ratio Metrics]({folder.name}_top10_ratio_metrics.pdf)"
                )
                plots.append(f"![Value Metrics]({folder.name}_value_metrics.pdf)")
            if has_regressor_data:
                plots.append(
                    f"![Regressor Comparison]({folder.name}_regressor_comparison.pdf)"
                )
                if has_custom:
                    plots.append(
                        f"![Regressor Summary]({folder.name}_regressor_summary.pdf)"
                    )

            if plots:
                f.write("- **Generated plots**:\n")
                for plot in plots:
                    f.write(f"  - {plot}\n")

            f.write("\n")

    logger.info(f"Summary report saved to: {output_file}")


def main():
    """Main function to visualize all results."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for all active learning experiment results"
    )
    # Plot median line instead of mean line
    parser.add_argument(
        "--plot-type",
        "-t",
        type=str,
        default="mean",
        help="Plot type: mean, median, sem",
    )
    parser.add_argument(
        "--base-path",
        "-b",
        type=str,
        default=".",
        help="Base directory to search for results folders (default: current directory)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots/)",
    )

    parser.add_argument(
        "--show-plots", "-s", action="store_true", help="Display plots interactively"
    )

    parser.add_argument(
        "--create-summary",
        "-r",
        action="store_true",
        help="Create a markdown summary report",
    )

    parser.add_argument(
        "--folder", "-f", type=str, help="Process only a specific results folder"
    )

    args = parser.parse_args()

    # Discover results folders
    if args.folder:
        # Process only the specified folder
        folder_path = Path(args.folder)
        if not folder_path.exists():
            logger.error(f"Specified folder does not exist: {folder_path}")
            return
        results_folders = [folder_path]
    else:
        # Discover all results folders
        results_folders = discover_results_folders(args.base_path)

    if not results_folders:
        logger.warning(f"No results folders found in {args.base_path}")
        return

    logger.info(f"Found {len(results_folders)} results folders:")
    for folder in results_folders:
        logger.info(f"  - {folder}")

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plots will be saved to: {output_dir}")

    # Process each folder
    logger.info("\n" + "=" * 60)
    logger.info("Starting visualization generation...")
    logger.info("=" * 60)

    for folder in results_folders:
        try:
            visualize_folder(
                folder_path=folder,
                output_dir=output_dir / folder.name,
                show_plots=args.show_plots,
                plot_type=args.plot_type,
            )
        except Exception as e:
            logger.error(f"Error processing folder {folder}: {e}")
            continue

    # Create summary report if requested
    if args.create_summary:
        summary_file = output_dir / "results_summary.md"
        create_summary_report(results_folders, str(summary_file))

    logger.info("\n" + "=" * 60)
    logger.info("Visualization generation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
