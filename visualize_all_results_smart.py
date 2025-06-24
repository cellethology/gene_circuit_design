#!/usr/bin/env python3
"""
Smart visualization script that automatically detects available metrics
and generates appropriate plots for all experiment results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from utils.plotting import plot_active_learning_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_available_metrics(csv_path: Path) -> Tuple[List[str], List[str]]:
    """
    Detect what standard and custom metrics are available in a CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (standard_metrics, custom_metrics) that are available
    """
    try:
        df = pd.read_csv(csv_path)
        columns = set(df.columns)

        # Define possible standard metrics
        possible_standard = {
            "rmse",
            "r2",
            "pearson_correlation",
            "pearson_p_value",
            "spearman_correlation",
            "spearman_p_value",
        }

        # Define possible custom metrics
        possible_custom = {
            "top_10_ratio_intersected_indices",
            "top_10_ratio_intersected_indices_cumulative",
            "best_value_predictions_values",
            "normalized_predictions_predictions_values",
            "best_value_ground_truth_values",
            "normalized_predictions_ground_truth_values",
        }

        # Find available metrics
        available_standard = [m for m in possible_standard if m in columns]
        available_custom = [m for m in possible_custom if m in columns]

        return available_standard, available_custom

    except Exception as e:
        logger.error(f"Error reading {csv_path}: {e}")
        return [], []


def create_smart_plots(folder_path: Path, show_plots: bool = False) -> Dict[str, bool]:
    """
    Create plots based on automatically detected metrics.

    Args:
        folder_path: Path to results folder
        show_plots: Whether to display plots

    Returns:
        Dictionary indicating success of each plot type
    """
    results = {"standard": False, "custom": False}

    # Check standard metrics
    standard_file = folder_path / "combined_all_results.csv"
    if standard_file.exists():
        standard_metrics, _ = detect_available_metrics(standard_file)
        if standard_metrics and "strategy" in pd.read_csv(standard_file).columns:
            try:
                logger.info(f"  Found standard metrics: {standard_metrics}")
                plots_dir = Path("plots") / folder_path.name
                plots_dir.mkdir(parents=True, exist_ok=True)
                plot_active_learning_metrics(
                    results_folder_path=str(folder_path),
                    metrics=standard_metrics,
                    save_path=str(
                        plots_dir / f"{folder_path.name}_smart_standard_metrics.png"
                    ),
                    show_plot=show_plots,
                )
                results["standard"] = True
                logger.info("  ✓ Smart standard metrics plot created")
            except Exception as e:
                logger.error(f"  ✗ Error creating standard plot: {e}")

    # Check custom metrics
    custom_file = folder_path / "combined_all_custom_metrics.csv"
    if custom_file.exists():
        _, custom_metrics = detect_available_metrics(custom_file)
        if custom_metrics and "strategy" in pd.read_csv(custom_file).columns:
            try:
                logger.info(f"  Found custom metrics: {custom_metrics}")

                # Adjust figure size based on number of metrics
                n_metrics = len(custom_metrics)
                if n_metrics <= 3:
                    figsize = (18, 6)
                    fig_layout = (1, min(n_metrics, 3))
                elif n_metrics <= 6:
                    figsize = (24, 8)
                    fig_layout = (2, 3)
                else:
                    figsize = (30, 12)
                    fig_layout = (3, 4)

                # Create custom plot with detected metrics
                import matplotlib.pyplot as plt

                # Read data
                df = pd.read_csv(custom_file)
                required_columns = ["strategy", "train_size"] + custom_metrics

                if all(col in df.columns for col in required_columns):
                    fig, axes = plt.subplots(*fig_layout, figsize=figsize)
                    if n_metrics == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten() if hasattr(axes, "flatten") else axes

                    # Plot each metric
                    for i, metric in enumerate(custom_metrics):
                        if i >= len(axes):
                            break

                        ax = axes[i]

                        for strategy in df["strategy"].unique():
                            strategy_data = df[df["strategy"] == strategy]

                            # Calculate mean and std for each train_size
                            stats = (
                                strategy_data.groupby("train_size")[metric]
                                .agg(["mean", "std"])
                                .reset_index()
                            )

                            # Plot mean line
                            ax.plot(
                                stats["train_size"],
                                stats["mean"],
                                marker="o",
                                label=strategy,
                            )

                            # Add fill between mean ± std
                            ax.fill_between(
                                stats["train_size"],
                                stats["mean"] - stats["std"],
                                stats["mean"] + stats["std"],
                                alpha=0.2,
                            )

                        # Format the plot
                        ax.set_title(
                            f'{metric.replace("_", " ").title()} vs Train Size'
                        )
                        ax.set_xlabel("Train Size")
                        ax.set_ylabel(metric.replace("_", " ").title())
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                    # Hide unused subplots
                    for i in range(len(custom_metrics), len(axes)):
                        axes[i].set_visible(False)

                    plt.tight_layout()

                    # Save plot
                    plots_dir = Path("plots") / folder_path.name
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    save_path = (
                        plots_dir / f"{folder_path.name}_smart_custom_metrics.png"
                    )
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")
                    logger.info(f"Plot saved to: {save_path}")

                    if show_plots:
                        plt.show()
                    else:
                        plt.close()

                    results["custom"] = True
                    logger.info("  ✓ Smart custom metrics plot created")
                else:
                    logger.warning("  Missing required columns for custom metrics")

            except Exception as e:
                logger.error(f"  ✗ Error creating custom plot: {e}")

    return results


def main():
    """Generate smart visualizations for all results folders."""
    # Discover all results folders
    base_path = Path(".")
    results_folders = []

    # Look for folders that start with "results" at the top level
    for folder in base_path.glob("results*"):
        if folder.is_dir():
            results_folders.append(folder)

    # Also look inside the results/ directory
    results_dir = base_path / "results"
    if results_dir.exists():
        for folder in results_dir.glob("results*"):
            if folder.is_dir():
                results_folders.append(folder)

    results_folders = sorted(results_folders)

    if not results_folders:
        logger.warning("No results folders found")
        return

    logger.info(f"Found {len(results_folders)} results folders")

    # Process each folder
    logger.info("\n" + "=" * 60)
    logger.info("Generating smart visualizations...")
    logger.info("=" * 60)

    summary_stats = {"processed": 0, "standard_plots": 0, "custom_plots": 0}

    for folder in results_folders:
        logger.info(f"\nProcessing: {folder}")

        try:
            plot_results = create_smart_plots(folder)
            summary_stats["processed"] += 1

            if plot_results["standard"]:
                summary_stats["standard_plots"] += 1
            if plot_results["custom"]:
                summary_stats["custom_plots"] += 1

        except Exception as e:
            logger.error(f"Error processing {folder}: {e}")
            continue

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"Processed folders: {summary_stats['processed']}/{len(results_folders)}"
    )
    logger.info(f"Standard plots created: {summary_stats['standard_plots']}")
    logger.info(f"Custom plots created: {summary_stats['custom_plots']}")
    logger.info(
        f"Total plots created: {summary_stats['standard_plots'] + summary_stats['custom_plots']}"
    )


if __name__ == "__main__":
    main()
