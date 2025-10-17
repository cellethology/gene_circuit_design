#!/usr/bin/env python3
"""
Plotting function for comparing top_k_selections across different regressors.

This script takes combined_all_custom_metrics.csv files from different folders
and creates line plots comparing top_10_ratio_intersected_indices metrics
across different regressors and strategies.
# Basic usage - compare folders
  python plotting/compare_top_k_selections.py results_folder1/
  results_folder2/ results_folder3/

  # Specify output directory and metric
  python plotting/compare_top_k_selections.py results_folder1/
  results_folder2/ \
      --output-dir plots/top_k_comparison/ \
      --metric "normalized_predictions_ground_truth_values_cumulative"

  # Custom figure size
  python plotting/compare_top_k_selections.py results_folder1/
  results_folder2/ \
      --figsize 15 10
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define consistent styling
STRATEGY_COLORS = {
    "high_expression": "#1f77b4",  # Blue
    "log_likelihood": "#ff7f0e",  # Orange
    "random": "#2ca02c",  # Green
}

REGRESSOR_MARKERS = {
    "linear_regression": "o",
    "random_forest": "s",
    "knn_regression": "^",
    "ridge_regression": "D",
    "lasso_regression": "v",
}

REGRESSOR_LABELS = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "knn_regression": "KNN Regression",
    "ridge_regression": "Ridge Regression",
    "lasso_regression": "Lasso Regression",
}


def load_custom_metrics_data(folder_paths: List[Path]) -> pd.DataFrame:
    """
    Load and combine custom metrics data from multiple folders.

    Args:
        folder_paths: List of paths to results folders

    Returns:
        Combined DataFrame with all custom metrics data
    """
    all_data = []

    for folder_path in folder_paths:
        csv_file = folder_path / "combined_all_custom_metrics.csv"

        if not csv_file.exists():
            logger.warning(f"No combined_all_custom_metrics.csv found in {folder_path}")
            continue

        try:
            df = pd.read_csv(csv_file)
            df["experiment_folder"] = folder_path.name
            all_data.append(df)
            logger.info(f"Loaded {len(df)} rows from {csv_file}")

        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
            continue

    if not all_data:
        raise ValueError("No valid custom metrics data found in any folder")

    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset shape: {combined_df.shape}")

    return combined_df


def plot_top_k_comparison(
    df: pd.DataFrame,
    metric_col: str = "top_10_ratio_intersected_indices",
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 8),
) -> None:
    """
    Create a single plot comparing all regressors and strategies with legends.

    Args:
        df: Combined DataFrame with custom metrics
        metric_col: Column name for the metric to plot
        output_path: Path to save the plot (optional)
        figsize: Figure size tuple
    """
    # Set up the plot style
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get unique strategies and regressors
    strategies = sorted(df["strategy"].unique())
    regressors = sorted(df["regression_model"].unique())

    logger.info(f"Found strategies: {strategies}")
    logger.info(f"Found regressors: {regressors}")

    # Generate colors for each strategy-regressor combination
    colors = plt.cm.Set3(range(len(strategies) * len(regressors)))
    color_idx = 0

    # Plot each strategy-regressor combination
    for strategy in strategies:
        strategy_data = df[df["strategy"] == strategy]

        for regressor in regressors:
            regressor_data = strategy_data[
                strategy_data["regression_model"] == regressor
            ]

            if regressor_data.empty:
                color_idx += 1
                continue

            # Calculate mean and std across seeds
            grouped = regressor_data.groupby("train_size")[metric_col].agg(
                ["mean", "std", "count"]
            )

            if grouped.empty:
                color_idx += 1
                continue

            # Create label - if only one strategy, don't include it in label
            if len(strategies) == 1:
                label = REGRESSOR_LABELS.get(regressor, regressor)
            else:
                label = f"{strategy.replace('_', ' ').title()} - {REGRESSOR_LABELS.get(regressor, regressor)}"

            # Plot mean line with shaded confidence interval
            ax.plot(
                grouped.index,
                grouped["mean"],
                marker=REGRESSOR_MARKERS.get(regressor, "o"),
                label=label,
                linewidth=3,
                markersize=8,
                color=colors[color_idx],
                alpha=0.9,
            )

            # Add shaded confidence interval
            ax.fill_between(
                grouped.index,
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                color=colors[color_idx],
                alpha=0.2,
            )

            color_idx += 1

    ax.set_title(
        "Top 10 Ratio Comparison: All Strategies and Regressors",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("Top 10 Ratio Intersected Indices", fontsize=12)

    # Create legend with multiple columns to save space
    # n_items = len([line for line in ax.lines])
    n_items = len(list(ax.lines))
    ncol = min(3, max(1, n_items // 6))  # Adjust columns based on number of items
    ax.legend(fontsize=10, ncol=ncol, bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.grid(True, alpha=0.3)

    # Adjust layout to accommodate legend
    plt.tight_layout()

    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_embedding_comparison_by_regressor(
    df: pd.DataFrame,
    metric_col: str = "top_10_ratio_intersected_indices",
    output_path: Optional[Path] = None,
    figsize: tuple = (15, 10),
) -> None:
    """
    Create comparison plots separated by regressor, comparing embedding methods across strategies.

    Args:
        df: Combined DataFrame with custom metrics
        metric_col: Column name for the metric to plot
        output_path: Path to save the plot (optional)
        figsize: Figure size tuple
    """
    regressors = sorted(df["regression_model"].unique())
    strategies = sorted(df["strategy"].unique())
    embedding_methods = sorted(df["experiment_folder"].unique())

    n_regressors = len(regressors)
    n_cols = min(3, n_regressors)  # Max 3 columns
    n_rows = (n_regressors + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    if n_regressors == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    # Auto-generate colors for embedding methods
    embedding_colors = {}
    colors = plt.cm.Set1(range(len(embedding_methods)))
    for i, method in enumerate(embedding_methods):
        embedding_colors[method] = colors[i]

    # Marker scheme for strategies
    strategy_markers = {"highExpression": "o", "log_likelihood": "s", "random": "^"}

    for i, regressor in enumerate(regressors):
        ax = axes[i]
        regressor_data = df[df["regression_model"] == regressor]

        # Plot each strategy-embedding combination
        for strategy in strategies:
            strategy_data = regressor_data[regressor_data["strategy"] == strategy]

            for embedding_method in embedding_methods:
                embedding_data = strategy_data[
                    strategy_data["experiment_folder"] == embedding_method
                ]

                if embedding_data.empty:
                    continue

                # Calculate statistics
                grouped = embedding_data.groupby("train_size")[metric_col].agg(
                    ["mean", "std"]
                )

                if grouped.empty:
                    continue

                # Create label - if only one strategy, don't include it in label
                embedding_label = embedding_method.replace("_", " ").title()
                if len(strategies) == 1:
                    label = embedding_label
                else:
                    strategy_label = strategy.replace("_", " ").title()
                    label = f"{embedding_label} - {strategy_label}"

                # Plot line with color coding by embedding method
                color = embedding_colors.get(embedding_method, "#2ca02c")
                ax.plot(
                    grouped.index,
                    grouped["mean"],
                    marker="o",
                    label=label,
                    color=color,
                    linewidth=3,
                    markersize=8,
                    alpha=0.9,
                )

                # Add shaded confidence interval
                ax.fill_between(
                    grouped.index,
                    grouped["mean"] - grouped["std"],
                    grouped["mean"] + grouped["std"],
                    color=color,
                    alpha=0.2,
                )

        # Formatting
        regressor_name = REGRESSOR_LABELS.get(regressor, regressor)
        ax.set_title(f"{regressor_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Set Size", fontsize=10)
        if i % n_cols == 0:  # First column
            ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_regressors, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle(
        f'Embedding Method Comparison by Regressor: {metric_col.replace("_", " ").title()}',
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def create_summary_table(
    df: pd.DataFrame, metric_col: str = "top_10_ratio_intersected_indices"
) -> pd.DataFrame:
    """
    Create a summary table of final performance for each regressor/strategy combination.

    Args:
        df: Combined DataFrame with custom metrics
        metric_col: Column name for the metric to summarize

    Returns:
        Summary DataFrame
    """
    # Get final training size for each experiment
    final_results = df.loc[
        df.groupby(["strategy", "regression_model", "seed"])["train_size"].idxmax()
    ]

    # Calculate summary statistics
    summary = (
        final_results.groupby(["strategy", "regression_model"])[metric_col]
        .agg(["mean", "std", "count"])
        .round(4)
    )

    summary.columns = ["Mean", "Std", "N_Seeds"]
    return summary.reset_index()


def main():
    """Main function to run the plotting script."""
    parser = argparse.ArgumentParser(
        description="Compare top_k_selections across regressors from multiple experiment folders"
    )

    parser.add_argument(
        "folders", nargs="+", type=Path, help="Paths to experiment result folders"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots (default: plots/)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="top_10_ratio_intersected_indices",
        help="Metric column to plot (default: top_10_ratio_intersected_indices)",
    )

    parser.add_argument(
        "--figsize",
        nargs=2,
        type=int,
        default=[12, 8],
        help="Figure size (width height)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading custom metrics data...")
    df = load_custom_metrics_data(args.folders)

    # Filter to only high expression (top-k selection) strategy
    df = df[df["strategy"] == "highExpression"]
    logger.info(f"Filtered to highExpression strategy only. New shape: {df.shape}")

    # Create plots
    logger.info("Creating comparison plots...")

    # Plot 1: All strategies and regressors on single plot
    plot_top_k_comparison(
        df,
        metric_col=args.metric,
        output_path=args.output_dir / f"{args.metric}_unified_comparison.png",
        figsize=tuple(args.figsize),
    )

    # Plot 2: Embedding method comparison by regressor
    plot_embedding_comparison_by_regressor(
        df,
        metric_col=args.metric,
        output_path=args.output_dir
        / f"{args.metric}_embedding_comparison_by_regressor.png",
        figsize=(15, 10),
    )

    # Create and save summary table
    logger.info("Creating summary table...")
    summary_table = create_summary_table(df, args.metric)
    summary_path = args.output_dir / f"{args.metric}_summary_table.csv"
    summary_table.to_csv(summary_path, index=False)
    logger.info(f"Summary table saved to {summary_path}")

    logger.info("All plots and summary created successfully!")


if __name__ == "__main__":
    main()
