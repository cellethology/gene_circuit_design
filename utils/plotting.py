"""
Plotting utilities for visualizing active learning experiment results.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

FONT_SIZE = 14

STATEGY_LABELS = {
    "highExpression": "Top-K Selection",
    "random": "Random",
    "log_likelihood": "Zero Shot",
}

def q1(x):
    return x.quantile(0.25)

def q3(x):
    return x.quantile(0.75)

def sem(x):
    return x.sem()

def regressor_line_plot(ax, stats, regressor, plot_type, regressor_colors):
    if plot_type == "mean":
        ax.plot(
            stats["train_size"],
            stats["mean"],
            marker="o",
            label=regressor,
            color=regressor_colors[regressor],
            linewidth=2,
        )

        # Add fill between mean ± std
        ax.fill_between(
            stats["train_size"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.2,
            color=regressor_colors[regressor],
        )
    elif plot_type == "median":
        ax.plot(
            stats["train_size"],
            stats["median"],
            marker="o",
            label=regressor,
            color=regressor_colors[regressor],
            linewidth=2,
        )

        # Add fill between median ± std
        ax.fill_between(
            stats["train_size"],
            stats["q1"],
            stats["q3"],
            alpha=0.2,
            color=regressor_colors[regressor],
        )
    elif plot_type == "sem":
        ax.plot(
            stats["train_size"],
            stats["mean"],
            marker="o",
            label=regressor,
            color=regressor_colors[regressor],
            linewidth=2,
        )
        # Add fill between mean ± sem
        ax.fill_between(
            stats["train_size"],
            stats["mean"] - stats["sem"],
            stats["mean"] + stats["sem"],
            alpha=0.2,
            color=regressor_colors[regressor],
        )

def plot_active_learning_metrics(
    results_folder_path: str,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot active learning metrics from experiment results.

    Args:
        results_folder_path: Path to the folder containing combined_all_results.csv
        metrics: List of metric names to plot. If None, uses default metrics.
        figsize: Figure size as (width, height)
        save_path: Path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot

    Raises:
        FileNotFoundError: If the combined results file is not found
        ValueError: If required columns are missing from the data
    """
    # Default metrics if none provided
    if metrics is None:
        # Check what columns are actually available
        results_path = Path(results_folder_path) / "combined_all_results.csv"
        if results_path.exists():
            temp_df = pd.read_csv(results_path)
            available_metrics = [
                col
                for col in temp_df.columns
                if col
                not in [
                    "round",
                    "strategy",
                    "seq_mod_method",
                    "seed",
                    "train_size",
                    "unlabeled_size",
                ]
            ]
            metrics = (
                available_metrics[:3] if available_metrics else ["train_size"]
            )  # fallback
        else:
            metrics = ["train_size"]  # fallback if file doesn't exist

    # Construct path to combined results file
    results_path = Path(results_folder_path) / "combined_all_results.csv"

    # Check if file exists
    if not results_path.exists():
        raise FileNotFoundError(f"Combined results file not found: {results_path}")

    # Read the data
    df = pd.read_csv(results_path)

    # Validate required columns
    required_columns = ["strategy", "train_size"] + metrics
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in data: {missing_columns}")

    # Set up the plot
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

    # Handle case where there's only one metric (axes is not a list)
    if len(metrics) == 1:
        axes = [axes]

    # Plot each metric
    for metric, ax in zip(metrics, axes):
        for strategy in df["strategy"].unique():
            strategy_data = df[df["strategy"] == strategy]

            # Calculate mean and std for each train_size
            stats = (
                strategy_data.groupby("train_size")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )

            # Plot mean line
            ax.plot(stats["train_size"], stats["mean"], marker="o", label=strategy)

            # Add fill between mean ± std
            ax.fill_between(
                stats["train_size"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.2,
            )

        # Format the plot
        ax.set_title(
            f'{metric.replace("_", " ").title()} vs Train Size', fontsize=FONT_SIZE
        )
        ax.set_xlabel("Train Size", fontsize=FONT_SIZE)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=FONT_SIZE)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    return fig


def plot_top10_ratio_metrics(
    results_folder_path: str,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot top 10 ratio metrics from active learning experiments.

    Args:
        results_folder_path: Path to the folder containing combined_all_custom_metrics.csv
        save_path: Path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot
    """
    metrics = [
        "top_10_ratio_intersected_indices",
        "top_10_ratio_intersected_indices_cumulative",
    ]

    # Construct path to combined custom metrics file
    results_path = Path(results_folder_path) / "combined_all_custom_metrics.csv"

    # Check if file exists
    if not results_path.exists():
        raise FileNotFoundError(
            f"Combined custom metrics file not found: {results_path}"
        )

    # Read the data
    df = pd.read_csv(results_path)

    # Validate required columns
    required_columns = ["strategy", "train_size"] + metrics
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in data: {missing_columns}")

    # Set up the plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot each metric
    for i, metric in enumerate(metrics):
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
            ax.plot(stats["train_size"], stats["mean"], marker="o", label=STATEGY_LABELS[strategy])

            # Add fill between mean ± std
            ax.fill_between(
                stats["train_size"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.2,
            )

        # Format the plot
        ax.set_title(
            f'{metric.replace("_", " ").title()} vs Train Size', fontsize=FONT_SIZE
        )
        ax.set_xlabel("Train Size", fontsize=FONT_SIZE)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=FONT_SIZE)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    return fig


def plot_value_metrics(
    results_folder_path: str,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot value-based metrics from active learning experiments.

    Args:
        results_folder_path: Path to the folder containing combined_all_custom_metrics.csv
        save_path: Path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot
    """
    metrics = [
        "best_value_predictions_values",
        "best_value_predictions_values_cumulative",
        "normalized_predictions_predictions_values",
        "normalized_predictions_predictions_values_cumulative",
        "best_value_ground_truth_values",
        "best_value_ground_truth_values_cumulative",
        "normalized_predictions_ground_truth_values",
        "normalized_predictions_ground_truth_values_cumulative",
    ]

    # Construct path to combined custom metrics file
    results_path = Path(results_folder_path) / "combined_all_custom_metrics.csv"

    # Check if file exists
    if not results_path.exists():
        raise FileNotFoundError(
            f"Combined custom metrics file not found: {results_path}"
        )

    # Read the data
    df = pd.read_csv(results_path)

    # Validate required columns
    required_columns = ["strategy", "train_size"] + metrics
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in data: {missing_columns}")

    # Set up the plot - calculate grid size based on number of metrics
    n_metrics = len(metrics)
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 12))

    # Handle different cases for axes
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()

    # Plot each metric
    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break  # safety check
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
            ax.plot(stats["train_size"], stats["mean"], marker="o", label=STATEGY_LABELS[strategy])

            # Add fill between mean ± std
            ax.fill_between(
                stats["train_size"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.2,
            )

        # Format the plot
        ax.set_title(
            f'{metric.replace("_", " ").title()} vs Train Size', fontsize=FONT_SIZE
        )
        ax.set_xlabel("Train Size", fontsize=FONT_SIZE)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=FONT_SIZE)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    return fig


def plot_custom_metrics(
    results_folder_path: str,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (30, 30),
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot custom metrics from active learning experiments.

    Args:
        results_folder_path: Path to the folder containing combined_all_custom_metrics.csv
        metrics: List of custom metric names to plot. If None, uses default custom metrics.
        figsize: Figure size as (width, height)
        save_path: Path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot

    Raises:
        FileNotFoundError: If the combined custom metrics file is not found
        ValueError: If required columns are missing from the data
    """
    # Default custom metrics if none provided
    if metrics is None:
        metrics = [
            "top_10_ratio_intersected_indices",
            "top_10_ratio_intersected_indices_cumulative",
            "best_value_predictions_values",
            "best_value_predictions_values_cumulative",
            "normalized_predictions_predictions_values",
            "normalized_predictions_predictions_values_cumulative",
            "best_value_ground_truth_values",
            "best_value_ground_truth_values_cumulative",
            "normalized_predictions_ground_truth_values",
            "normalized_predictions_ground_truth_values_cumulative",
        ]

    # Construct path to combined custom metrics file
    results_path = Path(results_folder_path) / "combined_all_custom_metrics.csv"

    # Check if file exists
    if not results_path.exists():
        raise FileNotFoundError(
            f"Combined custom metrics file not found: {results_path}"
        )

    # Read the data
    df = pd.read_csv(results_path)

    # Validate required columns
    required_columns = ["strategy", "train_size"] + metrics
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in data: {missing_columns}")

    # Set up the plot - calculate grid size based on number of metrics
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle different cases for axes
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()

    # Plot each metric
    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break  # safety check
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
            ax.plot(stats["train_size"], stats["mean"], marker="o", label=STATEGY_LABELS[strategy])

            # Add fill between mean ± std
            ax.fill_between(
                stats["train_size"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.2,
            )

        # Format the plot
        ax.set_title(
            f'{metric.replace("_", " ").title()} vs Train Size', fontsize=FONT_SIZE
        )
        ax.set_xlabel("Train Size", fontsize=FONT_SIZE)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=FONT_SIZE)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    return fig


def plot_regressor_comparison(
    results_folder_path: str,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (35, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True,
    strategy: Optional[str] = None,
    plot_type: str = "mean",
) -> plt.Figure:
    """
    Plot comparison between different regression models for each strategy.

    Args:
        results_folder_path: Path to the folder containing combined_all_results.csv
        metrics: List of metric names to plot. If None, uses default metrics.
        figsize: Figure size as (width, height)
        save_path: Path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot
        strategy: Strategy to plot. If None, plots all strategies.
        plot_type: Type of plot to use. If "mean", plots the mean of the metric. If "median", plots the median of the metric.
    Raises:
        FileNotFoundError: If the combined results file is not found
        ValueError: If required columns are missing from the data
    """
    # Default metrics if none provided
    if metrics is None:
        metrics = [
            "best_value_predictions_values_cumulative",
            "normalized_predictions_predictions_values_cumulative",
            "best_value_ground_truth_values_cumulative",
            "normalized_predictions_ground_truth_values_cumulative",
        ]


    # Construct path to combined results file
    results_path = Path(results_folder_path) / "combined_all_custom_metrics.csv"

    # Check if file exists
    if not results_path.exists():
        raise FileNotFoundError(f"Combined results file not found: {results_path}")

    # Read the data
    df = pd.read_csv(results_path)

    # Check if regression_model column exists
    if "regression_model" not in df.columns:
        print(
            "Warning: No regression_model column found. This might be single-regressor data."
        )
        return plot_active_learning_metrics(
            results_folder_path, metrics, figsize, save_path, show_plot
        )

    # Validate required columns
    required_columns = ["strategy", "regression_model", "train_size"] + metrics
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in data: {missing_columns}")

    # Get unique strategies and regression models
    strategies = df["strategy"].unique()
    # If strategy is provided, only plot that strategy
    if strategy:
        strategies = [strategy]
    regressors = df["regression_model"].unique()

    # Set up the plot - strategies as rows, metrics as columns
    n_strategies = len(strategies)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_strategies, n_metrics, figsize=figsize)

    # Handle different cases for axes
    if n_strategies == 1 and n_metrics == 1:
        axes = [[axes]]
    elif n_strategies == 1:
        axes = [axes]
    elif n_metrics == 1:
        axes = [[ax] for ax in axes]

    # Color map for regressors
    colors = plt.cm.Set1(range(len(regressors)))
    regressor_colors = dict(zip(regressors, colors))

    # Plot each strategy-metric combination
    for i, strategy in enumerate(strategies):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]

            for regressor in regressors:
                # Filter data for this strategy and regressor
                data = df[
                    (df["strategy"] == strategy) & (df["regression_model"] == regressor)
                ]

                if len(data) == 0:
                    continue

                # Calculate mean and std for each train_size
                stats = (
                    data.groupby("train_size")[metric]
                    .agg(["mean", "std", "median", q1, q3, sem])
                    .reset_index()
                )

                # Plot the regressor line
                regressor_line_plot(ax, stats, regressor, plot_type, regressor_colors)

            # Format the plot
            ax.set_title(
                f'{strategy} - {metric.replace("_", " ").title()} {(plot_type)}', fontsize=FONT_SIZE - 4
            )
            ax.set_xlabel("Train Size", fontsize=FONT_SIZE - 2)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=FONT_SIZE - 2)

            # NOTE: add this back if needed
            # Only show legend on the first subplot to avoid clutter
            # if i == 0 and j == 0:
            ax.legend(title="Regressor", fontsize=FONT_SIZE - 2, loc='upper left')

            # ax.grid(True, alpha=0.3)

            # Improve tick label formatting
            ax.tick_params(labelsize=FONT_SIZE - 4)

    # Adjust layout
    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regressor comparison plot saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    return fig


def plot_regressor_summary(
    results_folder_path: str,
    metric: str = "pearson_correlation",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot a summary comparison of regression models across all strategies for a single metric.

    Args:
        results_folder_path: Path to the folder containing combined_all_results.csv
        metric: Metric to compare (default: "pearson_correlation")
        figsize: Figure size as (width, height)
        save_path: Path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot

    Returns:
        The matplotlib figure object
    """
    # Construct path to combined results file
    results_path = Path(results_folder_path) / "combined_all_results.csv"

    # Check if file exists
    if not results_path.exists():
        raise FileNotFoundError(f"Combined results file not found: {results_path}")

    # Read the data
    df = pd.read_csv(results_path)

    # Check if regression_model column exists
    if "regression_model" not in df.columns:
        raise ValueError("No regression_model column found in data")

    # Get final performance for each combination
    final_df = (
        df.groupby(["strategy", "regression_model", "seed"])[metric]
        .last()
        .reset_index()
    )

    # Create box plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create a combined strategy-regressor identifier for plotting
    final_df["strategy_regressor"] = (
        final_df["strategy"] + "_" + final_df["regression_model"]
    )

    # Get unique combinations and sort them
    combinations = sorted(final_df["strategy_regressor"].unique())

    # Prepare data for box plot
    data_for_boxplot = []
    labels = []

    for combo in combinations:
        combo_data = final_df[final_df["strategy_regressor"] == combo][metric]
        data_for_boxplot.append(combo_data)
        # Format label to be more readable
        strategy, regressor = combo.split("_", 1)
        labels.append(f"{strategy}\n{regressor}")

    # Create box plot
    bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)

    # Color boxes by regressor
    regressors = [label.split("\n")[1] for label in labels]
    unique_regressors = list(set(regressors))
    colors = plt.cm.Set1(range(len(unique_regressors)))
    regressor_colors = dict(zip(unique_regressors, colors))

    for patch, regressor in zip(bp["boxes"], regressors):
        patch.set_facecolor(regressor_colors[regressor])
        patch.set_alpha(0.7)

    # Format the plot
    ax.set_title(
        f'Final {metric.replace("_", " ").title()} by Strategy and Regressor',
        fontsize=FONT_SIZE,
    )
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=FONT_SIZE)
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Add legend for regressors
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=regressor_colors[reg], alpha=0.7)
        for reg in unique_regressors
    ]
    ax.legend(handles, unique_regressors, title="Regressor", loc="upper right")

    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regressor summary plot saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    return fig


def list_available_results_folders(base_path: str = ".") -> List[str]:
    """
    List all available results folders that contain combined results files.

    Args:
        base_path: Base directory to search for results folders

    Returns:
        List of folder paths that contain combined results files
    """
    results_folders = []
    base_path = Path(base_path)

    # Look for folders that start with "results"
    for folder in base_path.glob("results*"):
        if folder.is_dir():
            combined_results = folder / "combined_all_results.csv"
            if combined_results.exists():
                results_folders.append(str(folder))

    return sorted(results_folders)


# Example usage function
def plot_example():
    """Example of how to use the plotting functions."""
    # List available results folders
    folders = list_available_results_folders()
    print("Available results folders:")
    for folder in folders:
        print(f"  - {folder}")

    if folders:
        # Plot the first available results folder
        folder_path = folders[0]
        print(f"\nPlotting results from: {folder_path}")

        # Plot standard metrics
        plot_active_learning_metrics(
            folder_path, save_path=f"{folder_path}/metrics_plot.pdf"
        )

        # Plot custom metrics if available
        custom_metrics_file = Path(folder_path) / "combined_all_custom_metrics.csv"
        if custom_metrics_file.exists():
            plot_custom_metrics(
                folder_path, save_path=f"{folder_path}/custom_metrics_plot.pdf"
            )

def create_combined_results_from_files(output_path: Path) -> None:
    """
    Create combined results files from individual experiment files.
    This is useful when experiments are interrupted but individual files exist.
    """
    import re

    # Find all individual results files (exclude combined files)
    results_files = [
        f
        for f in output_path.glob("*_results.csv")
        if "_all_seeds_" not in f.name and "combined_all_" not in f.name
    ]
    custom_metrics_files = [
        f
        for f in output_path.glob("*_custom_metrics.csv")
        if "_all_seeds_" not in f.name and "combined_all_" not in f.name
    ]

    if not results_files:
        # NOTE: use to be logger
        print("No individual results files found to combine")
        return

    # Combine results files
    all_results = []
    for file_path in results_files:
        filename = file_path.stem
        # Parse filename: strategy_seqmod_regressor_seed_X_results
        # Handle complex regressor names like "KNN_regression" or "linear_regresion"
        pattern = r"([^_]+)_([^_]+)_(.+?)_seed_(\d+)_results"
        match = re.match(pattern, filename)

        if not match:
            print(f"Could not parse filename {filename}")
            continue

        strategy, seq_mod_method, regression_model, seed = match.groups()
        seed = int(seed)

        try:
            df = pd.read_csv(file_path)
            # Add metadata columns if missing
            df["strategy"] = strategy
            df["seq_mod_method"] = seq_mod_method
            df["regression_model"] = regression_model
            df["seed"] = seed
            all_results.append(df)
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
            continue

    if all_results:
        combined_df = pd.DataFrame(pd.concat(all_results, ignore_index=True))
        combined_output_path = output_path / "combined_all_results.csv"
        combined_df.to_csv(combined_output_path, index=False)
        print(
            f"Combined results from {len(results_files)} files saved to {combined_output_path}"
        )

    # Combine custom metrics files
    all_custom_metrics = []
    for file_path in custom_metrics_files:
        filename = file_path.stem.replace("_custom_metrics", "")
        pattern = r"([^_]+)_([^_]+)_(.+?)_seed_(\d+)"
        match = re.match(pattern, filename)

        if not match:
            print(f"Could not parse custom metrics filename {filename}")
            continue

        strategy, seq_mod_method, regression_model, seed = match.groups()
        seed = int(seed)

        try:
            df = pd.read_csv(file_path)
            # Add metadata columns if missing
            if "strategy" not in df.columns:
                df["strategy"] = strategy
            if "seq_mod_method" not in df.columns:
                df["seq_mod_method"] = seq_mod_method
            if "regression_model" not in df.columns:
                df["regression_model"] = regression_model
            if "seed" not in df.columns:
                df["seed"] = seed
            all_custom_metrics.append(df)
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
            continue

    if all_custom_metrics:
        combined_custom_df = pd.DataFrame(
            pd.concat(all_custom_metrics, ignore_index=True)
        )
        combined_custom_output_path = output_path / "combined_all_custom_metrics.csv"
        combined_custom_df.to_csv(combined_custom_output_path, index=False)
        print(
            f"Combined custom metrics from {len(custom_metrics_files)} files saved to {combined_custom_output_path}"
        )


if __name__ == "__main__":
    plot_example()
