"""
Plotting utilities for visualizing active learning experiment results.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


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
        ax.set_title(f'{metric.replace("_", " ").title()} vs Train Size')
        ax.set_xlabel("Train Size")
        ax.set_ylabel(metric.replace("_", " ").title())
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
            ax.plot(stats["train_size"], stats["mean"], marker="o", label=strategy)

            # Add fill between mean ± std
            ax.fill_between(
                stats["train_size"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.2,
            )

        # Format the plot
        ax.set_title(f'{metric.replace("_", " ").title()} vs Train Size')
        ax.set_xlabel("Train Size")
        ax.set_ylabel(metric.replace("_", " ").title())
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
            ax.plot(stats["train_size"], stats["mean"], marker="o", label=strategy)

            # Add fill between mean ± std
            ax.fill_between(
                stats["train_size"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.2,
            )

        # Format the plot
        ax.set_title(f'{metric.replace("_", " ").title()} vs Train Size')
        ax.set_xlabel("Train Size")
        ax.set_ylabel(metric.replace("_", " ").title())
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
            ax.plot(stats["train_size"], stats["mean"], marker="o", label=strategy)

            # Add fill between mean ± std
            ax.fill_between(
                stats["train_size"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.2,
            )

        # Format the plot
        ax.set_title(f'{metric.replace("_", " ").title()} vs Train Size')
        ax.set_xlabel("Train Size")
        ax.set_ylabel(metric.replace("_", " ").title())
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
            folder_path, save_path=f"{folder_path}/metrics_plot.png"
        )

        # Plot custom metrics if available
        custom_metrics_file = Path(folder_path) / "combined_all_custom_metrics.csv"
        if custom_metrics_file.exists():
            plot_custom_metrics(
                folder_path, save_path=f"{folder_path}/custom_metrics_plot.png"
            )


if __name__ == "__main__":
    plot_example()
