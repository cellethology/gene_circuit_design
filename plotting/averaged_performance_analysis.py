#!/usr/bin/env python3
"""
Averaged Performance Analysis for Gene Circuit Design

This script creates comprehensive heatmaps showing averaged performance across:
- Different embedding methods (onehotPca, onehotRaw, evo, sei)
- Different models (KNN, LinearRegression, RandomForest, XGBoost)
- Different preprocessing strategies (active_learning, random, etc.)

The analysis provides multiple views:
1. Average performance by embedding method across all models
2. Average performance by model across all embeddings
3. Average performance by preprocessing strategy
4. Combined analysis showing best embedding-model combinations
5. Statistical significance testing between methods

Example Usage:
    python plotting/averaged_performance_analysis.py
    python plotting/averaged_performance_analysis.py --results-base-path /path/to/results
"""

import argparse
import os
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu

# Import the existing plotting functions
try:
    from .area_under_curve_grid_plot import collect_all_results
except ImportError:
    from area_under_curve_grid_plot import collect_all_results

matplotlib.use("Agg")  # Use non-interactive backend
warnings.filterwarnings("ignore", category=FutureWarning)


def aggregate_performance_by_dimension(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    aggregation_method: Literal["mean", "median", "std", "count"] = "mean",
    group_by: Literal["embedding", "regressor", "strategy", "dataset"] = "embedding",
) -> pd.DataFrame:
    """
    Aggregate performance metrics by specified dimension.

    Args:
        results_df: DataFrame with results
        metric: Metric column to aggregate
        aggregation_method: How to aggregate (mean, median, std, count)
        group_by: Dimension to group by

    Returns:
        Aggregated DataFrame
    """
    if results_df.empty:
        return pd.DataFrame()

    # Filter out NaN values for the metric
    valid_data = results_df.dropna(subset=[metric])

    if valid_data.empty:
        return pd.DataFrame()

    # Group by the specified dimension and aggregate
    if aggregation_method == "mean":
        agg_func = "mean"
    elif aggregation_method == "median":
        agg_func = "median"
    elif aggregation_method == "std":
        agg_func = "std"
    elif aggregation_method == "count":
        agg_func = "count"
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

    aggregated = valid_data.groupby(group_by)[metric].agg(agg_func).reset_index()
    aggregated.columns = [group_by, f"{metric}_{aggregation_method}"]

    return aggregated.sort_values(f"{metric}_{aggregation_method}", ascending=False)


def create_embedding_performance_heatmap(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    aggregation_method: Literal["mean", "median"] = "median",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dataset_type: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create heatmap showing average performance by embedding method across datasets and models.

    Args:
        results_df: Results DataFrame
        metric: Metric to visualize
        figsize: Figure size
        save_path: Optional path to save the plot
        aggregation_method: How to aggregate values (mean or median)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        dataset_type: Optional dataset type label ("trans", "cis", or "together") to include in title

    Returns:
        Figure and Axes objects
    """
    if results_df.empty:
        print("No data found for embedding performance heatmap")
        return None, None

    # Create pivot table: dataset x (embedding + regressor)
    results_df = results_df.copy()
    results_df["embedding_regressor"] = (
        results_df["embedding"] + "_" + results_df["regressor"]
    )

    # Filter out random strategy for this analysis
    non_random_data = results_df[results_df["strategy"] != "random"]

    if non_random_data.empty:
        print("No non-random data found")
        return None, None

    # Create pivot table
    pivot_data = non_random_data.pivot_table(
        index="dataset",
        columns="embedding_regressor",
        values=metric,
        aggfunc=aggregation_method,
    )

    if pivot_data.empty:
        print("No pivot data available")
        return None, None

    # Clean data: Drop rows/columns that are all NaN
    pivot_data = pivot_data.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if pivot_data.empty:
        print("No valid data after removing NaN rows/columns")
        return None, None

    # Replace remaining NaN values with 0 for clustering (but keep track for masking)
    # We need finite values for distance computation
    pivot_data_clean = pivot_data.fillna(0)

    # Check for any non-finite values and replace them
    if not np.isfinite(pivot_data_clean.values).all():
        print("Warning: Found non-finite values, replacing with 0")
        pivot_data_clean = pivot_data_clean.replace([np.inf, -np.inf], 0)
        pivot_data_clean = pivot_data_clean.fillna(0)

    # Ensure we have at least 2 rows and 2 columns for clustering
    if pivot_data_clean.shape[0] < 2 or pivot_data_clean.shape[1] < 2:
        print(f"Not enough data for clustering (shape: {pivot_data_clean.shape})")
        # Use regular heatmap instead
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot_data_clean,
            cmap="viridis",
            annot=True,
            fmt=".3f",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": f"{metric.replace('_', ' ').title()}"},
        )
        title_suffix = f" ({dataset_type.upper()})" if dataset_type else ""
        ax.set_title(
            f"{aggregation_method.title()} Performance by Embedding-Model Combination{title_suffix}\n{metric.replace('_', ' ').title()}"
        )
        ax.set_xlabel("Embedding-Model Combination")
        ax.set_ylabel("Dataset")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved embedding performance heatmap to: {save_path}")
        return fig, ax

    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for original NaN values (to show them differently)
    mask = pivot_data.isna()

    # Use clustermap for better visualization
    g = sns.clustermap(
        pivot_data_clean,
        cmap="viridis",
        annot=True,
        fmt=".3f",
        method="average",
        metric="euclidean",
        figsize=figsize,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": f"{metric.replace('_', ' ').title()}"},
        mask=mask,
    )

    title_suffix = f" ({dataset_type.upper()})" if dataset_type else ""
    g.fig.suptitle(
        f"{aggregation_method.title()} Performance by Embedding-Model Combination{title_suffix}\n{metric.replace('_', ' ').title()}",
        y=1.02,
    )
    g.ax_heatmap.set_xlabel("Embedding-Model Combination")
    g.ax_heatmap.set_ylabel("Dataset")
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        g.fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved embedding performance heatmap to: {save_path}")

    return g.fig, g.ax_heatmap


def create_model_performance_heatmap(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    aggregation_method: Literal["mean", "median"] = "median",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dataset_type: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create heatmap showing average performance by model across datasets and embeddings.

    Args:
        results_df: Results DataFrame
        metric: Metric to visualize
        figsize: Figure size
        save_path: Optional path to save the plot
        aggregation_method: How to aggregate values (mean or median)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        dataset_type: Optional dataset type label ("trans", "cis", or "together") to include in title

    Returns:
        Figure and Axes objects
    """
    if results_df.empty:
        print("No data found for model performance heatmap")
        return None, None

    # Filter out random strategy
    non_random_data = results_df[results_df["strategy"] != "random"]

    if non_random_data.empty:
        print("No non-random data found")
        return None, None

    # Create pivot table: dataset x regressor
    pivot_data = non_random_data.pivot_table(
        index="dataset", columns="regressor", values=metric, aggfunc=aggregation_method
    )

    if pivot_data.empty:
        print("No pivot data available")
        return None, None

    # Clean data: Drop rows/columns that are all NaN
    pivot_data = pivot_data.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if pivot_data.empty:
        print("No valid data after removing NaN rows/columns")
        return None, None

    # Replace remaining NaN values with 0 for clustering (but keep track for masking)
    # We need finite values for distance computation
    pivot_data_clean = pivot_data.fillna(0)

    # Check for any non-finite values and replace them
    if not np.isfinite(pivot_data_clean.values).all():
        print("Warning: Found non-finite values, replacing with 0")
        pivot_data_clean = pivot_data_clean.replace([np.inf, -np.inf], 0)
        pivot_data_clean = pivot_data_clean.fillna(0)

    # Ensure we have at least 2 rows and 2 columns for clustering
    if pivot_data_clean.shape[0] < 2 or pivot_data_clean.shape[1] < 2:
        print(f"Not enough data for clustering (shape: {pivot_data_clean.shape})")
        # Use regular heatmap instead
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot_data_clean,
            cmap="viridis",
            annot=True,
            fmt=".3f",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": f"{metric.replace('_', ' ').title()}"},
        )
        title_suffix = f" ({dataset_type.upper()})" if dataset_type else ""
        ax.set_title(
            f"{aggregation_method.title()} Performance by Model{title_suffix}\n{metric.replace('_', ' ').title()}"
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Dataset")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved model performance heatmap to: {save_path}")
        return fig, ax

    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for original NaN values (to show them differently)
    mask = pivot_data.isna()

    g = sns.clustermap(
        pivot_data_clean,
        cmap="viridis",
        annot=True,
        fmt=".3f",
        method="average",
        metric="euclidean",
        figsize=figsize,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": f"{metric.replace('_', ' ').title()}"},
        mask=mask,
    )

    title_suffix = f" ({dataset_type.upper()})" if dataset_type else ""
    g.fig.suptitle(
        f"{aggregation_method.title()} Performance by Model{title_suffix}\n{metric.replace('_', ' ').title()}",
        y=1.02,
    )
    g.ax_heatmap.set_xlabel("Model")
    g.ax_heatmap.set_ylabel("Dataset")
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        g.fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved model performance heatmap to: {save_path}")

    return g.fig, g.ax_heatmap


def create_embedding_summary_plot(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    aggregation_method: Literal["mean", "median"] = "mean",
    dataset_type: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create summary plot showing average performance by embedding method with error bars.

    Args:
        results_df: Results DataFrame
        metric: Metric to visualize
        figsize: Figure size
        save_path: Optional path to save the plot
        aggregation_method: How to aggregate values (mean or median)
        dataset_type: Optional dataset type label ("trans", "cis", or "together") to include in title

    Returns:
        Figure and Axes objects
    """
    if results_df.empty:
        print("No data found for embedding summary plot")
        return None, None

    # Filter out random strategy
    non_random_data = results_df[results_df["strategy"] != "random"]

    if non_random_data.empty:
        print("No non-random data found")
        return None, None

    # Aggregate by embedding method
    embedding_stats = (
        non_random_data.groupby("embedding")[metric]
        .agg(["mean", "std", "count", "median"])
        .reset_index()
    )

    # Calculate confidence intervals (assuming normal distribution)
    embedding_stats["ci_lower"] = embedding_stats["mean"] - 1.96 * embedding_stats[
        "std"
    ] / np.sqrt(embedding_stats["count"])
    embedding_stats["ci_upper"] = embedding_stats["mean"] + 1.96 * embedding_stats[
        "std"
    ] / np.sqrt(embedding_stats["count"])

    # Sort by chosen aggregation method
    sort_col = aggregation_method
    embedding_stats = embedding_stats.sort_values(sort_col, ascending=True)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar plot with error bars
    value_col = aggregation_method
    bars = ax1.bar(
        embedding_stats["embedding"],
        embedding_stats[value_col],
        yerr=[
            embedding_stats["mean"] - embedding_stats["ci_lower"],
            embedding_stats["ci_upper"] - embedding_stats["mean"],
        ],
        capsize=5,
        color="skyblue",
        edgecolor="navy",
        alpha=0.7,
    )

    title_suffix = f" ({dataset_type.upper()})" if dataset_type else ""
    ax1.set_title(
        f"{aggregation_method.title()} {metric.replace('_', ' ').title()} by Embedding Method{title_suffix}"
    )
    ax1.set_xlabel("Embedding Method")
    ax1.set_ylabel(f"{metric.replace('_', ' ').title()}")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, val in zip(bars, embedding_stats[value_col]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Box plot
    embedding_data = [
        non_random_data[non_random_data["embedding"] == emb][metric].values
        for emb in embedding_stats["embedding"]
    ]

    box_plot = ax2.boxplot(
        embedding_data, labels=embedding_stats["embedding"], patch_artist=True
    )

    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(embedding_data)))
    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_title(
        f"Distribution of {metric.replace('_', ' ').title()} by Embedding Method{title_suffix}"
    )
    ax2.set_xlabel("Embedding Method")
    ax2.set_ylabel(f"{metric.replace('_', ' ').title()}")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved embedding summary plot to: {save_path}")

    return fig, (ax1, ax2)


def create_model_summary_plot(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    aggregation_method: Literal["mean", "median"] = "mean",
    dataset_type: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create summary plot showing average performance by model with error bars.

    Args:
        results_df: Results DataFrame
        metric: Metric to visualize
        figsize: Figure size
        save_path: Optional path to save the plot
        aggregation_method: How to aggregate values (mean or median)
        dataset_type: Optional dataset type label ("trans", "cis", or "together") to include in title

    Returns:
        Figure and Axes objects
    """
    if results_df.empty:
        print("No data found for model summary plot")
        return None, None

    # Filter out random strategy
    non_random_data = results_df[results_df["strategy"] != "random"]

    if non_random_data.empty:
        print("No non-random data found")
        return None, None

    # Aggregate by model
    model_stats = (
        non_random_data.groupby("regressor")[metric]
        .agg(["mean", "std", "count", "median"])
        .reset_index()
    )

    # Calculate confidence intervals
    model_stats["ci_lower"] = model_stats["mean"] - 1.96 * model_stats["std"] / np.sqrt(
        model_stats["count"]
    )
    model_stats["ci_upper"] = model_stats["mean"] + 1.96 * model_stats["std"] / np.sqrt(
        model_stats["count"]
    )

    # Sort by chosen aggregation method
    sort_col = aggregation_method
    model_stats = model_stats.sort_values(sort_col, ascending=True)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar plot with error bars
    value_col = aggregation_method
    bars = ax1.bar(
        model_stats["regressor"],
        model_stats[value_col],
        yerr=[
            model_stats["mean"] - model_stats["ci_lower"],
            model_stats["ci_upper"] - model_stats["mean"],
        ],
        capsize=5,
        color="lightcoral",
        edgecolor="darkred",
        alpha=0.7,
    )

    title_suffix = f" ({dataset_type.upper()})" if dataset_type else ""
    ax1.set_title(
        f"{aggregation_method.title()} {metric.replace('_', ' ').title()} by Model{title_suffix}"
    )
    ax1.set_xlabel("Model")
    ax1.set_ylabel(f"{metric.replace('_', ' ').title()}")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, val in zip(bars, model_stats[value_col]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Box plot
    model_data = [
        non_random_data[non_random_data["regressor"] == model][metric].values
        for model in model_stats["regressor"]
    ]

    box_plot = ax2.boxplot(
        model_data, labels=model_stats["regressor"], patch_artist=True
    )

    # Color the boxes
    colors = plt.cm.plasma(np.linspace(0, 1, len(model_data)))
    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_title(
        f"Distribution of {metric.replace('_', ' ').title()} by Model{title_suffix}"
    )
    ax2.set_xlabel("Model")
    ax2.set_ylabel(f"{metric.replace('_', ' ').title()}")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved model summary plot to: {save_path}")

    return fig, (ax1, ax2)


def create_embedding_model_heatmap(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    aggregation_method: Literal["mean", "median"] = "mean",
    compare_to_baseline: bool = False,
    baseline_aggregation: Literal["mean", "median"] = "mean",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dataset_type: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create heatmap with embeddings on one axis and models on the other axis.

    Args:
        results_df: Results DataFrame
        metric: Metric to visualize
        figsize: Figure size
        save_path: Optional path to save the plot
        aggregation_method: How to aggregate across datasets (mean or median)
        compare_to_baseline: If True, show relative performance vs random baseline
        baseline_aggregation: How to aggregate baseline improvements across datasets (mean or median)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        dataset_type: Optional dataset type label ("trans", "cis", or "together") to include in title

    Returns:
        Figure and Axes objects
    """
    if results_df.empty:
        print("No data found for embedding-model heatmap")
        return None, None

    # Filter out random strategy for non-baseline data
    non_random_data = results_df[results_df["strategy"] != "random"]

    if non_random_data.empty:
        print("No non-random data found")
        return None, None

    if compare_to_baseline:
        # Get random baseline data
        random_data = results_df[results_df["strategy"] == "random"]

        if random_data.empty:
            print("No random baseline data found for comparison")
            return None, None

        # Create pivot table: dataset x (embedding + regressor) including random
        all_data = pd.concat([non_random_data, random_data], ignore_index=True)
        all_data["embedding_regressor"] = (
            all_data["embedding"] + "_" + all_data["regressor"]
        )

        # For random strategy, use "random" as the method label
        all_data.loc[all_data["strategy"] == "random", "embedding_regressor"] = "random"
        pivot_data = all_data.pivot_table(
            index="dataset",
            columns="embedding_regressor",
            values=metric,
            aggfunc=aggregation_method,
        )

        if pivot_data.empty:
            print("No pivot data available for baseline comparison")
            return None, None

        if "random" not in pivot_data.columns:
            print("No random baseline column found in pivot data")
            return None, None

        # Calculate percentage improvement per dataset: (method - baseline) / baseline * 100
        baseline_series = pivot_data["random"]
        delta = (
            pivot_data.subtract(baseline_series, axis=0).div(baseline_series, axis=0)
        ) * 100

        # Remove the random column and get only embedding-regressor combinations
        method_columns = [col for col in delta.columns if col != "random"]
        delta_methods = delta[method_columns]

        # Get unique embeddings from original data to properly split combined names
        unique_embeddings = sorted(all_data["embedding"].unique())

        # Convert back to embedding x regressor format
        embedding_regressor_data = []
        for col in method_columns:
            if "_" in col:
                # Try to match known embeddings first (longest match first)
                embedding = None
                regressor = None
                for emb in sorted(unique_embeddings, key=len, reverse=True):
                    if col.startswith(emb + "_"):
                        embedding = emb
                        regressor = col[len(emb) + 1 :]  # Remove embedding + "_"
                        break

                # Fallback: if no match found, use simple split on first underscore
                if embedding is None:
                    embedding, regressor = col.split("_", 1)
                # Use specified aggregation method for baseline improvements
                if aggregation_method == "median":
                    avg_improvement = delta_methods[col].median()
                else:  # mean
                    avg_improvement = delta_methods[col].mean()
                embedding_regressor_data.append(
                    {
                        "embedding": embedding,
                        "regressor": regressor,
                        "improvement": avg_improvement,
                    }
                )

        if embedding_regressor_data:
            df_temp = pd.DataFrame(embedding_regressor_data)
            pivot_data = df_temp.pivot_table(
                index="embedding", columns="regressor", values="improvement"
            )
        else:
            print("No embedding-regressor combinations found")
            return None, None

        title_suffix = (
            f"Relative to Random Baseline (% Improvement, {aggregation_method.title()})"
        )
        cbar_label = f"% Improvement vs Random ({aggregation_method.title()})"
    else:
        # Create pivot table: embedding x model (raw values)
        pivot_data = non_random_data.pivot_table(
            index="embedding",
            columns="regressor",
            values=metric,
            aggfunc=aggregation_method,
        )
        title_suffix = (
            f"{aggregation_method.title()} {metric.replace('_', ' ').title()}"
        )
        cbar_label = f"{metric.replace('_', ' ').title()}"

    if pivot_data.empty:
        print("No pivot data available")
        return None, None

    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Use seaborn heatmap for better styling
    # Choose colormap based on whether we're comparing to baseline
    cmap = "coolwarm" if compare_to_baseline else "viridis"
    center = 0.0 if compare_to_baseline else None
    fmt = ".1f" if compare_to_baseline else ".3f"  # Show percentage with 1 decimal

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        vmin=vmin if not compare_to_baseline else None,
        vmax=vmax if not compare_to_baseline else None,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )

    dataset_label = f" ({dataset_type.upper()})" if dataset_type else ""
    ax.set_title(
        f"Performance Heatmap: Embeddings vs Models{dataset_label}\n{title_suffix}",
        fontsize=14,
        pad=20,
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Embedding Method", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved embedding-model heatmap to: {save_path}")

    return fig, ax


def create_embedding_model_heatmap_with_stats(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    aggregation_method: Literal["mean", "median"] = "mean",
    compare_to_baseline: bool = False,
    baseline_aggregation: Literal["mean", "median"] = "mean",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dataset_type: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create heatmap with embeddings vs models, including statistical information.

    Args:
        results_df: Results DataFrame
        metric: Metric to visualize
        figsize: Figure size
        save_path: Optional path to save the plot
        aggregation_method: How to aggregate across datasets (mean or median)
        compare_to_baseline: If True, show relative performance vs random baseline
        baseline_aggregation: How to aggregate baseline improvements across datasets (mean or median)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        dataset_type: Optional dataset type label ("trans", "cis", or "together") to include in title

    Returns:
        Figure and Axes objects
    """
    if results_df.empty:
        print("No data found for embedding-model heatmap with stats")
        return None, None

    # Filter out random strategy for non-baseline data
    non_random_data = results_df[results_df["strategy"] != "random"]

    if non_random_data.empty:
        print("No non-random data found")
        return None, None

    if compare_to_baseline:
        # Get random baseline data
        random_data = results_df[results_df["strategy"] == "random"]

        if random_data.empty:
            print("No random baseline data found for comparison")
            return None, None

        # Create pivot table: dataset x (embedding + regressor) including random
        all_data = pd.concat([non_random_data, random_data], ignore_index=True)
        all_data["embedding_regressor"] = (
            all_data["embedding"] + "_" + all_data["regressor"]
        )

        # For random strategy, use "random" as the method label
        all_data.loc[all_data["strategy"] == "random", "embedding_regressor"] = "random"
        pivot_data = all_data.pivot_table(
            index="dataset",
            columns="embedding_regressor",
            values=metric,
            aggfunc=aggregation_method,
        )

        if pivot_data.empty:
            print("No pivot data available for baseline comparison")
            return None, None

        if "random" not in pivot_data.columns:
            print("No random baseline column found in pivot data")
            return None, None

        # Calculate percentage improvement per dataset: (method - baseline) / baseline * 100
        baseline_series = pivot_data["random"]
        delta = (
            pivot_data.subtract(baseline_series, axis=0).div(baseline_series, axis=0)
        ) * 100

        # Remove the random column and get only embedding-regressor combinations
        method_columns = [col for col in delta.columns if col != "random"]
        delta_methods = delta[method_columns]

        # Get unique embeddings from original data to properly split combined names
        unique_embeddings = sorted(all_data["embedding"].unique())

        # Convert back to embedding x regressor format
        embedding_regressor_data = []
        for col in method_columns:
            if "_" in col:
                # Try to match known embeddings first (longest match first)
                embedding = None
                regressor = None
                for emb in sorted(unique_embeddings, key=len, reverse=True):
                    if col.startswith(emb + "_"):
                        embedding = emb
                        regressor = col[len(emb) + 1 :]  # Remove embedding + "_"
                        break

                # Fallback: if no match found, use simple split on first underscore
                if embedding is None:
                    embedding, regressor = col.split("_", 1)
                # Use specified aggregation method for baseline improvements
                if aggregation_method == "median":
                    avg_improvement = delta_methods[col].median()
                else:  # mean
                    avg_improvement = delta_methods[col].mean()
                embedding_regressor_data.append(
                    {
                        "embedding": embedding,
                        "regressor": regressor,
                        "improvement": avg_improvement,
                    }
                )

        if embedding_regressor_data:
            df_temp = pd.DataFrame(embedding_regressor_data)
            pivot_data = df_temp.pivot_table(
                index="embedding", columns="regressor", values="improvement"
            )
        else:
            print("No embedding-regressor combinations found")
            return None, None

        title_suffix = (
            f"Relative to Random Baseline (% Improvement, {aggregation_method.title()})"
        )
        cbar_label = f"% Improvement vs Random ({aggregation_method.title()})"
    else:
        # Create pivot table: embedding x model (raw values)
        pivot_data = non_random_data.pivot_table(
            index="embedding",
            columns="regressor",
            values=metric,
            aggfunc=aggregation_method,
        )
        title_suffix = (
            f"{aggregation_method.title()} {metric.replace('_', ' ').title()}"
        )
        cbar_label = f"{metric.replace('_', ' ').title()}"

    if pivot_data.empty:
        print("No pivot data available")
        return None, None

    # Calculate count and std for each cell
    count_data = non_random_data.pivot_table(
        index="embedding", columns="regressor", values=metric, aggfunc="count"
    )

    std_data = non_random_data.pivot_table(
        index="embedding", columns="regressor", values=metric, aggfunc="std"
    )

    # Create the heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Main performance heatmap
    # Choose colormap based on whether we're comparing to baseline
    cmap = "coolwarm" if compare_to_baseline else "viridis"
    center = 0.0 if compare_to_baseline else None
    fmt = ".1f" if compare_to_baseline else ".3f"  # Show percentage with 1 decimal

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        vmin=vmin if not compare_to_baseline else None,
        vmax=vmax if not compare_to_baseline else None,
        cbar_kws={"label": cbar_label},
        ax=ax1,
    )
    dataset_label = f" ({dataset_type.upper()})" if dataset_type else ""
    ax1.set_title(f"Performance{dataset_label}: {title_suffix}")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Embedding Method")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax1.get_yticklabels(), rotation=0)

    # Count heatmap (number of samples per cell)
    sns.heatmap(
        count_data,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar_kws={"label": "Number of Samples"},
        ax=ax2,
    )
    ax2.set_title(f"Sample Count per Embedding-Model Combination{dataset_label}")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Embedding Method")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved embedding-model heatmap with stats to: {save_path}")

    return fig, (ax1, ax2)


def create_combined_analysis_figure(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    figsize: Tuple[int, int] = (24, 20),
    save_path: Optional[str] = None,
    aggregation_method: Literal["mean", "median"] = "mean",
    compare_to_baseline: bool = False,
    baseline_aggregation: Literal["mean", "median"] = "mean",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dataset_type: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a combined figure with all 6 analysis plots in a single figure.

    Args:
        results_df: Results DataFrame
        metric: Metric to visualize
        figsize: Overall figure size (will be adjusted for subplots)
        save_path: Optional path to save the plot
        aggregation_method: How to aggregate values (mean or median)
        compare_to_baseline: If True, show relative performance vs random baseline
        baseline_aggregation: How to aggregate baseline improvements across datasets
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        dataset_type: Optional dataset type label ("trans", "cis", or "together")

    Returns:
        Figure and array of Axes objects
    """
    if results_df.empty:
        print("No data found for combined analysis figure")
        return None, None

    # Create figure with 3 rows and 2 columns
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()

    # Calculate global color scale if not provided
    non_random_for_scale = results_df[results_df["strategy"] != "random"]
    if vmin is None or vmax is None:
        if not non_random_for_scale.empty:
            valid_metric_values = non_random_for_scale[metric].dropna()
            if not valid_metric_values.empty:
                if vmin is None:
                    vmin = float(valid_metric_values.min())
                if vmax is None:
                    vmax = float(valid_metric_values.max())

    # 1. Embedding performance heatmap (top-left)
    print("Creating embedding performance heatmap (combined figure)...")
    if non_random_for_scale.empty:
        axes[0].text(0.5, 0.5, "No data available", ha="center", va="center")
        axes[0].set_title("1. Embedding Performance Heatmap")
    else:
        # Create pivot table for embedding performance
        results_df_copy = results_df.copy()
        results_df_copy["embedding_regressor"] = (
            results_df_copy["embedding"] + "_" + results_df_copy["regressor"]
        )
        non_random_data = results_df_copy[results_df_copy["strategy"] != "random"]
        if not non_random_data.empty:
            pivot_data = non_random_data.pivot_table(
                index="dataset",
                columns="embedding_regressor",
                values=metric,
                aggfunc=aggregation_method,
            )
            pivot_data = pivot_data.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if not pivot_data.empty:
                pivot_data_clean = pivot_data.fillna(0).replace([np.inf, -np.inf], 0)
                mask = pivot_data.isna()
                sns.heatmap(
                    pivot_data_clean,
                    cmap="viridis",
                    annot=True,
                    fmt=".3f",
                    ax=axes[0],
                    vmin=vmin,
                    vmax=vmax,
                    mask=mask,
                    cbar_kws={"label": f"{metric.replace('_', ' ').title()}"},
                )
                title_suffix = f" ({dataset_type.upper()})" if dataset_type else ""
                axes[0].set_title(
                    f"1. Embedding-Model Performance Heatmap{title_suffix}",
                    fontsize=10,
                )
                axes[0].set_xlabel("Embedding-Model Combination", fontsize=9)
                axes[0].set_ylabel("Dataset", fontsize=9)
                plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=7)
                plt.setp(axes[0].get_yticklabels(), fontsize=7)
            else:
                axes[0].text(0.5, 0.5, "No data available", ha="center", va="center")
                axes[0].set_title("1. Embedding Performance Heatmap")
        else:
            axes[0].text(0.5, 0.5, "No data available", ha="center", va="center")
            axes[0].set_title("1. Embedding Performance Heatmap")

    # 2. Model performance heatmap (top-right)
    print("Creating model performance heatmap (combined figure)...")
    if non_random_for_scale.empty:
        axes[1].text(0.5, 0.5, "No data available", ha="center", va="center")
        axes[1].set_title("2. Model Performance Heatmap")
    else:
        non_random_data = results_df[results_df["strategy"] != "random"]
        if not non_random_data.empty:
            pivot_data = non_random_data.pivot_table(
                index="dataset",
                columns="regressor",
                values=metric,
                aggfunc=aggregation_method,
            )
            pivot_data = pivot_data.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if not pivot_data.empty:
                pivot_data_clean = pivot_data.fillna(0).replace([np.inf, -np.inf], 0)
                mask = pivot_data.isna()
                sns.heatmap(
                    pivot_data_clean,
                    cmap="viridis",
                    annot=True,
                    fmt=".3f",
                    ax=axes[1],
                    vmin=vmin,
                    vmax=vmax,
                    mask=mask,
                    cbar_kws={"label": f"{metric.replace('_', ' ').title()}"},
                )
                title_suffix = f" ({dataset_type.upper()})" if dataset_type else ""
                axes[1].set_title(
                    f"2. Model Performance Heatmap{title_suffix}", fontsize=10
                )
                axes[1].set_xlabel("Model", fontsize=9)
                axes[1].set_ylabel("Dataset", fontsize=9)
                plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=7)
                plt.setp(axes[1].get_yticklabels(), fontsize=7)
            else:
                axes[1].text(0.5, 0.5, "No data available", ha="center", va="center")
                axes[1].set_title("2. Model Performance Heatmap")
        else:
            axes[1].text(0.5, 0.5, "No data available", ha="center", va="center")
            axes[1].set_title("2. Model Performance Heatmap")

    # 3. Embedding summary plot (middle-left) - bar plot only
    print("Creating embedding summary plot (combined figure)...")
    if non_random_for_scale.empty:
        axes[2].text(0.5, 0.5, "No data available", ha="center", va="center")
        axes[2].set_title("3. Embedding Summary")
    else:
        non_random_data = results_df[results_df["strategy"] != "random"]
        if not non_random_data.empty:
            embedding_stats = (
                non_random_data.groupby("embedding")[metric]
                .agg(["mean", "std", "count", "median"])
                .reset_index()
            )
            embedding_stats["ci_lower"] = embedding_stats[
                "mean"
            ] - 1.96 * embedding_stats["std"] / np.sqrt(embedding_stats["count"])
            embedding_stats["ci_upper"] = embedding_stats[
                "mean"
            ] + 1.96 * embedding_stats["std"] / np.sqrt(embedding_stats["count"])
            sort_col = aggregation_method
            embedding_stats = embedding_stats.sort_values(sort_col, ascending=True)
            value_col = aggregation_method
            bars = axes[2].bar(
                embedding_stats["embedding"],
                embedding_stats[value_col],
                yerr=[
                    embedding_stats["mean"] - embedding_stats["ci_lower"],
                    embedding_stats["ci_upper"] - embedding_stats["mean"],
                ],
                capsize=5,
                color="skyblue",
                edgecolor="navy",
                alpha=0.7,
            )
            title_suffix = f" ({dataset_type.upper()})" if dataset_type else ""
            axes[2].set_title(f"3. Embedding Summary{title_suffix}", fontsize=10)
            axes[2].set_xlabel("Embedding Method", fontsize=9)
            axes[2].set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=9)
            axes[2].tick_params(axis="x", rotation=45, labelsize=7)
            axes[2].tick_params(axis="y", labelsize=7)
            for bar, val in zip(bars, embedding_stats[value_col]):
                axes[2].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
        else:
            axes[2].text(0.5, 0.5, "No data available", ha="center", va="center")
            axes[2].set_title("3. Embedding Summary")

    # 4. Model summary plot (middle-right) - bar plot only
    print("Creating model summary plot (combined figure)...")
    if non_random_for_scale.empty:
        axes[3].text(0.5, 0.5, "No data available", ha="center", va="center")
        axes[3].set_title("4. Model Summary")
    else:
        non_random_data = results_df[results_df["strategy"] != "random"]
        if not non_random_data.empty:
            model_stats = (
                non_random_data.groupby("regressor")[metric]
                .agg(["mean", "std", "count", "median"])
                .reset_index()
            )
            model_stats["ci_lower"] = model_stats["mean"] - 1.96 * model_stats[
                "std"
            ] / np.sqrt(model_stats["count"])
            model_stats["ci_upper"] = model_stats["mean"] + 1.96 * model_stats[
                "std"
            ] / np.sqrt(model_stats["count"])
            sort_col = aggregation_method
            model_stats = model_stats.sort_values(sort_col, ascending=True)
            value_col = aggregation_method
            bars = axes[3].bar(
                model_stats["regressor"],
                model_stats[value_col],
                yerr=[
                    model_stats["mean"] - model_stats["ci_lower"],
                    model_stats["ci_upper"] - model_stats["mean"],
                ],
                capsize=5,
                color="lightcoral",
                edgecolor="darkred",
                alpha=0.7,
            )
            title_suffix = f" ({dataset_type.upper()})" if dataset_type else ""
            axes[3].set_title(f"4. Model Summary{title_suffix}", fontsize=10)
            axes[3].set_xlabel("Model", fontsize=9)
            axes[3].set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=9)
            axes[3].tick_params(axis="x", rotation=45, labelsize=7)
            axes[3].tick_params(axis="y", labelsize=7)
            for bar, val in zip(bars, model_stats[value_col]):
                axes[3].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
        else:
            axes[3].text(0.5, 0.5, "No data available", ha="center", va="center")
            axes[3].set_title("4. Model Summary")

    # 5. Embedding vs Model heatmap (bottom-left)
    print("Creating embedding vs model heatmap (combined figure)...")
    if non_random_for_scale.empty:
        axes[4].text(0.5, 0.5, "No data available", ha="center", va="center")
        axes[4].set_title("5. Embedding vs Model Heatmap")
    else:
        non_random_data = results_df[results_df["strategy"] != "random"]
        if not non_random_data.empty:
            if compare_to_baseline:
                random_data = results_df[results_df["strategy"] == "random"]
                if not random_data.empty:
                    all_data = pd.concat(
                        [non_random_data, random_data], ignore_index=True
                    )
                    all_data["embedding_regressor"] = (
                        all_data["embedding"] + "_" + all_data["regressor"]
                    )
                    all_data.loc[
                        all_data["strategy"] == "random", "embedding_regressor"
                    ] = "random"
                    pivot_data = all_data.pivot_table(
                        index="dataset",
                        columns="embedding_regressor",
                        values=metric,
                        aggfunc=aggregation_method,
                    )
                    if "random" in pivot_data.columns:
                        baseline_series = pivot_data["random"]
                        delta = (
                            pivot_data.subtract(baseline_series, axis=0).div(
                                baseline_series, axis=0
                            )
                        ) * 100
                        method_columns = [
                            col for col in delta.columns if col != "random"
                        ]
                        delta_methods = delta[method_columns]
                        unique_embeddings = sorted(all_data["embedding"].unique())
                        embedding_regressor_data = []
                        for col in method_columns:
                            if "_" in col:
                                embedding = None
                                regressor = None
                                for emb in sorted(
                                    unique_embeddings, key=len, reverse=True
                                ):
                                    if col.startswith(emb + "_"):
                                        embedding = emb
                                        regressor = col[len(emb) + 1 :]
                                        break
                                if embedding is None:
                                    embedding, regressor = col.split("_", 1)
                                if aggregation_method == "median":
                                    avg_improvement = delta_methods[col].median()
                                else:
                                    avg_improvement = delta_methods[col].mean()
                                embedding_regressor_data.append(
                                    {
                                        "embedding": embedding,
                                        "regressor": regressor,
                                        "improvement": avg_improvement,
                                    }
                                )
                        if embedding_regressor_data:
                            df_temp = pd.DataFrame(embedding_regressor_data)
                            pivot_data = df_temp.pivot_table(
                                index="embedding",
                                columns="regressor",
                                values="improvement",
                            )
                            cmap = "coolwarm"
                            center = 0.0
                            fmt = ".1f"
                            cbar_label = f"% Improvement vs Random ({aggregation_method.title()})"
                        else:
                            pivot_data = pd.DataFrame()
                    else:
                        pivot_data = pd.DataFrame()
                else:
                    pivot_data = non_random_data.pivot_table(
                        index="embedding",
                        columns="regressor",
                        values=metric,
                        aggfunc=aggregation_method,
                    )
                    cmap = "viridis"
                    center = None
                    fmt = ".3f"
                    cbar_label = f"{metric.replace('_', ' ').title()}"
            else:
                pivot_data = non_random_data.pivot_table(
                    index="embedding",
                    columns="regressor",
                    values=metric,
                    aggfunc=aggregation_method,
                )
                cmap = "viridis"
                center = None
                fmt = ".3f"
                cbar_label = f"{metric.replace('_', ' ').title()}"

            if not pivot_data.empty:
                sns.heatmap(
                    pivot_data,
                    annot=True,
                    fmt=fmt,
                    cmap=cmap,
                    center=center,
                    vmin=vmin if not compare_to_baseline else None,
                    vmax=vmax if not compare_to_baseline else None,
                    cbar_kws={"label": cbar_label},
                    ax=axes[4],
                )
                dataset_label = f" ({dataset_type.upper()})" if dataset_type else ""
                title_suffix = (
                    f"Relative to Random Baseline (% Improvement, {aggregation_method.title()})"
                    if compare_to_baseline
                    else f"{aggregation_method.title()} {metric.replace('_', ' ').title()}"
                )
                axes[4].set_title(
                    f"5. Embeddings vs Models{dataset_label}\n{title_suffix}",
                    fontsize=10,
                )
                axes[4].set_xlabel("Model", fontsize=9)
                axes[4].set_ylabel("Embedding Method", fontsize=9)
                plt.setp(axes[4].get_xticklabels(), rotation=45, ha="right", fontsize=7)
                plt.setp(axes[4].get_yticklabels(), fontsize=7)
            else:
                axes[4].text(0.5, 0.5, "No data available", ha="center", va="center")
                axes[4].set_title("5. Embedding vs Model Heatmap")
        else:
            axes[4].text(0.5, 0.5, "No data available", ha="center", va="center")
            axes[4].set_title("5. Embedding vs Model Heatmap")

    # 6. Embedding vs Model heatmap with statistics (bottom-right) - performance only
    print("Creating embedding vs model heatmap with stats (combined figure)...")
    if non_random_for_scale.empty:
        axes[5].text(0.5, 0.5, "No data available", ha="center", va="center")
        axes[5].set_title("6. Embedding vs Model Heatmap (with stats)")
    else:
        non_random_data = results_df[results_df["strategy"] != "random"]
        if not non_random_data.empty:
            # Same logic as plot 5 for the performance heatmap
            if compare_to_baseline:
                random_data = results_df[results_df["strategy"] == "random"]
                if not random_data.empty:
                    all_data = pd.concat(
                        [non_random_data, random_data], ignore_index=True
                    )
                    all_data["embedding_regressor"] = (
                        all_data["embedding"] + "_" + all_data["regressor"]
                    )
                    all_data.loc[
                        all_data["strategy"] == "random", "embedding_regressor"
                    ] = "random"
                    pivot_data = all_data.pivot_table(
                        index="dataset",
                        columns="embedding_regressor",
                        values=metric,
                        aggfunc=aggregation_method,
                    )
                    if "random" in pivot_data.columns:
                        baseline_series = pivot_data["random"]
                        delta = (
                            pivot_data.subtract(baseline_series, axis=0).div(
                                baseline_series, axis=0
                            )
                        ) * 100
                        method_columns = [
                            col for col in delta.columns if col != "random"
                        ]
                        delta_methods = delta[method_columns]
                        unique_embeddings = sorted(all_data["embedding"].unique())
                        embedding_regressor_data = []
                        for col in method_columns:
                            if "_" in col:
                                embedding = None
                                regressor = None
                                for emb in sorted(
                                    unique_embeddings, key=len, reverse=True
                                ):
                                    if col.startswith(emb + "_"):
                                        embedding = emb
                                        regressor = col[len(emb) + 1 :]
                                        break
                                if embedding is None:
                                    embedding, regressor = col.split("_", 1)
                                if aggregation_method == "median":
                                    avg_improvement = delta_methods[col].median()
                                else:
                                    avg_improvement = delta_methods[col].mean()
                                embedding_regressor_data.append(
                                    {
                                        "embedding": embedding,
                                        "regressor": regressor,
                                        "improvement": avg_improvement,
                                    }
                                )
                        if embedding_regressor_data:
                            df_temp = pd.DataFrame(embedding_regressor_data)
                            pivot_data = df_temp.pivot_table(
                                index="embedding",
                                columns="regressor",
                                values="improvement",
                            )
                            cmap = "coolwarm"
                            center = 0.0
                            fmt = ".1f"
                            cbar_label = f"% Improvement vs Random ({aggregation_method.title()})"
                        else:
                            pivot_data = pd.DataFrame()
                    else:
                        pivot_data = pd.DataFrame()
                else:
                    pivot_data = non_random_data.pivot_table(
                        index="embedding",
                        columns="regressor",
                        values=metric,
                        aggfunc=aggregation_method,
                    )
                    cmap = "viridis"
                    center = None
                    fmt = ".3f"
                    cbar_label = f"{metric.replace('_', ' ').title()}"
            else:
                pivot_data = non_random_data.pivot_table(
                    index="embedding",
                    columns="regressor",
                    values=metric,
                    aggfunc=aggregation_method,
                )
                cmap = "viridis"
                center = None
                fmt = ".3f"
                cbar_label = f"{metric.replace('_', ' ').title()}"

            if not pivot_data.empty:
                # Calculate count for annotation
                count_data = non_random_data.pivot_table(
                    index="embedding",
                    columns="regressor",
                    values=metric,
                    aggfunc="count",
                )
                # Create annotation with count in parentheses
                annot_data = pivot_data.copy()
                for emb in pivot_data.index:
                    for reg in pivot_data.columns:
                        val = pivot_data.loc[emb, reg]
                        count = (
                            count_data.loc[emb, reg]
                            if emb in count_data.index and reg in count_data.columns
                            else 0
                        )
                        if pd.notna(val):
                            if fmt == ".1f":
                                annot_data.loc[emb, reg] = (
                                    f"{val:.1f}\n(n={int(count)})"
                                )
                            elif fmt == ".3f":
                                annot_data.loc[emb, reg] = (
                                    f"{val:.3f}\n(n={int(count)})"
                                )
                            else:
                                annot_data.loc[emb, reg] = (
                                    f"{val:.2f}\n(n={int(count)})"
                                )
                        else:
                            annot_data.loc[emb, reg] = ""
                sns.heatmap(
                    pivot_data,
                    annot=annot_data,
                    fmt="",
                    cmap=cmap,
                    center=center,
                    vmin=vmin if not compare_to_baseline else None,
                    vmax=vmax if not compare_to_baseline else None,
                    cbar_kws={"label": cbar_label},
                    ax=axes[5],
                )
                dataset_label = f" ({dataset_type.upper()})" if dataset_type else ""
                title_suffix = (
                    f"Relative to Random Baseline (% Improvement, {aggregation_method.title()})"
                    if compare_to_baseline
                    else f"{aggregation_method.title()} {metric.replace('_', ' ').title()} (with sample counts)"
                )
                axes[5].set_title(
                    f"6. Embeddings vs Models{dataset_label}\n{title_suffix}",
                    fontsize=10,
                )
                axes[5].set_xlabel("Model", fontsize=9)
                axes[5].set_ylabel("Embedding Method", fontsize=9)
                plt.setp(axes[5].get_xticklabels(), rotation=45, ha="right", fontsize=7)
                plt.setp(axes[5].get_yticklabels(), fontsize=7)
            else:
                axes[5].text(0.5, 0.5, "No data available", ha="center", va="center")
                axes[5].set_title("6. Embedding vs Model Heatmap (with stats)")
        else:
            axes[5].text(0.5, 0.5, "No data available", ha="center", va="center")
            axes[5].set_title("6. Embedding vs Model Heatmap (with stats)")

    # Add overall title
    dataset_label = f" ({dataset_type.upper()})" if dataset_type else ""
    fig.suptitle(
        f"Comprehensive Performance Analysis{dataset_label}\n{aggregation_method.title()} {metric.replace('_', ' ').title()}",
        fontsize=14,
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved combined analysis figure to: {save_path}")

    return fig, axes


def perform_statistical_tests(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """
    Perform statistical tests to compare performance across different dimensions.

    Args:
        results_df: Results DataFrame
        metric: Metric to test
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if results_df.empty:
        return {}

    # Filter out random strategy
    non_random_data = results_df[results_df["strategy"] != "random"]

    if non_random_data.empty:
        return {}

    test_results = {}

    # Test embedding methods
    embedding_groups = [
        group[metric].values for name, group in non_random_data.groupby("embedding")
    ]
    embedding_names = list(non_random_data["embedding"].unique())

    if len(embedding_groups) > 1:
        try:
            # Kruskal-Wallis test (non-parametric ANOVA)
            h_stat, p_value = kruskal(*embedding_groups)
            test_results["embedding_kruskal"] = {
                "test": "Kruskal-Wallis",
                "statistic": h_stat,
                "p_value": p_value,
                "significant": p_value < alpha,
                "groups": embedding_names,
            }

            # Pairwise Mann-Whitney U tests
            embedding_pairs = list(combinations(embedding_names, 2))
            embedding_pairwise = {}
            for emb1, emb2 in embedding_pairs:
                group1 = non_random_data[non_random_data["embedding"] == emb1][
                    metric
                ].values
                group2 = non_random_data[non_random_data["embedding"] == emb2][
                    metric
                ].values
                try:
                    u_stat, p_val = mannwhitneyu(
                        group1, group2, alternative="two-sided"
                    )
                    embedding_pairwise[f"{emb1}_vs_{emb2}"] = {
                        "statistic": u_stat,
                        "p_value": p_val,
                        "significant": p_val < alpha,
                    }
                except ValueError:
                    embedding_pairwise[f"{emb1}_vs_{emb2}"] = {
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "significant": False,
                    }
            test_results["embedding_pairwise"] = embedding_pairwise

        except Exception as e:
            print(f"Error in embedding statistical tests: {e}")

    # Test models
    model_groups = [
        group[metric].values for name, group in non_random_data.groupby("regressor")
    ]
    model_names = list(non_random_data["regressor"].unique())

    if len(model_groups) > 1:
        try:
            # Kruskal-Wallis test
            h_stat, p_value = kruskal(*model_groups)
            test_results["model_kruskal"] = {
                "test": "Kruskal-Wallis",
                "statistic": h_stat,
                "p_value": p_value,
                "significant": p_value < alpha,
                "groups": model_names,
            }

            # Pairwise Mann-Whitney U tests
            model_pairs = list(combinations(model_names, 2))
            model_pairwise = {}
            for model1, model2 in model_pairs:
                group1 = non_random_data[non_random_data["regressor"] == model1][
                    metric
                ].values
                group2 = non_random_data[non_random_data["regressor"] == model2][
                    metric
                ].values
                try:
                    u_stat, p_val = mannwhitneyu(
                        group1, group2, alternative="two-sided"
                    )
                    model_pairwise[f"{model1}_vs_{model2}"] = {
                        "statistic": u_stat,
                        "p_value": p_val,
                        "significant": p_val < alpha,
                    }
                except ValueError:
                    model_pairwise[f"{model1}_vs_{model2}"] = {
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "significant": False,
                    }
            test_results["model_pairwise"] = model_pairwise

        except Exception as e:
            print(f"Error in model statistical tests: {e}")

    return test_results


def print_summary_statistics(
    results_df: pd.DataFrame, metric: str = "auc_normalized_pred"
):
    """Print comprehensive summary statistics."""
    if results_df.empty:
        print("No data available for summary statistics")
        return

    print(f"\n{'=' * 60}")
    print(f"SUMMARY STATISTICS FOR {metric.upper()}")
    print(f"{'=' * 60}")

    # Overall statistics
    valid_data = results_df.dropna(subset=[metric])
    if not valid_data.empty:
        print("\nOverall Statistics:")
        print(f"  Count: {len(valid_data)}")
        print(f"  Mean:  {valid_data[metric].mean():.4f}")
        print(f"  Std:   {valid_data[metric].std():.4f}")
        print(f"  Min:   {valid_data[metric].min():.4f}")
        print(f"  Max:   {valid_data[metric].max():.4f}")
        print(f"  Median: {valid_data[metric].median():.4f}")

    # By embedding method
    print("\nBy Embedding Method:")
    for embedding in sorted(valid_data["embedding"].unique()):
        emb_data = valid_data[valid_data["embedding"] == embedding][metric]
        print(
            f"  {embedding:12}: Mean={emb_data.mean():.4f}, Std={emb_data.std():.4f}, N={len(emb_data)}"
        )

    # By model
    print("\nBy Model:")
    for model in sorted(valid_data["regressor"].unique()):
        model_data = valid_data[valid_data["regressor"] == model][metric]
        print(
            f"  {model:15}: Mean={model_data.mean():.4f}, Std={model_data.std():.4f}, N={len(model_data)}"
        )

    # By dataset
    print("\nBy Dataset:")
    for dataset in sorted(valid_data["dataset"].unique()):
        dataset_data = valid_data[valid_data["dataset"] == dataset][metric]
        print(
            f"  {dataset:15}: Mean={dataset_data.mean():.4f}, Std={dataset_data.std():.4f}, N={len(dataset_data)}"
        )


def summarize_performance_by_dataset(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    include_random: bool = False,
) -> pd.DataFrame:
    """Summarize performance per dataset for a given metric.

    This computes mean, std, count, and median of the metric per dataset.

    Args:
        results_df: Long-form results dataframe containing a `dataset` column and the metric column.
        metric: Name of the metric column to summarize.
        include_random: Whether to include rows with strategy == "random". Defaults to False.

    Returns:
        A dataframe with one row per dataset and columns: mean, std, count, median.
    """
    if results_df.empty:
        return pd.DataFrame()

    data = results_df.dropna(subset=[metric]).copy()

    # Non-random summary (raw performance)
    non_random = data
    if not include_random and "strategy" in data.columns:
        non_random = data[data["strategy"] != "random"]

    if non_random.empty:
        return pd.DataFrame()

    non_random_summary = (
        non_random.groupby("dataset")[metric]
        .agg(["mean", "std", "count", "median"])
        .reset_index()
    )
    non_random_summary = non_random_summary.rename(
        columns={
            "mean": f"{metric}_mean",
            "std": f"{metric}_std",
            "count": f"{metric}_count",
            "median": f"{metric}_median",
        }
    )

    # Baseline (random strategy) per dataset if available
    baseline_summary = pd.DataFrame()
    if "strategy" in data.columns:
        random_rows = data[data["strategy"] == "random"]
        if not random_rows.empty:
            baseline_summary = (
                random_rows.groupby("dataset")[metric]
                .agg(["mean", "std", "count", "median"])
                .reset_index()
                .rename(
                    columns={
                        "mean": f"{metric}_baseline_mean",
                        "std": f"{metric}_baseline_std",
                        "count": f"{metric}_baseline_count",
                        "median": f"{metric}_baseline_median",
                    }
                )
            )

    # Merge and compute % improvements
    if not baseline_summary.empty:
        merged = pd.merge(
            non_random_summary, baseline_summary, on="dataset", how="left"
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_pct = (
                (merged[f"{metric}_mean"] - merged[f"{metric}_baseline_mean"])
                / merged[f"{metric}_baseline_mean"]
            ) * 100.0
            median_pct = (
                (merged[f"{metric}_median"] - merged[f"{metric}_baseline_median"])
                / merged[f"{metric}_baseline_median"]
            ) * 100.0
        merged[f"{metric}_improvement_vs_baseline_mean_pct"] = mean_pct.replace(
            [np.inf, -np.inf], np.nan
        )
        merged[f"{metric}_improvement_vs_baseline_median_pct"] = median_pct.replace(
            [np.inf, -np.inf], np.nan
        )
        result = merged
    else:
        result = non_random_summary

    return result.sort_values(by=f"{metric}_mean", ascending=False)


def calculate_percentage_improvement_vs_baseline(
    results_df: pd.DataFrame, metric: str = "auc_normalized_pred"
) -> pd.DataFrame:
    """
    Calculate percentage improvement vs baseline for each method.

    Args:
        results_df: Results DataFrame with baseline (random) and method data
        metric: Metric column to calculate improvement for

    Returns:
        DataFrame with added percentage improvement column
    """
    if results_df.empty:
        return results_df

    # Create a copy to avoid modifying the original
    df = results_df.copy()

    # Initialize the improvement column
    df[f"{metric}_improvement_vs_baseline_pct"] = np.nan

    # Get unique datasets
    datasets = df["dataset"].unique()

    for dataset in datasets:
        dataset_data = df[df["dataset"] == dataset]

        # Get baseline (random strategy) performance for this dataset
        baseline_data = dataset_data[dataset_data["strategy"] == "random"]

        if baseline_data.empty:
            continue

        # Get baseline value (should be the same for all random entries in a dataset)
        baseline_value = baseline_data[metric].iloc[0]

        if pd.isna(baseline_value) or baseline_value == 0:
            continue

        # Calculate percentage improvement for non-random strategies
        non_random_mask = dataset_data["strategy"] != "random"
        non_random_indices = dataset_data[non_random_mask].index

        for idx in non_random_indices:
            method_value = df.loc[idx, metric]
            if not pd.isna(method_value):
                improvement_pct = (
                    (method_value - baseline_value) / baseline_value
                ) * 100
                df.loc[idx, f"{metric}_improvement_vs_baseline_pct"] = improvement_pct

    return df


def main():
    """Main function to generate averaged performance analysis."""
    parser = argparse.ArgumentParser(
        description="Generate averaged performance analysis"
    )
    parser.add_argument(
        "--results-base-path",
        type=str,
        default="results/166k_2024_regulators_auto_gen",
        help="Base path for results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: derived from results-base-path)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="auc_normalized_pred",
        help="Metric to analyze",
    )
    parser.add_argument(
        "--aggregation-method",
        type=str,
        choices=["mean", "median"],
        default="median",
        help="Aggregation method to use for plots (mean or median). Default: median",
    )
    parser.add_argument(
        "--compare-to-baseline",
        action="store_true",
        help="Compare performance to random baseline instead of showing raw values",
    )
    parser.add_argument(
        "--baseline-aggregation",
        type=str,
        choices=["mean", "median"],
        default="mean",
        help="How to aggregate baseline improvements across datasets (mean or median)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["trans", "cis", "together"],
        default=None,
        help="Dataset type label to include in plot titles (trans, cis, or together)",
    )
    args = parser.parse_args()

    # Check if path exists
    if not os.path.exists(args.results_base_path):
        print(f"Results path not found: {args.results_base_path}")
        return

    # Derive output directory from results_base_path if not provided
    if args.output_dir is None:
        results_path = Path(args.results_base_path)
        results_base_name = results_path.name or results_path.stem
        output_dir = Path("plots") / results_base_name / "averaged_performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(output_dir)
        print(f"Output directory derived from input path: {args.output_dir}")
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset type (default to None if not specified)
    dataset_type = args.dataset_type

    # Set up cache file path in input directory
    cache_file = os.path.join(args.results_base_path, "collected_all_results_cache.csv")

    print("Collecting results...")
    results_df = collect_all_results(args.results_base_path, cache_file=cache_file)

    if results_df.empty:
        print("No results found!")
        return

    print(f"Found {len(results_df)} result combinations")
    print(f"Datasets: {sorted(results_df['dataset'].unique())}")
    print(f"Embeddings: {sorted(results_df['embedding'].unique())}")
    print(f"Models: {sorted(results_df['regressor'].unique())}")
    print(f"Strategies: {sorted(results_df['strategy'].unique())}")

    # Calculate percentage improvement vs baseline
    print("Calculating percentage improvement vs baseline...")
    results_df = calculate_percentage_improvement_vs_baseline(results_df, args.metric)

    # Print summary of improvements
    improvement_col = f"{args.metric}_improvement_vs_baseline_pct"
    non_random_data = results_df[results_df["strategy"] != "random"]
    if not non_random_data.empty and improvement_col in non_random_data.columns:
        valid_improvements = non_random_data[improvement_col].dropna()
        if not valid_improvements.empty:
            print("\nPercentage Improvement vs Baseline Summary:")
            print(f"  Mean improvement: {valid_improvements.mean():.2f}%")
            print(f"  Median improvement: {valid_improvements.median():.2f}%")
            print(f"  Min improvement: {valid_improvements.min():.2f}%")
            print(f"  Max improvement: {valid_improvements.max():.2f}%")
            print(
                f"  Methods with positive improvement: {(valid_improvements > 0).sum()}/{len(valid_improvements)}"
            )

    # Print summary statistics
    print_summary_statistics(results_df, args.metric)

    # Calculate global color scale for consistent coloring across plots
    non_random_for_scale = results_df[results_df["strategy"] != "random"]
    if not non_random_for_scale.empty:
        valid_metric_values = non_random_for_scale[args.metric].dropna()
        if not valid_metric_values.empty:
            global_vmin = float(valid_metric_values.min())
            global_vmax = float(valid_metric_values.max())
            print(
                f"\nGlobal color scale for {args.metric}: [{global_vmin:.4f}, {global_vmax:.4f}]"
            )
        else:
            global_vmin = None
            global_vmax = None
            print(f"Warning: No valid values found for {args.metric}, using auto scale")
    else:
        global_vmin = None
        global_vmax = None
        print("Warning: No non-random data found, using auto scale")

    # Create subdirectory for aggregation method
    aggregation_output_dir = os.path.join(args.output_dir, args.aggregation_method)
    os.makedirs(aggregation_output_dir, exist_ok=True)

    # 1. Embedding performance heatmap
    print("\nCreating embedding performance heatmap...")
    fig, ax = create_embedding_performance_heatmap(
        results_df,
        metric=args.metric,
        aggregation_method=args.aggregation_method,
        vmin=global_vmin,
        vmax=global_vmax,
        dataset_type=dataset_type,
        save_path=os.path.join(
            aggregation_output_dir,
            f"embedding_performance_heatmap_{args.metric}.pdf",
        ),
    )
    if fig:
        plt.close(fig)

    # 2. Model performance heatmap
    print("Creating model performance heatmap...")
    fig, ax = create_model_performance_heatmap(
        results_df,
        metric=args.metric,
        aggregation_method=args.aggregation_method,
        vmin=global_vmin,
        vmax=global_vmax,
        dataset_type=dataset_type,
        save_path=os.path.join(
            aggregation_output_dir,
            f"model_performance_heatmap_{args.metric}.pdf",
        ),
    )
    if fig:
        plt.close(fig)

    # 3. Embedding summary plot
    print("Creating embedding summary plot...")
    fig, ax = create_embedding_summary_plot(
        results_df,
        metric=args.metric,
        aggregation_method=args.aggregation_method,
        dataset_type=dataset_type,
        save_path=os.path.join(
            aggregation_output_dir,
            f"embedding_summary_{args.metric}.pdf",
        ),
    )
    if fig:
        plt.close(fig)

    # 4. Model summary plot
    print("Creating model summary plot...")
    fig, ax = create_model_summary_plot(
        results_df,
        metric=args.metric,
        aggregation_method=args.aggregation_method,
        dataset_type=dataset_type,
        save_path=os.path.join(
            aggregation_output_dir,
            f"model_summary_{args.metric}.pdf",
        ),
    )
    if fig:
        plt.close(fig)

    # 5. Embedding vs Model heatmap
    print("Creating embedding vs model heatmap...")
    fig, ax = create_embedding_model_heatmap(
        results_df,
        metric=args.metric,
        aggregation_method=args.aggregation_method,
        compare_to_baseline=args.compare_to_baseline,
        baseline_aggregation=args.baseline_aggregation,
        vmin=global_vmin,
        vmax=global_vmax,
        dataset_type=dataset_type,
        save_path=os.path.join(
            aggregation_output_dir,
            f"embedding_model_heatmap_{args.metric}.pdf",
        ),
    )
    if fig:
        plt.close(fig)

    # 6. Embedding vs Model heatmap with statistics
    print("Creating embedding vs model heatmap with statistics...")
    fig, ax = create_embedding_model_heatmap_with_stats(
        results_df,
        metric=args.metric,
        aggregation_method=args.aggregation_method,
        compare_to_baseline=args.compare_to_baseline,
        baseline_aggregation=args.baseline_aggregation,
        vmin=global_vmin,
        vmax=global_vmax,
        dataset_type=dataset_type,
        save_path=os.path.join(
            aggregation_output_dir,
            f"embedding_model_heatmap_stats_{args.metric}.pdf",
        ),
    )
    if fig:
        plt.close(fig)

    # 7. Combined figure with all 6 plots
    print("\nCreating combined analysis figure with all 6 plots...")
    fig, axes = create_combined_analysis_figure(
        results_df,
        metric=args.metric,
        aggregation_method=args.aggregation_method,
        compare_to_baseline=args.compare_to_baseline,
        baseline_aggregation=args.baseline_aggregation,
        vmin=global_vmin,
        vmax=global_vmax,
        dataset_type=dataset_type,
        save_path=os.path.join(
            aggregation_output_dir,
            f"combined_analysis_figure_{args.metric}.pdf",
        ),
    )
    if fig:
        plt.close(fig)

    # 8. Statistical tests
    print("Performing statistical tests...")
    test_results = perform_statistical_tests(results_df, args.metric)

    # Print statistical test results
    print(f"\n{'=' * 60}")
    print("STATISTICAL TEST RESULTS")
    print(f"{'=' * 60}")

    for test_name, test_data in test_results.items():
        if "kruskal" in test_name:
            print(f"\n{test_name.upper()}:")
            print(f"  Test: {test_data['test']}")
            print(f"  Statistic: {test_data['statistic']:.4f}")
            print(f"  P-value: {test_data['p_value']:.4f}")
            print(f"  Significant: {test_data['significant']}")
            print(f"  Groups: {test_data['groups']}")
        elif "pairwise" in test_name:
            print(f"\n{test_name.upper()}:")
            for pair_name, pair_data in test_data.items():
                print(f"  {pair_name}:")
                print(f"    Statistic: {pair_data['statistic']:.4f}")
                print(f"    P-value: {pair_data['p_value']:.4f}")
                print(f"    Significant: {pair_data['significant']}")

    # Save results to CSV
    results_output_path = os.path.join(
        aggregation_output_dir,
        f"averaged_performance_results_{args.metric}.csv",
    )
    results_df.to_csv(results_output_path, index=False)
    print(f"\nSaved detailed results to: {results_output_path}")

    # Save per-dataset summary CSV (excluding random strategy by default)
    dataset_summary_df = summarize_performance_by_dataset(
        results_df, metric=args.metric, include_random=False
    )
    dataset_summary_path = os.path.join(
        aggregation_output_dir,
        f"per_dataset_summary_{args.metric}.csv",
    )
    if not dataset_summary_df.empty:
        dataset_summary_df.to_csv(dataset_summary_path, index=False)
        print(f"Saved per-dataset summary to: {dataset_summary_path}")
    else:
        print("No data available to write per-dataset summary CSV")

    print(f"\nAnalysis complete! All plots saved to: {aggregation_output_dir}")


if __name__ == "__main__":
    main()
