#!/usr/bin/env python3
"""
Grid plot for area under curve metrics across datasets, embedding methods, and regressors.

This script creates a heatmap where:
- Y-axis: datasets (Feng_2023, Angenent-Mari_2020, alcantar_2025)
- X-axis: embedding method + regressor combinations (e.g., onehot_KNN, sei_LinearRegression)
- Color: area under the curve of cumulative metrics

Example Usage:
- Single:
    python plotting/area_under_curve_grid_plot.py --results-base-path /path/to/directory
- Double:
    python plotting/area_under_curve_grid_plot.py \
    --results-base-path-1 /path/to/directory1 \
    --results-base-path-2 /path/to/directory2 \
    --title-1 "Experiment Trans" \
    --title-2 "Experiment Cis"
"""

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from new_file_plot_gen import create_grid_plot_avg_col as create_grid_plot
from scipy import integrate

matplotlib.use("Agg")  # Use non-interactive backend


def extract_info_from_path(file_path):
    """Extract dataset, embedding, and regressor info from file path."""
    path_parts = Path(file_path).parts

    # Find dataset name
    dataset = None
    dataset_keywords = [
        "AD",
        "CIS",
    ]
    for part in path_parts:
        if any(d in part for d in dataset_keywords):
            dataset = part
            break

    # Find embedding method (check in order of specificity)
    embedding = None
    for part in path_parts:
        if "onehot_pca" in part:
            embedding = "onehot_pca"
            break
        elif "onehotRaw" in part:
            embedding = "onehotRaw"
            break
        elif "evo" in part or "evo2" in part:
            embedding = "evo"
            break
        elif "sei" in part:
            embedding = "sei"
            break

    return dataset, embedding


def extract_regressor_from_filename(filename):
    """Extract regressor name from filename."""
    if "KNN_regression" in filename:
        return "KNN"
    elif "linear_regression" in filename:
        return "LinearRegression"
    elif "random_forest" in filename:
        return "RandomForest"
    elif "xg_boost" in filename:
        return "XGBoost"
    return "Unknown"


def _auc_one_seed(df, metric_column, normalize=True):
    df = df.sort_values("round")
    x = df["train_size"].to_numpy()
    y = df[metric_column].to_numpy()
    if x.size < 2 or np.all(x == x[0]):  # need at least 2 distinct x points
        return np.nan
    auc = integrate.trapezoid(y, x)  # if using SciPy
    # auc = np.trapz(y, x)                   # NumPy alternative
    if normalize:
        xr = x.max() - x.min()
        if xr > 0:
            auc = auc / xr
    return float(auc)


def calculate_auc_by_seed(
    data: pd.DataFrame,
    metric_column: str = "normalized_predictions_ground_truth_values_cumulative",
    normalize: bool = True,
):
    """
    Returns (mean_auc_across_seeds, per_seed_auc_series)
    """
    if metric_column not in data.columns:
        return np.nan, pd.Series(dtype=float)

    per_seed = data.groupby("seed", group_keys=False).apply(
        lambda df: _auc_one_seed(df, metric_column, normalize=normalize),
        include_groups=False,
    )
    mean_auc = per_seed.mean(skipna=True)
    return mean_auc


def calculate_auc_from_cumulative(
    data,
    metric_column="normalized_predictions_predictions_values_cumulative",
    file_path=None,
):
    """Calculate area under curve from cumulative metric values."""
    if metric_column not in data.columns:
        return np.nan

    # Sort by training round
    data_sorted = data.sort_values("round")
    x = data_sorted["train_size"].values
    y = data_sorted[metric_column].values

    if len(x) < 2:
        return np.nan

    # Calculate AUC using trapezoidal rule
    auc = integrate.trapezoid(y, x)

    # Normalize by the range of x to get average performance
    x_range = x.max() - x.min()
    if x_range > 0:
        auc = auc / x_range

    if auc > 1:
        print(f"Warning: AUC {auc} > 1 for file_path {file_path}")
        # Don't return NaN, let's keep the value for now
        # return np.nan

    return auc


def collect_all_results(results_base_path):
    """Collect all AUC results from the results directory."""
    results_list = []

    # Walk through all directories
    for root, _dirs, files in os.walk(results_base_path):
        # print(f"SANITY CHECK PRINTS: files {root}")
        for file in files:
            if file == "combined_all_custom_metrics.csv":
                file_path = os.path.join(root, file)
                print(file_path)

                # Extract dataset and embedding info
                dataset, embedding = extract_info_from_path(file_path)

                if dataset is None or embedding is None:
                    continue

                try:
                    # Read the data
                    df = pd.read_csv(file_path)

                    # Get unique regressors in this file
                    regressors = df["regression_model"].unique()

                    for regressor in regressors:
                        # Filter data for this regressor
                        regressor_data = df[df["regression_model"] == regressor]

                        if len(regressor_data) == 0:
                            continue

                        # Get unique strategies for this regressor
                        strategies = regressor_data["strategy"].unique()
                        for strategy in strategies:
                            # Filter data for this strategy
                            strategy_data = regressor_data[
                                regressor_data["strategy"] == strategy
                            ]

                            if len(strategy_data) == 0:
                                continue

                            # Calculate AUC for different metrics
                            auc_normalized = calculate_auc_by_seed(
                                strategy_data,
                                "normalized_predictions_ground_truth_values_cumulative",
                            )

                            os.makedirs("./debug_result", exist_ok=True)
                            strategy_data.to_csv(
                                f"./debug_result/{dataset}_{embedding}_{regressor}_{strategy}.csv"
                            )

                            # For random strategy, make it independent of regressor and embedding
                            if strategy == "random":
                                method_label = "random"
                            else:
                                method_label = f"{embedding}_{regressor}"

                            # Store results
                            results_list.append(
                                {
                                    "dataset": dataset,
                                    "embedding": embedding,
                                    "regressor": regressor,
                                    "strategy": strategy,
                                    "method_label": method_label,
                                    "auc_normalized_pred": auc_normalized,
                                    "file_path": file_path,
                                }
                            )
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

    final_frame = pd.DataFrame(results_list)
    final_frame.to_csv("final_frame.csv")
    return final_frame


def create_dual_heatmap(
    results_df1: pd.DataFrame,
    results_df2: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    title1: str = "Directory 1",
    title2: str = "Directory 2",
    figsize: tuple[int, int] = (56, 32),
    baseline_label: str = "random",
    drop_baseline: bool = True,
):
    """
    Create two heatmaps side by side with the same colormap scale.

    Args:
        results_df1: First DataFrame with results.
        results_df2: Second DataFrame with results.
        metric: Metric column to visualize.
        title1: Title for the first heatmap.
        title2: Title for the second heatmap.
        figsize: Figure size (width, height).
        baseline_label: Column used as baseline.
        drop_baseline: If True, drop the baseline column from the plot.

    Returns:
        (fig, axes): Matplotlib Figure and list of Axes, or (None, None) if no data.
    """
    if results_df1.empty and results_df2.empty:
        print("No data found to plot")
        return None, None

    # Process both dataframes
    def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for heatmap: returns (pivot_data, mat_norm)."""
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        df_copy = df.copy()
        if "method_label" not in df_copy.columns:
            df_copy["method_label"] = (
                df_copy["embedding"].astype(str)
                + "_"
                + df_copy["regressor"].astype(str)
            )

        # Build pivot with one aggregated baseline per dataset
        random_results = (
            df_copy[df_copy["strategy"] == "random"]
            .groupby("dataset", as_index=False)
            .agg({metric: "median"})
        )
        random_results["method_label"] = baseline_label
        non_random_results = df_copy[df_copy["strategy"] != "random"]
        plot_data = pd.concat([non_random_results, random_results], ignore_index=True)

        pivot_data = plot_data.pivot_table(
            index="dataset", columns="method_label", values=metric, aggfunc="median"
        )
        if pivot_data.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Calculate delta vs baseline
        if baseline_label not in pivot_data.columns:
            mat = pivot_data.copy()
        else:
            baseline_series = pivot_data[baseline_label]
            mat = pivot_data.subtract(baseline_series, axis=0)

        # Optionally drop the baseline column
        if drop_baseline and baseline_label in mat.columns:
            mat = mat.drop(columns=[baseline_label])

        # Column-wise minmax normalization
        col_min = mat.min(axis=0)
        col_max = mat.max(axis=0)
        denom = (col_max - col_min).replace(0, np.nan)
        mat_norm = (mat - col_min) / denom * 100.0  # 0–100 per column

        return pivot_data, mat_norm

    # Prepare data for both directories
    pivot1, mat_norm1 = prepare_data(results_df1)
    pivot2, mat_norm2 = prepare_data(results_df2)

    if mat_norm1.empty and mat_norm2.empty:
        print("No valid data to plot")
        return None, None

    # Find common scale for both heatmaps
    all_values = []
    if not mat_norm1.empty:
        all_values.extend(mat_norm1.values.flatten())
    if not mat_norm2.empty:
        all_values.extend(mat_norm2.values.flatten())

    all_values = [v for v in all_values if pd.notna(v) and np.isfinite(v)]
    if not all_values:
        print("No finite values found")
        return None, None

    vmin = min(all_values)
    vmax = max(all_values)
    center_val = 50.0  # For minmax normalized data

    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(
        f"Area Under Curve Comparison: {metric.replace('_', ' ').title()}",
        y=1.02,
        fontsize=16,
    )

    # Plot first heatmap (without colorbar)
    if not mat_norm1.empty:
        ax1 = axes[0]
        # Rank rows by mean
        row_scores1 = mat_norm1.mean(axis=1, skipna=True)
        row_order1 = row_scores1.sort_values(ascending=False, na_position="last").index
        mat_norm1_ordered = mat_norm1.reindex(row_order1)

        sns.heatmap(
            mat_norm1_ordered,
            ax=ax1,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            center=center_val,
            annot=True,
            fmt=".1f",
            cbar=False,  # No colorbar for first plot
        )
        ax1.set_title(title1, fontsize=14, pad=20)
        ax1.set_xlabel("Method", fontsize=12)
        ax1.set_ylabel("Dataset", fontsize=12)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax1.get_yticklabels(), rotation=0)
    else:
        axes[0].text(
            0.5, 0.5, "No data", ha="center", va="center", transform=axes[0].transAxes
        )
        axes[0].set_title(title1, fontsize=14, pad=20)

    # Plot second heatmap (with shared colorbar)
    if not mat_norm2.empty:
        ax2 = axes[1]
        # Rank rows by mean
        row_scores2 = mat_norm2.mean(axis=1, skipna=True)
        row_order2 = row_scores2.sort_values(ascending=False, na_position="last").index
        mat_norm2_ordered = mat_norm2.reindex(row_order2)

        # Create heatmap with colorbar
        sns.heatmap(
            mat_norm2_ordered,
            ax=ax2,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            center=center_val,
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "Column-normalized ΔAUC (0-100)", "shrink": 0.8},
        )
        ax2.set_title(title2, fontsize=14, pad=20)
        ax2.set_xlabel("Method", fontsize=12)
        ax2.set_ylabel("", fontsize=12)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax2.get_yticklabels(), rotation=0)
    else:
        axes[1].text(
            0.5, 0.5, "No data", ha="center", va="center", transform=axes[1].transAxes
        )
        axes[1].set_title(title2, fontsize=14, pad=20)

    plt.tight_layout()
    return fig, axes


def main():
    """Main function to generate the grid plot."""

    parser = argparse.ArgumentParser(description="Generate grid plot for AUC results")
    parser.add_argument(
        "--results-base-path",
        type=str,
        default=None,
        help="Base path for results (single directory mode)",
    )
    parser.add_argument(
        "--results-base-path-1",
        type=str,
        default=None,
        help="Base path for first results directory (dual directory mode)",
    )
    parser.add_argument(
        "--results-base-path-2",
        type=str,
        default=None,
        help="Base path for second results directory (dual directory mode)",
    )
    parser.add_argument(
        "--title-1",
        type=str,
        default="Directory 1",
        help="Title for the first heatmap",
    )
    parser.add_argument(
        "--title-2",
        type=str,
        default="Directory 2",
        help="Title for the second heatmap",
    )
    args = parser.parse_args()

    # Determine mode: single or dual directory
    dual_mode = (
        args.results_base_path_1 is not None and args.results_base_path_2 is not None
    )

    if dual_mode:
        # Dual directory mode
        results_base_path1 = args.results_base_path_1
        results_base_path2 = args.results_base_path_2

        # Check if paths exist
        if not os.path.exists(results_base_path1):
            print(f"Results path 1 not found: {results_base_path1}")
            return
        if not os.path.exists(results_base_path2):
            print(f"Results path 2 not found: {results_base_path2}")
            return

        # Derive output directory from both paths
        path1 = Path(results_base_path1)
        path2 = Path(results_base_path2)
        base_name1 = path1.name or path1.stem
        base_name2 = path2.name or path2.stem
        output_dir = Path("plots") / f"{base_name1}_vs_{base_name2}" / "AUC_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Collecting results from first directory...")
        results_df1 = collect_all_results(results_base_path1)

        print("Collecting results from second directory...")
        results_df2 = collect_all_results(results_base_path2)

        if results_df1.empty and results_df2.empty:
            print("No results found in either directory!")
            return

        print(f"Found {len(results_df1)} result combinations in directory 1")
        print(f"Found {len(results_df2)} result combinations in directory 2")

        # Create plots for different metrics
        metrics_to_plot = ["auc_normalized_pred"]
        metric_names = ["Normalized Predictions AUC"]

        for metric, name in zip(metrics_to_plot, metric_names):
            print(f"\nCreating dual plot for {name}...")

            # Filter out NaN values for this metric
            valid_data1 = (
                results_df1.dropna(subset=[metric])
                if not results_df1.empty
                else pd.DataFrame()
            )
            valid_data2 = (
                results_df2.dropna(subset=[metric])
                if not results_df2.empty
                else pd.DataFrame()
            )

            if valid_data1.empty and valid_data2.empty:
                print(f"No valid data for {metric}")
                continue

            fig, axes = create_dual_heatmap(
                valid_data1,
                valid_data2,
                metric=metric,
                title1=args.title_1,
                title2=args.title_2,
                figsize=(56, 32),
            )

            if fig is not None:
                # Save the plot
                output_path = output_dir / f"auc_grid_dual_{metric}.png"
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Saved plot to: {output_path}")

                # Close the plot to save memory
                plt.close(fig)

        # Print summary statistics
        print("\nSummary Statistics:")
        for metric in metrics_to_plot:
            valid_data1 = (
                results_df1.dropna(subset=[metric])
                if not results_df1.empty
                else pd.DataFrame()
            )
            valid_data2 = (
                results_df2.dropna(subset=[metric])
                if not results_df2.empty
                else pd.DataFrame()
            )

            if not valid_data1.empty:
                print(f"\n{metric} - Directory 1:")
                print(f"  Mean: {valid_data1[metric].mean():.4f}")
                print(f"  Std:  {valid_data1[metric].std():.4f}")
                print(f"  Min:  {valid_data1[metric].min():.4f}")
                print(f"  Max:  {valid_data1[metric].max():.4f}")

            if not valid_data2.empty:
                print(f"\n{metric} - Directory 2:")
                print(f"  Mean: {valid_data2[metric].mean():.4f}")
                print(f"  Std:  {valid_data2[metric].std():.4f}")
                print(f"  Min:  {valid_data2[metric].min():.4f}")
                print(f"  Max:  {valid_data2[metric].max():.4f}")

    else:
        # Single directory mode (original behavior)
        results_base_path = args.results_base_path or "results/auc_result"

        # Check if path exists
        if not os.path.exists(results_base_path):
            print(f"Results path not found: {results_base_path}")
            return

        # Derive output directory from results_base_path
        results_path = Path(results_base_path)
        results_base_name = results_path.name or results_path.stem
        output_dir = Path("plots") / results_base_name / "AUC_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Collecting results...")
        results_df = collect_all_results(results_base_path)

        if results_df.empty:
            print("No results found!")
            return

        print(f"Found {len(results_df)} result combinations")
        print(f"Datasets: {sorted(results_df['dataset'].unique())}")
        print(f"Embeddings: {sorted(results_df['embedding'].unique())}")
        print(f"Regressors: {sorted(results_df['regressor'].unique())}")

        # Create plots for different metrics
        metrics_to_plot = ["auc_normalized_pred"]
        metric_names = ["Normalized Predictions AUC"]

        for metric, name in zip(metrics_to_plot, metric_names):
            print(f"\nCreating plot for {name}...")

            # Filter out NaN values for this metric
            valid_data = results_df.dropna(subset=[metric])

            if valid_data.empty:
                print(f"No valid data for {metric}")
                continue

            fig, _ax = create_grid_plot(valid_data, metric=metric, figsize=(28, 32))

            if fig is not None:
                # Save the plot using the output_dir derived from result_base_path
                output_path = output_dir / f"auc_grid_avg_col_{metric}.png"
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Saved plot to: {output_path}")

                # Close the plot to save memory
                plt.close(fig)

        # Print summary statistics
        print("\nSummary Statistics:")
        for metric in metrics_to_plot:
            valid_data = results_df.dropna(subset=[metric])
            if not valid_data.empty:
                print(f"\n{metric}:")
                print(f"  Mean: {valid_data[metric].mean():.4f}")
                print(f"  Std:  {valid_data[metric].std():.4f}")
                print(f"  Min:  {valid_data[metric].min():.4f}")
                print(f"  Max:  {valid_data[metric].max():.4f}")


if __name__ == "__main__":
    main()
