#!/usr/bin/env python3
"""
Grid plot for area under curve metrics across datasets, embedding methods, and regressors.

This script creates a heatmap where:
- Y-axis: datasets (Feng_2023, Angenent-Mari_2020, alcantar_2025)
- X-axis: embedding method + regressor combinations (e.g., onehot_KNN, sei_LinearRegression)
- Color: area under the curve of cumulative metrics

Example Usage:
    python plotting/area_under_curve_grid_plot.py
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from scipy import integrate
from new_file_plot_gen import create_grid_plot_avg_col as create_grid_plot

matplotlib.use("Agg")  # Use non-interactive backend


def extract_info_from_path(file_path):
    """Extract dataset, embedding, and regressor info from file path."""
    path_parts = Path(file_path).parts

    # Find dataset name
    dataset = None
    for part in path_parts:
        if any(
            d in part
            for d in ["AD"]
            # TODO: need to revisit the plotting logics here
            # for d in ["Feng_2023", "angenent-Mari_2020", "alcantar_2025", "166k_2024", ""]
        ):
            dataset = part
            break

    # Find embedding method
    embedding = None
    for part in path_parts:
        if any(e in part for e in ["onehotPca", "onehotRaw", "evo", "sei"]):
            if "onehotPca" in part:
                embedding = "onehotPca"
            elif "onehotRaw" in part:
                embedding = "onehotRaw"
            elif "evo" in part:
                embedding = "evo"
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
        lambda df: _auc_one_seed(df, metric_column, normalize=normalize)
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


def main():
    """Main function to generate the grid plot."""

    parser = argparse.ArgumentParser(description="Generate grid plot for AUC results")
    parser.add_argument(
        "--results-base-path",
        type=str,
        default="results/auc_result",
        help="Base path for results",
    )
    args = parser.parse_args()
    results_base_path = args.results_base_path

    # results_base_path = "results/auc_result"

    # Check if path exists
    if not os.path.exists(results_base_path):
        print(f"Results path not found: {results_base_path}")
        return

    print("Collecting results...")
    results_df = collect_all_results(results_base_path)

    # results_df.to_csv("hello_auc_result.csv")
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

        fig, _ax = create_grid_plot(valid_data, metric=metric, figsize=(14, 12))

        if fig is not None:
            # Save the plot
            output_path = f"plots/AUC_results/auc_grid_avg_col_{metric}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
