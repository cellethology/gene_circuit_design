#!/usr/bin/env python3
"""
Grid plot for area under curve metrics across datasets, embedding methods, and regressors.

This script creates a heatmap where:
- Y-axis: datasets (Feng_2023, Angenent-Mari_2020, alcantar_2025)
- X-axis: embedding method + regressor combinations (e.g., onehot_KNN, sei_LinearRegression)
- Color: area under the curve of cumulative metrics
"""
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate

matplotlib.use("Agg")  # Use non-interactive backend


def extract_info_from_path(file_path):
    """Extract dataset, embedding, and regressor info from file path."""
    path_parts = Path(file_path).parts

    # Find dataset name
    dataset = None
    for part in path_parts:
        if any(d in part for d in ["Feng_2023", "Angenent-Mari_2020", "alcantar_2025"]):
            dataset = part
            break

    # Find embedding method
    embedding = None
    for part in path_parts:
        if any(e in part for e in ["onehot", "evo", "sei"]):
            if "onehot" in part:
                embedding = "onehot"
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


def calculate_auc_from_cumulative(
    data, metric_column="top_10_ratio_intersected_indices_cumulative"
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

    return auc


def collect_all_results(results_base_path):
    """Collect all AUC results from the results directory."""
    results_list = []

    # Walk through all directories
    for root, _dirs, files in os.walk(results_base_path):
        for file in files:
            if file == "combined_all_custom_metrics.csv":
                file_path = os.path.join(root, file)

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

                        # Calculate AUC for different metrics
                        auc_top10 = calculate_auc_from_cumulative(
                            regressor_data,
                            "top_10_ratio_intersected_indices_cumulative",
                        )
                        auc_normalized = calculate_auc_from_cumulative(
                            regressor_data,
                            "normalized_predictions_predictions_values_cumulative",
                        )

                        # Store results
                        results_list.append(
                            {
                                "dataset": dataset,
                                "embedding": embedding,
                                "regressor": regressor,
                                "auc_top10_ratio": auc_top10,
                                "auc_normalized_pred": auc_normalized,
                                "file_path": file_path,
                            }
                        )

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

    return pd.DataFrame(results_list)


def create_grid_plot(results_df, metric="auc_top10_ratio", figsize=(12, 6)):
    """Create a grid plot of AUC values."""
    if results_df.empty:
        print("No data found to plot")
        return None, None

    # Create combination column for x-axis
    results_df = results_df.copy()
    results_df["embedding_regressor"] = (
        results_df["embedding"] + "_" + results_df["regressor"]
    )

    # Create pivot table for heatmap
    pivot_data = results_df.pivot_table(
        index="dataset",
        columns="embedding_regressor",
        values=metric,
        aggfunc="mean",  # Average across seeds if multiple
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": f'{metric.replace("_", " ").title()}'},
        ax=ax,
    )

    # Customize the plot
    ax.set_title(
        f'Area Under Curve: {metric.replace("_", " ").title()}\nAcross Datasets and Methods'
    )
    ax.set_xlabel("Embedding Method + Regressor")
    ax.set_ylabel("Dataset")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    return fig, ax


def main():
    """Main function to generate the grid plot."""
    # Define the base path for results
    results_base_path = (
        "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/results/example"
    )

    # Check if path exists
    if not os.path.exists(results_base_path):
        print(f"Results path not found: {results_base_path}")
        return

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
    metrics_to_plot = ["auc_top10_ratio", "auc_normalized_pred"]
    metric_names = ["Top 10% Intersection Ratio AUC", "Normalized Predictions AUC"]

    for metric, name in zip(metrics_to_plot, metric_names):
        print(f"\nCreating plot for {name}...")

        # Filter out NaN values for this metric
        valid_data = results_df.dropna(subset=[metric])

        if valid_data.empty:
            print(f"No valid data for {metric}")
            continue

        fig, _ax = create_grid_plot(valid_data, metric=metric, figsize=(14, 8))

        if fig is not None:
            # Save the plot
            output_path = f"/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/plotting/auc_grid_{metric}.png"
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
