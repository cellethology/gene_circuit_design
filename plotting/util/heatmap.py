from pathlib import Path
import pandas as pd
from typing import Literal
import os
import numpy as np
from scipy import integrate


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

    per_seed = (
        data.drop(columns=["seed"])
        .groupby(data["seed"], group_keys=False)
        .apply(lambda df: _auc_one_seed(df, metric_column, normalize=normalize))
    )
    mean_auc = per_seed.mean(skipna=True)
    return mean_auc


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


def collect_all_results(results_base_path) -> pd.DataFrame:
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
