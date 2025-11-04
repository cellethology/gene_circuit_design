from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_grid_plot_no_norm(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    figsize: tuple[int, int] = (12, 6),
    baseline_label: str = "random",
    drop_baseline: bool = True,
):
    """Create a grid plot of (method - baseline) AUC values per dataset.

    Args:
        results_df: Long-form results with columns ['dataset','embedding','regressor','strategy','method_label', metric].
        metric: Metric column to visualize (defaults to 'auc_normalized_pred').
        figsize: Figure size.
        baseline_label: Column name in the pivot used as baseline (default: 'random').
        drop_baseline: If True, remove the baseline column from the heatmap; if False, keep it (will be zeros).

    Returns:
        (fig, ax): Matplotlib Figure and Axes or (None, None) if no data.
    """
    if results_df.empty:
        print("No data found to plot")
        return None, None

    # Ensure method_label exists and aggregate "random" across methods per dataset
    results_df = results_df.copy()
    if "method_label" in results_df.columns:
        random_results = (
            results_df[results_df["strategy"] == "random"]
            .groupby("dataset", as_index=False)
            .agg({metric: "median"})
        )
        random_results["method_label"] = baseline_label

        non_random_results = results_df[results_df["strategy"] != "random"]
        plot_data = pd.concat([non_random_results, random_results], ignore_index=True)
    else:
        results_df["method_label"] = (
            results_df["embedding"] + "_" + results_df["regressor"]
        )
        plot_data = results_df

    # Pivot to dataset x method_label
    pivot_data = plot_data.pivot_table(
        index="dataset", columns="method_label", values=metric, aggfunc="median"
    )

    if baseline_label not in pivot_data.columns:
        print(f'Baseline column "{baseline_label}" not found. Showing raw values.')
        # Fallback to raw heatmap
        # Order columns: others sorted, baseline at end if present
        other_cols = sorted([c for c in pivot_data.columns if c != baseline_label])
        if baseline_label in pivot_data.columns:
            other_cols += [baseline_label]
        pivot_ordered = pivot_data[other_cols] if other_cols else pivot_data

        # Check if we have enough data for clustering (need at least 2 rows and 2 columns)
        can_cluster_rows = len(pivot_ordered) >= 2
        can_cluster_cols = len(pivot_ordered.columns) >= 2

        g = sns.clustermap(
            pivot_ordered,
            cmap="viridis",
            annot=True,
            fmt=".3f",
            method="average",  # clustering linkage method
            metric="euclidean",  # distance metric
            row_cluster=can_cluster_rows,  # Disable row clustering if < 2 rows
            col_cluster=can_cluster_cols,  # Disable col clustering if < 2 cols
            cbar_kws={"label": metric.replace("_", " ").title()},
            figsize=figsize,
        )

        # Adjust layout and title
        g.fig.suptitle(f"Area Under Curve: {metric.replace('_', ' ').title()}", y=1.02)
        g.ax_heatmap.set_xlabel("Method")
        g.ax_heatmap.set_ylabel("Dataset")
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

        # Explicitly set all y-tick labels to ensure every row label is shown
        if not pivot_ordered.empty:
            num_rows = len(pivot_ordered)

            # Get the row indices in the order shown after clustering
            try:
                row_indices = g.dendrogram_row.reordered_ind
                if row_indices is not None:
                    # Use the clustered row order
                    reordered_labels = [pivot_ordered.index[i] for i in row_indices]
                else:
                    # Fallback to original order if no clustering
                    reordered_labels = pivot_ordered.index.tolist()
            except AttributeError:
                # No dendrogram available, use original order
                reordered_labels = pivot_ordered.index.tolist()

            # Set ticks at every row position
            yticks = np.arange(num_rows) + 0.5  # Center of each cell
            g.ax_heatmap.set_yticks(yticks)
            g.ax_heatmap.set_yticklabels(reordered_labels, rotation=0)

            # Adjust font size based on number of rows
            if num_rows > 15:
                plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)
            elif num_rows > 10:
                plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=9)

        plt.show()
        return g.fig, g.ax_heatmap

    # Compute (% difference vs baseline) per dataset
    baseline_series = pivot_data[baseline_label]
    delta = (
        pivot_data.subtract(baseline_series, axis=0).div(baseline_series, axis=0)
    ) * 100

    # Reorder columns: alphabetical methods, baseline last (or drop)
    other_cols = sorted([c for c in delta.columns if c != baseline_label])
    if not drop_baseline:
        col_order = other_cols + [baseline_label]
        # Baseline deltas are exactly 0 by construction
        delta_subset = delta[col_order]
    else:
        col_order = other_cols
        delta_subset = delta[col_order]

    # Check if we have enough data for clustering after filtering
    can_cluster_rows_delta = len(delta_subset) >= 2
    can_cluster_cols_delta = len(delta_subset.columns) >= 2

    # Plot clustered heatmap centered at zero
    try:
        g = sns.clustermap(
            delta_subset,
            cmap="coolwarm",
            center=0.0,
            annot=True,  # requires seaborn >= 0.13
            fmt=".1f",
            method="average",  # linkage: "average", "complete", "single", "ward"
            metric="euclidean",  # distance: "euclidean", "correlation", etc.
            row_cluster=can_cluster_rows_delta,  # Disable row clustering if < 2 rows
            col_cluster=can_cluster_cols_delta,  # Disable col clustering if < 2 cols
            figsize=figsize,
            cbar_kws={"label": "Δ AUC vs random | ((method - random) / random) * 100"},
        )
    except TypeError:
        # Fallback for seaborn < 0.13 (no native annot)
        g = sns.clustermap(
            delta_subset,
            cmap="coolwarm",
            center=0.0,
            method="average",
            metric="euclidean",
            row_cluster=can_cluster_rows_delta,  # Disable row clustering if < 2 rows
            col_cluster=can_cluster_cols_delta,  # Disable col clustering if < 2 cols
            figsize=figsize,
            cbar_kws={"label": "Δ AUC vs random | ((method - random) / random) * 100"},
        )
        # Manual annotations
        ax = g.ax_heatmap
        row_labels = [t.get_text() for t in ax.get_yticklabels()]
        col_labels = [t.get_text() for t in ax.get_xticklabels()]
        shown = delta_subset.loc[row_labels, col_labels]
        for i, r in enumerate(shown.index):
            for j, c in enumerate(shown.columns):
                v = shown.loc[r, c]
                if pd.notna(v):
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{v:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    # Titles and labels
    g.fig.suptitle(
        "AUC percentage change vs Random\n(method - random) across Datasets and Methods",
        y=1.05,
    )
    g.ax_heatmap.set_xlabel(
        "Method (Embedding_Regressor{} )".format("" if drop_baseline else " or Random")
    )
    g.ax_heatmap.set_ylabel("Dataset")
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    # Explicitly set all y-tick labels to ensure every row label is shown
    # Get the actual row order after clustering (from dendrogram)
    if not delta.empty:
        num_rows = len(delta)

        # Get the row indices in the order shown after clustering
        try:
            row_indices = g.dendrogram_row.reordered_ind
            if row_indices is not None:
                # Use the clustered row order
                reordered_labels = [delta.index[i] for i in row_indices]
            else:
                # Fallback to original order if no clustering
                reordered_labels = delta.index.tolist()
        except AttributeError:
            # No dendrogram available, use original order
            reordered_labels = delta.index.tolist()

        # Set ticks at every row position
        yticks = np.arange(num_rows) + 0.5  # Center of each cell
        g.ax_heatmap.set_yticks(yticks)
        g.ax_heatmap.set_yticklabels(reordered_labels, rotation=0)

        # Adjust font size based on number of rows
        if num_rows > 15:
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)
        elif num_rows > 10:
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=9)

    plt.tight_layout()

    return g.fig, g.ax_heatmap


def create_grid_plot_avg_row(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    figsize: tuple[int, int] = (12, 6),
    baseline_label: str = "random",
    drop_baseline: bool = True,
    *,
    row_norm: Literal["minmax", "zscore", "none"] = "minmax",
    col_rank_stat: Literal["mean", "median"] = "mean",
):
    """Create a heatmap of ΔAUC vs baseline, with per-row normalization and columns ranked
    by their aggregate (mean/median) after normalization.

    Pipeline:
      1) Build pivot: dataset × method_label with metric medians.
      2) Δ = method - baseline, then (optionally) normalize per row.
      3) Rank columns by their average/median across datasets.
      4) Plot heatmap.

    Args:
        results_df: Long-form DataFrame with columns:
            ['dataset','embedding','regressor','strategy','method_label', metric].
        metric: Metric column to visualize.
        figsize: Matplotlib figure size.
        baseline_label: Column used as baseline (default 'random').
        drop_baseline: If True, drop the baseline column from the heatmap.
        row_norm: Row-wise normalization: 'minmax' → [0, 100], 'zscore' → mean 0, std 1, 'none' → no row norm.
        col_rank_stat: Statistic to rank columns ('mean' or 'median') AFTER row normalization.

    Returns:
        (fig, ax): Figure and Axes, or (None, None) if no data.
    """
    if results_df.empty:
        print("No data found to plot")
        return None, None

    # --- Ensure method_label and aggregate the random baseline across methods per dataset ---
    df = results_df.copy()
    if "method_label" not in df.columns:
        df["method_label"] = (
            df["embedding"].astype(str) + "_" + df["regressor"].astype(str)
        )

    random_results = (
        df[df["strategy"] == "random"]
        .groupby("dataset", as_index=False)
        .agg({metric: "median"})
    )
    random_results["method_label"] = baseline_label
    non_random_results = df[df["strategy"] != "random"]
    plot_data = pd.concat([non_random_results, random_results], ignore_index=True)

    # --- Pivot: dataset × method_label ---
    pivot_data = plot_data.pivot_table(
        index="dataset", columns="method_label", values=metric, aggfunc="median"
    )

    if pivot_data.empty:
        print("No pivoted data to plot")
        return None, None

    if baseline_label not in pivot_data.columns:
        # Fallback: show raw values (no Δ), ranked columns by chosen stat
        col_stat = getattr(pivot_data, col_rank_stat)(axis=0)
        col_order = col_stat.sort_values(ascending=False, na_position="last").index
        pivot_ordered = pivot_data[col_order]

        fig, ax = plt.subplots(figsize=figsize)
        g = sns.clustermap(
            pivot_ordered,
            cmap="viridis",
            annot=True,
            fmt=".3f",
            method="average",  # clustering linkage method
            metric="euclidean",  # distance metric
            cbar_kws={"label": metric.replace("_", " ").title()},
            figsize=figsize,
        )

        # Adjust layout and title
        g.fig.suptitle(f"Area Under Curve: {metric.replace('_', ' ').title()}", y=1.02)
        g.ax_heatmap.set_xlabel("Method")
        g.ax_heatmap.set_ylabel("Dataset")
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

        # Explicitly set all y-tick labels to ensure every row label is shown
        if not pivot_ordered.empty:
            num_rows = len(pivot_ordered)

            # Get the row indices in the order shown after clustering
            try:
                row_indices = g.dendrogram_row.reordered_ind
                if row_indices is not None:
                    # Use the clustered row order
                    reordered_labels = [pivot_ordered.index[i] for i in row_indices]
                else:
                    # Fallback to original order if no clustering
                    reordered_labels = pivot_ordered.index.tolist()
            except AttributeError:
                # No dendrogram available, use original order
                reordered_labels = pivot_ordered.index.tolist()

            # Set ticks at every row position
            yticks = np.arange(num_rows) + 0.5  # Center of each cell
            g.ax_heatmap.set_yticks(yticks)
            g.ax_heatmap.set_yticklabels(reordered_labels, rotation=0)

            # Adjust font size based on number of rows
            if num_rows > 15:
                plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)
            elif num_rows > 10:
                plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=9)

        plt.show()
        return g.fig, g.ax_heatmap

    # --- Δ vs baseline (no percentage scaling here; row normalization happens next) ---
    baseline_series = pivot_data[baseline_label]
    delta = pivot_data.subtract(baseline_series, axis=0)

    # Optionally drop baseline column from visualization
    method_cols = [
        c for c in delta.columns if (c != baseline_label) or (not drop_baseline)
    ]
    delta = delta[method_cols]

    # --- Row-wise normalization ---
    if row_norm == "minmax":
        row_min = delta.min(axis=1)
        row_max = delta.max(axis=1)
        denom = (row_max - row_min).replace(0, np.nan)
        delta_norm = (delta.sub(row_min, axis=0)).div(
            denom, axis=0
        ) * 100.0  # 0–100 per row
        heat_center = 50.0
        annot_fmt = ".1f"
        cbar_label = "Row-normalized ΔAUC (0–100)"
    elif row_norm == "zscore":
        row_mean = delta.mean(axis=1)
        row_std = delta.std(axis=1, ddof=0).replace(0, np.nan)
        delta_norm = delta.sub(row_mean, axis=0).div(row_std, axis=0)
        heat_center = 0.0
        annot_fmt = ".2f"
        cbar_label = "Row z-score of ΔAUC"
    elif row_norm == "none":
        delta_norm = delta.copy()
        heat_center = 0.0
        annot_fmt = ".3f"
        cbar_label = "ΔAUC vs baseline"
    else:
        raise ValueError(f"Unsupported row_norm: {row_norm!r}")

    # --- Rank columns by their aggregate (after row normalization) ---
    col_stat_values = getattr(delta_norm, col_rank_stat)(axis=0)
    col_order = col_stat_values.sort_values(
        ascending=False, na_position="last"
    ).index.tolist()
    delta_norm = delta_norm[col_order]

    # --- Plot ---
    # fig, ax = plt.subplots(figsize=figsize)
    g = sns.clustermap(
        delta_norm,
        cmap="coolwarm",
        annot=True,
        fmt=annot_fmt,
        center=heat_center,
        method="average",  # or "ward", "complete", etc.
        metric="euclidean",
        figsize=figsize,
        cbar_kws={"label": cbar_label},
    )

    g.fig.suptitle(
        "Clustered ΔAUC vs Baseline\n(rows & columns clustered by similarity)",
        y=1.05,
    )
    g.ax_heatmap.set_xlabel("Method")
    g.ax_heatmap.set_ylabel("Dataset")
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    # Explicitly set all y-tick labels to ensure every row label is shown
    if not delta_norm.empty:
        num_rows = len(delta_norm)

        # Get the row indices in the order shown after clustering
        try:
            row_indices = g.dendrogram_row.reordered_ind
            if row_indices is not None:
                # Use the clustered row order
                reordered_labels = [delta_norm.index[i] for i in row_indices]
            else:
                # Fallback to original order if no clustering
                reordered_labels = delta_norm.index.tolist()
        except AttributeError:
            # No dendrogram available, use original order
            reordered_labels = delta_norm.index.tolist()

        # Set ticks at every row position
        yticks = np.arange(num_rows) + 0.5  # Center of each cell
        g.ax_heatmap.set_yticks(yticks)
        g.ax_heatmap.set_yticklabels(reordered_labels, rotation=0)

        # Adjust font size based on number of rows
        if num_rows > 15:
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)
        elif num_rows > 10:
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=9)

    plt.show()
    return g.fig, g.ax_heatmap


def create_grid_plot_avg_col(
    results_df: pd.DataFrame,
    metric: str = "auc_normalized_pred",
    figsize: tuple[int, int] = (12, 6),
    baseline_label: str = "random",
    drop_baseline: bool = True,
    *,
    col_norm: Literal["minmax", "zscore", "none"] = "minmax",
    row_rank_stat: Literal["mean", "median"] = "mean",
):
    """Clustered heatmap of ΔAUC vs baseline with:
       • COLUMN-wise normalization (per method), and
       • ROW ordering by aggregate (mean/median) after normalization.

    Pipeline:
      1) Pivot: dataset × method_label (median of `metric`).
      2) Δ = method - baseline.
      3) Normalize per COLUMN (minmax → [0,100], zscore → mean 0/std 1, none).
      4) Rank ROWS by their aggregate across columns (mean/median).
      5) Plot with seaborn.clustermap (columns clustered; rows fixed to the ranking).

    Args:
        results_df: Long-form DataFrame with columns
          ['dataset','embedding','regressor','strategy','method_label', metric].
        metric: Metric column to visualize.
        figsize: Figure size.
        baseline_label: Column used as baseline (default 'random').
        drop_baseline: If True, drop the baseline column from the plot.
        col_norm: Column-wise normalization: 'minmax' | 'zscore' | 'none'.
        row_rank_stat: Statistic to rank rows after normalization: 'mean' | 'median'.

    Returns:
        (fig, ax): Matplotlib Figure and heatmap Axes, or (None, None) if no data.
    """
    if results_df.empty:
        print("No data found to plot")
        return None, None

    df = results_df.copy()
    if "method_label" not in df.columns:
        df["method_label"] = (
            df["embedding"].astype(str) + "_" + df["regressor"].astype(str)
        )

    # Build pivot with one aggregated baseline per dataset
    random_results = (
        df[df["strategy"] == "random"]
        .groupby("dataset", as_index=False)
        .agg({metric: "median"})
    )
    random_results["method_label"] = baseline_label
    non_random_results = df[df["strategy"] != "random"]
    plot_data = pd.concat([non_random_results, random_results], ignore_index=True)

    pivot_data = plot_data.pivot_table(
        index="dataset", columns="method_label", values=metric, aggfunc="median"
    )
    if pivot_data.empty:
        print("No pivoted data to plot")
        return None, None

    # If no baseline: operate on raw values (still do col-norm + row ranking as requested)
    if baseline_label not in pivot_data.columns:
        mat = pivot_data.copy()
    else:
        # Δ vs baseline
        baseline_series = pivot_data[baseline_label]
        mat = pivot_data.subtract(baseline_series, axis=0)

    # Optionally drop the baseline column from visualization
    if drop_baseline and baseline_label in mat.columns:
        mat = mat.drop(columns=[baseline_label])

    # ---------- COLUMN-wise normalization ----------
    if col_norm == "minmax":
        col_min = mat.min(axis=0)
        col_max = mat.max(axis=0)
        denom = (col_max - col_min).replace(0, np.nan)
        mat_norm = (mat - col_min) / denom * 100.0  # 0–100 per column
        heat_center = 50.0
        cbar_label = "Column-normalized ΔAUC (0–100)"
        annot_fmt = ".1f"
    elif col_norm == "zscore":
        col_mean = mat.mean(axis=0)
        col_std = mat.std(axis=0, ddof=0).replace(0, np.nan)
        mat_norm = (mat - col_mean) / col_std
        heat_center = 0.0
        cbar_label = "Column z-score of ΔAUC"
        annot_fmt = ".2f"
    elif col_norm == "none":
        mat_norm = mat.copy()
        heat_center = 0.0
        cbar_label = "ΔAUC vs baseline"
        annot_fmt = ".3f"
    else:
        raise ValueError(f"Unsupported col_norm: {col_norm!r}")

    # ---------- Rank ROWS by aggregate after normalization ----------
    if row_rank_stat == "mean":
        row_scores = mat_norm.mean(axis=1, skipna=True)
    elif row_rank_stat == "median":
        row_scores = mat_norm.median(axis=1, skipna=True)
    else:
        raise ValueError(f"Unsupported row_rank_stat: {row_rank_stat!r}")

    row_order = row_scores.sort_values(ascending=False, na_position="last").index
    mat_norm = mat_norm.reindex(row_order)

    # ---------- Clustered heatmap ----------
    # We preserve row ranking (row_cluster=False), but allow column clustering.
    # ----- choose explicit color limits so both sides render -----
    if col_norm == "minmax":
        # data are in [0, 100]; center at 50 -> use full range
        vmin, vmax = 0.0, 100.0
        center_val = 50.0
    elif col_norm == "zscore":
        # symmetric around 0 using the global max abs value
        m = np.nanmax(np.abs(mat_norm.values))
        if not np.isfinite(m) or m == 0:
            m = 1.0
        vmin, vmax = -m, m
        center_val = 0.0
    else:  # "none"
        # symmetric around 0 using the global max abs value
        m = np.nanmax(np.abs(mat_norm.values))
        if not np.isfinite(m) or m == 0:
            m = 1.0
        vmin, vmax = -m, m
        center_val = 0.0

    g = sns.clustermap(
        mat_norm,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        center=center_val,  # <- explicit limits
        annot=True,
        fmt=annot_fmt,
        figsize=figsize,
        method="average",
        metric="euclidean",
        row_cluster=True,
        col_cluster=True,
        cbar_kws={"label": cbar_label},
    )

    title_prefix = (
        "ΔAUC vs Baseline" if baseline_label in pivot_data.columns else "Metric (raw)"
    )
    g.fig.suptitle(
        f"{title_prefix} — column-normalized; rows ranked by {row_rank_stat}",
        y=1.05,
    )
    g.ax_heatmap.set_xlabel("Method")
    g.ax_heatmap.set_ylabel("Dataset")
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    # Explicitly set all y-tick labels to ensure every row label is shown
    if not mat_norm.empty:
        num_rows = len(mat_norm)

        # Get the row indices in the order shown after clustering
        try:
            row_indices = g.dendrogram_row.reordered_ind
            if row_indices is not None:
                # Use the clustered row order
                reordered_labels = [mat_norm.index[i] for i in row_indices]
            else:
                # Fallback to original order if no clustering
                reordered_labels = mat_norm.index.tolist()
        except AttributeError:
            # No dendrogram available, use original order
            reordered_labels = mat_norm.index.tolist()

        # Set ticks at every row position
        yticks = np.arange(num_rows) + 0.5  # Center of each cell
        g.ax_heatmap.set_yticks(yticks)
        g.ax_heatmap.set_yticklabels(reordered_labels, rotation=0)

        # Adjust font size based on number of rows
        if num_rows > 15:
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)
        elif num_rows > 10:
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=9)

    plt.tight_layout()
    return g.fig, g.ax_heatmap
