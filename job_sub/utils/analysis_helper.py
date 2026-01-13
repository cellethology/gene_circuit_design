import numpy as np
from scipy import stats


def mannwhitney_p(values, baseline_values, alternative="two-sided"):
    if len(values) == 0 or len(baseline_values) == 0:
        return np.nan
    try:
        return float(
            stats.mannwhitneyu(values, baseline_values, alternative=alternative).pvalue
        )
    except ValueError:
        return np.nan


def fold_change(strategy_median, baseline_median):
    return strategy_median / baseline_median


# # Heatmaps of fold-change vs baseline (clustered, log2-scaled)
# avg_top_pivot = baseline_comparison.pivot(
#     index="dataset_name", columns="strategy", values="avg_top_fold_change"
# )

# overall_pivot = baseline_comparison.pivot(
#     index="dataset_name", columns="strategy", values="overall_fold_change"
# )

# avg_top_pivot = avg_top_pivot.sort_index()
# overall_pivot = overall_pivot.reindex(avg_top_pivot.index)

# avg_top_log2 = np.log2(avg_top_pivot)
# overall_log2 = np.log2(overall_pivot)

# P_VALUE_CUTOFF = 1e-4
# avg_top_pvals = baseline_comparison.pivot(
#     index="dataset_name", columns="strategy", values="avg_top_p_value"
# )
# overall_pvals = baseline_comparison.pivot(
#     index="dataset_name", columns="strategy", values="overall_p_value"
# )
# avg_top_pvals = avg_top_pvals.reindex(avg_top_pivot.index).reindex(
#     avg_top_pivot.columns, axis=1
# )
# overall_pvals = overall_pvals.reindex(overall_pivot.index).reindex(
#     overall_pivot.columns, axis=1
# )

# avg_top_mask = (avg_top_pvals > P_VALUE_CUTOFF) | ~np.isfinite(avg_top_log2)
# overall_mask = (overall_pvals > P_VALUE_CUTOFF) | ~np.isfinite(overall_log2)

# fold_cmap = sns.color_palette("vlag", as_cmap=True)
# fold_cmap.set_bad("black")

# output_dir = Path("../../results")
# output_dir.mkdir(parents=True, exist_ok=True)

# # Typography tuning for readability.
# TITLE_FONTSIZE = 28
# LABEL_FONTSIZE = 24
# TICK_FONTSIZE = 16


# def compute_figsize(pivot, col_scale=0.6, row_scale=0.6, base_w=8, base_h=6):
#     width = max(12, col_scale * pivot.shape[1] + base_w)
#     height = max(8, row_scale * pivot.shape[0] + base_h)
#     return (width, height)


# def format_clustermap(grid, title, pad=0.01, width=0.015):
#     grid.ax_row_dendrogram.set_visible(False)
#     grid.ax_col_dendrogram.set_visible(False)
#     grid.ax_heatmap.set_title(title, fontsize=TITLE_FONTSIZE)
#     grid.ax_heatmap.yaxis.tick_left()
#     grid.ax_heatmap.yaxis.set_label_position("left")
#     grid.ax_heatmap.tick_params(axis="y", pad=2, labelsize=TICK_FONTSIZE)
#     grid.ax_heatmap.tick_params(axis="x", pad=2, labelsize=TICK_FONTSIZE)
#     grid.cax.yaxis.label.set_size(LABEL_FONTSIZE)
#     grid.cax.tick_params(labelsize=TICK_FONTSIZE)
#     heatmap_pos = grid.ax_heatmap.get_position()
#     grid.cax.set_position(
#         [heatmap_pos.x1 + pad, heatmap_pos.y0, width, heatmap_pos.height]
#     )


# data = avg_top_log2.replace([np.inf, -np.inf], np.nan).mask(avg_top_mask)
# avg_top_cluster = sns.clustermap(
#     data.fillna(0),
#     cmap=fold_cmap,
#     center=0,
#     row_cluster=True,
#     col_cluster=True,
#     cbar_kws={"label": "AUC Log2 Fold-Change"},
#     figsize=compute_figsize(avg_top_log2),
#     mask=data.isna(),
# )
# format_clustermap(avg_top_cluster, "avg top Log2 Fold-Change vs Baseline")
# # auc_cluster.savefig(
# #     output_dir / "avg_top_log2_fold_change_heatmap.pdf", dpi=300, bbox_inches="tight"
# # )
# plt.show()

# overall_cluster = sns.clustermap(
#     overall_log2,
#     cmap=fold_cmap,
#     center=0,
#     row_cluster=True,
#     col_cluster=True,
#     cbar_kws={"label": "Overall Log2 Fold-Change"},
#     figsize=compute_figsize(overall_log2),
#     mask=overall_mask,
# )
# format_clustermap(overall_cluster, "Overall Log2 Fold-Change vs Baseline")
# # overall_cluster.savefig(
# #     output_dir / "overall_true_log2_fold_change_heatmap.pdf",
# #     dpi=300,
# #     bbox_inches="tight",
# # )
# plt.show()

# spearman_df = overall_log2.T.corr(method="spearman")

# spearman_mask = np.eye(spearman_df.shape[0], dtype=bool)

# spearman_cluster = sns.clustermap(
#     spearman_df,
#     cmap="RdBu_r",
#     row_cluster=True,
#     col_cluster=True,
#     cbar_kws={"label": "Spearman Correlation (Overall Log2)"},
#     figsize=compute_figsize(spearman_df),
#     mask=spearman_mask,
# )
# spearman_cluster.ax_heatmap.set_title(
#     "Spearman Correlation Across Datasets (Overall Log2)",
#     fontsize=TITLE_FONTSIZE,
# )
# spearman_cluster.ax_heatmap.tick_params(axis="y", pad=2, labelsize=TICK_FONTSIZE)
# spearman_cluster.ax_heatmap.tick_params(axis="x", pad=2, labelsize=TICK_FONTSIZE)
# plt.setp(spearman_cluster.ax_heatmap.get_xticklabels(), rotation=90)
# plt.setp(spearman_cluster.ax_heatmap.get_yticklabels(), rotation=0)
# spearman_cluster.cax.yaxis.label.set_size(LABEL_FONTSIZE)
# spearman_cluster.cax.tick_params(labelsize=TICK_FONTSIZE)
# spearman_cluster.ax_col_dendrogram.set_visible(True)
# spearman_cluster.ax_row_dendrogram.set_visible(True)
# # spearman_cluster.savefig(
# #     output_dir / "overall_log2_spearman_dataset_heatmap.pdf",
# #     dpi=300,
# #     bbox_inches="tight",
# # )
# plt.show()
