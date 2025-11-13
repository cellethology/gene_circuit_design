#!/usr/bin/env python3
"""
Alpha-specific visualization script for active learning results.

For a given results folder that contains:
    - combined_all_results.csv
    - combined_all_custom_metrics.csv  (optional)

and where these CSVs包含一列 `alpha`，

本脚本会对每一个 alpha：
    - 把该 alpha 下所有 seeds 的结果筛选出来
    - 临时写成一个“只含该 alpha”的 combined_all_results/combined_all_custom_metrics
    - 调用 utils.plotting 中原有的绘图函数
    - 在输出目录中生成类似原先的 PDF 图，但文件名中带上 alpha
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import pandas as pd

from utils.plotting import (
    plot_active_learning_metrics,
    plot_regressor_comparison,
    plot_top10_ratio_metrics,
    plot_value_metrics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_alpha_values(df: pd.DataFrame) -> List[float]:
    """从 DataFrame 中提取所有唯一的 alpha 值（排除 NaN）"""
    if "alpha" not in df.columns:
        raise ValueError("DataFrame 中不存在 'alpha' 列，无法按 alpha 可视化。")
    alphas = sorted(df["alpha"].dropna().unique().tolist())
    return alphas


def visualize_folder_by_alpha(
    folder_path: Path,
    output_dir: Path,
    show_plots: bool = False,
    plot_type: str = "mean",
    only_alphas: Optional[List[float]] = None,
) -> None:
    """
    对单个 results 文件夹按 alpha 生成图。

    Args:
        folder_path: 结果文件夹路径（例如 results/Feng_2023/evo_pca_512_combined_std_exp_alpha_range）
        output_dir: 输出 plots 的根目录，如 plots/alpha_results
        show_plots: 是否在屏幕上显示图
        plot_type: 传给 plot_regressor_comparison 的 plot_type（mean/median/sem）
        only_alphas: 如果不为 None，则只处理这个列表中的 alpha
    """
    folder_path = folder_path.resolve()
    logger.info(f"Processing folder (by alpha): {folder_path}")

    standard_file = folder_path / "combined_all_results.csv"
    custom_file = folder_path / "combined_all_custom_metrics.csv"

    if not standard_file.exists():
        logger.warning(f"  ✗ {standard_file} 不存在，跳过该 folder。")
        return

    df_std = pd.read_csv(standard_file)
    if df_std.empty:
        logger.warning("  ✗ combined_all_results.csv 为空，跳过。")
        return

    if "alpha" not in df_std.columns:
        logger.warning("  ✗ combined_all_results.csv 中没有 'alpha' 列，无法按 alpha 可视化。")
        return

    # 标准 metrics 中的 alpha 列
    all_alphas = get_alpha_values(df_std)

    if only_alphas is not None:
        # 只保留用户想看的 alpha
        all_alphas = [a for a in all_alphas if a in only_alphas]

    if not all_alphas:
        logger.warning("  ✗ 没有可用的 alpha（或者全部被 only_alphas 过滤掉），跳过。")
        return

    # 如果存在 custom metrics，就读出来；否则置为 None
    df_custom = None
    if custom_file.exists():
        df_custom = pd.read_csv(custom_file)
        if "alpha" not in df_custom.columns:
            logger.warning(
                "  ⚠ combined_all_custom_metrics.csv 中没有 'alpha' 列，将无法按 alpha 拆分 custom metrics。"
            )
            df_custom = None

    # 临时工作目录：只存放按 alpha 过滤后的 combined_all_* 文件
    temp_dir = folder_path / "_alpha_temp_work"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 输出目录：例如 plots/alpha_results/evo_pca_512_combined_std_exp_alpha_range/
    folder_output_dir = output_dir / folder_path.name
    folder_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Found alphas: {all_alphas}")
    logger.info(f"  Plots will be saved under: {folder_output_dir}")

    for alpha_val in all_alphas:
        logger.info(f"\n  === Alpha = {alpha_val} ===")
        # 1. 按 alpha 过滤标准 metrics，并始终包含 RANDOM 作为对照
        if "strategy" in df_std.columns:
            df_std_alpha = df_std[
                (df_std["alpha"] == alpha_val) | (df_std["strategy"] == "random")
            ].copy()
        else:
            df_std_alpha = df_std[df_std["alpha"] == alpha_val].copy()

        if df_std_alpha.empty:
            logger.warning(f"  ✗ 对 alpha={alpha_val}，standard metrics 为空，跳过。")
            continue

        # 写入临时 combined_all_results.csv
        temp_std_file = temp_dir / "combined_all_results.csv"
        df_std_alpha.to_csv(temp_std_file, index=False)

        # 2. 按 alpha 过滤 custom metrics（如果有），并始终包含 RANDOM 作为对照
        has_custom_alpha = False
        if df_custom is not None:
            if "strategy" in df_custom.columns:
                df_custom_alpha = df_custom[
                    (df_custom["alpha"] == alpha_val) | (df_custom["strategy"] == "random")
                ].copy()
            else:
                df_custom_alpha = df_custom[df_custom["alpha"] == alpha_val].copy()

            if not df_custom_alpha.empty:
                temp_custom_file = temp_dir / "combined_all_custom_metrics.csv"
                df_custom_alpha.to_csv(temp_custom_file, index=False)
                has_custom_alpha = True
            else:
                logger.warning(f"  ⚠ 对 alpha={alpha_val}，custom metrics 为空，将不会生成 custom metrics 图。")

        # 3. 对这个 alpha 调用原有绘图函数（使用 temp_dir 作为 results_folder_path）
        try:
            logger.info("    - Generating standard metrics plot...")
            save_path = folder_output_dir / f"{folder_path.name}_alpha_{alpha_val}_standard_metrics.pdf"
            plot_active_learning_metrics(
                results_folder_path=str(temp_dir),
                save_path=str(save_path),
                show_plot=show_plots,
            )
            logger.info(f"      ✓ Standard metrics plot saved to: {save_path}")
        except Exception as e:
            logger.error(f"      ✗ Error creating standard metrics plot for alpha={alpha_val}: {e}")

        if has_custom_alpha:
            try:
                logger.info("    - Generating top 10 ratio metrics plot...")
                save_path = folder_output_dir / f"{folder_path.name}_alpha_{alpha_val}_top10_ratio_metrics.pdf"
                plot_top10_ratio_metrics(
                    results_folder_path=str(temp_dir),
                    save_path=str(save_path),
                    show_plot=show_plots,
                )
                logger.info(f"      ✓ Top 10 ratio metrics plot saved to: {save_path}")
            except Exception as e:
                logger.error(f"      ✗ Error creating top10 ratio metrics plot for alpha={alpha_val}: {e}")

            try:
                logger.info("    - Generating value metrics plot...")
                save_path = folder_output_dir / f"{folder_path.name}_alpha_{alpha_val}_value_metrics.pdf"
                plot_value_metrics(
                    results_folder_path=str(temp_dir),
                    save_path=str(save_path),
                    show_plot=show_plots,
                )
                logger.info(f"      ✓ Value metrics plot saved to: {save_path}")
            except Exception as e:
                logger.error(f"      ✗ Error creating value metrics plot for alpha={alpha_val}: {e}")

            # 如果 combined_all_custom_metrics 中有 regression_model 列，plot_regressor_comparison 也会起作用
            try:
                logger.info("    - Generating regressor comparison plot...")
                save_path = folder_output_dir / f"{folder_path.name}_alpha_{alpha_val}_regressor_comparison.pdf"
                plot_regressor_comparison(
                    results_folder_path=str(temp_dir),
                    save_path=str(save_path),
                    show_plot=show_plots,
                    plot_type=plot_type,
                    strategy="highExpression",  # 保持和 visualize_all_results.py 一致
                )
                logger.info(f"      ✓ Regressor comparison plot saved to: {save_path}")
            except Exception as e:
                logger.error(f"      ✗ Error creating regressor comparison plot for alpha={alpha_val}: {e}")

    # 清理临时目录（如果你希望保留可注释掉）
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-alpha visualizations for active learning experiment results"
    )

    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        required=True,
        help="Path to a specific results folder containing combined_all_results.csv",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="plots_alpha",
        help="Output directory for alpha-specific plots (default: plots_alpha/)",
    )

    parser.add_argument(
        "--plot-type",
        "-t",
        type=str,
        default="mean",
        help="Plot type for regressor comparison: mean, median, sem (default: mean)",
    )

    parser.add_argument(
        "--alphas",
        "-a",
        type=float,
        nargs="*",
        help="Specific alpha values to visualize. If omitted, visualize all alphas found.",
    )

    parser.add_argument(
        "--show-plots",
        "-s",
        action="store_true",
        help="Display plots interactively instead of only saving",
    )

    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists():
        logger.error(f"指定的结果文件夹不存在: {folder_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_folder_by_alpha(
        folder_path=folder_path,
        output_dir=output_dir,
        show_plots=args.show_plots,
        plot_type=args.plot_type,
        only_alphas=args.alphas,
    )


if __name__ == "__main__":
    main()