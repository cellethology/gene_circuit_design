#!/usr/bin/env python3
"""
Script to generate regressor comparison plots from 3-regressor experimental results.
"""

import argparse
from pathlib import Path

from utils.plotting import plot_regressor_comparison, plot_regressor_summary


def main():
    parser = argparse.ArgumentParser(description="Generate regressor comparison plots")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/results_3_regressors/embeddings_all_strategies",
        help="Path to results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (defaults to results directory)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["pearson_correlation", "spearman_correlation", "r2", "rmse"],
        help="Metrics to plot",
    )
    parser.add_argument(
        "--summary-metric",
        type=str,
        default="pearson_correlation",
        help="Metric for summary plot",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (only save them)",
    )

    args = parser.parse_args()

    # Set up paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Check if combined results file exists
    combined_results_file = results_dir / "combined_all_results.csv"
    if not combined_results_file.exists():
        print(f"Error: Combined results file not found: {combined_results_file}")
        print("Make sure to run the experiments first to generate results.")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating regressor comparison plots from: {results_dir}")
    print(f"Saving plots to: {output_dir}")

    try:
        # Generate detailed comparison plot
        print("\nGenerating detailed regressor comparison plot...")
        detailed_plot_path = output_dir / "regressor_comparison_detailed.png"
        plot_regressor_comparison(
            results_folder_path=str(results_dir),
            metrics=args.metrics,
            figsize=(20, 15),
            save_path=str(detailed_plot_path),
            show_plot=not args.no_show,
        )

        # Generate summary plot
        print(f"\nGenerating summary plot for {args.summary_metric}...")
        summary_plot_path = output_dir / f"regressor_summary_{args.summary_metric}.png"
        plot_regressor_summary(
            results_folder_path=str(results_dir),
            metric=args.summary_metric,
            figsize=(14, 10),
            save_path=str(summary_plot_path),
            show_plot=not args.no_show,
        )

        print("\nPlots generated successfully!")
        print(f"  - Detailed comparison: {detailed_plot_path}")
        print(f"  - Summary plot: {summary_plot_path}")

    except Exception as e:
        print(f"Error generating plots: {e}")
        print("Make sure the results data contains the regression_model column.")


if __name__ == "__main__":
    main()
