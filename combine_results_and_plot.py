#!/usr/bin/env python3
"""
Combine existing individual results files and generate regressor comparison plots.
"""
import sys
from pathlib import Path

from run_experiments import create_combined_results_from_files
from utils.plotting import plot_regressor_comparison

# Add the current directory to Python path so we can import from run_experiments
sys.path.append(str(Path(__file__).parent))


def main():
    results_dir = Path("results/results_3_regressors/embeddings_all_strategies")

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Combining results from: {results_dir}")

    # Create combined results from individual files
    create_combined_results_from_files(results_dir)

    # Check if combined results file was created
    combined_file = results_dir / "combined_all_results.csv"
    if not combined_file.exists():
        print("Error: Could not create combined results file")
        return

    print(f"Combined results saved to: {combined_file}")

    # Check what metrics are available
    import pandas as pd

    df = pd.read_csv(combined_file)
    available_columns = df.columns.tolist()
    print(f"Available columns: {available_columns}")

    # For no_test experiments, we only have basic columns, so let's use custom metrics
    custom_metrics_file = results_dir / "combined_all_custom_metrics.csv"

    if custom_metrics_file.exists():
        print("\nGenerating plots from custom metrics...")
        try:
            from utils.plotting import plot_custom_metrics

            # Plot custom metrics instead
            custom_plot_path = results_dir / "regressor_custom_metrics_comparison.png"
            plot_custom_metrics(
                results_folder_path=str(results_dir),
                figsize=(24, 18),
                save_path=str(custom_plot_path),
                show_plot=False,
            )
            print(f"Custom metrics plot saved to: {custom_plot_path}")

            print("\nCustom metrics plots generated successfully!")

        except Exception as e:
            print(f"Error generating custom metrics plots: {e}")
    else:
        print("No custom metrics file found. Skipping plots.")

    # If we have any performance metrics, try to plot them
    if len(available_columns) > 7:  # More than just basic columns
        print("\nTrying to generate basic comparison plots...")
        try:
            # Use available metrics (excluding metadata columns)
            metadata_columns = [
                "round",
                "strategy",
                "seq_mod_method",
                "regression_model",
                "seed",
                "train_size",
                "unlabeled_size",
            ]
            metric_columns = [
                col for col in available_columns if col not in metadata_columns
            ]

            if metric_columns:
                print(f"Using metrics: {metric_columns}")
                detailed_plot_path = (
                    results_dir / "regressor_comparison_available_metrics.png"
                )
                plot_regressor_comparison(
                    results_folder_path=str(results_dir),
                    metrics=metric_columns[:4],  # Use first 4 available metrics
                    figsize=(20, 15),
                    save_path=str(detailed_plot_path),
                    show_plot=False,
                )
                print(f"Available metrics plot saved to: {detailed_plot_path}")

        except Exception as e:
            print(f"Error generating available metrics plots: {e}")


if __name__ == "__main__":
    main()
