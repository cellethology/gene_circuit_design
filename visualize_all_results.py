#!/usr/bin/env python3
"""
Comprehensive visualization script for all active learning experiment results.

This script automatically discovers all results folders and generates visualizations
for both standard metrics and custom metrics using the plotting utilities.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

from utils.plotting import (
    plot_active_learning_metrics,
    plot_top10_ratio_metrics,
    plot_value_metrics,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discover_results_folders(base_path: str = ".") -> List[Path]:
    """
    Discover all results folders in the base path.

    Args:
        base_path: Base directory to search for results folders

    Returns:
        List of Path objects for results folders
    """
    base_path = Path(base_path)
    results_folders = []

    # Look for folders that start with "results" at the top level
    for folder in base_path.glob("results*"):
        if folder.is_dir():
            results_folders.append(folder)

    # Also look inside the results/ directory for nested results folders
    results_dir = base_path / "results"
    if results_dir.exists() and results_dir.is_dir():
        for folder in results_dir.glob("results*"):
            if folder.is_dir():
                results_folders.append(folder)

    return sorted(results_folders)


def check_folder_contents(folder_path: Path) -> Tuple[bool, bool]:
    """
    Check what type of results files are available in a folder.

    Args:
        folder_path: Path to the results folder

    Returns:
        Tuple of (has_standard_metrics, has_custom_metrics)
    """
    standard_file = folder_path / "combined_all_results.csv"
    custom_file = folder_path / "combined_all_custom_metrics.csv"

    has_standard = standard_file.exists()
    has_custom = custom_file.exists()

    return has_standard, has_custom


def visualize_folder(
    folder_path: Path, output_dir: Path = None, show_plots: bool = False
) -> None:
    """
    Create visualizations for a single results folder.

    Args:
        folder_path: Path to the results folder
        output_dir: Optional output directory for plots. If None, saves in plots/ folder
        show_plots: Whether to display plots interactively
    """
    if output_dir is None:
        output_dir = Path("plots")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_standard, has_custom = check_folder_contents(folder_path)

    logger.info(f"Processing folder: {folder_path}")
    logger.info(f"  Standard metrics: {'✓' if has_standard else '✗'}")
    logger.info(f"  Custom metrics: {'✓' if has_custom else '✗'}")

    # Plot standard metrics if available
    if has_standard:
        try:
            logger.info("  Generating standard metrics plot...")
            plot_active_learning_metrics(
                results_folder_path=str(folder_path),
                save_path=str(output_dir / f"{folder_path.name}_standard_metrics.png"),
                show_plot=show_plots,
            )
            logger.info("  ✓ Standard metrics plot created")
        except Exception as e:
            logger.error(f"  ✗ Error creating standard metrics plot: {e}")

    # Plot custom metrics if available
    if has_custom:
        try:
            logger.info("  Generating top 10 ratio metrics plot...")
            plot_top10_ratio_metrics(
                results_folder_path=str(folder_path),
                save_path=str(
                    output_dir / f"{folder_path.name}_top10_ratio_metrics.png"
                ),
                show_plot=show_plots,
            )
            logger.info("  ✓ Top 10 ratio metrics plot created")
        except Exception as e:
            logger.error(f"  ✗ Error creating top 10 ratio metrics plot: {e}")

        try:
            logger.info("  Generating value metrics plot...")
            plot_value_metrics(
                results_folder_path=str(folder_path),
                save_path=str(output_dir / f"{folder_path.name}_value_metrics.png"),
                show_plot=show_plots,
            )
            logger.info("  ✓ Value metrics plot created")
        except Exception as e:
            logger.error(f"  ✗ Error creating value metrics plot: {e}")

    if not has_standard and not has_custom:
        logger.warning(f"  No suitable data files found in {folder_path}")


def create_summary_report(
    results_folders: List[Path], output_file: str = "results_summary.md"
) -> None:
    """
    Create a markdown summary report of all results folders.

    Args:
        results_folders: List of results folder paths
        output_file: Output markdown file name
    """
    with open(output_file, "w") as f:
        f.write("# Active Learning Experiments Results Summary\n\n")
        f.write(f"Found {len(results_folders)} results folders:\n\n")

        for folder in results_folders:
            has_standard, has_custom = check_folder_contents(folder)

            f.write(f"## {folder.name}\n\n")
            f.write(f"- **Path**: `{folder}`\n")
            f.write(
                f"- **Standard metrics**: {'✅ Available' if has_standard else '❌ Not found'}\n"
            )
            f.write(
                f"- **Custom metrics**: {'✅ Available' if has_custom else '❌ Not found'}\n"
            )

            # List generated plots
            plots = []
            if has_standard:
                plots.append(f"![Standard Metrics]({folder.name}_standard_metrics.png)")
            if has_custom:
                plots.append(f"![Custom Metrics]({folder.name}_custom_metrics.png)")

            if plots:
                f.write("- **Generated plots**:\n")
                for plot in plots:
                    f.write(f"  - {plot}\n")

            f.write("\n")

    logger.info(f"Summary report saved to: {output_file}")


def main():
    """Main function to visualize all results."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for all active learning experiment results"
    )

    parser.add_argument(
        "--base-path",
        "-b",
        type=str,
        default=".",
        help="Base directory to search for results folders (default: current directory)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots/)",
    )

    parser.add_argument(
        "--show-plots", "-s", action="store_true", help="Display plots interactively"
    )

    parser.add_argument(
        "--create-summary",
        "-r",
        action="store_true",
        help="Create a markdown summary report",
    )

    parser.add_argument(
        "--folder", "-f", type=str, help="Process only a specific results folder"
    )

    args = parser.parse_args()

    # Discover results folders
    if args.folder:
        # Process only the specified folder
        folder_path = Path(args.folder)
        if not folder_path.exists():
            logger.error(f"Specified folder does not exist: {folder_path}")
            return
        results_folders = [folder_path]
    else:
        # Discover all results folders
        results_folders = discover_results_folders(args.base_path)

    if not results_folders:
        logger.warning(f"No results folders found in {args.base_path}")
        return

    logger.info(f"Found {len(results_folders)} results folders:")
    for folder in results_folders:
        logger.info(f"  - {folder}")

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plots will be saved to: {output_dir}")

    # Process each folder
    logger.info("\n" + "=" * 60)
    logger.info("Starting visualization generation...")
    logger.info("=" * 60)

    for folder in results_folders:
        try:
            visualize_folder(
                folder_path=folder,
                output_dir=output_dir / folder.name,
                show_plots=args.show_plots,
            )
        except Exception as e:
            logger.error(f"Error processing folder {folder}: {e}")
            continue

    # Create summary report if requested
    if args.create_summary:
        summary_file = output_dir / "results_summary.md"
        create_summary_report(results_folders, str(summary_file))

    logger.info("\n" + "=" * 60)
    logger.info("Visualization generation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
