#!/usr/bin/env python3
"""
Simple script to generate plots for existing results.
"""

import subprocess
import sys


def main():
    """Generate plots for all existing results."""
    print("Generating plots for all results...")

    # Run the visualization script
    try:
        print("Running visualization script...")
        result = subprocess.run(
            ["uv", "run", "python", "plotting/visualize_all_results.py"], check=True
        )
        print("✓ Visualization plots generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Visualization failed: {e}")
        return False

    # Run the regressor comparison script
    try:
        print("Running regressor comparison script...")
        result = subprocess.run(
            ["uv", "run", "python", "plotting/plot_regressor_comparison.py"], check=True
        )
        print("✓ Regressor comparison plots generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Regressor comparison failed: {e}")
        return False

    print("\nAll plots generated successfully!")
    print("Check the 'plots/' directory for the generated visualizations.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
