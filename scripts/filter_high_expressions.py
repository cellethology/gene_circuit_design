#!/usr/bin/env python3
"""
Filter out data points with expression values higher than 500 from safetensors files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file, save_file


def plot_filtering_analysis(
    expressions, log_likelihoods, keep_indices, max_expression, output_dir
):
    """
    Create plots showing the distribution of filtered data and correlation with log likelihood.

    Args:
        expressions: Array of expression values
        log_likelihoods: Array of log likelihood values
        keep_indices: Boolean array indicating which points to keep
        max_expression: Maximum expression threshold
        output_dir: Directory to save plots
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split data into kept and removed
    kept_expressions = expressions[keep_indices]
    removed_expressions = expressions[~keep_indices]
    kept_log_likelihoods = log_likelihoods[keep_indices]
    removed_log_likelihoods = log_likelihoods[~keep_indices]

    # Figure 1: Expression distribution comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(
        kept_expressions,
        bins=50,
        alpha=0.7,
        label=f"Kept (n={len(kept_expressions)})",
        color="green",
    )
    plt.hist(
        removed_expressions,
        bins=50,
        alpha=0.7,
        label=f"Removed (n={len(removed_expressions)})",
        color="red",
    )
    plt.axvline(
        x=max_expression,
        color="black",
        linestyle="--",
        label=f"Threshold={max_expression}",
    )
    plt.xlabel("Expression Value")
    plt.ylabel("Frequency")
    plt.title("Expression Distribution: Kept vs Removed")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(
        kept_log_likelihoods,
        bins=50,
        alpha=0.7,
        label=f"Kept (n={len(kept_log_likelihoods)})",
        color="green",
    )
    plt.hist(
        removed_log_likelihoods,
        bins=50,
        alpha=0.7,
        label=f"Removed (n={len(removed_log_likelihoods)})",
        color="red",
    )
    plt.xlabel("Log Likelihood")
    plt.ylabel("Frequency")
    plt.title("Log Likelihood Distribution: Kept vs Removed")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "filtering_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Figure 2: Scatter plot of expression vs log likelihood
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(
        kept_expressions,
        kept_log_likelihoods,
        alpha=0.5,
        s=1,
        c="green",
        label=f"Kept (n={len(kept_expressions)})",
    )
    plt.scatter(
        removed_expressions,
        removed_log_likelihoods,
        alpha=0.5,
        s=1,
        c="red",
        label=f"Removed (n={len(removed_expressions)})",
    )
    plt.axvline(
        x=max_expression,
        color="black",
        linestyle="--",
        alpha=0.8,
        label=f"Threshold={max_expression}",
    )
    plt.xlabel("Expression Value")
    plt.ylabel("Log Likelihood")
    plt.title("Expression vs Log Likelihood (All Data)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Focus on kept data only
    plt.scatter(kept_expressions, kept_log_likelihoods, alpha=0.6, s=1, c="green")
    plt.xlabel("Expression Value")
    plt.ylabel("Log Likelihood")
    plt.title(
        f"Expression vs Log Likelihood (Kept Data Only, n={len(kept_expressions)})"
    )
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "expression_vs_log_likelihood.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Calculate and print correlation statistics
    all_corr = np.corrcoef(expressions, log_likelihoods)[0, 1]
    kept_corr = np.corrcoef(kept_expressions, kept_log_likelihoods)[0, 1]
    removed_corr = np.corrcoef(removed_expressions, removed_log_likelihoods)[0, 1]

    print("\nCorrelation Analysis:")
    print(f"All data correlation (expression vs log_likelihood): {all_corr:.4f}")
    print(f"Kept data correlation: {kept_corr:.4f}")
    print(f"Removed data correlation: {removed_corr:.4f}")

    # Summary statistics
    print("\nSummary Statistics:")
    print(
        f"Kept data - Expression: mean={kept_expressions.mean():.3f}, std={kept_expressions.std():.3f}"
    )
    print(
        f"Kept data - Log likelihood: mean={kept_log_likelihoods.mean():.3f}, std={kept_log_likelihoods.std():.3f}"
    )
    print(
        f"Removed data - Expression: mean={removed_expressions.mean():.3f}, std={removed_expressions.std():.3f}"
    )
    print(
        f"Removed data - Log likelihood: mean={removed_log_likelihoods.mean():.3f}, std={removed_log_likelihoods.std():.3f}"
    )

    print(f"\nPlots saved to: {output_dir}")


def filter_high_expressions(
    input_path: str, output_path: str, max_expression: float = 750.0
):
    """
    Filter out data points with expression values higher than the threshold.

    Args:
        input_path: Path to input safetensors file
        output_path: Path to output safetensors file
        max_expression: Maximum allowed expression value
    """
    print(f"Loading data from {input_path}")

    # Load the safetensors file
    tensors = load_file(input_path)

    # Print available keys
    print(f"Available keys: {list(tensors.keys())}")

    # Load expressions and find indices to keep
    if "expression" in tensors:
        expressions = tensors["expression"].float().numpy()
        expression_key = "expression"
    elif "expressions" in tensors:
        expressions = tensors["expressions"].float().numpy()
        expression_key = "expressions"
    else:
        raise ValueError(
            f"No expression data found. Available keys: {list(tensors.keys())}"
        )

    # Load log likelihoods for correlation analysis
    if "log_likelihood" in tensors:
        log_likelihoods = tensors["log_likelihood"].float().numpy()
    elif "log_likelihoods" in tensors:
        log_likelihoods = tensors["log_likelihoods"].float().numpy()
    else:
        print("Warning: No log likelihood data found for correlation analysis")
        log_likelihoods = None

    print(f"Original data shape: {expressions.shape}")
    print(f"Expression range: {expressions.min():.3f} to {expressions.max():.3f}")

    # Find indices where expression <= max_expression
    keep_indices = expressions <= max_expression
    num_kept = keep_indices.sum()
    num_removed = len(expressions) - num_kept

    print(f"Removing {num_removed} data points with expression > {max_expression}")
    print(f"Keeping {num_kept} data points")

    # Create plots if log likelihood data is available
    if log_likelihoods is not None:
        output_dir = Path(output_path).parent / "filtering_analysis"
        plot_filtering_analysis(
            expressions, log_likelihoods, keep_indices, max_expression, output_dir
        )

    # Filter all tensors
    filtered_tensors = {}
    for key, tensor in tensors.items():
        if len(tensor.shape) > 0 and tensor.shape[0] == len(expressions):
            # This tensor has the same first dimension as expressions, so filter it
            filtered_tensor = tensor[keep_indices]
            filtered_tensors[key] = filtered_tensor
            print(f"Filtered {key}: {tensor.shape} -> {filtered_tensor.shape}")
        else:
            # This tensor doesn't match the data dimension, keep as is
            filtered_tensors[key] = tensor
            print(f"Kept unchanged {key}: {tensor.shape}")

    # Verify the filtering worked
    filtered_expressions = filtered_tensors[expression_key].float().numpy()
    print(f"total number of data points: {len(expressions)}")
    print(f"total number of filtered data points: {len(filtered_expressions)}")
    print(
        f"Filtered expression range: {filtered_expressions.min():.3f} to {filtered_expressions.max():.3f}"
    )

    # Save the filtered data
    print(f"Saving filtered data to {output_path}")
    save_file(filtered_tensors, output_path)
    print("Done!")


def main():
    """Main function to filter both files."""

    # File paths
    pca_input = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/166k_data/166k_rice/post_embeddings/pca_results_32cores.safetensors"
    pca_output = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/166k_data/166k_rice/post_embeddings/pca_results_32cores_filtered.safetensors"

    embeddings_input = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/166k_data/166k_rice/post_embeddings/all_tensor_data.safetensors"
    embeddings_output = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/166k_data/166k_rice/post_embeddings/all_tensor_data_filtered.safetensors"

    # Filter PCA results
    print("=" * 60)
    print("Filtering PCA results")
    print("=" * 60)
    if Path(pca_input).exists():
        filter_high_expressions(pca_input, pca_output)
    else:
        print(f"PCA file not found: {pca_input}")

    print("\n" + "=" * 60)
    print("Filtering embeddings data")
    print("=" * 60)
    if Path(embeddings_input).exists():
        filter_high_expressions(embeddings_input, embeddings_output)
    else:
        print(f"Embeddings file not found: {embeddings_input}")


if __name__ == "__main__":
    main()
