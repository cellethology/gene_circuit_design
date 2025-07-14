#!/usr/bin/env python3
"""
Filter out data points with expression values higher than 500 from safetensors files.
"""

from pathlib import Path

from safetensors.torch import load_file, save_file


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

    print(f"Original data shape: {expressions.shape}")
    print(f"Expression range: {expressions.min():.3f} to {expressions.max():.3f}")

    # Find indices where expression <= max_expression
    keep_indices = expressions <= max_expression
    num_kept = keep_indices.sum()
    num_removed = len(expressions) - num_kept

    print(f"Removing {num_removed} data points with expression > {max_expression}")
    print(f"Keeping {num_kept} data points")

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
