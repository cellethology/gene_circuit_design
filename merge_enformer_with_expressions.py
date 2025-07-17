#!/usr/bin/env python3
"""
Merge enformer embeddings with expression data from all_tensor_data.safetensors
using variant_ids as the matching key.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def merge_enformer_with_expressions(
    enformer_path: str, expression_data_path: str, output_path: str
):
    """
    Merge enformer embeddings with expression data using variant_ids.

    Args:
        enformer_path: Path to enformer safetensors file (embeddings + variant_ids)
        expression_data_path: Path to all_tensor_data.safetensors (expressions + variant_ids)
        output_path: Path to output merged safetensors file
    """

    print(f"Loading enformer data from: {enformer_path}")
    enformer_data = load_file(enformer_path)

    print(f"Loading expression data from: {expression_data_path}")
    expression_data = load_file(expression_data_path)

    print(f"Enformer data keys: {list(enformer_data.keys())}")
    print(f"Expression data keys: {list(expression_data.keys())}")

    # Extract data from enformer file
    enformer_embeddings = enformer_data["embeddings"]
    enformer_variant_ids = enformer_data["variant_ids"]

    print(f"Enformer embeddings shape: {enformer_embeddings.shape}")
    print(f"Enformer variant_ids shape: {enformer_variant_ids.shape}")

    # Extract data from expression file
    expression_variant_ids = expression_data["variant_ids"]
    expressions = expression_data["expressions"]

    print(f"Expression variant_ids shape: {expression_variant_ids.shape}")
    print(f"Expressions shape: {expressions.shape}")

    # Check if log_likelihoods exist in expression data
    has_log_likelihoods = "log_likelihoods" in expression_data
    if has_log_likelihoods:
        log_likelihoods = expression_data["log_likelihoods"]
        print(f"Log likelihoods shape: {log_likelihoods.shape}")
    else:
        print("No log likelihoods found in expression data")

    # Convert to numpy for easier manipulation
    enformer_variant_ids_np = enformer_variant_ids.numpy()
    expression_variant_ids_np = expression_variant_ids.numpy()

    # Find matching variant_ids
    print("Finding matching variant_ids...")

    # Create mapping from variant_id to index in expression data
    expr_variant_to_idx = {
        int(vid): idx for idx, vid in enumerate(expression_variant_ids_np)
    }

    # Find indices in enformer data that have matching expressions
    matched_enformer_indices = []
    matched_expression_indices = []
    matched_variant_ids = []

    for enformer_idx, variant_id in enumerate(enformer_variant_ids_np):
        variant_id_int = int(variant_id)
        if variant_id_int in expr_variant_to_idx:
            matched_enformer_indices.append(enformer_idx)
            matched_expression_indices.append(expr_variant_to_idx[variant_id_int])
            matched_variant_ids.append(variant_id_int)

    print(f"Found {len(matched_enformer_indices)} matching variant_ids")
    print(f"Total enformer variants: {len(enformer_variant_ids_np)}")
    print(f"Total expression variants: {len(expression_variant_ids_np)}")

    if len(matched_enformer_indices) == 0:
        raise ValueError(
            "No matching variant_ids found between enformer and expression data"
        )

    # Extract matched data
    matched_embeddings = enformer_embeddings[matched_enformer_indices]
    matched_expressions = expressions[matched_expression_indices]
    matched_variant_ids_tensor = torch.tensor(matched_variant_ids)

    print(f"Matched embeddings shape: {matched_embeddings.shape}")
    print(f"Matched expressions shape: {matched_expressions.shape}")

    # Create output data dictionary
    output_data = {
        "embeddings": matched_embeddings,
        "expressions": matched_expressions,
        "variant_ids": matched_variant_ids_tensor,
    }

    # Add log likelihoods if available
    if has_log_likelihoods:
        matched_log_likelihoods = log_likelihoods[matched_expression_indices]
        output_data["log_likelihoods"] = matched_log_likelihoods
        print(f"Matched log likelihoods shape: {matched_log_likelihoods.shape}")

    # Save merged data
    print(f"Saving merged data to: {output_path}")
    save_file(output_data, output_path)

    # Print summary
    print("\nMerge Summary:")
    print(f"- Input enformer embeddings: {enformer_embeddings.shape}")
    print(f"- Input expressions: {expressions.shape}")
    print(f"- Matched samples: {len(matched_enformer_indices)}")
    print(f"- Output file: {output_path}")

    # Print sample statistics
    print("\nExpression statistics:")
    print(f"- Min: {matched_expressions.min():.3f}")
    print(f"- Max: {matched_expressions.max():.3f}")
    print(f"- Mean: {matched_expressions.mean():.3f}")
    print(f"- Std: {matched_expressions.std():.3f}")

    if has_log_likelihoods:
        # Count non-NaN log likelihoods
        valid_log_likelihoods = ~torch.isnan(matched_log_likelihoods)
        num_valid = valid_log_likelihoods.sum().item()
        print("\nLog likelihood statistics:")
        print(f"- Valid values: {num_valid}/{len(matched_log_likelihoods)}")
        if num_valid > 0:
            valid_ll = matched_log_likelihoods[valid_log_likelihoods]
            print(f"- Min: {valid_ll.min():.3f}")
            print(f"- Max: {valid_ll.max():.3f}")
            print(f"- Mean: {valid_ll.mean():.3f}")

    print("\nMerged data saved successfully!")


def main():
    # File paths
    enformer_path = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/166k_data/166k_rice/post_embeddings/all_embeddings_parallel_enformer.safetensors"
    expression_data_path = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/166k_data/166k_rice/post_embeddings/all_tensor_data.safetensors"
    output_path = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/166k_data/166k_rice/post_embeddings/all_embeddings_parallel_enformer_with_expressions.safetensors"

    # Check if input files exist
    if not Path(enformer_path).exists():
        print(f"Error: Enformer file not found: {enformer_path}")
        return

    if not Path(expression_data_path).exists():
        print(f"Error: Expression data file not found: {expression_data_path}")
        return

    try:
        merge_enformer_with_expressions(
            enformer_path, expression_data_path, output_path
        )
    except Exception as e:
        print(f"Error during merge: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
