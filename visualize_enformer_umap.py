#!/usr/bin/env python3
"""
Visualize enformer embeddings using UMAP clustering and color by expression values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from safetensors.torch import load_file

try:
    import umap
except ImportError:
    print("UMAP not available. Install with: pip install umap-learn")
    exit(1)


def load_enformer_data(data_path: str):
    """Load enformer embeddings and expression data."""
    print(f"Loading data from: {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load safetensors file
    tensors = load_file(data_path)

    print(f"Available keys: {list(tensors.keys())}")

    # Extract data
    embeddings = tensors["embeddings"].float().numpy()
    expressions = tensors["expressions"].float().numpy()
    variant_ids = tensors["variant_ids"].numpy()

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Expressions shape: {expressions.shape}")
    print(f"Variant IDs shape: {variant_ids.shape}")

    # Check for log likelihoods
    has_log_likelihoods = "log_likelihoods" in tensors
    if has_log_likelihoods:
        log_likelihoods = tensors["log_likelihoods"].float().numpy()
        print(f"Log likelihoods shape: {log_likelihoods.shape}")
    else:
        log_likelihoods = None
        print("No log likelihoods found")

    return embeddings, expressions, variant_ids, log_likelihoods


def create_umap_embedding(
    embeddings, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
):
    """Create UMAP embedding of the high-dimensional embeddings."""
    print(
        f"Creating UMAP embedding with {embeddings.shape[0]} samples and {embeddings.shape[1]} dimensions..."
    )

    # Initialize UMAP
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        verbose=True,
    )

    # Fit and transform
    umap_embedding = umap_model.fit_transform(embeddings)

    print(f"UMAP embedding shape: {umap_embedding.shape}")
    return umap_embedding


def create_visualizations(
    umap_embedding, expressions, variant_ids, log_likelihoods=None, output_dir="plots"
):
    """Create various visualizations of the UMAP embedding."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("viridis")

    # Figure 1: Color by expression values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Scatter plot colored by expression
    scatter = ax1.scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        c=expressions,
        cmap="viridis",
        alpha=0.6,
        s=1,
    )
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_title("Enformer Embeddings - UMAP Colored by Expression")
    plt.colorbar(scatter, ax=ax1, label="Expression Value")

    # Plot 2: Expression distribution
    ax2.hist(expressions, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    ax2.set_xlabel("Expression Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Expression Value Distribution")
    ax2.axvline(
        expressions.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {expressions.mean():.2f}",
    )
    ax2.axvline(
        np.median(expressions),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(expressions):.2f}",
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        output_dir / "enformer_umap_expression.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Figure 2: Expression quartiles
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Calculate quartiles
    q1 = np.percentile(expressions, 25)
    q2 = np.percentile(expressions, 50)
    q3 = np.percentile(expressions, 75)

    # Create quartile labels
    quartile_labels = []
    for expr in expressions:
        if expr <= q1:
            quartile_labels.append("Q1 (Low)")
        elif expr <= q2:
            quartile_labels.append("Q2 (Med-Low)")
        elif expr <= q3:
            quartile_labels.append("Q3 (Med-High)")
        else:
            quartile_labels.append("Q4 (High)")

    # Plot with quartile colors
    unique_labels = ["Q1 (Low)", "Q2 (Med-Low)", "Q3 (Med-High)", "Q4 (High)"]
    colors = ["blue", "green", "orange", "red"]

    for i, label in enumerate(unique_labels):
        mask = np.array(quartile_labels) == label
        ax.scatter(
            umap_embedding[mask, 0],
            umap_embedding[mask, 1],
            c=colors[i],
            label=label,
            alpha=0.6,
            s=1,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Enformer Embeddings - UMAP Colored by Expression Quartiles")
    ax.legend()

    plt.savefig(
        output_dir / "enformer_umap_quartiles.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Figure 3: Log likelihoods if available
    if log_likelihoods is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Filter out NaN values for plotting
        valid_mask = ~np.isnan(log_likelihoods)
        valid_umap = umap_embedding[valid_mask]
        valid_ll = log_likelihoods[valid_mask]

        if len(valid_ll) > 0:
            # Plot 1: Scatter plot colored by log likelihood
            scatter = ax1.scatter(
                valid_umap[:, 0],
                valid_umap[:, 1],
                c=valid_ll,
                cmap="plasma",
                alpha=0.6,
                s=1,
            )
            ax1.set_xlabel("UMAP 1")
            ax1.set_ylabel("UMAP 2")
            ax1.set_title("Enformer Embeddings - UMAP Colored by Log Likelihood")
            plt.colorbar(scatter, ax=ax1, label="Log Likelihood")

            # Plot 2: Log likelihood distribution
            ax2.hist(valid_ll, bins=50, alpha=0.7, color="purple", edgecolor="black")
            ax2.set_xlabel("Log Likelihood")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Log Likelihood Distribution")
            ax2.axvline(
                valid_ll.mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {valid_ll.mean():.2f}",
            )
            ax2.legend()
        else:
            ax1.text(
                0.5,
                0.5,
                "No valid log likelihood data",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax2.text(
                0.5,
                0.5,
                "No valid log likelihood data",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        plt.tight_layout()
        plt.savefig(
            output_dir / "enformer_umap_log_likelihood.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Figure 4: High vs Low expression comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Define high and low expression thresholds
    high_threshold = np.percentile(expressions, 90)
    low_threshold = np.percentile(expressions, 10)

    # Create masks
    high_mask = expressions >= high_threshold
    low_mask = expressions <= low_threshold
    medium_mask = ~(high_mask | low_mask)

    # Plot each group
    ax.scatter(
        umap_embedding[medium_mask, 0],
        umap_embedding[medium_mask, 1],
        c="lightgray",
        alpha=0.3,
        s=1,
        label=f"Medium ({np.sum(medium_mask)} samples)",
    )
    ax.scatter(
        umap_embedding[low_mask, 0],
        umap_embedding[low_mask, 1],
        c="blue",
        alpha=0.7,
        s=2,
        label=f"Low Expression ({np.sum(low_mask)} samples)",
    )
    ax.scatter(
        umap_embedding[high_mask, 0],
        umap_embedding[high_mask, 1],
        c="red",
        alpha=0.7,
        s=2,
        label=f"High Expression ({np.sum(high_mask)} samples)",
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Enformer Embeddings - High vs Low Expression")
    ax.legend()

    plt.savefig(
        output_dir / "enformer_umap_high_low_expression.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("CLUSTERING ANALYSIS SUMMARY")
    print("=" * 50)

    print(f"Total samples: {len(expressions)}")
    print(f"Expression range: {expressions.min():.3f} to {expressions.max():.3f}")
    print(f"Expression mean: {expressions.mean():.3f}")
    print(f"Expression std: {expressions.std():.3f}")

    print("\nExpression quartiles:")
    print(f"Q1 (25%): {q1:.3f}")
    print(f"Q2 (50%): {q2:.3f}")
    print(f"Q3 (75%): {q3:.3f}")

    print("\nHigh/Low expression analysis:")
    print(
        f"High expression (top 10%): {np.sum(high_mask)} samples, threshold: {high_threshold:.3f}"
    )
    print(
        f"Low expression (bottom 10%): {np.sum(low_mask)} samples, threshold: {low_threshold:.3f}"
    )

    if log_likelihoods is not None:
        valid_ll_count = np.sum(~np.isnan(log_likelihoods))
        print(
            f"\nLog likelihood data: {valid_ll_count}/{len(log_likelihoods)} valid values"
        )

    print(f"\nPlots saved to: {output_dir}")
    print("=" * 50)


def main():
    # File paths
    data_path = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/166k_data/166k_rice/post_embeddings/all_embeddings_parallel_enformer_with_expressions.safetensors"
    output_dir = "plots/enformer_umap"

    try:
        # Load data
        embeddings, expressions, variant_ids, log_likelihoods = load_enformer_data(
            data_path
        )

        # Create UMAP embedding
        umap_embedding = create_umap_embedding(embeddings)

        # Create visualizations
        create_visualizations(
            umap_embedding, expressions, variant_ids, log_likelihoods, output_dir
        )

        print("\nUMAP clustering visualization completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
