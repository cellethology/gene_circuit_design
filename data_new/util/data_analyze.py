import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from safetensors.torch import load_file


def visualize_embeddings_umap(
    safetensors_path,
    embedding_key="pca_components",
    expression_key="expression",
    expression_threshold=20,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
    figsize=(12, 8),
    alpha=0.7,
    point_size=20,
    colormap="viridis",
    save_path=None,
    show_detailed_plots=True,
    verbose=True,
):
    """
    Create UMAP visualization of embeddings colored by expression levels.

    Parameters:
    -----------
    safetensors_path : str
        Path to the safetensors file containing embeddings and expression data
    embedding_key : str, default='pca_components'
        Key for embeddings in the safetensors file
    expression_key : str, default='expression'
        Key for expression values in the safetensors file
    expression_threshold : float, default=20
        Filter samples with expression values below this threshold
    n_components : int, default=2
        Number of UMAP components (2 or 3)
    n_neighbors : int, default=15
        UMAP n_neighbors parameter
    min_dist : float, default=0.1
        UMAP min_dist parameter
    metric : str, default='euclidean'
        UMAP distance metric
    random_state : int, default=42
        Random state for reproducibility
    figsize : tuple, default=(12, 8)
        Figure size for the main plot
    alpha : float, default=0.7
        Transparency of points
    point_size : int, default=20
        Size of scatter plot points
    colormap : str, default='viridis'
        Matplotlib colormap name
    save_path : str, optional
        Path to save the main plot (if None, uses default name)
    show_detailed_plots : bool, default=True
        Whether to show the detailed comparison plots
    verbose : bool, default=True
        Whether to print progress and statistics

    Returns:
    --------
    dict : Dictionary containing:
        - 'umap_embedding': 2D/3D UMAP coordinates
        - 'filtered_embeddings': Original embeddings after filtering
        - 'filtered_expression': Expression values after filtering
        - 'reducer': Fitted UMAP reducer object
        - 'filter_mask': Boolean mask used for filtering
    """

    # Load data
    if verbose:
        print("Loading embeddings...")
    data = load_file(safetensors_path)
    embeddings = data[embedding_key].flatten().numpy()
    expression_values = data[expression_key].numpy()

    if verbose:
        print(f"Original embeddings shape: {embeddings.shape}")
        print(f"Original expression values shape: {expression_values.shape}")

    # Filter data
    mask = expression_values < expression_threshold
    filtered_embeddings = embeddings[mask]
    filtered_expression = expression_values[mask]

    if verbose:
        print(f"Filtered embeddings shape: {filtered_embeddings.shape}")
        print(f"Kept {np.sum(mask)}/{len(mask)} samples ({np.mean(mask)*100:.1f}%)")

    # Create and fit UMAP
    if verbose:
        print("Fitting UMAP on filtered data...")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    embedding_2d = reducer.fit_transform(filtered_embeddings)

    if verbose:
        print(f"UMAP embedding shape: {embedding_2d.shape}")

    # Create main plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=filtered_expression,
        cmap=colormap,
        alpha=alpha,
        s=point_size,
    )

    plt.colorbar(scatter, label="Expression Level")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(
        f"UMAP of Embeddings (Expression < {expression_threshold})\nColored by Expression Level"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save main plot
    if save_path is None:
        save_path = f"umap_embeddings_filtered_{expression_threshold}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Print statistics
    if verbose:
        print("\nUMAP Statistics:")
        print(
            f"UMAP 1 range: [{embedding_2d[:, 0].min():.2f}, {embedding_2d[:, 0].max():.2f}]"
        )
        print(
            f"UMAP 2 range: [{embedding_2d[:, 1].min():.2f}, {embedding_2d[:, 1].max():.2f}]"
        )
        print(
            f"Expression range: [{filtered_expression.min():.2f}, {filtered_expression.max():.2f}]"
        )

    # Create detailed comparison plots
    if show_detailed_plots:
        plt.figure(figsize=(15, 5))

        # Plot 1: Standard colormap
        plt.subplot(1, 3, 1)
        plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=filtered_expression,
            cmap="viridis",
            alpha=alpha,
            s=15,
        )
        plt.colorbar(label="Expression")
        plt.title("UMAP - Viridis")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

        # Plot 2: Different colormap
        plt.subplot(1, 3, 2)
        plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=filtered_expression,
            cmap="RdYlBu_r",
            alpha=alpha,
            s=15,
        )
        plt.colorbar(label="Expression")
        plt.title("UMAP - RdYlBu")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

        # Plot 3: Binned expression levels
        plt.subplot(1, 3, 3)
        expression_bins = pd.cut(
            filtered_expression,
            bins=5,
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
        )
        colors = ["blue", "cyan", "green", "orange", "red"]

        for i, bin_label in enumerate(
            ["Very Low", "Low", "Medium", "High", "Very High"]
        ):
            mask_bin = expression_bins == bin_label
            if np.any(mask_bin):
                plt.scatter(
                    embedding_2d[mask_bin, 0],
                    embedding_2d[mask_bin, 1],
                    c=colors[i],
                    label=bin_label,
                    alpha=alpha,
                    s=15,
                )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title("UMAP - Binned Expression")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

        plt.tight_layout()
        detailed_save_path = save_path.replace(".png", "_detailed.png")
        plt.savefig(detailed_save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # Return results
    return {
        "umap_embedding": embedding_2d,
        "filtered_embeddings": filtered_embeddings,
        "filtered_expression": filtered_expression,
        "reducer": reducer,
        "filter_mask": mask,
    }
