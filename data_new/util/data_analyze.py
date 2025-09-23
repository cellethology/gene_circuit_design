import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors.torch import load_file, save_file
import umap
import torch
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from typing import Tuple

def _estimate_k90(
    X_sub: np.ndarray,
    target_var: float = 0.90,
    caps = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
) -> int:
    """
    Incrementally increase n_components for randomized PCA on a subset
    until cumulative explained variance >= target_var. Returns the
    smallest integer k hitting the target (or the last cap if not reached).
    """
    p = X_sub.shape[1]
    best_k = min(p - 1, caps[-1])
    for cap in caps:
        cap = min(cap, p - 1)
        pca_probe = PCA(n_components=cap, svd_solver="randomized", random_state=42)
        pca_probe.fit(X_sub)
        cum = np.cumsum(pca_probe.explained_variance_ratio_)
        if cum[-1] >= target_var:
            k = int(np.searchsorted(cum, target_var) + 1)
            return k
        best_k = cap  # track the largest tried
    # If we never reached the target, return the largest attempted cap
    return best_k


def pca_down(
    in_path: str,
    out_path: str,
    expression_key: str,
    target_var: float = 0.90,
    subset_size: int = 20000,
    arpack_threshold: int = 256  # if k <= this, prefer arpack; else randomized
) -> Tuple[int, float]:
    
    # --- Load ---
    data = load_file(in_path)
    emb: torch.Tensor = data["embeddings"].float()   # ensure float before numpy
    expr: torch.Tensor = data[expression_key].float()

    if 'log_likelihoods' in data.keys():
        log_likelihood: torch.Tensor = data['log_likelihoods'].float()

    # --- Flatten [N, 960, 16] -> [N, 15360] ---
    N = emb.shape[0]
    X = emb.contiguous().view(N, -1).numpy()  # contiguous for safe view->numpy
    # Keep expressions as torch; we’ll save them unchanged
    # y = expr.numpy()  # if you need numpy downstream

    # --- Pilot subset ---
    rng = check_random_state(42)
    idx = rng.choice(N, size=min(subset_size, N), replace=False)
    X_sub = X[idx]

    # k90 = _estimate_k90(X_sub, target_var=target_var)
    # print(f"[Pilot] Estimated k for >={target_var:.0%} variance: {k90}")

    k90 = 256
    # --- Final PCA on full data ---
    # Choose solver based on k size (ARPACK prefers k << min(N, p))
    solver = "arpack" if k90 <= arpack_threshold else "randomized"
    print(f"[Final] Using svd_solver='{solver}'")

    pca = PCA(n_components=k90, svd_solver=solver, random_state=42)
    X_pca = pca.fit_transform(X)

    var_sum = float(pca.explained_variance_ratio_.sum())
    print(f"Original shape: {X.shape} -> Reduced shape: {X_pca.shape}")
    print(f"Explained variance (sum): {var_sum:.4f}")

    # --- Save to safetensors ---
    emb_pca = torch.from_numpy(X_pca.copy()).float()  # [N, k90]
    
    if 'log_likelihood' in data.keys():
        save_file({"embeddings": emb_pca.contiguous(), "expressions": expr.contiguous(), 'log_likelihoods': log_likelihood.contiguous()}, out_path)
    else:
        save_file({"embeddings": emb_pca.contiguous(), "expressions": expr.contiguous()}, out_path)
    print(f"Saved PCA-reduced tensors to: {out_path}")

    return k90, var_sum


def visualize_embeddings_umap(
    safetensors_path,
    embedding_key='embeddings',
    expression_key='expressions',
    expression_threshold=2000,
    n_pca_components=100,
    apply_pca=True,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    figsize=(12, 8),
    alpha=0.7,
    point_size=20,
    colormap='viridis',
    save_path=None,
    show_detailed_plots=True,
    verbose=True
):
    """
    Create UMAP visualization of embeddings colored by expression levels.
    Includes optional PCA preprocessing for high-dimensional data.
    
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
    n_pca_components : int, default=100
        Number of PCA components to keep (only used if apply_pca=True)
    apply_pca : bool, default=True
        Whether to apply PCA preprocessing before UMAP
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
        - 'pca_embeddings': PCA-transformed embeddings (if apply_pca=True)
        - 'filtered_expression': Expression values after filtering
        - 'reducer': Fitted UMAP reducer object
        - 'pca': Fitted PCA object (if apply_pca=True, else None)
        - 'filter_mask': Boolean mask used for filtering
    """
    
    # Load data
    if verbose:
        print("Loading embeddings...")
    data = load_file(safetensors_path)
    embeddings = data[embedding_key].view(data[embedding_key].shape[0], -1).numpy()
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
    
    # Apply PCA if requested
    pca = None
    pca_embeddings = None
    embeddings_for_umap = filtered_embeddings
    
    if apply_pca:
        if verbose:
            print(f"Applying PCA: {filtered_embeddings.shape} → ({filtered_embeddings.shape[0]}, {n_pca_components})")
        
        pca = PCA(n_components=n_pca_components, random_state=random_state)
        pca_embeddings = pca.fit_transform(filtered_embeddings)
        embeddings_for_umap = pca_embeddings
        
        if verbose:
            print(f"PCA explained variance ratio (first 10): {pca.explained_variance_ratio_[:10].round(4)}")
            print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Create and fit UMAP
    if verbose:
        input_shape = embeddings_for_umap.shape
        print(f"Fitting UMAP on {'PCA-transformed' if apply_pca else 'original'} data: {input_shape}")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    
    embedding_2d = reducer.fit_transform(embeddings_for_umap)
    
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
        s=point_size
    )
    
    plt.colorbar(scatter, label='Expression Level')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    pca_suffix = " (with PCA)" if apply_pca else ""
    plt.title(f'UMAP of Embeddings{pca_suffix} (Expression < {expression_threshold})\nColored by Expression Level')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save main plot
    if save_path is None:
        pca_tag = "_pca" if apply_pca else ""
        save_path = f'umap_embeddings_filtered_{expression_threshold}{pca_tag}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    if verbose:
        print(f"\nUMAP Statistics:")
        print(f"UMAP 1 range: [{embedding_2d[:, 0].min():.2f}, {embedding_2d[:, 0].max():.2f}]")
        print(f"UMAP 2 range: [{embedding_2d[:, 1].min():.2f}, {embedding_2d[:, 1].max():.2f}]")
        print(f"Expression range: [{filtered_expression.min():.2f}, {filtered_expression.max():.2f}]")
    
    # Create detailed comparison plots
    if show_detailed_plots:
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Standard colormap
        plt.subplot(1, 3, 1)
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=filtered_expression, 
                   cmap='viridis', alpha=alpha, s=15)
        plt.colorbar(label='Expression')
        plt.title('UMAP - Viridis')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        
        # Plot 2: Different colormap
        plt.subplot(1, 3, 2)
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=filtered_expression, 
                   cmap='RdYlBu_r', alpha=alpha, s=15)
        plt.colorbar(label='Expression')
        plt.title('UMAP - RdYlBu')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        
        # Plot 3: Binned expression levels
        plt.subplot(1, 3, 3)
        expression_bins = pd.cut(filtered_expression, bins=5, 
                               labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        colors = ['blue', 'cyan', 'green', 'orange', 'red']
        
        for i, bin_label in enumerate(['Very Low', 'Low', 'Medium', 'High', 'Very High']):
            mask_bin = expression_bins == bin_label
            if np.any(mask_bin):
                plt.scatter(embedding_2d[mask_bin, 0], embedding_2d[mask_bin, 1], 
                           c=colors[i], label=bin_label, alpha=alpha, s=15)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('UMAP - Binned Expression')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        
        plt.tight_layout()
        detailed_save_path = save_path.replace('.png', '_detailed.png')
        plt.savefig(detailed_save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # Return results
    return {
        'umap_embedding': embedding_2d,
        'filtered_embeddings': filtered_embeddings,
        'pca_embeddings': pca_embeddings,
        'filtered_expression': filtered_expression,
        'reducer': reducer,
        'pca': pca,
        'filter_mask': mask
    }