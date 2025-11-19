from typing import Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.decomposition import PCA


def pca_down(
    in_path: str,
    out_path: str,
    target_var: float = 0.90,
    subset_size: int = 20000,
    arpack_threshold: int = 256,  # if k <= this, prefer arpack; else randomized
) -> Tuple[int, float]:
    """
    Perform PCA down-projection on embeddings stored in a safetensors file
    and save the result as a compressed NPZ:

        np.savez_compressed(out_path, ids=ids, embeddings=embeddings)

    where:
        - ids: taken from the 'ids' tensor in the input file
        - embeddings: PCA-reduced embeddings as float32

    Args:
        in_path: Path to input .safetensors file containing 'embeddings' and 'ids'.
        out_path: Path to output .npz file.
        target_var: Target cumulative explained variance (currently unused if k fixed).
        subset_size: Number of samples used for pilot estimation (if using _estimate_k90).
        arpack_threshold: Threshold on k for choosing between 'arpack' and 'randomized' solvers.

    Returns:
        Tuple of (k90, var_sum):
            k90: Number of principal components used.
            var_sum: Sum of explained variance ratios for the retained components.
    """
    # --- Load ---
    data = load_file(in_path)
    emb: torch.Tensor = data["embeddings"].float()  # [N, ...]
    if "ids" not in data:
        raise KeyError("Input safetensors file must contain an 'ids' tensor.")
    ids = data["ids"]

    # --- Flatten [N, ...] -> [N, p] ---
    N = emb.shape[0]
    X = emb.contiguous().view(N, -1).numpy()

    # --- Optionally estimate k via pilot subset ---
    # rng = check_random_state(42)
    # idx = rng.choice(N, size=min(subset_size, N), replace=False)
    # X_sub = X[idx]
    # k90 = _estimate_k90(X_sub, target_var=target_var)
    # print(f"[Pilot] Estimated k for >={target_var:.0%} variance: {k90}")

    k90 = 512

    # --- Final PCA on full data ---
    solver = "arpack" if k90 <= arpack_threshold else "randomized"
    print(f"[Final] Using svd_solver='{solver}'")

    pca = PCA(n_components=k90, svd_solver=solver, random_state=42)
    X_pca = pca.fit_transform(X)  # shape: [N, k90]

    var_sum = float(pca.explained_variance_ratio_.sum())
    print(f"Original shape: {X.shape} -> Reduced shape: {X_pca.shape}")
    print(f"Explained variance (sum): {var_sum:.4f}")

    # --- Prepare arrays for NPZ ---
    # embeddings: float32 PCA representation
    embeddings = X_pca.astype(np.float32, copy=False)

    # ids: convert to numpy
    if isinstance(ids, torch.Tensor):
        ids_np = ids.cpu().numpy()
    else:
        ids_np = np.asarray(ids)

    # --- Save NPZ ---
    np.savez_compressed(out_path, ids=ids_np, embeddings=embeddings)
    print(f"Saved PCA-reduced embeddings NPZ to: {out_path}")

    return k90, var_sum
