import logging
from typing import List

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def encode_sequences(
    indices: List[int],
    embeddings: np.ndarray,
) -> np.ndarray:
    """Encode sequences via pretrained embeddings."""
    return embeddings[indices]


def select_initial_batch_kmeans_from_features(
    X_all: np.ndarray, initial_sample_size: int, seed: int
) -> List[int]:
    """
    Select initial indices using K-means on provided feature matrix.

    Args:
        X_all: Feature matrix for the full dataset.
        initial_sample_size: Number of clusters/initial selections.
        seed: Random seed for KMeans reproducibility.

    Returns:
        List of selected global indices.
    """
    kmeans = KMeans(n_clusters=initial_sample_size, random_state=seed)
    cluster_labels = kmeans.fit_predict(X_all)
    cluster_centers = kmeans.cluster_centers_

    selected_indices: List[int] = []
    for cluster_id in range(initial_sample_size):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_points = X_all[cluster_indices]
        cluster_center = cluster_centers[cluster_id]
        distances_to_center = np.linalg.norm(cluster_points - cluster_center, axis=1)
        closest_idx_in_cluster = int(np.argmin(distances_to_center))
        selected_indices.append(int(cluster_indices[closest_idx_in_cluster]))

    return selected_indices
