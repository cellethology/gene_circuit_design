import logging
from typing import List, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans

from utils.sequence_utils import (
    SequenceModificationMethod,
    flatten_one_hot_sequences,
    flatten_one_hot_sequences_with_pca,
    one_hot_encode_sequences,
)

logger = logging.getLogger(__name__)


def encode_sequences(
    indices: List[int],
    all_sequences: Sequence[str],
    embeddings: Optional[np.ndarray],
    seq_mod_method: SequenceModificationMethod,
    use_pca: bool = False,
    pca_components: int = 4096,
) -> np.ndarray:
    """
    Encode sequences at given indices using pre-computed embeddings or one-hot encoding.

    Args:
        indices: Indices of sequences to encode.
        all_sequences: Collection of raw sequences (strings) aligned to indices.
        embeddings: Precomputed embedding matrix aligned to indices, or None.
        seq_mod_method: Strategy for sequence processing/encoding.
        use_pca: Whether to reduce one-hot features with PCA.
        pca_components: Number of PCA components if PCA is used.

    Returns:
        Feature matrix for the selected indices.
    """
    if embeddings is not None:
        return embeddings[indices]

    sequences = [all_sequences[i] for i in indices]
    encoded = one_hot_encode_sequences(sequences, seq_mod_method)
    if use_pca:
        logger.info(f"PCA enabled with {pca_components} components")
        return flatten_one_hot_sequences_with_pca(encoded, n_components=pca_components)
    return flatten_one_hot_sequences(encoded)


def select_initial_batch_kmeans_from_features(
    X_all: np.ndarray, initial_sample_size: int, random_seed: int
) -> List[int]:
    """
    Select initial indices using K-means on provided feature matrix.

    Args:
        X_all: Feature matrix for the full dataset.
        initial_sample_size: Number of clusters/initial selections.
        random_seed: Random seed for KMeans reproducibility.

    Returns:
        List of selected global indices.
    """
    kmeans = KMeans(n_clusters=initial_sample_size, random_state=random_seed, n_init=10)
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
