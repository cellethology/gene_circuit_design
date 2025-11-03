from typing import List
import numpy as np
from sklearn.cluster import KMeans
import logging


def encode_sequences(self, indices: List[int]) -> np.ndarray:
    """
    Encode sequences at given indices using pre-computed embeddings or one-hot encoding.

    Args:
        indices: List of sequence indices to encode

    Returns:
        Encoded sequences (either pre-computed embeddings or flattened one-hot encoded sequences)
    """
    if self.embeddings is not None:
        # Use pre-computed embeddings from safetensors file
        return self.embeddings[indices]
    else:
        # Fall back to one-hot encoding for CSV files
        # For DNA data, all_sequences is a list of strings
        sequences = [self.all_sequences[i] for i in indices]

        encoded = one_hot_encode_sequences(sequences, self.seq_mod_method)

        # Apply PCA if specified
        if self.use_pca:
            logger.info(f"PCA enabled with {self.pca_components} components")
            return flatten_one_hot_sequences_with_pca(
                encoded, n_components=self.pca_components
            )
        else:
            return flatten_one_hot_sequences(encoded)


def select_initial_batch_kmeans_clustering(
    all_sequences: List[str],
    all_expressions: List[float],
    initial_sample_size: int,
    random_seed: int,
) -> List[int]:
    """
    Select initial batch using K-means clustering on the whole dataset.

    Steps:
    1. Use K-means clustering on the whole dataset
    2. Set the centroid number to be the same as the initial sample size
    3. Select the data point in each cluster which is closest to that cluster's centroid

    Returns:
        List of indices for initial batch
    """
    # Set up logger if not already present
    logger = logging.getLogger(__name__)

    # Encode all sequences to get feature representations
    all_indices = list(range(len(all_sequences)))
    X_all = encode_sequences(all_indices)

    # Apply K-means clustering with k = initial_sample_size
    kmeans = KMeans(
        n_clusters=initial_sample_size,
        random_state=random_seed,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(X_all)
    cluster_centers = kmeans.cluster_centers_

    selected_indices = []

    # For each cluster, find the point closest to the cluster centroid
    for cluster_id in range(initial_sample_size):
        # Get indices of points in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            # If cluster is empty (shouldn't happen with proper k-means), skip
            continue

        # Get feature vectors for points in this cluster
        cluster_points = X_all[cluster_indices]
        cluster_center = cluster_centers[cluster_id]

        # Calculate distances from each point in cluster to cluster centroid
        distances_to_center = np.linalg.norm(cluster_points - cluster_center, axis=1)

        # Find the point in this cluster closest to cluster centroid
        closest_idx_in_cluster = np.argmin(distances_to_center)
        closest_global_idx = cluster_indices[closest_idx_in_cluster]

        selected_indices.append(closest_global_idx)

    # Log selection info
    selected_expressions = all_expressions[selected_indices]
    logger.info(
        f"KMEANS_CLUSTERING: Selected {len(selected_indices)} sequences "
        f"from {initial_sample_size} clusters with actual expressions: "
        f"[{', '.join(f'{expr:.1f}' for expr in selected_expressions)}]"
    )

    return selected_indices
