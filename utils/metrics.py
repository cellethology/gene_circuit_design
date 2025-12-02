"""
Utility functions for calculating metrics.
"""

import numpy as np


def proportion_of_selected_indices_in_top_labels(
    selected_indices: np.ndarray, labels: np.ndarray, top_percentage: float = 0.1
) -> float:
    # Get indices of top 10% values
    num_top = int(len(labels) * top_percentage)
    top_labels = np.argsort(labels)[-num_top:]

    # Calculate ratio of intersection size to total size
    intersection_ratio = len(np.intersect1d(selected_indices, top_labels)) / len(
        selected_indices
    )
    return intersection_ratio
