"""
Utility functions for calculating metrics.
"""

import numpy as np

def normalized_to_best_val_metric(y_pred, all_y_true):
    """
    Normalize the predicted values to the best possible value in the sequence pool.
    """
    # Best in the sequence pool -> this should be a constant value
    highest_expression_val = np.max(all_y_true)
    highest_y_pred = np.max(y_pred)
    return highest_y_pred / highest_expression_val


def top_10_ratio_intersection_metric(y_pred_indices, all_y_true):
    """
    Calculate the intersection ratio between top 10 predicted and true indices.

    Args:
        y_pred_indices: Array of predicted values
        all_y_true: Array of true values

    Returns:
        float: Ratio of overlapping indices between top 10 predicted and true values
    """
    # Get indices of top 10% values
    num_top_precent = 0.3  # Take top 10% values
    num_top = int(len(all_y_true) * num_top_precent) # rounded down to the nearest integer
    # top_pred_indices = np.argsort(y_pred_indices)[-num_top:]
    top_true_indices = np.argsort(all_y_true)[-num_top:]

    # Find intersection
    intersection = np.intersect1d(y_pred_indices, top_true_indices)

    # Calculate ratio of intersection size to total size
    intersection_ratio = len(intersection) / num_top
    return intersection_ratio

def get_best_value_metric(y_pred):
    """
    Get the best value the model picked.
    """
    return np.max(y_pred)
