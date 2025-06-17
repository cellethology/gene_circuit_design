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
    return y_pred / highest_expression_val


def top_10_ratio_intersection_metric(y_pred, all_y_true):
    """
    Calculate the ratio of the top 10% of predicted values to the top 10% of true values.
    """
    # Get the top 10% of predicted values
    top_10_pred = np.sort(y_pred)[-10:]

    # top_10_true should always be the same
    top_10_true = np.sort(all_y_true)[-10:]
    return np.mean(top_10_pred) / np.mean(top_10_true)

def get_best_value_metric(y_pred):
    """
    Get the best value the model picked.
    """
    return np.max(y_pred)
