"""
Utility functions for calculating metrics.
"""

import numpy as np

def normalized_to_best(y_pred, all_y_true):
    """
    Normalize the predicted values to the best possible value.
    """
    # Best in the sequence pool -> this should be a constant value
    highest_expression_val = np.max(all_y_true)
    return y_pred / highest_expression_val
