"""
Configuration for model setup for gene circuit design
"""

from enum import Enum

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


# NOTE: Can the Enum class be directly referring to the model
class RegressionModel(str, Enum):
    """Enumeration of available selection strategies."""

    KNN = "KNN_regression"
    LINEAR = "linear_regression"
    RANDOM_FOREST = "random_forest"


def return_model(model: str, random_state: int = 42):
    """Given the input model type return the functional class"""
    if model == RegressionModel.LINEAR:
        return LinearRegression()
    elif model == RegressionModel.RANDOM_FOREST:
        return RandomForestRegressor(random_state=random_state)
    elif model == RegressionModel.KNN:
        # KNN doesn't have random_state, but we can set n_neighbors for consistency
        return KNeighborsRegressor(n_neighbors=5)
