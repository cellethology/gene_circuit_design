"""
Configuration for model setup for gene circuit design
"""

from enum import Enum

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


# NOTE: Can the Enum class be directly referring to the model
class RegressionModelType(str, Enum):
    """Enumeration of available selection strategies."""

    KNN = "KNN_regression"
    LINEAR = "linear_regression"
    RANDOM_FOREST = "random_forest"


def return_model(model: str, random_state: int = 42):
    """Given the input model type return the functional class"""
    if model == RegressionModelType.LINEAR:
        return LinearRegression()
    elif model == RegressionModelType.RANDOM_FOREST:
        # TODO: n_jobs is how many cores it is using, -1 means all cores
        return RandomForestRegressor(random_state=random_state, n_jobs=-1)
    elif model == RegressionModelType.KNN:
        # KNN doesn't have random_state, but we can set n_neighbors for consistency
        return KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
