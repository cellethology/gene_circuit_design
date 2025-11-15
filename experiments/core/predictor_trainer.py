"""
Model training and evaluation utilities for active learning experiments.
"""

import logging
from typing import Any, Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, root_mean_squared_error

logger = logging.getLogger(__name__)


class PredictorTrainer:
    """
    Handles model training and evaluation for active learning experiments.
    """

    def __init__(self, predictor: Any) -> None:
        """
        Initialize the predictor trainer.

        Args:
            predictor: Scikit-learn compatible regression predictor
        """
        self.predictor = predictor

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, train_indices: List[int]
    ) -> None:
        """
        Train the model on training data.

        Args:
            X_train: Training features
            y_train: Training targets
            train_indices: Indices of training samples (for logging)
        """
        logger.info(f"Training model with {len(train_indices)} samples")
        logger.info(f"Predictor type: {self.predictor.__class__.__name__}")

        self.predictor.fit(X_train, y_train)

        # Log training performance
        train_pred = self.predictor.predict(X_train)
        train_rmse = root_mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)

        logger.info(f"Training RMSE: {train_rmse:.2f}, R²: {train_r2:.3f}")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        no_test: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test set.

        Args:
            X_test: Test features
            y_test: Test targets
            no_test: Whether to skip evaluation (returns empty dict)

        Returns:
            Dictionary with evaluation metrics (empty if no_test=True)
        """
        if no_test:
            return {}

        y_pred = self.predictor.predict(X_test)

        # Calculate metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(y_test, y_pred)
        spearman_corr, spearman_p = spearmanr(y_test, y_pred)

        metrics = {
            "rmse": rmse,
            "r2": r2,
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
        }

        logger.info(
            f"Test metrics - RMSE: {rmse:.2f}, R²: {r2:.3f}, "
            f"Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}"
        )

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Features to predict on

        Returns:
            Predictions array
        """
        return self.predictor.predict(X)
