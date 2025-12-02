"""
Model training and evaluation utilities for active learning experiments.
"""

import logging
from typing import Any, Optional

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PredictorTrainer:
    """
    Handles model training and evaluation for active learning experiments.
    """

    def __init__(
        self,
        predictor: RegressorMixin,
        normalize_features: bool = False,
        normalize_labels: bool = False,
    ) -> None:
        """
        Initialize the predictor trainer.

        Args:
            predictor: Scikit-learn compatible regression predictor
            normalize_features: Whether to standardize features each round
            normalize_labels: Whether to standardize labels each round
        """
        self.base_predictor = predictor
        self.normalize_features = normalize_features
        self.normalize_labels = normalize_labels
        self.model_: Optional[Any] = None

        logger.info(
            f"PredictorTrainer initialized with normalize_features={normalize_features} and normalize_labels={normalize_labels}"
        )

    def _build_estimator(self) -> Any:
        """Create a fresh estimator (Pipeline + optional target transformer)."""
        steps = []
        if self.normalize_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(("estimator", clone(self.base_predictor)))

        if len(steps) == 1 and not self.normalize_features:
            estimator: Any = steps[0][1]
        else:
            estimator = Pipeline(steps)

        if self.normalize_labels:
            estimator = TransformedTargetRegressor(
                regressor=estimator,
                transformer=StandardScaler(),
            )

        return estimator

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train a fresh model on training data, re-normalizing features/labels if specified.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        if len(X_train) == 0:
            raise ValueError("Training requires at least one sample.")

        logger.info(f"Total training samples: {len(X_train)}")

        estimator = self._build_estimator()
        estimator.fit(X_train, y_train)
        self.model_ = estimator

        train_pred = self.model_.predict(X_train)
        train_rmse = root_mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)

        logger.info(f"Train RMSE: {train_rmse:.2f}, R²: {train_r2:.3f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Features to predict on

        Returns:
            Predictions array
        """
        if self.model_ is None:
            raise ValueError("PredictorTrainer.train must be called before predict.")
        return self.model_.predict(X)

    def get_model(self) -> Optional[Any]:
        """Return the underlying fitted model (pipeline or estimator)."""
        return self.model_
