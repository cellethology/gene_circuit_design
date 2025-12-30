"""
Model training and evaluation utilities for active learning experiments.
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from core.uncertainty import UncertaintyWrapper

logger = logging.getLogger(__name__)


class PredictorTrainer:
    """
    Handles model training and evaluation for active learning experiments.
    """

    def __init__(
        self,
        predictor: RegressorMixin,
        feature_transform: Optional[List[Tuple[str, Any]]] = None,
        target_transform: Optional[List[Tuple[str, Any]]] = None,
    ) -> None:
        """
        Initialize the predictor trainer.

        Args:
            predictor: Scikit-learn compatible regression predictor
            feature_transform: List of (name, transformer) steps to apply to the *features*
            target_transform: List of (name, transformer) steps to apply to the *targets*
        """
        self.base_predictor = predictor
        self.feature_transform = feature_transform
        self.target_transform = target_transform
        self.model_: Optional[Any] = None

        if feature_transform:
            logger.info(
                f"PredictorTrainer initialized with feature_transform={self.feature_transform}"
            )
        if target_transform:
            logger.info(
                f"PredictorTrainer initialized with target_transform={self.target_transform}"
            )

    def _build_estimator(
        self,
        feature_transform: Optional[List[Tuple[str, Any]]] = None,
        target_transform: Optional[List[Tuple[str, Any]]] = None,
    ) -> Any:
        """
        Create a fresh estimator with optional feature and target transformers.

        Parameters
        ----------
        feature_transform :
            List of (name, transformer) steps to apply to the *features*
            before the base predictor, e.g.
            [("scaler", StandardScaler()), ("pca", PCA())].

        target_transform :
            List of (name, transformer) steps to apply to the *targets*
            via a Pipeline wrapped in TransformedTargetRegressor, e.g.
            [("log", FunctionTransformer(np.log1p, np.expm1))].
        """
        feature_transform = feature_transform or []
        target_transform = target_transform or []

        if feature_transform:
            pipeline_steps = feature_transform + [
                ("estimator", clone(self.base_predictor))
            ]
            estimator: Any = Pipeline(pipeline_steps)
        else:
            estimator = clone(self.base_predictor)

        if target_transform:
            y_pipeline = Pipeline(target_transform)

            estimator = TransformedTargetRegressor(
                regressor=estimator,
                transformer=y_pipeline,
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

        estimator = self._build_estimator(
            feature_transform=self.feature_transform,
            target_transform=self.target_transform,
        )
        estimator.fit(X_train, y_train)
        self.model_ = estimator

    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        Make predictions using the trained model.
        """
        if self.model_ is None:
            raise ValueError("PredictorTrainer.train must be called before predict.")

        preds = self.model_.predict(X)

        if not return_std:
            return preds

        stds = UncertaintyWrapper(self.model_).compute_std(X)
        return preds, stds

    def get_model(self) -> Optional[Any]:
        """Return the underlying fitted model (pipeline or estimator)."""
        return self.model_
