"""
Unit tests for PredictorTrainer class.

Tests model training, prediction, and normalization functionality.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from core.predictor_trainer import PredictorTrainer


class VarianceAwareRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y, y_var=None):
        self.y_mean_ = float(np.mean(y))
        self.y_var_ = None if y_var is None else np.asarray(y_var, dtype=float)
        return self

    def predict(self, X):
        return np.full(len(X), self.y_mean_)


class TestPredictorTrainer:
    """Test cases for PredictorTrainer class."""

    def test_train_basic(self):
        """Training produces a fitted estimator stored on the trainer."""
        model = LinearRegression()
        trainer = PredictorTrainer(model)

        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        trainer.train(X_train, y_train)

        trained_model = trainer.get_model()
        assert trained_model is not None
        preds = trainer.predict(X_train)
        assert preds.shape == (3,)

    def test_train_with_different_models(self):
        """Test training with different model types."""
        models = [
            LinearRegression(),
            RandomForestRegressor(n_estimators=10, random_state=42),
            KNeighborsRegressor(n_neighbors=2),
        ]

        X_train = np.random.randn(10, 5)
        y_train = np.random.randn(10)
        for model in models:
            trainer = PredictorTrainer(model)
            trainer.train(X_train, y_train)
            trained_model = trainer.get_model()
            assert trained_model is not None
            preds = trainer.predict(X_train)
            assert preds.shape == (10,)

    def test_predict(self):
        """Test making predictions."""
        trainer = PredictorTrainer(LinearRegression())

        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        trainer.train(X_train, y_train)

        predictions = trainer.predict(np.array([[2, 3], [4, 5]]))

        assert predictions.shape == (2,)
        assert isinstance(predictions, np.ndarray)

    def test_feature_and_target_transforms(self):
        """Feature pipeline and target transformer are applied correctly."""
        feature_steps = [("scaler", StandardScaler())]
        target_steps = [
            (
                "log",
                FunctionTransformer(np.log1p, np.expm1),
            )
        ]
        trainer = PredictorTrainer(
            LinearRegression(),
            feature_transform=feature_steps,
            target_transform=target_steps,
        )

        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_train = np.array([10.0, 20.0, 30.0, 40.0])
        trainer.train(X_train, y_train)

        preds = trainer.predict(np.array([[5.0], [6.0]]))
        assert preds.shape == (2,)

    def test_train_passes_observation_variance_to_supported_estimator(self):
        trainer = PredictorTrainer(VarianceAwareRegressor())

        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([1.0, 2.0, 3.0])
        y_var_train = np.array([0.1, 0.2, 0.3])
        trainer.train(X_train, y_train, y_var_train=y_var_train)

        model = trainer.get_model()
        np.testing.assert_allclose(model.y_var_, y_var_train)

    def test_train_transforms_observation_variance_with_target_transform(self):
        target_steps = [
            ("log", FunctionTransformer(np.log1p, np.expm1)),
            ("scaler", StandardScaler()),
        ]
        trainer = PredictorTrainer(
            VarianceAwareRegressor(),
            target_transform=target_steps,
        )

        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([1.0, 2.0, 3.0])
        y_var_train = np.array([0.1, 0.1, 0.1])
        trainer.train(X_train, y_train, y_var_train=y_var_train)

        model = trainer.get_model()
        log_scale = np.std(np.log1p(y_train), ddof=0)
        expected = y_var_train / ((1.0 + y_train) ** 2 * log_scale**2)
        np.testing.assert_allclose(model.regressor_.y_var_, expected, rtol=1e-9)
