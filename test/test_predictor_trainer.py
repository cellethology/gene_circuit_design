"""
Unit tests for PredictorTrainer class.

Tests model training, prediction, and normalization functionality.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from experiments.core.predictor_trainer import PredictorTrainer


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

    def test_train_with_normalization(self):
        """Ensure feature/target normalization works per round."""
        trainer = PredictorTrainer(
            LinearRegression(), normalize_features=True, normalize_labels=True
        )

        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_train = np.array([10.0, 20.0, 30.0, 40.0])
        trainer.train(X_train, y_train)

        preds = trainer.predict(np.array([[5.0], [6.0]]))
        assert np.allclose(preds, np.array([50.0, 60.0]), atol=1e-3)
