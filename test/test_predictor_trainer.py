"""
Unit tests for PredictorTrainer class.

Tests model training, evaluation, prediction, and normalization functionality.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from experiments.core.predictor_trainer import PredictorTrainer


class TestPredictorTrainer:
    """Test cases for PredictorTrainer class."""

    def test_train_basic(self):
        """Test basic model training."""
        model = LinearRegression()
        trainer = PredictorTrainer(model)

        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        trainer.train(X_train, y_train, [0, 1, 2])

        assert hasattr(model, "coef_")

    def test_train_with_different_models(self):
        """Test training with different model types."""
        models = [
            LinearRegression(),
            RandomForestRegressor(n_estimators=10, random_state=42),
            KNeighborsRegressor(n_neighbors=2),
        ]

        X_train = np.random.randn(10, 5)
        y_train = np.random.randn(10)
        train_indices = list(range(10))

        for model in models:
            trainer = PredictorTrainer(model)
            trainer.train(X_train, y_train, train_indices)
            assert hasattr(model, "predict")

    def test_evaluate_with_test_set(self):
        """Test evaluation on test set."""
        model = LinearRegression()
        trainer = PredictorTrainer(model)

        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        trainer.train(X_train, y_train, [0, 1, 2])

        X_test = np.array([[2, 3], [4, 5]])
        y_test = np.array([1.5, 2.5])
        metrics = trainer.evaluate(X_test, y_test, no_test=False)

        assert "rmse" in metrics
        assert "r2" in metrics
        assert "pearson_correlation" in metrics
        assert "spearman_correlation" in metrics

    def test_evaluate_no_test(self):
        """Test evaluation when no_test=True."""
        trainer = PredictorTrainer(LinearRegression())

        metrics = trainer.evaluate(
            np.array([[2, 3], [4, 5]]), np.array([1.5, 2.5]), no_test=True
        )

        assert metrics == {}

    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        trainer = PredictorTrainer(LinearRegression())

        X_train = np.array([[1], [2], [3]])
        y_train = np.array([2, 4, 6])
        trainer.train(X_train, y_train, [0, 1, 2])

        X_test = np.array([[4], [5]])
        y_test = np.array([8, 10])
        metrics = trainer.evaluate(X_test, y_test, no_test=False)

        assert abs(metrics["r2"] - 1.0) < 1e-6
        assert abs(metrics["rmse"]) < 1e-6

    def test_predict(self):
        """Test making predictions."""
        trainer = PredictorTrainer(LinearRegression())

        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        trainer.train(X_train, y_train, [0, 1, 2])

        predictions = trainer.predict(np.array([[2, 3], [4, 5]]))

        assert predictions.shape == (2,)
        assert isinstance(predictions, np.ndarray)

    def test_train_evaluate_workflow(self):
        """Test complete train-evaluate workflow."""
        trainer = PredictorTrainer(LinearRegression())

        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        y_train = X_train.sum(axis=1) + np.random.randn(50) * 0.1

        X_test = np.random.randn(20, 5)
        y_test = X_test.sum(axis=1) + np.random.randn(20) * 0.1

        trainer.train(X_train, y_train, list(range(50)))
        metrics = trainer.evaluate(X_test, y_test, no_test=False)

        assert metrics["r2"] > 0.5
        assert metrics["rmse"] > 0

    def test_train_with_normalization(self):
        """Ensure feature/target normalization works per round."""
        trainer = PredictorTrainer(
            LinearRegression(), normalize_features=True, normalize_targets=True
        )

        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_train = np.array([10.0, 20.0, 30.0, 40.0])
        trainer.train(X_train, y_train, list(range(len(X_train))))

        preds = trainer.predict(np.array([[5.0], [6.0]]))
        assert np.allclose(preds, np.array([50.0, 60.0]), atol=1e-3)

    def test_evaluate_with_normalization(self):
        """Evaluation should use inverse-transformed predictions."""
        trainer = PredictorTrainer(
            LinearRegression(), normalize_features=True, normalize_targets=True
        )

        X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_train = np.array([0.0, 3.0, 6.0, 9.0])
        trainer.train(X_train, y_train, list(range(len(X_train))))

        X_test = np.array([[4.0], [5.0]])
        y_test = np.array([12.0, 15.0])

        metrics = trainer.evaluate(X_test, y_test, no_test=False)
        assert metrics["rmse"] < 1e-6
        assert abs(metrics["r2"] - 1.0) < 1e-6
