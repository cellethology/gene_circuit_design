"""
Unit tests for ModelTrainer class.

Tests model training, evaluation, and prediction functionality.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from experiments.core.model_trainer import ModelTrainer


class TestModelTrainer:
    """Test cases for ModelTrainer class."""

    def test_train_basic(self):
        """Test basic model training."""
        model = LinearRegression()
        trainer = ModelTrainer(model)

        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        train_indices = [0, 1, 2]

        trainer.train(X_train, y_train, train_indices)

        # Model should be trained
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
            trainer = ModelTrainer(model)
            trainer.train(X_train, y_train, train_indices)

            # All models should be trainable
            assert hasattr(model, "predict")

    def test_evaluate_with_test_set(self):
        """Test evaluation on test set."""
        model = LinearRegression()
        trainer = ModelTrainer(model)

        # Train model
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        trainer.train(X_train, y_train, [0, 1, 2])

        # Evaluate
        X_test = np.array([[2, 3], [4, 5]])
        y_test = np.array([1.5, 2.5])
        metrics = trainer.evaluate(X_test, y_test, no_test=False)

        # Check metrics are present
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "pearson_correlation" in metrics
        assert "spearman_correlation" in metrics
        assert "pearson_p_value" in metrics
        assert "spearman_p_value" in metrics

        # Check metric types
        assert isinstance(metrics["rmse"], (float, np.floating))
        assert isinstance(metrics["r2"], (float, np.floating))
        assert isinstance(metrics["pearson_correlation"], (float, np.floating))

    def test_evaluate_no_test(self):
        """Test evaluation when no_test=True."""
        model = LinearRegression()
        trainer = ModelTrainer(model)

        X_test = np.array([[2, 3], [4, 5]])
        y_test = np.array([1.5, 2.5])
        metrics = trainer.evaluate(X_test, y_test, no_test=True)

        # Should return empty dict
        assert metrics == {}

    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        model = LinearRegression()
        trainer = ModelTrainer(model)

        # Train on simple linear relationship
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([2, 4, 6])  # y = 2x
        trainer.train(X_train, y_train, [0, 1, 2])

        # Test on same relationship
        X_test = np.array([[4], [5]])
        y_test = np.array([8, 10])  # y = 2x
        metrics = trainer.evaluate(X_test, y_test, no_test=False)

        # Should have perfect R²
        assert abs(metrics["r2"] - 1.0) < 1e-6
        assert abs(metrics["rmse"]) < 1e-6
        assert abs(metrics["pearson_correlation"] - 1.0) < 1e-6

    def test_predict(self):
        """Test making predictions."""
        model = LinearRegression()
        trainer = ModelTrainer(model)

        # Train model
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        trainer.train(X_train, y_train, [0, 1, 2])

        # Make predictions
        X_new = np.array([[2, 3], [4, 5]])
        predictions = trainer.predict(X_new)

        assert predictions.shape == (2,)
        assert isinstance(predictions, np.ndarray)

    def test_train_evaluate_workflow(self):
        """Test complete train-evaluate workflow."""
        model = LinearRegression()
        trainer = ModelTrainer(model)

        # Generate synthetic data
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        y_train = X_train.sum(axis=1) + np.random.randn(50) * 0.1

        X_test = np.random.randn(20, 5)
        y_test = X_test.sum(axis=1) + np.random.randn(20) * 0.1

        # Train
        trainer.train(X_train, y_train, list(range(50)))

        # Evaluate
        metrics = trainer.evaluate(X_test, y_test, no_test=False)

        # Should have reasonable performance
        assert metrics["r2"] > 0.5  # Should fit reasonably well
        assert metrics["rmse"] > 0  # Should have some error
        assert abs(metrics["pearson_correlation"]) > 0.5
