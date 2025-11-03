"""
Tests for reproducibility and deterministic behavior.

This module tests that the active learning pipeline produces consistent,
reproducible results across runs with the same parameters:
- Random seed consistency
- Model training determinism
- Data processing consistency
- Cross-validation reproducibility
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from utils.metrics import (
    get_best_value_metric,
    normalized_to_best_val_metric,
    top_10_ratio_intersected_indices_metric,
)
from utils.model_loader import RegressionModelType, return_model
from utils.sequence_utils import (
    SequenceModificationMethod,
    flatten_one_hot_sequences,
    flatten_one_hot_sequences_with_pca,
    load_sequence_data,
    one_hot_encode_sequences,
)


class TestRandomSeedConsistency:
    """Test that random seed settings produce consistent results."""

    def test_numpy_random_seed_consistency(self):
        """Test that numpy random operations are reproducible."""
        # Set seed and generate random data
        np.random.seed(42)
        data1 = np.random.uniform(0, 1, 100)

        # Reset seed and generate again
        np.random.seed(42)
        data2 = np.random.uniform(0, 1, 100)

        # Should be identical
        assert np.array_equal(data1, data2)

    def test_torch_random_seed_consistency(self):
        """Test that PyTorch random operations are reproducible."""
        # Set seed and generate random tensor
        torch.manual_seed(42)
        tensor1 = torch.randn(10, 10)

        # Reset seed and generate again
        torch.manual_seed(42)
        tensor2 = torch.randn(10, 10)

        # Should be identical
        assert torch.equal(tensor1, tensor2)

    def test_sklearn_random_seed_consistency(self):
        """Test that sklearn operations are reproducible."""
        X = np.random.RandomState(42).uniform(0, 1, (100, 4))
        y = np.random.RandomState(42).uniform(0, 1, 100)

        # Split with same random state
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Should be identical
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(X_test1, X_test2)
        assert np.array_equal(y_train1, y_train2)
        assert np.array_equal(y_test1, y_test2)

    def test_combined_seed_consistency(self):
        """Test consistency when multiple random generators are used."""

        def generate_mixed_random_data(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Mix of numpy and torch operations
            numpy_data = np.random.uniform(0, 1, 50)
            torch_data = torch.randn(50).numpy()

            return numpy_data, torch_data

        # Generate data twice with same seed
        np_data1, torch_data1 = generate_mixed_random_data(123)
        np_data2, torch_data2 = generate_mixed_random_data(123)

        assert np.array_equal(np_data1, np_data2)
        assert np.array_equal(torch_data1, torch_data2)


class TestSequenceProcessingDeterminism:
    """Test that sequence processing operations are deterministic."""

    def test_one_hot_encoding_determinism(self):
        """Test that one-hot encoding produces identical results."""
        sequences = ["ATCG", "GCTA", "TTAA", "CCGG"]

        # Encode multiple times
        results1 = torch.tensor(
            np.vstack(
                one_hot_encode_sequences(
                    sequences, seq_mod_method=SequenceModificationMethod.EMBEDDING
                )
            ),
            dtype=torch.float32,
        )
        results2 = torch.tensor(
            np.vstack(
                one_hot_encode_sequences(
                    sequences, seq_mod_method=SequenceModificationMethod.EMBEDDING
                )
            ),
            dtype=torch.float32,
        )
        results3 = torch.tensor(
            np.vstack(
                one_hot_encode_sequences(
                    sequences, seq_mod_method=SequenceModificationMethod.EMBEDDING
                )
            ),
            dtype=torch.float32,
        )

        # All results should be identical
        for r1, r2, r3 in zip(results1, results2, results3):
            assert torch.equal(r1, r2)
            assert torch.equal(r2, r3)

    def test_flattening_determinism(self):
        """Test that sequence flattening is deterministic."""
        sequences = ["ATCG"] * 10
        encoded = one_hot_encode_sequences(
            sequences, seq_mod_method=SequenceModificationMethod.EMBEDDING
        )

        # Flatten multiple times
        flat1 = torch.tensor(flatten_one_hot_sequences(encoded), dtype=torch.float32)
        flat2 = torch.tensor(flatten_one_hot_sequences(encoded), dtype=torch.float32)
        flat3 = torch.tensor(flatten_one_hot_sequences(encoded), dtype=torch.float32)

        assert torch.equal(flat1, flat2)
        assert torch.equal(flat2, flat3)

    def test_pca_determinism_with_seed(self):
        """Test that PCA is deterministic when random_state is set."""
        # Create sample data
        data = torch.tensor(
            np.random.RandomState(42).uniform(0, 1, (100, 20)), dtype=torch.float32
        )

        # Apply PCA multiple times
        reduced1 = torch.tensor(
            flatten_one_hot_sequences_with_pca(data, n_components=5),
            dtype=torch.float32,
        )
        reduced2 = torch.tensor(
            flatten_one_hot_sequences_with_pca(data, n_components=5),
            dtype=torch.float32,
        )

        # Results should be very close (allowing for numerical precision)
        assert torch.allclose(reduced1, reduced2, atol=1e-10)

    def test_sequence_order_independence(self):
        """Test that processing doesn't depend on sequence order."""
        sequences1 = ["ATCG", "GCTA", "TTAA"]
        sequences2 = ["GCTA", "ATCG", "TTAA"]  # Different order

        results1 = torch.tensor(
            one_hot_encode_sequences(
                sequences1, seq_mod_method=SequenceModificationMethod.PAD
            ),
            dtype=torch.float32,
        )
        results2 = torch.tensor(
            one_hot_encode_sequences(
                sequences2, seq_mod_method=SequenceModificationMethod.PAD
            ),
            dtype=torch.float32,
        )

        # Results should be deterministic for each sequence individually
        # ATCG should always encode the same way
        atcg_result1 = results1[0]
        atcg_result2 = results2[1]

        assert torch.equal(atcg_result1, atcg_result2)


class TestModelTrainingDeterminism:
    """Test that model training produces consistent results."""

    def test_linear_regression_determinism(self):
        """Test that linear regression training is deterministic."""
        # Generate consistent training data
        X = torch.tensor(
            np.random.RandomState(42).uniform(0, 1, (100, 5)), dtype=torch.float32
        )
        y = torch.tensor(
            np.random.RandomState(42).uniform(0, 1, 100), dtype=torch.float32
        )

        # Train model multiple times
        model1 = return_model(RegressionModelType.LINEAR, random_state=42)
        model1.fit(X, y)

        model2 = return_model(RegressionModelType.LINEAR, random_state=42)
        model2.fit(X, y)

        # Predictions should be identical
        test_X = np.random.RandomState(123).uniform(0, 1, (10, 5))
        pred1 = model1.predict(test_X)
        pred2 = model2.predict(test_X)

        assert np.allclose(pred1, pred2, atol=1e-10)

    def test_random_forest_determinism(self):
        """Test that random forest training is deterministic with random_state."""
        X = np.random.RandomState(42).uniform(0, 1, (100, 5))
        y = np.random.RandomState(42).uniform(0, 1, 100)

        # Train model multiple times with same random_state
        model1 = return_model(RegressionModelType.RANDOM_FOREST, random_state=42)
        model1.fit(X, y)

        model2 = return_model(RegressionModelType.RANDOM_FOREST, random_state=42)
        model2.fit(X, y)

        # Predictions should be identical
        test_X = np.random.RandomState(123).uniform(0, 1, (10, 5))
        pred1 = model1.predict(test_X)
        pred2 = model2.predict(test_X)

        assert np.allclose(pred1, pred2, atol=1e-10)

    def test_knn_determinism(self):
        """Test that KNN training is deterministic."""
        X = np.random.RandomState(42).uniform(0, 1, (100, 5))
        y = np.random.RandomState(42).uniform(0, 1, 100)

        # Train model multiple times
        model1 = return_model(RegressionModelType.KNN, random_state=42)
        model1.fit(X, y)

        model2 = return_model(RegressionModelType.KNN, random_state=42)
        model2.fit(X, y)

        # Predictions should be identical
        test_X = np.random.RandomState(123).uniform(0, 1, (10, 5))
        pred1 = model1.predict(test_X)
        pred2 = model2.predict(test_X)

        assert np.allclose(pred1, pred2, atol=1e-10)

    def test_model_parameter_consistency(self):
        """Test that model parameters are consistent across instantiations."""
        # Create multiple instances of the same model type
        model1 = return_model(RegressionModelType.RANDOM_FOREST, random_state=42)
        model2 = return_model(RegressionModelType.RANDOM_FOREST, random_state=42)

        # Parameters should be identical
        assert model1.get_params() == model2.get_params()


class TestMetricsDeterminism:
    """Test that metrics calculations are deterministic."""

    def test_normalized_metric_determinism(self):
        """Test that normalized metric calculation is deterministic."""
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        all_y_true = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        # Calculate metric multiple times
        result1 = normalized_to_best_val_metric(y_pred, all_y_true)
        result2 = normalized_to_best_val_metric(y_pred, all_y_true)
        result3 = normalized_to_best_val_metric(y_pred, all_y_true)

        assert result1 == result2 == result3

    def test_top_ratio_metric_determinism(self):
        """Test that top ratio metric calculation is deterministic."""
        all_y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_pred_indices = np.array([8, 9])

        # Calculate metric multiple times
        result1 = top_10_ratio_intersected_indices_metric(y_pred_indices, all_y_true)
        result2 = top_10_ratio_intersected_indices_metric(y_pred_indices, all_y_true)
        result3 = top_10_ratio_intersected_indices_metric(y_pred_indices, all_y_true)

        assert result1 == result2 == result3

    def test_best_value_metric_determinism(self):
        """Test that best value metric calculation is deterministic."""
        y_pred = np.array([1.0, 5.0, 3.0, 2.0, 4.0])

        # Calculate metric multiple times
        result1 = get_best_value_metric(y_pred)
        result2 = get_best_value_metric(y_pred)
        result3 = get_best_value_metric(y_pred)

        assert result1 == result2 == result3

    def test_metrics_with_identical_inputs(self):
        """Test metrics consistency with identical but separately created inputs."""
        # Create identical arrays separately
        y_pred1 = np.array([1.0, 2.0, 3.0])
        y_pred2 = np.array([1.0, 2.0, 3.0])

        all_y_true1 = np.array([1.5, 2.5, 3.5])
        all_y_true2 = np.array([1.5, 2.5, 3.5])

        # Metrics should be identical
        norm1 = normalized_to_best_val_metric(y_pred1, all_y_true1)
        norm2 = normalized_to_best_val_metric(y_pred2, all_y_true2)

        best1 = get_best_value_metric(y_pred1)
        best2 = get_best_value_metric(y_pred2)

        assert norm1 == norm2
        assert best1 == best2


class TestDataLoadingConsistency:
    """Test that data loading produces consistent results."""

    def test_csv_loading_consistency(self):
        """Test that CSV loading is consistent across multiple reads."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Sequence,Expression\n")
            f.write("ATCG,1.5\n")
            f.write("GCTA,2.3\n")
            f.write("TTAA,0.8\n")
            temp_path = f.name

        try:
            # Load data multiple times
            data1 = load_sequence_data(
                temp_path, seq_mod_method=SequenceModificationMethod.PAD
            )
            data2 = load_sequence_data(
                temp_path, seq_mod_method=SequenceModificationMethod.PAD
            )
            data3 = load_sequence_data(
                temp_path, seq_mod_method=SequenceModificationMethod.PAD
            )

            # All results should be identical
            assert np.array_equal(data1, data2)
            assert np.array_equal(data2, data3)

        finally:
            Path(temp_path).unlink()

    def test_data_processing_pipeline_consistency(self):
        """Test that entire data processing pipeline is consistent."""
        # Create test data
        sequences = ["ATCG", "GCTA", "TTAA", "CCGG"]

        def process_sequences(seqs):
            """Complete processing pipeline."""
            encoded = one_hot_encode_sequences(
                seqs, seq_mod_method=SequenceModificationMethod.PAD
            )
            flattened = flatten_one_hot_sequences(encoded)
            return flattened

        # Process multiple times
        result1 = process_sequences(sequences)
        result2 = process_sequences(sequences)
        result3 = process_sequences(sequences)

        assert np.array_equal(result1, result2)
        assert np.array_equal(result2, result3)


class TestCrossValidationReproducibility:
    """Test reproducibility of cross-validation procedures."""

    def test_train_test_split_reproducibility(self):
        """Test that train/test splits are reproducible."""
        # Create sample data
        X = np.random.RandomState(42).uniform(0, 1, (100, 5))
        y = np.random.RandomState(42).uniform(0, 1, 100)

        # Perform splits multiple times with same random_state
        splits = []
        for _ in range(3):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            splits.append((X_train, X_test, y_train, y_test))

        # All splits should be identical
        for i in range(1, len(splits)):
            assert np.array_equal(splits[0][0], splits[i][0])  # X_train
            assert np.array_equal(splits[0][1], splits[i][1])  # X_test
            assert np.array_equal(splits[0][2], splits[i][2])  # y_train
            assert np.array_equal(splits[0][3], splits[i][3])  # y_test

    def test_complete_pipeline_reproducibility(self):
        """Test reproducibility of complete training pipeline."""

        def run_complete_pipeline(seed):
            """Simulate complete active learning pipeline."""
            # Set all random seeds
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Generate data
            sequences = ["ATCG", "GCTA", "TTAA", "CCGG"] * 25  # 100 sequences
            expressions = np.random.uniform(0, 10, 100)

            # Process sequences
            encoded = one_hot_encode_sequences(
                sequences, seq_mod_method=SequenceModificationMethod.EMBEDDING
            )
            X = flatten_one_hot_sequences(encoded)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, expressions, test_size=0.2, random_state=seed
            )

            # Train model
            model = return_model(RegressionModelType.RANDOM_FOREST, random_state=seed)
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # Calculate metrics
            norm_metric = normalized_to_best_val_metric(predictions, expressions)
            best_metric = get_best_value_metric(predictions)

            return predictions, norm_metric, best_metric

        # Run pipeline multiple times with same seed
        result1 = run_complete_pipeline(42)
        result2 = run_complete_pipeline(42)

        # Results should be identical
        assert np.allclose(result1[0], result2[0], atol=1e-10)  # predictions
        assert abs(result1[1] - result2[1]) < 1e-10  # norm_metric
        assert abs(result1[2] - result2[2]) < 1e-10  # best_metric

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""

        def simple_random_pipeline(seed):
            np.random.seed(seed)
            return np.random.uniform(0, 1, 10)

        # Run with different seeds
        result1 = simple_random_pipeline(42)
        result2 = simple_random_pipeline(123)

        # Results should be different
        assert not np.array_equal(result1, result2)

    def test_state_isolation_between_runs(self):
        """Test that random state is properly isolated between runs."""

        def stateful_operation():
            # Don't set seed - should use current state
            return np.random.uniform(0, 1, 5)

        # Set seed and run operation
        np.random.seed(42)
        result1 = stateful_operation()

        # Reset seed and run again
        np.random.seed(42)
        result2 = stateful_operation()

        # Should be identical due to seed reset
        assert np.array_equal(result1, result2)

        # Run without reset - should be different
        result3 = stateful_operation()
        assert not np.array_equal(result1, result3)
