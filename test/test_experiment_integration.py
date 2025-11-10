"""
Integration tests for ActiveLearningExperiment.

Tests the complete experiment workflow using the refactored components.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from experiments.core.experiment import ActiveLearningExperiment
from utils.config_loader import SelectionStrategy
from utils.model_loader import RegressionModelType
from utils.sequence_utils import SequenceModificationMethod


class TestActiveLearningExperiment:
    """Integration tests for ActiveLearningExperiment."""

    def create_test_safetensors(self, tmp_path, n_samples=20):
        """Helper to create test safetensors file."""
        safetensors_path = tmp_path / "test_data.safetensors"

        embeddings = torch.randn(n_samples, 10)
        expressions = torch.randn(n_samples)
        log_likelihoods = torch.randn(n_samples)

        save_file(
            {
                "embeddings": embeddings,
                "expressions": expressions,
                "log_likelihoods": log_likelihoods,
            },
            str(safetensors_path),
        )

        return str(safetensors_path)

    def create_test_csv(self, tmp_path, n_samples=20):
        """Helper to create test CSV file."""
        csv_path = tmp_path / "test_data.csv"

        df = pd.DataFrame(
            {
                "Sequence": [f"ATGC{i}" for i in range(n_samples)],
                "Expression": np.random.randn(n_samples),
                "Log_Likelihood": np.random.randn(n_samples),
            }
        )
        df.to_csv(csv_path, index=False)

        return str(csv_path)

    def test_experiment_initialization_safetensors(self, tmp_path):
        """Test experiment initialization with safetensors."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=20)

        experiment = ActiveLearningExperiment(
            data_path=data_path,
            selection_strategy=SelectionStrategy.RANDOM,
            regression_model=RegressionModelType.LINEAR,
            initial_sample_size=5,
            batch_size=3,
            test_size=5,
            random_seed=42,
            seq_mod_method=SequenceModificationMethod.EMBEDDING,
            no_test=False,
            normalize_input_output=False,
        )

        # Check data is loaded
        assert len(experiment.all_sequences) == 20
        assert len(experiment.all_expressions) == 20
        assert experiment.embeddings is not None
        assert experiment.embeddings.shape == (20, 10)

        # Check data splits
        assert len(experiment.train_indices) == 5
        assert len(experiment.test_indices) == 5
        assert len(experiment.unlabeled_indices) == 10

    def test_experiment_initialization_csv(self, tmp_path):
        """Test experiment initialization with CSV."""
        data_path = self.create_test_csv(tmp_path, n_samples=15)

        experiment = ActiveLearningExperiment(
            data_path=data_path,
            selection_strategy=SelectionStrategy.RANDOM,
            regression_model=RegressionModelType.LINEAR,
            initial_sample_size=4,
            batch_size=2,
            test_size=3,
            random_seed=42,
            seq_mod_method=SequenceModificationMethod.EMBEDDING,
            no_test=False,
            normalize_input_output=False,
        )

        assert len(experiment.all_sequences) == 15
        assert experiment.embeddings is None  # CSV doesn't have embeddings

    def test_run_experiment_single_round(self, tmp_path):
        """Test running experiment for a single round."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=15)

        experiment = ActiveLearningExperiment(
            data_path=data_path,
            selection_strategy=SelectionStrategy.RANDOM,
            regression_model=RegressionModelType.LINEAR,
            initial_sample_size=5,
            batch_size=3,
            test_size=3,
            random_seed=42,
            no_test=False,
            normalize_input_output=False,
        )

        results = experiment.run_experiment(max_rounds=1)

        # Check results
        assert len(results) == 1
        assert results[0]["round"] == 1
        assert "rmse" in results[0]
        assert "r2" in results[0]

        # Check that training set was updated
        assert len(experiment.train_indices) == 8  # 5 initial + 3 from batch

        # Check custom metrics were calculated
        assert len(experiment.custom_metrics) >= 1

    def test_run_experiment_multiple_rounds(self, tmp_path):
        """Test running experiment for multiple rounds."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=20)

        experiment = ActiveLearningExperiment(
            data_path=data_path,
            selection_strategy=SelectionStrategy.RANDOM,
            regression_model=RegressionModelType.LINEAR,
            initial_sample_size=5,
            batch_size=3,
            test_size=5,
            random_seed=42,
            no_test=False,
            normalize_input_output=False,
        )

        results = experiment.run_experiment(max_rounds=3)

        # Check results
        assert len(results) == 3
        assert all(r["round"] == i + 1 for i, r in enumerate(results))

        # Check training set grows
        assert len(experiment.train_indices) == 14  # 5 + 3*3

        # Check custom metrics
        assert len(experiment.custom_metrics) >= 3

    def test_run_experiment_no_test(self, tmp_path):
        """Test running experiment without test set."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=15)

        experiment = ActiveLearningExperiment(
            data_path=data_path,
            selection_strategy=SelectionStrategy.RANDOM,
            regression_model=RegressionModelType.LINEAR,
            initial_sample_size=5,
            batch_size=3,
            test_size=0,
            random_seed=42,
            no_test=True,
            normalize_input_output=False,
        )

        results = experiment.run_experiment(max_rounds=2)

        # Results should not have test metrics
        assert len(results) == 2
        assert "rmse" not in results[0] or results[0].get("rmse") is None

    def test_save_results(self, tmp_path):
        """Test saving experiment results."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=15)

        experiment = ActiveLearningExperiment(
            data_path=data_path,
            selection_strategy=SelectionStrategy.RANDOM,
            regression_model=RegressionModelType.LINEAR,
            initial_sample_size=5,
            batch_size=3,
            test_size=3,
            random_seed=42,
            no_test=False,
            normalize_input_output=False,
        )

        experiment.run_experiment(max_rounds=2)

        output_path = str(tmp_path / "test_results.csv")
        experiment.save_results(output_path)

        # Check files exist
        assert Path(output_path).exists()
        assert Path(str(tmp_path / "test_results_custom_metrics.csv")).exists()
        assert Path(str(tmp_path / "test_results_selected_variants.csv")).exists()

        # Check results file
        df = pd.read_csv(output_path)
        assert len(df) == 2

    def test_backward_compatibility_properties(self, tmp_path):
        """Test backward compatibility properties."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=15)

        experiment = ActiveLearningExperiment(
            data_path=data_path,
            selection_strategy=SelectionStrategy.RANDOM,
            regression_model=RegressionModelType.LINEAR,
            initial_sample_size=5,
            batch_size=3,
            random_seed=42,
            no_test=True,
            normalize_input_output=False,
        )

        # Test all backward compatibility properties
        assert isinstance(experiment.all_sequences, list)
        assert isinstance(experiment.all_expressions, np.ndarray)
        assert isinstance(experiment.all_log_likelihoods, np.ndarray)
        assert isinstance(experiment.train_indices, list)
        assert isinstance(experiment.test_indices, list)
        assert isinstance(experiment.unlabeled_indices, list)
        assert isinstance(experiment.custom_metrics, list)
        assert isinstance(experiment.selected_variants, list)

    def test_different_selection_strategies(self, tmp_path):
        """Test experiment with different selection strategies."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=20)

        strategies = [
            SelectionStrategy.RANDOM,
            SelectionStrategy.HIGH_EXPRESSION,
        ]

        for strategy in strategies:
            experiment = ActiveLearningExperiment(
                data_path=data_path,
                selection_strategy=strategy,
                regression_model=RegressionModelType.LINEAR,
                initial_sample_size=5,
                batch_size=3,
                random_seed=42,
                no_test=True,
                normalize_input_output=False,
            )

            results = experiment.run_experiment(max_rounds=2)
            assert len(results) == 2

    def test_different_regression_models(self, tmp_path):
        """Test experiment with different regression models."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=15)

        models = [
            RegressionModelType.LINEAR,
            RegressionModelType.KNN,
            RegressionModelType.RANDOM_FOREST,
        ]

        for model in models:
            experiment = ActiveLearningExperiment(
                data_path=data_path,
                selection_strategy=SelectionStrategy.RANDOM,
                regression_model=model,
                initial_sample_size=5,
                batch_size=3,
                random_seed=42,
                no_test=True,
                normalize_input_output=False,
            )

            results = experiment.run_experiment(max_rounds=1)
            assert len(results) == 1

    def test_get_final_performance(self, tmp_path):
        """Test getting final performance metrics."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=15)

        experiment = ActiveLearningExperiment(
            data_path=data_path,
            selection_strategy=SelectionStrategy.RANDOM,
            regression_model=RegressionModelType.LINEAR,
            initial_sample_size=5,
            batch_size=3,
            test_size=3,
            random_seed=42,
            no_test=False,
            normalize_input_output=False,
        )

        experiment.run_experiment(max_rounds=2)

        final_perf = experiment.get_final_performance()

        # Should have metrics but not metadata
        assert "round" not in final_perf
        assert "strategy" not in final_perf
        assert "seed" not in final_perf

        # Should have performance metrics if test set was used
        if not experiment.no_test:
            assert "rmse" in final_perf or "r2" in final_perf

    def test_experiment_stops_when_no_unlabeled(self, tmp_path):
        """Test that experiment stops when no unlabeled data remains."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=10)

        experiment = ActiveLearningExperiment(
            data_path=data_path,
            selection_strategy=SelectionStrategy.RANDOM,
            regression_model=RegressionModelType.LINEAR,
            initial_sample_size=5,
            batch_size=3,
            test_size=2,
            random_seed=42,
            no_test=False,
            normalize_input_output=False,
        )

        # Only 3 unlabeled samples, but batch_size is 3
        # After 1 round, should have 0 unlabeled
        results = experiment.run_experiment(max_rounds=10)

        # Should stop early when no unlabeled data
        assert len(results) <= 2  # At most 2 rounds before running out
