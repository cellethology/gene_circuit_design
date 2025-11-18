"""
Integration tests for ActiveLearningExperiment without a test split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from sklearn.linear_model import LinearRegression

from experiments.core.experiment import ActiveLearningExperiment
from experiments.core.query_strategies import Random, TopPredictions


class TestActiveLearningExperiment:
    """Integration tests covering the no-test active learning workflow."""

    def create_test_safetensors(self, tmp_path, n_samples: int = 20) -> str:
        """Helper to create a safetensors file with embeddings and labels."""
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

    def create_test_csv(self, tmp_path, n_samples: int = 20) -> str:
        """Helper to create a CSV dataset."""
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
        """Experiment initializes with random initial pool and no test indices."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=18)
        experiment = ActiveLearningExperiment(
            data_path=data_path,
            query_strategy=Random(seed=42),
            predictor=LinearRegression(),
            initial_sample_size=6,
            batch_size=3,
            normalize_input_output=False,
        )

        assert len(experiment.all_sequences) == 18
        assert experiment.embeddings is not None
        assert len(experiment.train_indices) == 6
        assert len(experiment.unlabeled_indices) == 12
        assert experiment.test_indices == []

    def test_experiment_initialization_csv(self, tmp_path):
        """CSV datasets also work without embeddings."""
        data_path = self.create_test_csv(tmp_path, n_samples=15)
        experiment = ActiveLearningExperiment(
            data_path=data_path,
            query_strategy=Random(seed=123),
            predictor=LinearRegression(),
            initial_sample_size=5,
            batch_size=2,
            normalize_input_output=False,
        )

        assert experiment.embeddings is None
        assert len(experiment.all_expressions) == 15

    def test_run_experiment_multiple_rounds(self, tmp_path):
        """Running multiple rounds grows the training pool and produces metrics."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=25)
        experiment = ActiveLearningExperiment(
            data_path=data_path,
            query_strategy=TopPredictions(),
            predictor=LinearRegression(),
            initial_sample_size=5,
            batch_size=4,
            normalize_input_output=False,
        )

        results = experiment.run_experiment(max_rounds=3)

        assert len(results) == 3
        assert results[-1]["train_size"] == 5 + (4 * 3)
        assert isinstance(experiment.custom_metrics, list)
        assert experiment.custom_metrics, "Custom metrics should accumulate each round"

        final_metrics = experiment.get_final_performance()
        assert "top_10_ratio_intersected_indices" in final_metrics
        assert "best_value_ground_truth_values_cumulative" in final_metrics

    def test_save_results(self, tmp_path):
        """Saving results writes results, custom metrics, and selected variants."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=20)
        experiment = ActiveLearningExperiment(
            data_path=data_path,
            query_strategy=Random(seed=99),
            predictor=LinearRegression(),
            initial_sample_size=5,
            batch_size=5,
            normalize_input_output=False,
        )

        experiment.run_experiment(max_rounds=2)

        output_path = Path(tmp_path) / "results.csv"
        experiment.save_results(str(output_path))

        assert output_path.exists()
        assert (tmp_path / "results_custom_metrics.csv").exists()
        assert (tmp_path / "results_selected_variants.csv").exists()

    def test_backward_compatibility_properties(self, tmp_path):
        """Compatibility properties stay available even without a test split."""
        data_path = self.create_test_safetensors(tmp_path, n_samples=12)
        experiment = ActiveLearningExperiment(
            data_path=data_path,
            query_strategy=Random(seed=1),
            predictor=LinearRegression(),
            initial_sample_size=4,
            batch_size=4,
            normalize_input_output=False,
        )

        assert isinstance(experiment.all_sequences, list)
        assert isinstance(experiment.all_expressions, np.ndarray)
        assert isinstance(experiment.all_log_likelihoods, np.ndarray)
        assert isinstance(experiment.train_indices, list)
        assert isinstance(experiment.unlabeled_indices, list)
        assert experiment.test_indices == []
        assert isinstance(experiment.custom_metrics, list)
        assert isinstance(experiment.selected_variants, list)
