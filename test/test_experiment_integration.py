"""
Integration tests for ActiveLearningExperiment without a test split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from experiments.core.experiment import ActiveLearningExperiment
from experiments.core.initial_selection_strategies import RandomInitialSelection
from experiments.core.query_strategies import Random, TopPredictions


class TestActiveLearningExperiment:
    """Integration tests covering the no-test active learning workflow."""

    def create_dataset(self, tmp_path, n_samples: int = 20, dim: int = 10):
        """Create paired embeddings and metadata files."""
        embeddings = np.random.randn(n_samples, dim).astype(np.float32)
        ids = np.arange(n_samples, dtype=np.int32)
        emb_path = tmp_path / "embeddings.npz"
        np.savez_compressed(emb_path, embeddings=embeddings, ids=ids)

        df = pd.DataFrame(
            {
                "Sequence": [f"ATGC{i}" for i in range(n_samples)],
                "Expression": np.random.randn(n_samples),
            }
        )
        csv_path = tmp_path / "metadata.csv"
        df.to_csv(csv_path, index=False)

        return str(emb_path), str(csv_path)

    def _build_experiment(
        self,
        embeddings_path: str,
        metadata_path: str,
        query_strategy,
        initial_sample_size: int,
        batch_size: int,
        initial_selection_strategy=None,
    ) -> ActiveLearningExperiment:
        initial_strategy = initial_selection_strategy or RandomInitialSelection(seed=0)
        return ActiveLearningExperiment(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            initial_selection_strategy=initial_strategy,
            query_strategy=query_strategy,
            predictor=LinearRegression(),
            initial_sample_size=initial_sample_size,
            batch_size=batch_size,
            normalize_features=False,
            normalize_labels=False,
            label_key="Expression",
        )

    def test_experiment_initialization(self, tmp_path):
        """Experiment initializes with random initial pool and no test indices."""
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=18)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=Random(seed=42),
            initial_sample_size=6,
            batch_size=3,
            initial_selection_strategy=RandomInitialSelection(seed=42),
        )

        assert len(experiment.all_samples) == 18
        assert experiment.embeddings is not None
        assert len(experiment.train_indices) == 6
        assert len(experiment.unlabeled_indices) == 12

    def test_run_experiment_multiple_rounds(self, tmp_path):
        """Running multiple rounds grows the training pool and produces metrics."""
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=25)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=TopPredictions(),
            initial_sample_size=5,
            batch_size=4,
            initial_selection_strategy=RandomInitialSelection(seed=42),
        )

        results = experiment.run_experiment(max_rounds=3)

        assert len(results) == 3
        assert results[-1]["train_size"] == 5 + (4 * 3)
        assert isinstance(experiment.custom_metrics, list)
        assert experiment.custom_metrics

        final_metrics = experiment.get_final_performance()
        assert "top_proportion" in final_metrics
        assert "best_true" in final_metrics

    def test_save_results(self, tmp_path):
        """Saving results writes results, custom metrics, and selected variants."""
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=20)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=Random(seed=99),
            initial_sample_size=5,
            batch_size=5,
            initial_selection_strategy=RandomInitialSelection(seed=99),
        )

        experiment.run_experiment(max_rounds=2)

        output_path = Path(tmp_path) / "results.csv"
        experiment.save_results(output_path)

        assert output_path.exists()
        assert (tmp_path / "results_custom_metrics.csv").exists()
        assert (tmp_path / "results_selected_variants.csv").exists()

    def test_backward_compatibility_properties(self, tmp_path):
        """Compatibility properties stay available even without a test split."""
        emb_path, csv_path = self.create_dataset(tmp_path, n_samples=12)
        experiment = self._build_experiment(
            embeddings_path=emb_path,
            metadata_path=csv_path,
            query_strategy=Random(seed=1),
            initial_sample_size=4,
            batch_size=4,
            initial_selection_strategy=RandomInitialSelection(seed=1),
        )

        assert isinstance(experiment.all_samples, list)
        assert isinstance(experiment.all_expressions, np.ndarray)
        assert isinstance(experiment.train_indices, list)
        assert isinstance(experiment.unlabeled_indices, list)
        assert isinstance(experiment.custom_metrics, list)
        assert isinstance(experiment.selected_variants, list)
