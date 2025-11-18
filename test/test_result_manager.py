"""
Unit tests for ResultManager class.

Tests result saving functionality.
"""

from pathlib import Path

import pandas as pd

from experiments.core.result_manager import ResultManager


class TestResultManager:
    """Test cases for ResultManager class."""

    def test_initialization(self):
        """Test ResultManager initialization."""
        manager = ResultManager(
            strategy="highExpression",
            seq_mod_method="embedding",
            predictor_name="LinearRegression",
            seed=42,
            initial_sample_size=8,
            batch_size=8,
        )

        assert manager.strategy == "highExpression"
        assert manager.seq_mod_method == "embedding"
        assert manager.predictor_name == "LinearRegression"
        assert manager.seed == 42

    def test_save_results_basic(self, tmp_path):
        """Test saving basic results."""
        manager = ResultManager(
            strategy="highExpression",
            seq_mod_method="embedding",
            predictor_name="LinearRegression",
            seed=42,
            initial_sample_size=8,
            batch_size=8,
        )

        results = [
            {
                "round": 1,
                "train_size": 8,
                "rmse": 1.5,
                "r2": 0.8,
                "pearson_correlation": 0.9,
            },
            {
                "round": 2,
                "train_size": 16,
                "rmse": 1.2,
                "r2": 0.85,
                "pearson_correlation": 0.92,
            },
        ]

        output_path = str(tmp_path / "test_results.csv")
        manager.save_results(output_path, results, [], [])

        # Check file exists
        assert Path(output_path).exists()

        # Check file contents
        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert "round" in df.columns
        assert "rmse" in df.columns
        assert df.iloc[0]["round"] == 1

    def test_save_custom_metrics(self, tmp_path):
        """Test saving custom metrics."""
        manager = ResultManager(
            strategy="highExpression",
            seq_mod_method="embedding",
            predictor_name="LinearRegression",
            seed=42,
            initial_sample_size=8,
            batch_size=8,
        )

        custom_metrics = [
            {
                "top_10_ratio_intersected_indices": 0.5,
                "best_value_predictions_values": 5.0,
                "normalized_predictions_predictions_values": 0.8,
                "best_value_ground_truth_values": 5.0,
                "normalized_predictions_ground_truth_values": 1.0,
                "top_10_ratio_intersected_indices_cumulative": 0.5,
                "best_value_predictions_values_cumulative": 5.0,
                "normalized_predictions_predictions_values_cumulative": 0.8,
                "best_value_ground_truth_values_cumulative": 5.0,
                "normalized_predictions_ground_truth_values_cumulative": 1.0,
            },
            {
                "top_10_ratio_intersected_indices": 0.3,
                "best_value_predictions_values": 6.0,
                "normalized_predictions_predictions_values": 0.9,
                "best_value_ground_truth_values": 6.0,
                "normalized_predictions_ground_truth_values": 1.0,
                "top_10_ratio_intersected_indices_cumulative": 0.8,
                "best_value_predictions_values_cumulative": 6.0,
                "normalized_predictions_predictions_values_cumulative": 0.9,
                "best_value_ground_truth_values_cumulative": 6.0,
                "normalized_predictions_ground_truth_values_cumulative": 1.0,
            },
        ]

        output_path = str(tmp_path / "test_results.csv")
        manager.save_results(output_path, [], custom_metrics, [])

        # Check custom metrics file
        custom_metrics_path = str(tmp_path / "test_results_custom_metrics.csv")
        assert Path(custom_metrics_path).exists()

        df = pd.read_csv(custom_metrics_path)
        assert len(df) == 2
        assert "strategy" in df.columns
        assert "seq_mod_method" in df.columns
        assert "predictor" in df.columns
        assert "seed" in df.columns
        assert "round" in df.columns
        assert "train_size" in df.columns

        # Check train_size calculation
        assert df.iloc[0]["train_size"] == 8  # initial_sample_size
        assert df.iloc[1]["train_size"] == 16  # initial_sample_size + batch_size

    def test_save_selected_variants(self, tmp_path):
        """Test saving selected variants."""
        manager = ResultManager(
            strategy="highExpression",
            seq_mod_method="embedding",
            predictor_name="LinearRegression",
            seed=42,
            initial_sample_size=8,
            batch_size=8,
        )

        selected_variants = [
            {
                "round": 0,
                "strategy": "highExpression",
                "seed": 42,
                "variant_index": 0,
                "expression": 1.0,
                "sample_id": "sample_0",
            },
            {
                "round": 1,
                "strategy": "highExpression",
                "seed": 42,
                "variant_index": 5,
                "expression": 2.0,
                "sample_id": "sample_5",
            },
        ]

        output_path = str(tmp_path / "test_results.csv")
        manager.save_results(output_path, [], [], selected_variants)

        # Check selected variants file
        variants_path = str(tmp_path / "test_results_selected_variants.csv")
        assert Path(variants_path).exists()

        df = pd.read_csv(variants_path)
        assert len(df) == 2
        assert "round" in df.columns
        assert "variant_index" in df.columns
        assert "expression" in df.columns
        assert "sample_id" in df.columns
        assert "predictor" in df.columns
        assert "seq_mod_method" in df.columns

    def test_save_all_outputs(self, tmp_path):
        """Test saving all output types together."""
        manager = ResultManager(
            strategy="highExpression",
            seq_mod_method="embedding",
            regression_model_name="LinearRegression",
            seed=42,
            initial_sample_size=8,
            batch_size=8,
        )

        results = [{"round": 1, "train_size": 8, "rmse": 1.5}]
        custom_metrics = [
            {
                "top_10_ratio_intersected_indices": 0.5,
                "best_value_predictions_values": 5.0,
                "normalized_predictions_predictions_values": 0.8,
                "best_value_ground_truth_values": 5.0,
                "normalized_predictions_ground_truth_values": 1.0,
                "top_10_ratio_intersected_indices_cumulative": 0.5,
                "best_value_predictions_values_cumulative": 5.0,
                "normalized_predictions_predictions_values_cumulative": 0.8,
                "best_value_ground_truth_values_cumulative": 5.0,
                "normalized_predictions_ground_truth_values_cumulative": 1.0,
            }
        ]
        selected_variants = [
            {
                "round": 0,
                "strategy": "highExpression",
                "seed": 42,
                "variant_index": 0,
                "expression": 1.0,
                "sample_id": "sample_0",
            }
        ]

        output_path = str(tmp_path / "test_results.csv")
        manager.save_results(output_path, results, custom_metrics, selected_variants)

        # Check all files exist
        assert Path(output_path).exists()
        assert Path(str(tmp_path / "test_results_custom_metrics.csv")).exists()
        assert Path(str(tmp_path / "test_results_selected_variants.csv")).exists()

    def test_save_empty_results(self, tmp_path):
        """Test saving when results are empty."""
        manager = ResultManager(
            strategy="highExpression",
            seq_mod_method="embedding",
            regression_model_name="LinearRegression",
            seed=42,
            initial_sample_size=8,
            batch_size=8,
        )

        output_path = str(tmp_path / "test_results.csv")
        # Should not raise error with empty lists
        manager.save_results(output_path, [], [], [])

        # Main results file should not exist if results are empty
        # (but this depends on implementation - checking it doesn't crash)

    def test_metadata_columns_order(self, tmp_path):
        """Test that metadata columns are ordered correctly."""
        manager = ResultManager(
            strategy="highExpression",
            seq_mod_method="embedding",
            regression_model_name="LinearRegression",
            seed=42,
            initial_sample_size=8,
            batch_size=8,
        )

        custom_metrics = [
            {
                "top_10_ratio_intersected_indices": 0.5,
                "best_value_predictions_values": 5.0,
                "normalized_predictions_predictions_values": 0.8,
                "best_value_ground_truth_values": 5.0,
                "normalized_predictions_ground_truth_values": 1.0,
                "top_10_ratio_intersected_indices_cumulative": 0.5,
                "best_value_predictions_values_cumulative": 5.0,
                "normalized_predictions_predictions_values_cumulative": 0.8,
                "best_value_ground_truth_values_cumulative": 5.0,
                "normalized_predictions_ground_truth_values_cumulative": 1.0,
            }
        ]

        output_path = str(tmp_path / "test_results.csv")
        manager.save_results(output_path, [], custom_metrics, [])

        df = pd.read_csv(str(tmp_path / "test_results_custom_metrics.csv"))

        # Check that metadata columns come first
        expected_first_cols = [
            "round",
            "strategy",
            "seed",
            "train_size",
            "regression_model",
            "seq_mod_method",
        ]
        actual_first_cols = list(df.columns[: len(expected_first_cols)])
        assert actual_first_cols == expected_first_cols
