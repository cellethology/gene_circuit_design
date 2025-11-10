"""
Unit tests for MetricsCalculator class.

Tests custom metrics calculation and cumulative tracking.
"""

import numpy as np

from experiments.core.metrics_calculator import MetricsCalculator
from utils.config_loader import SelectionStrategy


class TestMetricsCalculator:
    """Test cases for MetricsCalculator class."""

    def test_initialization(self):
        """Test MetricsCalculator initialization."""
        all_expressions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        calculator = MetricsCalculator(all_expressions)

        assert calculator.all_expressions.shape == (5,)
        assert len(calculator.cumulative_metrics) == 0

    def test_calculate_round_metrics_basic(self):
        """Test calculating basic round metrics."""
        all_expressions = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        calculator = MetricsCalculator(all_expressions)

        # Select top 2 indices
        selected_indices = [8, 9]  # Indices with values 9 and 10
        predictions = np.array([9.0, 10.0])

        metrics = calculator.calculate_round_metrics(
            selected_indices=selected_indices,
            selection_strategy=SelectionStrategy.HIGH_EXPRESSION,
            predictions=predictions,
        )

        # Check all required metrics are present
        assert "top_10_ratio_intersected_indices" in metrics
        assert "best_value_predictions_values" in metrics
        assert "normalized_predictions_predictions_values" in metrics
        assert "best_value_ground_truth_values" in metrics
        assert "normalized_predictions_ground_truth_values" in metrics

        # Best value should be 10.0 (max of selected)
        assert metrics["best_value_ground_truth_values"] == 10.0
        assert metrics["best_value_predictions_values"] == 10.0

    # TODO: ZELUN fix this later
    # def test_calculate_round_metrics_log_likelihood_strategy(self):
    #     """Test metrics calculation for LOG_LIKELIHOOD strategy."""
    #     all_expressions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    #     calculator = MetricsCalculator(all_expressions)

    #     selected_indices = [3, 4]
    #     # For LOG_LIKELIHOOD, predictions should be ignored
    #     predictions = np.array([100.0, 200.0])  # Should be ignored

    #     metrics = calculator.calculate_round_metrics(
    #         selected_indices=selected_indices,
    #         selection_strategy=SelectionStrategy.LOG_LIKELIHOOD,
    #         predictions=predictions,
    #     )

    #     # Should use true values, not predictions
    #     assert metrics["best_value_predictions_values"] == 5.0  # Max of true values
    #     assert (
    #         metrics["best_value_predictions_values"]
    #         == metrics["best_value_ground_truth_values"]
    #     )

    # TODO: ZELUN fix this later
    # def test_calculate_round_metrics_no_predictions(self):
    #     """Test metrics calculation without predictions."""
    #     all_expressions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    #     calculator = MetricsCalculator(all_expressions)

    #     selected_indices = [2, 3]

    #     metrics = calculator.calculate_round_metrics(
    #         selected_indices=selected_indices,
    #         selection_strategy=SelectionStrategy.HIGH_EXPRESSION,
    #         predictions=None,
    #     )

    #     # Should fallback to true values
    #     assert metrics["best_value_predictions_values"] == 4.0
    #     assert (
    #         metrics["best_value_predictions_values"]
    #         == metrics["best_value_ground_truth_values"]
    #     )

    def test_update_cumulative_first_round(self):
        """Test cumulative metrics for first round."""
        all_expressions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        calculator = MetricsCalculator(all_expressions)

        round_metrics = {
            "top_10_ratio_intersected_indices": 0.5,
            "best_value_predictions_values": 5.0,
            "normalized_predictions_predictions_values": 0.8,
            "best_value_ground_truth_values": 5.0,
            "normalized_predictions_ground_truth_values": 1.0,
        }

        cumulative = calculator.update_cumulative(round_metrics)

        # First round: cumulative should equal current
        assert cumulative["top_10_ratio_intersected_indices_cumulative"] == 0.5
        assert cumulative["best_value_predictions_values_cumulative"] == 5.0
        assert len(calculator.cumulative_metrics) == 1

    def test_update_cumulative_multiple_rounds(self):
        """Test cumulative metrics across multiple rounds."""
        all_expressions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        calculator = MetricsCalculator(all_expressions)

        # Round 1
        round1_metrics = {
            "top_10_ratio_intersected_indices": 0.3,
            "best_value_predictions_values": 4.0,
            "normalized_predictions_predictions_values": 0.6,
            "best_value_ground_truth_values": 4.0,
            "normalized_predictions_ground_truth_values": 0.8,
        }
        cumulative1 = calculator.update_cumulative(round1_metrics)

        # Round 2
        round2_metrics = {
            "top_10_ratio_intersected_indices": 0.2,
            "best_value_predictions_values": 5.0,  # Better than round 1
            "normalized_predictions_predictions_values": 0.7,  # Better than round 1
            "best_value_ground_truth_values": 5.0,
            "normalized_predictions_ground_truth_values": 1.0,
        }
        cumulative2 = calculator.update_cumulative(round2_metrics)

        # Check cumulative values
        # Top 10 ratio should be summed
        assert (
            cumulative2["top_10_ratio_intersected_indices_cumulative"] == 0.5
        )  # 0.3 + 0.2

        # Best values should be max
        assert (
            cumulative2["best_value_predictions_values_cumulative"] == 5.0
        )  # max(4.0, 5.0)
        assert (
            cumulative2["normalized_predictions_predictions_values_cumulative"] == 0.7
        )  # max(0.6, 0.7)

        assert len(calculator.cumulative_metrics) == 2

    def test_get_all_metrics(self):
        """Test retrieving all metrics."""
        all_expressions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        calculator = MetricsCalculator(all_expressions)

        # Add some metrics
        for i in range(3):
            round_metrics = {
                "top_10_ratio_intersected_indices": 0.1 * i,
                "best_value_predictions_values": float(i),
                "normalized_predictions_predictions_values": 0.1 * i,
                "best_value_ground_truth_values": float(i),
                "normalized_predictions_ground_truth_values": 0.1 * i,
            }
            calculator.update_cumulative(round_metrics)

        all_metrics = calculator.get_all_metrics()

        assert len(all_metrics) == 3
        assert isinstance(all_metrics, list)
        assert all(isinstance(m, dict) for m in all_metrics)

    def test_metrics_with_top_10_calculation(self):
        """Test top 10 ratio intersection calculation."""
        # Create dataset with 100 samples
        all_expressions = np.arange(1, 101, dtype=float)
        calculator = MetricsCalculator(all_expressions)

        # Select indices that are in top 10% (indices 90-99)
        selected_indices = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        predictions = all_expressions[selected_indices]

        metrics = calculator.calculate_round_metrics(
            selected_indices=selected_indices,
            selection_strategy=SelectionStrategy.HIGH_EXPRESSION,
            predictions=predictions,
        )

        # All selected should be in top 10%, so ratio should be 1.0
        assert metrics["top_10_ratio_intersected_indices"] == 1.0

    def test_metrics_with_partial_top_10(self):
        """Test top 10 ratio with partial intersection."""
        all_expressions = np.arange(1, 101, dtype=float)
        calculator = MetricsCalculator(all_expressions)

        # Select 5 from top 10% and 5 from elsewhere
        selected_indices = [90, 91, 92, 93, 94, 0, 1, 2, 3, 4]
        predictions = all_expressions[selected_indices]

        metrics = calculator.calculate_round_metrics(
            selected_indices=selected_indices,
            selection_strategy=SelectionStrategy.HIGH_EXPRESSION,
            predictions=predictions,
        )

        # 5 out of 10 in top 10% = 0.5 ratio
        assert metrics["top_10_ratio_intersected_indices"] == 0.5
