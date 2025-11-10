"""
Unit tests for VariantTracker class.

Tests variant tracking across active learning rounds.
"""

import numpy as np

from experiments.core.variant_tracker import VariantTracker


class TestVariantTracker:
    """Test cases for VariantTracker class."""

    def test_initialization(self):
        """Test VariantTracker initialization."""
        all_expressions = np.array([1.0, 2.0, 3.0])
        all_log_likelihoods = np.array([-0.5, -0.3, -0.1])
        all_sequences = ["ATGC", "CGTA", "AAAA"]

        tracker = VariantTracker(
            all_expressions=all_expressions,
            all_log_likelihoods=all_log_likelihoods,
            all_sequences=all_sequences,
            variant_ids=None,
        )

        assert len(tracker.selected_variants) == 0

    def test_track_round_basic(self):
        """Test tracking variants in a round."""
        all_expressions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        all_log_likelihoods = np.array([-0.5, -0.3, -0.1, -0.05, -0.01])
        all_sequences = ["seq1", "seq2", "seq3", "seq4", "seq5"]

        tracker = VariantTracker(
            all_expressions=all_expressions,
            all_log_likelihoods=all_log_likelihoods,
            all_sequences=all_sequences,
            variant_ids=None,
        )

        selected_indices = [2, 4]
        tracker.track_round(
            round_num=1,
            selected_indices=selected_indices,
            strategy="highExpression",
            seed=42,
        )

        assert len(tracker.selected_variants) == 2

        # Check first variant
        variant1 = tracker.selected_variants[0]
        assert variant1["round"] == 1
        assert variant1["strategy"] == "highExpression"
        assert variant1["seed"] == 42
        assert variant1["variant_index"] == 2
        assert variant1["expression"] == 3.0
        assert variant1["log_likelihood"] == -0.1
        assert variant1["variant_id"] == "variant_2"
        assert variant1["sequence"] == "seq3"

    def test_track_round_with_variant_ids(self):
        """Test tracking with variant IDs."""
        all_expressions = np.array([1.0, 2.0, 3.0])
        all_log_likelihoods = np.array([-0.5, -0.3, -0.1])
        all_sequences = ["seq1", "seq2", "seq3"]
        variant_ids = np.array([100, 200, 300])

        tracker = VariantTracker(
            all_expressions=all_expressions,
            all_log_likelihoods=all_log_likelihoods,
            all_sequences=all_sequences,
            variant_ids=variant_ids,
        )

        tracker.track_round(
            round_num=0,
            selected_indices=[0, 2],
            strategy="random",
            seed=42,
        )

        assert tracker.selected_variants[0]["variant_id"] == 100
        assert tracker.selected_variants[1]["variant_id"] == 300

    def test_track_round_with_nan_log_likelihood(self):
        """Test tracking with NaN log likelihood values."""
        all_expressions = np.array([1.0, 2.0, 3.0])
        all_log_likelihoods = np.array([-0.5, np.nan, -0.1])
        all_sequences = ["seq1", "seq2", "seq3"]

        tracker = VariantTracker(
            all_expressions=all_expressions,
            all_log_likelihoods=all_log_likelihoods,
            all_sequences=all_sequences,
            variant_ids=None,
        )

        tracker.track_round(
            round_num=1,
            selected_indices=[1],
            strategy="highExpression",
            seed=42,
        )

        # NaN should be converted to None
        assert tracker.selected_variants[0]["log_likelihood"] is None

    def test_track_multiple_rounds(self):
        """Test tracking across multiple rounds."""
        all_expressions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        all_log_likelihoods = np.array([-0.5, -0.3, -0.1, -0.05, -0.01])
        all_sequences = [f"seq{i}" for i in range(5)]

        tracker = VariantTracker(
            all_expressions=all_expressions,
            all_log_likelihoods=all_log_likelihoods,
            all_sequences=all_sequences,
            variant_ids=None,
        )

        # Round 0
        tracker.track_round(0, [0, 1], "random", 42)
        # Round 1
        tracker.track_round(1, [2, 3], "highExpression", 42)
        # Round 2
        tracker.track_round(2, [4], "highExpression", 42)

        assert len(tracker.selected_variants) == 5

        # Check round numbers
        rounds = [v["round"] for v in tracker.selected_variants]
        assert rounds == [0, 0, 1, 1, 2]

    def test_get_all_variants(self):
        """Test retrieving all tracked variants."""
        all_expressions = np.array([1.0, 2.0, 3.0])
        all_log_likelihoods = np.array([-0.5, -0.3, -0.1])
        all_sequences = ["seq1", "seq2", "seq3"]

        tracker = VariantTracker(
            all_expressions=all_expressions,
            all_log_likelihoods=all_log_likelihoods,
            all_sequences=all_sequences,
            variant_ids=None,
        )

        tracker.track_round(0, [0, 1], "random", 42)
        tracker.track_round(1, [2], "highExpression", 42)

        all_variants = tracker.get_all_variants()

        assert len(all_variants) == 3
        assert isinstance(all_variants, list)
        assert all(isinstance(v, dict) for v in all_variants)

        # Should be a copy, not reference
        all_variants.append({"test": "data"})
        assert len(tracker.selected_variants) == 3  # Original unchanged

    def test_track_round_long_sequence_truncation(self):
        """Test that long sequences are truncated."""
        all_expressions = np.array([1.0])
        all_log_likelihoods = np.array([-0.5])
        # Create a very long sequence
        long_sequence = "A" * 100

        tracker = VariantTracker(
            all_expressions=all_expressions,
            all_log_likelihoods=all_log_likelihoods,
            all_sequences=[long_sequence],
            variant_ids=None,
        )

        tracker.track_round(0, [0], "random", 42)

        variant = tracker.selected_variants[0]
        assert len(variant["sequence"]) == 53  # 50 chars + "..."
        assert variant["sequence"].endswith("...")

    def test_track_round_short_sequence_no_truncation(self):
        """Test that short sequences are not truncated."""
        all_expressions = np.array([1.0])
        all_log_likelihoods = np.array([-0.5])
        short_sequence = "ATGC"

        tracker = VariantTracker(
            all_expressions=all_expressions,
            all_log_likelihoods=all_log_likelihoods,
            all_sequences=[short_sequence],
            variant_ids=None,
        )

        tracker.track_round(0, [0], "random", 42)

        variant = tracker.selected_variants[0]
        assert variant["sequence"] == "ATGC"
        assert not variant["sequence"].endswith("...")
