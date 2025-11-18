"""
Unit tests for VariantTracker class.
"""

import numpy as np

from experiments.core.variant_tracker import VariantTracker


class TestVariantTracker:
    """Test cases for VariantTracker class."""

    def test_initialization(self):
        tracker = VariantTracker(
            sample_ids=["sample_0", "sample_1"],
            all_expressions=np.array([1.0, 2.0]),
        )
        assert tracker.selected_variants == []

    def test_track_round_basic(self):
        tracker = VariantTracker(
            sample_ids=["a", "b", "c"],
            all_expressions=np.array([1.0, 2.0, 3.0]),
        )

        tracker.track_round(
            round_num=1,
            selected_indices=[0, 2],
            strategy="random",
            seed=42,
        )

        assert len(tracker.selected_variants) == 2
        variant = tracker.selected_variants[0]
        assert variant["variant_index"] == 0
        assert variant["sample_id"] == "a"
        assert variant["expression"] == 1.0

    def test_track_round_missing_ids(self):
        tracker = VariantTracker(
            sample_ids=["only_one"],
            all_expressions=np.array([1.0]),
        )

        tracker.track_round(
            round_num=0,
            selected_indices=[5],
            strategy="random",
            seed=42,
        )

        assert tracker.selected_variants[0]["sample_id"] == "sample_5"

    def test_track_multiple_rounds(self):
        tracker = VariantTracker(
            sample_ids=[f"id_{i}" for i in range(5)],
            all_expressions=np.linspace(0, 1, 5),
        )

        tracker.track_round(0, [0, 1], "random", 1)
        tracker.track_round(1, [2], "random", 1)
        tracker.track_round(2, [3, 4], "random", 1)

        rounds = [v["round"] for v in tracker.selected_variants]
        assert rounds == [0, 0, 1, 2, 2]

    def test_get_all_variants_returns_copy(self):
        tracker = VariantTracker(
            sample_ids=["x", "y"],
            all_expressions=np.array([1.0, 2.0]),
        )

        tracker.track_round(0, [0], "random", 42)
        variants = tracker.get_all_variants()
        variants.append({"round": 1})

        assert len(tracker.selected_variants) == 1
