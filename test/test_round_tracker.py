"""
Unit tests for RoundTracker class.
"""

import numpy as np
import pandas as pd
import pytest

from experiments.core.round_tracker import RoundTracker

DUMMY_METRICS = {
    "top_proportion": 0.5,
    "best_pred": 1.0,
    "normalized_pred": 0.5,
    "best_true": 1.0,
    "normalized_true": 0.5,
}


class TestRoundTracker:
    def test_initialization(self):
        tracker = RoundTracker(sample_ids=np.array([0, 1, 2]))
        assert tracker.rounds == []
        assert tracker.round_num == 0

    def test_track_round_basic(self):
        tracker = RoundTracker(sample_ids=np.array(["a", "b", "c"]))

        tracker.track_round(selected_indices=[0, 2], metrics=DUMMY_METRICS)

        assert len(tracker.rounds) == 1
        recorded = tracker.rounds[0]
        assert recorded["round"] == 0
        assert recorded["selected_sample_ids"] == ["a", "c"]
        assert recorded["unlabeled_pool_size"] == 1

    def test_track_round_missing_ids(self):
        tracker = RoundTracker(sample_ids=np.array(["only"]))

        with pytest.raises(IndexError):
            tracker.track_round(selected_indices=[5], metrics=DUMMY_METRICS)

    def test_track_multiple_rounds(self):
        tracker = RoundTracker(sample_ids=np.array([f"id_{i}" for i in range(5)]))

        tracker.track_round([0, 1], DUMMY_METRICS)
        tracker.track_round([2], DUMMY_METRICS)
        tracker.track_round([3, 4], DUMMY_METRICS)

        rounds = [entry["round"] for entry in tracker.rounds]
        assert rounds == [0, 1, 2]

    def test_save_to_csv(self, tmp_path):
        tracker = RoundTracker(sample_ids=np.array([0, 1]))
        tracker.track_round([0], DUMMY_METRICS)

        output_path = tmp_path / "rounds.csv"
        tracker.save_to_csv(output_path)

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert "selected_sample_ids" in df.columns
