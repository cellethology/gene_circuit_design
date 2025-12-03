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
        tracker = RoundTracker(sample_ids=np.array([0, 1, 2]))

        tracker.track_round(selected_indices=[0, 2], metrics=DUMMY_METRICS)

        assert len(tracker.rounds) == 1
        recorded = tracker.rounds[0]
        assert recorded["round"] == 0
        assert recorded["selected_sample_ids"] == [0, 2]
        assert recorded["unlabeled_pool_size"] == 1

    def test_track_round_missing_ids(self):
        tracker = RoundTracker(sample_ids=np.array([0, 1, 2]))

        with pytest.raises(IndexError):
            tracker.track_round(selected_indices=[3], metrics=DUMMY_METRICS)

    def test_compute_auc_with_no_rounds(self):
        tracker = RoundTracker(sample_ids=np.array([0, 1, 2]))

        with pytest.raises(
            ValueError, match="Cannot compute AUC: no rounds have been tracked yet"
        ):
            tracker.compute_auc(["normalized_true"])

    def test_compute_auc_and_missing_metric(self):
        tracker = RoundTracker(sample_ids=np.array([0, 1, 2]))
        tracker.track_round(
            selected_indices=[0], metrics={"normalized_true": 0.2, "best_true": 0.1}
        )
        tracker.track_round(
            selected_indices=[1], metrics={"normalized_true": 0.4, "best_true": 0.3}
        )
        tracker.track_round(
            selected_indices=[2], metrics={"normalized_true": 0.3, "best_true": 0.1}
        )

        aucs = tracker.compute_auc(["normalized_true", "best_true"])
        assert pytest.approx(1.0, rel=1e-6) == aucs["normalized_true"]
        assert pytest.approx(0.7, rel=1e-6) == aucs["best_true"]

        with pytest.raises(ValueError):
            tracker.compute_auc(["missing_metric"])

    def test_save_to_csv(self, tmp_path):
        tracker = RoundTracker(sample_ids=np.array([0, 1]))
        tracker.track_round([0], DUMMY_METRICS)

        output_path = tmp_path / "rounds.csv"
        tracker.save_to_csv(output_path)

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert "selected_sample_ids" in df.columns
