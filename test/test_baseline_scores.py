"""
Unit tests for baseline_scores helpers.
"""

import numpy as np
import pytest

from utils.baseline_scores import (
    compute_random_summary_metrics_history,
    draw_random_rounds,
)


def test_rounds_to_top_all_nan_when_no_hits():
    labels = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    top_mask = np.zeros(len(labels), dtype=bool)

    history = compute_random_summary_metrics_history(
        labels=labels,
        top_mask=top_mask,
        max_label=float(labels.max()),
        num_rounds=2,
        num_samples_per_round=2,
        seed=0,
    )

    assert len(history) == 2
    for record in history:
        assert np.isnan(record["rounds_to_top"])


def test_rounds_to_top_first_hit_persists():
    labels = np.linspace(1.0, 6.0, 6, dtype=float)
    num_rounds = 3
    num_samples_per_round = 2
    seed = 7

    rng = np.random.default_rng(seed)
    rounds = draw_random_rounds(
        num_samples=len(labels),
        num_rounds=num_rounds,
        num_samples_per_round=num_samples_per_round,
        rng=rng,
    )

    top_mask = np.zeros(len(labels), dtype=bool)
    top_mask[int(rounds[1][0])] = True

    history = compute_random_summary_metrics_history(
        labels=labels,
        top_mask=top_mask,
        max_label=float(labels.max()),
        num_rounds=num_rounds,
        num_samples_per_round=num_samples_per_round,
        seed=seed,
    )

    assert np.isnan(history[0]["rounds_to_top"])
    assert pytest.approx(2.0) == history[1]["rounds_to_top"]
    assert pytest.approx(2.0) == history[2]["rounds_to_top"]
