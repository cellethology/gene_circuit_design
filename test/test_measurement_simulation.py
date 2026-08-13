"""Tests for simulated replicate measurements."""

import numpy as np
import pandas as pd
import pytest

from core.measurement_simulation import (
    MeasurementSimulationConfig,
    MeasurementSimulator,
)


def test_noiseless_replicates_record_all_measurements_and_confirm_top(tmp_path):
    simulator = MeasurementSimulator(
        sample_ids=np.array([10, 11, 12, 13]),
        true_labels=np.array([1.0, 2.0, 3.0, 4.0]),
        metadata=None,
        label_key="Expression",
        config=MeasurementSimulationConfig(
            replicates_per_construct=2,
            noise_sigma_log10_expression=0.0,
        ),
        random_seed=0,
    )

    simulator.measure([2, 3], round_num=1)

    assert len(simulator.measurement_rows) == 4
    np.testing.assert_allclose(simulator.training_targets([2, 3]), [3.0, 4.0])
    np.testing.assert_allclose(
        simulator.training_y_var([2, 3]),
        [1e-12, 1e-12],
    )
    np.testing.assert_array_equal(
        simulator.confirmed_true_top_indices([2, 3], top_p=0.25),
        np.array([3]),
    )

    simulator.record_training_observations(
        indices=[2, 3],
        training_round=1,
        train_y_var=simulator.training_y_var([2, 3]),
    )
    simulator.save_outputs(tmp_path)

    measurements = pd.read_csv(tmp_path / "simulated_measurements.csv")
    training = pd.read_csv(tmp_path / "training_observations.csv")
    assert len(measurements) == 4
    assert len(training) == 2
    assert set(measurements["replicate"]) == {0, 1}
    assert "observed_score" in measurements.columns
    assert "train_yvar" in training.columns


def test_single_replicate_confirmation_uses_observed_score_only():
    simulator = MeasurementSimulator(
        sample_ids=np.array([0, 1]),
        true_labels=np.array([1.0, 2.0]),
        metadata=None,
        label_key="Expression",
        config=MeasurementSimulationConfig(replicates_per_construct=1),
        random_seed=0,
    )

    simulator.measure([1], round_num=0)

    assert simulator.training_y_var([1]) is None
    np.testing.assert_array_equal(
        simulator.confirmed_true_top_indices([1], top_p=0.5),
        np.array([1]),
    )


def test_induced_over_basal_simulation_uses_expression_columns():
    metadata = pd.DataFrame({"basal": [2.0], "induced": [10.0]})
    simulator = MeasurementSimulator(
        sample_ids=np.array([0]),
        true_labels=np.array([5.0]),
        metadata=metadata,
        label_key="Fold Change",
        config=MeasurementSimulationConfig(
            replicates_per_construct=1,
            noise_sigma_log10_expression=0.0,
            expression_columns=("basal", "induced"),
            score_mode="induced_over_basal",
        ),
        random_seed=0,
    )

    simulator.measure([0], round_num=0)

    assert simulator.measurement_rows[0]["observed_score"] == pytest.approx(5.0)
    assert simulator.measurement_rows[0]["true_expression_basal"] == pytest.approx(2.0)
    assert simulator.measurement_rows[0][
        "observed_expression_induced"
    ] == pytest.approx(10.0)
