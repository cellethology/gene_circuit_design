"""Tests for simulated replicate measurements."""

import numpy as np
import pandas as pd
import pytest

from core.measurement_simulation import (
    MeasurementSimulationConfig,
    MeasurementSimulator,
)


def _mean_expressions(measurement_rows, columns):
    return {
        column: np.mean(
            [row[f"observed_expression_{column}"] for row in measurement_rows]
        )
        for column in columns
    }


def _expected_ratio_yvar(
    measurement_rows, target, numerator_column, denominator_column
):
    replicate_count = len(measurement_rows)

    def log10_variance(column):
        values = [row[f"observed_expression_{column}"] for row in measurement_rows]
        return np.var(np.log10(values), ddof=1)

    relative_variance = (
        np.expm1(np.log(10.0) ** 2 * log10_variance(numerator_column)) / replicate_count
        + np.expm1(np.log(10.0) ** 2 * log10_variance(denominator_column))
        / replicate_count
    )
    return target**2 * relative_variance


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
    assert "aggregate_observed_score" in measurements.columns
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
    assert simulator.training_y_var([0]) is None


def test_induced_over_basal_averages_expression_before_scoring_and_propagates_yvar():
    metadata = pd.DataFrame({"basal": [2.0], "induced": [10.0]})
    simulator = MeasurementSimulator(
        sample_ids=np.array([0]),
        true_labels=np.array([5.0]),
        metadata=metadata,
        label_key="Fold Change",
        config=MeasurementSimulationConfig(
            replicates_per_construct=3,
            noise_sigma_log10_expression=0.1,
            expression_columns=("basal", "induced"),
            score_mode="induced_over_basal",
        ),
        random_seed=11,
    )

    simulator.measure([0], round_num=0)

    rows = simulator.measurement_rows
    means = _mean_expressions(rows, ("basal", "induced"))
    expected_target = means["induced"] / means["basal"]
    replicate_score_mean = np.mean([row["observed_score"] for row in rows])

    assert simulator.training_targets([0])[0] == pytest.approx(expected_target)
    assert simulator.training_targets([0])[0] != pytest.approx(
        replicate_score_mean,
        abs=1e-12,
    )
    assert simulator.training_y_var([0])[0] == pytest.approx(
        _expected_ratio_yvar(rows, expected_target, "induced", "basal")
    )
    assert all(
        row["aggregate_observed_score"] == pytest.approx(expected_target)
        for row in rows
    )


def test_expression_yvar_pools_log10_replicate_variance_and_records_estimates():
    metadata = pd.DataFrame(
        {
            "basal": [2.0, 4.0],
            "induced": [10.0, 12.0],
        }
    )
    simulator = MeasurementSimulator(
        sample_ids=np.array([10, 11]),
        true_labels=np.array([5.0, 3.0]),
        metadata=metadata,
        label_key="Fold Change",
        config=MeasurementSimulationConfig(
            replicates_per_construct=2,
            noise_sigma_log10_expression=0.1,
            expression_columns=("basal", "induced"),
            score_mode="induced_over_basal",
        ),
        random_seed=19,
    )

    simulator.measure([0, 1], round_num=0)

    expected_variances = {}
    for column in ("basal", "induced"):
        residual_sum_squares = 0.0
        for sample_index in (0, 1):
            values = simulator.expression_replicates[column][sample_index]
            log_values = np.log10(values)
            residual_sum_squares += np.sum((log_values - np.mean(log_values)) ** 2)
        expected_variances[column] = residual_sum_squares / 2

    pooled_variances = simulator.pooled_log10_expression_variances()
    assert pooled_variances == pytest.approx(expected_variances)

    train_y_var = simulator.training_y_var([0, 1])
    simulator.record_training_observations(
        indices=[0, 1],
        training_round=1,
        train_y_var=train_y_var,
    )

    for sample_index, row in enumerate(simulator.training_observation_rows):
        target = simulator.training_targets([sample_index])[0]
        expected_yvar = target**2 * sum(
            np.expm1(np.log(10.0) ** 2 * expected_variances[column]) / 2
            for column in ("basal", "induced")
        )
        assert train_y_var[sample_index] == pytest.approx(expected_yvar)
        assert row["aggregation_method"] == "score_from_mean_expression"
        assert row["observed_score"] == pytest.approx(target)
        assert row["train_yvar"] == pytest.approx(expected_yvar)
        for column in ("basal", "induced"):
            assert row[f"pooled_log10_expression_var_{column}"] == pytest.approx(
                expected_variances[column]
            )
            assert row[f"estimated_sigma_log10_expression_{column}"] == pytest.approx(
                np.sqrt(expected_variances[column])
            )


@pytest.mark.parametrize(
    ("score_mode", "expected_score"),
    [("and_score", 1.5), ("or_score", 3.0)],
)
def test_multi_input_score_modes_recompute_scores_from_four_states(
    score_mode, expected_score
):
    metadata = pd.DataFrame(
        {
            "basal": [2.0],
            "input_a": [8.0],
            "input_b": [6.0],
            "dual": [12.0],
        }
    )
    simulator = MeasurementSimulator(
        sample_ids=np.array([0]),
        true_labels=np.array([expected_score]),
        metadata=metadata,
        label_key=score_mode,
        config=MeasurementSimulationConfig(
            replicates_per_construct=1,
            noise_sigma_log10_expression=0.0,
            expression_columns=("basal", "input_a", "input_b", "dual"),
            score_mode=score_mode,
        ),
        random_seed=0,
    )

    simulator.measure([0], round_num=0)

    measurement = simulator.measurement_rows[0]
    assert measurement["observed_score"] == pytest.approx(expected_score)
    for column in ("basal", "input_a", "input_b", "dual"):
        assert measurement[f"true_expression_{column}"] == pytest.approx(
            metadata.loc[0, column]
        )
        assert measurement[f"observed_expression_{column}"] == pytest.approx(
            metadata.loc[0, column]
        )


@pytest.mark.parametrize("score_mode", ["and_score", "or_score"])
def test_multi_input_replicates_use_noisy_derived_scores(score_mode):
    metadata = pd.DataFrame(
        {
            "basal": [2.0],
            "input_a": [8.0],
            "input_b": [6.0],
            "dual": [12.0],
        }
    )
    simulator = MeasurementSimulator(
        sample_ids=np.array([0]),
        true_labels=np.array([1.5 if score_mode == "and_score" else 3.0]),
        metadata=metadata,
        label_key=score_mode,
        config=MeasurementSimulationConfig(
            replicates_per_construct=3,
            noise_sigma_log10_expression=0.1,
            expression_columns=("basal", "input_a", "input_b", "dual"),
            score_mode=score_mode,
        ),
        random_seed=7,
    )

    simulator.measure([0], round_num=0)

    observed_scores = []
    for measurement in simulator.measurement_rows:
        observed = {
            column: measurement[f"observed_expression_{column}"]
            for column in ("basal", "input_a", "input_b", "dual")
        }
        if score_mode == "and_score":
            expected = observed["dual"] / max(
                observed["basal"], observed["input_a"], observed["input_b"]
            )
        else:
            expected = (
                min(observed["input_a"], observed["input_b"], observed["dual"])
                / observed["basal"]
            )
        assert measurement["observed_score"] == pytest.approx(expected)
        observed_scores.append(expected)

    columns = ("basal", "input_a", "input_b", "dual")
    means = _mean_expressions(simulator.measurement_rows, columns)
    if score_mode == "and_score":
        denominator_column = max(
            ("basal", "input_a", "input_b"),
            key=means.__getitem__,
        )
        numerator_column = "dual"
        expected_target = means[numerator_column] / means[denominator_column]
    else:
        numerator_column = min(
            ("input_a", "input_b", "dual"),
            key=means.__getitem__,
        )
        denominator_column = "basal"
        expected_target = means[numerator_column] / means[denominator_column]

    assert simulator.training_targets([0])[0] == pytest.approx(expected_target)
    assert simulator.training_targets([0])[0] != pytest.approx(
        np.mean(observed_scores),
        abs=1e-12,
    )
    assert simulator.training_y_var([0])[0] == pytest.approx(
        _expected_ratio_yvar(
            simulator.measurement_rows,
            expected_target,
            numerator_column,
            denominator_column,
        )
    )
    assert all(
        row["aggregate_observed_score"] == pytest.approx(expected_target)
        for row in simulator.measurement_rows
    )


def test_multi_input_named_columns_do_not_depend_on_expression_column_order():
    metadata = pd.DataFrame(
        {"dual": [12.0], "input_b": [6.0], "basal": [2.0], "input_a": [8.0]}
    )
    simulator = MeasurementSimulator(
        sample_ids=np.array([0]),
        true_labels=np.array([1.5]),
        metadata=metadata,
        label_key="and_score",
        config=MeasurementSimulationConfig(
            score_mode="and_score",
            basal_column="basal",
            single_input_a_column="input_a",
            single_input_b_column="input_b",
            dual_input_column="dual",
        ),
        random_seed=0,
    )

    simulator.measure([0], round_num=0)

    assert simulator.measurement_rows[0]["observed_score"] == pytest.approx(1.5)


def test_multi_input_mode_allows_zero_expression():
    metadata = pd.DataFrame(
        {"basal": [0.0], "input_a": [2.0], "input_b": [3.0], "dual": [6.0]}
    )
    simulator = MeasurementSimulator(
        sample_ids=np.array([0]),
        true_labels=np.array([2.0]),
        metadata=metadata,
        label_key="and_score",
        config=MeasurementSimulationConfig(
            expression_columns=("basal", "input_a", "input_b", "dual"),
            score_mode="and_score",
            noise_sigma_log10_expression=0.1,
        ),
        random_seed=0,
    )

    simulator.measure([0], round_num=0)

    assert simulator.measurement_rows[0]["observed_expression_basal"] == 0.0
