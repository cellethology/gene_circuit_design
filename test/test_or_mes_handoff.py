from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from job_sub import submit_or_mes_allocations as submitter
from job_sub import validate_or_mes_sweep as validator


def test_complete_summary_requires_exact_or_configuration(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    payload = {
        "completed_rounds": submitter.EVALUATED_ROUNDS,
        "query_strategy": "MES",
        "dataset_name": "example",
        "seed": 4,
        "measurement_simulation": {
            "enabled": True,
            "score_mode": submitter.SCORE_MODE,
            "expression_columns": list(submitter.EXPRESSION_COLUMNS),
            "replicates_per_construct": 2,
            "noise_sigma_log10_expression": 0.1,
        },
    }
    summary_path.write_text(json.dumps(payload))

    assert submitter.complete_summary(
        summary_path,
        dataset_name="example",
        seed=4,
        replicates=2,
    )

    payload["measurement_simulation"]["score_mode"] = "and_score"
    summary_path.write_text(json.dumps(payload))
    assert not submitter.complete_summary(
        summary_path,
        dataset_name="example",
        seed=4,
        replicates=2,
    )


def test_active_manifest_keys_maps_only_queued_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "submission_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "allocation": "12x2",
                "dataset_name": "dataset_a",
                "job_ids": ["100_0", "100_1"],
                "seeds": [0, 1],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(submitter.shutil, "which", lambda _: "/fake/squeue")
    monkeypatch.setattr(
        submitter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="100_1\n"),
    )

    assert submitter.active_manifest_keys(tmp_path) == {("12x2", "dataset_a", 1)}


def _write_synthetic_or_run(
    sweep_dir: Path,
    *,
    dataset_name: str,
    allocation: str = "12x2",
    seed: int = 0,
) -> Path:
    constructs, replicates = submitter.ALLOCATIONS[allocation]
    run_dir = sweep_dir / allocation / dataset_name / f"seed_{seed}"
    run_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    measurement_rows: list[dict[str, object]] = []
    selected_by_round: list[list[int]] = []
    aggregate_by_sample: dict[int, float] = {}
    for round_num in range(validator.EVALUATED_ROUNDS):
        selected = []
        for construct_index in range(constructs):
            sample_id = round_num * 1000 + construct_index
            selected.append(sample_id)
            true_expression = {
                submitter.EXPRESSION_COLUMNS[0]: 10.0 + construct_index,
                submitter.EXPRESSION_COLUMNS[1]: 20.0 + construct_index,
                submitter.EXPRESSION_COLUMNS[2]: 30.0 + construct_index,
                submitter.EXPRESSION_COLUMNS[3]: 40.0 + construct_index,
            }
            true_derived = (
                min(
                    true_expression[submitter.EXPRESSION_COLUMNS[1]],
                    true_expression[submitter.EXPRESSION_COLUMNS[2]],
                    true_expression[submitter.EXPRESSION_COLUMNS[3]],
                )
                / true_expression[submitter.EXPRESSION_COLUMNS[0]]
            )
            calibration = 1.0 + construct_index / 100.0
            true_score = true_derived * calibration

            replicate_values: list[dict[str, float]] = []
            replicate_scores: list[float] = []
            for _ in range(replicates):
                observed: dict[str, float] = {}
                noise: dict[str, float] = {}
                for column in submitter.EXPRESSION_COLUMNS:
                    noise[column] = float(rng.normal(0.0, 0.1))
                    observed[column] = true_expression[column] * 10 ** noise[column]
                derived = (
                    min(
                        observed[submitter.EXPRESSION_COLUMNS[1]],
                        observed[submitter.EXPRESSION_COLUMNS[2]],
                        observed[submitter.EXPRESSION_COLUMNS[3]],
                    )
                    / observed[submitter.EXPRESSION_COLUMNS[0]]
                )
                replicate_values.append(
                    {**observed, **{f"noise:{k}": v for k, v in noise.items()}}
                )
                replicate_scores.append(derived * calibration)

            mean_expression = {
                column: float(np.mean([values[column] for values in replicate_values]))
                for column in submitter.EXPRESSION_COLUMNS
            }
            aggregate_derived = (
                min(
                    mean_expression[submitter.EXPRESSION_COLUMNS[1]],
                    mean_expression[submitter.EXPRESSION_COLUMNS[2]],
                    mean_expression[submitter.EXPRESSION_COLUMNS[3]],
                )
                / mean_expression[submitter.EXPRESSION_COLUMNS[0]]
            )
            aggregate_score = aggregate_derived * calibration
            aggregate_by_sample[sample_id] = aggregate_score

            for replicate, (values, observed_score) in enumerate(
                zip(replicate_values, replicate_scores, strict=True)
            ):
                row: dict[str, object] = {
                    "round": round_num,
                    "sample_index": sample_id,
                    "sample_id": sample_id,
                    "replicate": replicate,
                    "replicates_per_construct": replicates,
                    "noise_sigma_log10_expression": 0.1,
                    "score_mode": "or_score",
                    "true_score": true_score,
                    "observed_score": observed_score,
                    "expression_derived_score": observed_score / calibration,
                    "historical_score_calibration_factor": calibration,
                    "aggregate_observed_score": aggregate_score,
                    "aggregate_expression_derived_score": aggregate_derived,
                    "replicate_score_mean": float(np.mean(replicate_scores)),
                }
                for column in submitter.EXPRESSION_COLUMNS:
                    row[f"true_expression_{column}"] = true_expression[column]
                    row[f"observed_expression_{column}"] = values[column]
                    row[f"log10_noise_{column}"] = values[f"noise:{column}"]
                    row[f"mean_observed_expression_{column}"] = mean_expression[column]
                measurement_rows.append(row)
        selected_by_round.append(selected)

    pd.DataFrame(measurement_rows).to_csv(
        run_dir / "simulated_measurements.csv", index=False
    )

    results = pd.DataFrame(
        {
            "round": range(validator.EVALUATED_ROUNDS),
            "train_size": [
                constructs * value for value in range(validator.EVALUATED_ROUNDS)
            ],
            "selected_sample_ids": [str(values) for values in selected_by_round],
            "best_true_score_found": np.linspace(1.0, 2.0, validator.EVALUATED_ROUNDS),
        }
    )
    results.to_csv(run_dir / "results.csv", index=False)

    training_rows = []
    for training_round in range(1, validator.EVALUATED_ROUNDS):
        for selected_round in range(training_round):
            for sample_id in selected_by_round[selected_round]:
                row = {
                    "training_round": training_round,
                    "sample_id": sample_id,
                    "selected_round": selected_round,
                    "replicate_count": replicates,
                    "aggregation_method": "score_from_mean_expression",
                    "observed_score": aggregate_by_sample[sample_id],
                    "train_yvar": 0.01 if replicates > 1 else np.nan,
                }
                for column in submitter.EXPRESSION_COLUMNS:
                    row[f"estimated_sigma_log10_expression_{column}"] = (
                        0.1 if replicates > 1 else np.nan
                    )
                training_rows.append(row)
    pd.DataFrame(training_rows).to_csv(
        run_dir / "training_observations.csv", index=False
    )

    summary = {
        "completed_rounds": validator.EVALUATED_ROUNDS,
        "query_strategy": "MES",
        "embedding_model": submitter.EMBEDDING_MODEL,
        "dataset_name": dataset_name,
        "seed": seed,
        "measurement_simulation": {
            "enabled": True,
            "score_mode": "or_score",
            "expression_columns": list(submitter.EXPRESSION_COLUMNS),
            "replicates_per_construct": replicates,
            "noise_sigma_log10_expression": 0.1,
        },
        "summary_by_round": [
            {"round": value} for value in range(validator.EVALUATED_ROUNDS)
        ],
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary))
    return summary_path


def test_validator_accepts_consistent_or_run_and_rejects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validator, "EVALUATED_ROUNDS", 2)
    dataset_name = validator.load_dataset_names()[0]
    summary_paths = {
        allocation: _write_synthetic_or_run(
            tmp_path,
            dataset_name=dataset_name,
            allocation=allocation,
        )
        for allocation in submitter.ALLOCATIONS
    }

    report = validator.validate_sweep(tmp_path, expected_runs=3)
    assert report["valid"]
    assert report["validated_runs"] == 3

    summary_path = summary_paths["12x2"]
    measurements_path = summary_path.parent / "simulated_measurements.csv"
    measurements = pd.read_csv(measurements_path)
    observed_column = f"observed_expression_{submitter.EXPRESSION_COLUMNS[0]}"
    measurements.loc[0, observed_column] *= 2
    measurements.to_csv(measurements_path, index=False)

    corrupted = validator.validate_sweep(tmp_path, expected_runs=3)
    assert not corrupted["valid"]
    assert any(
        "true_expression * 10^noise" in failure
        for failure in corrupted["run_failures"][0]["failures"]
    )
