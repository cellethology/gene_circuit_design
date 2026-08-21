from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from job_sub.submit_or_mes_allocations import (
    ALLOCATIONS,
    DATASETS_PATH,
    EMBEDDING_MODEL,
    EVALUATED_ROUNDS,
    EXPRESSION_COLUMNS,
    NOISE_SIGMA_LOG10_EXPRESSION,
    NUM_SEEDS,
    SCORE_MODE,
)

ASSAYS_PER_ROUND = 24
FULL_RUN_COUNT = 33 * len(ALLOCATIONS) * NUM_SEEDS


@dataclass(frozen=True)
class RunKey:
    allocation: str
    dataset_name: str
    seed: int

    def display(self) -> str:
        return f"{self.allocation}/{self.dataset_name}/seed_{self.seed}"


@dataclass
class RunValidation:
    key: RunKey
    failures: list[str]
    empirical_noise_sigma: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate sigma=0.10 MES OR simulation outputs."
    )
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Defaults to <sweep-dir>/validation_report.json.",
    )
    return parser.parse_args()


def load_dataset_names() -> list[str]:
    config = OmegaConf.load(DATASETS_PATH)
    return [str(dataset.name) for dataset in config.datasets]


def key_from_summary_path(sweep_dir: Path, summary_path: Path) -> RunKey:
    relative = summary_path.relative_to(sweep_dir)
    if len(relative.parts) != 4 or relative.name != "summary.json":
        raise ValueError(f"Unexpected summary path: {summary_path}")
    allocation, dataset_name, seed_dir, _ = relative.parts
    if not seed_dir.startswith("seed_"):
        raise ValueError(f"Unexpected seed directory: {summary_path}")
    return RunKey(allocation, dataset_name, int(seed_dir.removeprefix("seed_")))


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _finite_numeric(frame: pd.DataFrame, columns: list[str]) -> bool:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return bool(np.isfinite(values).all())


def _allclose(left: Any, right: Any) -> bool:
    return bool(np.allclose(left, right, rtol=1e-8, atol=1e-10, equal_nan=False))


def validate_measurements(
    path: Path,
    *,
    constructs: int,
    replicates: int,
    failures: list[str],
) -> float | None:
    try:
        measurements = pd.read_csv(path)
    except Exception as exc:
        failures.append(f"cannot read simulated_measurements.csv: {exc}")
        return None

    true_columns = [f"true_expression_{column}" for column in EXPRESSION_COLUMNS]
    observed_columns = [
        f"observed_expression_{column}" for column in EXPRESSION_COLUMNS
    ]
    noise_columns = [f"log10_noise_{column}" for column in EXPRESSION_COLUMNS]
    mean_columns = [
        f"mean_observed_expression_{column}" for column in EXPRESSION_COLUMNS
    ]
    required = {
        "round",
        "sample_index",
        "sample_id",
        "replicate",
        "replicates_per_construct",
        "noise_sigma_log10_expression",
        "score_mode",
        "true_score",
        "observed_score",
        "expression_derived_score",
        "historical_score_calibration_factor",
        "aggregate_observed_score",
        "aggregate_expression_derived_score",
        "replicate_score_mean",
        *true_columns,
        *observed_columns,
        *noise_columns,
        *mean_columns,
    }
    missing = sorted(required - set(measurements.columns))
    if missing:
        failures.append(f"measurement columns missing: {missing}")
        return None

    expected_rows = EVALUATED_ROUNDS * ASSAYS_PER_ROUND
    _require(
        len(measurements) == expected_rows,
        f"measurement rows={len(measurements)}, expected {expected_rows}",
        failures,
    )
    _require(
        set(measurements["round"].astype(int)) == set(range(EVALUATED_ROUNDS)),
        "measurement rounds are not exactly 0..29",
        failures,
    )
    rows_per_round = measurements.groupby("round").size()
    _require(
        bool((rows_per_round == ASSAYS_PER_ROUND).all()),
        "not every round contains exactly 24 assays",
        failures,
    )
    _require(
        set(measurements["replicates_per_construct"].astype(int)) == {replicates},
        "replicates_per_construct does not match allocation",
        failures,
    )
    _require(
        set(measurements["score_mode"].astype(str)) == {SCORE_MODE},
        f"score_mode is not uniformly {SCORE_MODE}",
        failures,
    )
    _require(
        _allclose(
            measurements["noise_sigma_log10_expression"].to_numpy(float),
            NOISE_SIGMA_LOG10_EXPRESSION,
        ),
        "measurement noise sigma is not uniformly 0.10",
        failures,
    )

    group_columns = ["round", "sample_id"]
    construct_groups = measurements.groupby(group_columns, sort=False)
    group_sizes = construct_groups.size()
    _require(
        bool((group_sizes == replicates).all()),
        "one or more selected constructs has the wrong replicate count",
        failures,
    )
    constructs_per_round = (
        measurements[["round", "sample_id"]].drop_duplicates().groupby("round").size()
    )
    _require(
        bool((constructs_per_round == constructs).all()),
        "one or more rounds has the wrong number of unique constructs",
        failures,
    )
    replicate_sets = construct_groups["replicate"].agg(
        lambda values: tuple(sorted(values.astype(int)))
    )
    expected_replicates = tuple(range(replicates))
    _require(
        bool((replicate_sets == expected_replicates).all()),
        "replicate indices are not exactly 0..replicates-1",
        failures,
    )
    sample_round_counts = measurements.groupby("sample_id")["round"].nunique()
    _require(
        bool((sample_round_counts == 1).all()),
        "a construct was selected in more than one round",
        failures,
    )
    _require(
        measurements["sample_id"].nunique() == constructs * EVALUATED_ROUNDS,
        "total unique selected constructs is incorrect",
        failures,
    )

    numeric_columns = [
        "true_score",
        "observed_score",
        "expression_derived_score",
        "historical_score_calibration_factor",
        "aggregate_observed_score",
        "aggregate_expression_derived_score",
        *true_columns,
        *observed_columns,
        *noise_columns,
        *mean_columns,
    ]
    _require(
        _finite_numeric(measurements, numeric_columns),
        "measurement table contains non-finite numeric values",
        failures,
    )

    for true_column, observed_column, noise_column in zip(
        true_columns, observed_columns, noise_columns, strict=True
    ):
        expected_observed = measurements[true_column].to_numpy(
            float
        ) * 10 ** measurements[noise_column].to_numpy(float)
        _require(
            _allclose(measurements[observed_column].to_numpy(float), expected_observed),
            f"{observed_column} does not equal true_expression * 10^noise",
            failures,
        )

    basal, input_a, input_b, dual = observed_columns
    replicate_derived = np.minimum.reduce(
        [
            measurements[input_a].to_numpy(float),
            measurements[input_b].to_numpy(float),
            measurements[dual].to_numpy(float),
        ]
    ) / measurements[basal].to_numpy(float)
    factor = measurements["historical_score_calibration_factor"].to_numpy(float)
    _require(
        _allclose(
            measurements["expression_derived_score"].to_numpy(float),
            replicate_derived,
        ),
        "replicate expression-derived OR scores are inconsistent",
        failures,
    )
    _require(
        _allclose(
            measurements["observed_score"].to_numpy(float),
            replicate_derived * factor,
        ),
        "replicate observed OR scores are inconsistent with calibration",
        failures,
    )

    grouped_means = construct_groups[observed_columns].mean()
    aggregate_derived = np.minimum.reduce(
        [
            grouped_means[input_a].to_numpy(float),
            grouped_means[input_b].to_numpy(float),
            grouped_means[dual].to_numpy(float),
        ]
    ) / grouped_means[basal].to_numpy(float)
    grouped_factor = construct_groups["historical_score_calibration_factor"].first()
    grouped_aggregate = construct_groups["aggregate_observed_score"].first()
    grouped_derived = construct_groups["aggregate_expression_derived_score"].first()
    _require(
        _allclose(grouped_derived.to_numpy(float), aggregate_derived),
        "aggregate OR score was not computed from mean expression",
        failures,
    )
    _require(
        _allclose(
            grouped_aggregate.to_numpy(float),
            aggregate_derived * grouped_factor.to_numpy(float),
        ),
        "aggregate observed OR score is inconsistent with calibration",
        failures,
    )
    for mean_column, observed_column in zip(
        mean_columns, observed_columns, strict=True
    ):
        stored_means = construct_groups[mean_column].first()
        _require(
            _allclose(
                stored_means.to_numpy(float),
                grouped_means[observed_column].to_numpy(float),
            ),
            f"{mean_column} is inconsistent with replicate means",
            failures,
        )

    true_basal, true_input_a, true_input_b, true_dual = true_columns
    true_first = measurements.groupby("sample_id", sort=False)[
        [*true_columns, "historical_score_calibration_factor", "true_score"]
    ].first()
    true_derived = np.minimum.reduce(
        [
            true_first[true_input_a].to_numpy(float),
            true_first[true_input_b].to_numpy(float),
            true_first[true_dual].to_numpy(float),
        ]
    ) / true_first[true_basal].to_numpy(float)
    _require(
        _allclose(
            true_first["true_score"].to_numpy(float),
            true_derived
            * true_first["historical_score_calibration_factor"].to_numpy(float),
        ),
        "historical true OR scores are not anchored to true expression",
        failures,
    )

    noise_values = measurements[noise_columns].to_numpy(float).ravel()
    empirical_sigma = float(np.std(noise_values, ddof=1))
    _require(
        0.07 <= empirical_sigma <= 0.13,
        f"empirical log10 noise sigma {empirical_sigma:.5f} is implausible",
        failures,
    )
    return empirical_sigma


def validate_results(
    path: Path,
    *,
    constructs: int,
    failures: list[str],
) -> None:
    try:
        results = pd.read_csv(path)
    except Exception as exc:
        failures.append(f"cannot read results.csv: {exc}")
        return

    required = {"round", "train_size", "selected_sample_ids", "best_true_score_found"}
    missing = sorted(required - set(results.columns))
    if missing:
        failures.append(f"results columns missing: {missing}")
        return
    _require(
        len(results) == EVALUATED_ROUNDS,
        f"results rows={len(results)}, expected {EVALUATED_ROUNDS}",
        failures,
    )
    _require(
        results["round"].astype(int).tolist() == list(range(EVALUATED_ROUNDS)),
        "results rounds are not exactly 0..29 in order",
        failures,
    )
    expected_train_sizes = np.arange(EVALUATED_ROUNDS) * constructs
    _require(
        np.array_equal(results["train_size"].to_numpy(int), expected_train_sizes),
        "results train_size does not match cumulative constructs",
        failures,
    )
    try:
        selected = [ast.literal_eval(value) for value in results["selected_sample_ids"]]
        _require(
            all(len(values) == constructs for values in selected),
            "selected_sample_ids does not contain the allocation size each round",
            failures,
        )
        flat = [sample_id for values in selected for sample_id in values]
        _require(
            len(flat) == len(set(flat)),
            "results selected_sample_ids contains duplicate selections",
            failures,
        )
    except (SyntaxError, ValueError, TypeError) as exc:
        failures.append(f"cannot parse selected_sample_ids: {exc}")


def validate_training_observations(
    path: Path,
    *,
    constructs: int,
    replicates: int,
    failures: list[str],
) -> None:
    try:
        training = pd.read_csv(path)
    except Exception as exc:
        failures.append(f"cannot read training_observations.csv: {exc}")
        return

    sigma_columns = [
        f"estimated_sigma_log10_expression_{column}" for column in EXPRESSION_COLUMNS
    ]
    required = {
        "training_round",
        "sample_id",
        "selected_round",
        "replicate_count",
        "aggregation_method",
        "observed_score",
        "train_yvar",
        *sigma_columns,
    }
    missing = sorted(required - set(training.columns))
    if missing:
        failures.append(f"training observation columns missing: {missing}")
        return

    expected_rows = constructs * sum(range(1, EVALUATED_ROUNDS))
    _require(
        len(training) == expected_rows,
        f"training observation rows={len(training)}, expected {expected_rows}",
        failures,
    )
    round_counts = training.groupby("training_round").size()
    expected_counts = pd.Series(
        {
            training_round: constructs * training_round
            for training_round in range(1, EVALUATED_ROUNDS)
        }
    )
    _require(
        round_counts.equals(expected_counts),
        "training observation counts do not match the cumulative design",
        failures,
    )
    _require(
        set(training["replicate_count"].astype(int)) == {replicates},
        "training replicate_count does not match allocation",
        failures,
    )
    _require(
        set(training["aggregation_method"].astype(str))
        == {"score_from_mean_expression"},
        "training aggregation method is not score_from_mean_expression",
        failures,
    )
    if replicates == 1:
        _require(
            bool(training["train_yvar"].isna().all()),
            "24x1 should use observed-only GP noise without train_yvar",
            failures,
        )
    else:
        _require(
            bool(
                np.isfinite(training["train_yvar"].to_numpy(float)).all()
                and (training["train_yvar"].to_numpy(float) > 0).all()
            ),
            "replicated allocations require finite positive train_yvar",
            failures,
        )
        _require(
            _finite_numeric(training, sigma_columns),
            "replicated allocations require finite pooled expression sigma estimates",
            failures,
        )


def validate_run(sweep_dir: Path, summary_path: Path) -> RunValidation:
    try:
        key = key_from_summary_path(sweep_dir, summary_path)
    except Exception as exc:
        return RunValidation(RunKey("?", "?", -1), [str(exc)])

    failures: list[str] = []
    if key.allocation not in ALLOCATIONS:
        return RunValidation(key, [f"unknown allocation {key.allocation}"])
    constructs, replicates = ALLOCATIONS[key.allocation]

    try:
        with summary_path.open() as handle:
            summary = json.load(handle)
    except Exception as exc:
        return RunValidation(key, [f"cannot read summary.json: {exc}"])

    simulation = summary.get("measurement_simulation") or {}
    _require(
        int(summary.get("completed_rounds", -1)) == EVALUATED_ROUNDS,
        "summary completed_rounds is not 30",
        failures,
    )
    _require(
        str(summary.get("query_strategy", "")).upper() == "MES",
        "summary query strategy is not MES",
        failures,
    )
    _require(
        str(summary.get("embedding_model", "")) == EMBEDDING_MODEL,
        "summary embedding model is incorrect",
        failures,
    )
    _require(
        str(summary.get("dataset_name", "")) == key.dataset_name,
        "summary dataset name does not match path",
        failures,
    )
    _require(
        int(summary.get("seed", -1)) == key.seed,
        "summary seed does not match path",
        failures,
    )
    _require(
        bool(simulation.get("enabled")), "measurement simulation is disabled", failures
    )
    _require(
        str(simulation.get("score_mode", "")) == SCORE_MODE,
        "summary score mode is not or_score",
        failures,
    )
    _require(
        tuple(simulation.get("expression_columns") or ()) == EXPRESSION_COLUMNS,
        "summary expression columns are incorrect or out of order",
        failures,
    )
    _require(
        int(simulation.get("replicates_per_construct", -1)) == replicates,
        "summary replicate count does not match allocation",
        failures,
    )
    try:
        sigma = float(simulation.get("noise_sigma_log10_expression", -1))
    except (TypeError, ValueError):
        sigma = -1
    _require(
        abs(sigma - NOISE_SIGMA_LOG10_EXPRESSION) < 1e-12,
        "summary noise sigma is not 0.10",
        failures,
    )
    _require(
        len(summary.get("summary_by_round") or []) == EVALUATED_ROUNDS,
        "summary_by_round does not contain 30 rounds",
        failures,
    )

    run_dir = summary_path.parent
    empirical_sigma = validate_measurements(
        run_dir / "simulated_measurements.csv",
        constructs=constructs,
        replicates=replicates,
        failures=failures,
    )
    validate_results(run_dir / "results.csv", constructs=constructs, failures=failures)
    validate_training_observations(
        run_dir / "training_observations.csv",
        constructs=constructs,
        replicates=replicates,
        failures=failures,
    )
    return RunValidation(key, failures, empirical_sigma)


def expected_full_grid(dataset_names: list[str]) -> set[RunKey]:
    return {
        RunKey(allocation, dataset_name, seed)
        for allocation in ALLOCATIONS
        for dataset_name in dataset_names
        for seed in range(NUM_SEEDS)
    }


def validate_sweep(sweep_dir: Path, expected_runs: int) -> dict[str, Any]:
    sweep_dir = sweep_dir.expanduser().resolve()
    dataset_names = load_dataset_names()
    summary_paths = sorted(sweep_dir.glob("*/*/seed_*/summary.json"))
    validations = [validate_run(sweep_dir, path) for path in summary_paths]
    keys = [validation.key for validation in validations]

    global_failures: list[str] = []
    if expected_runs < 1:
        global_failures.append("expected_runs must be positive")
    if len(summary_paths) != expected_runs:
        global_failures.append(
            f"found {len(summary_paths)} summary files, expected {expected_runs}"
        )
    duplicate_keys = sorted(
        (key.display() for key, count in Counter(keys).items() if count > 1)
    )
    if duplicate_keys:
        global_failures.append(f"duplicate run keys: {duplicate_keys[:20]}")

    allowed_datasets = set(dataset_names)
    for key in keys:
        if key.dataset_name not in allowed_datasets:
            global_failures.append(f"unexpected dataset: {key.display()}")
        if key.seed not in range(NUM_SEEDS):
            global_failures.append(f"seed outside 0..29: {key.display()}")

    missing_full_grid: list[str] = []
    unexpected_full_grid: list[str] = []
    if expected_runs == FULL_RUN_COUNT:
        expected_keys = expected_full_grid(dataset_names)
        actual_keys = set(keys)
        missing_full_grid = sorted(key.display() for key in expected_keys - actual_keys)
        unexpected_full_grid = sorted(
            key.display() for key in actual_keys - expected_keys
        )
        if missing_full_grid:
            global_failures.append(
                f"full grid is missing {len(missing_full_grid)} runs"
            )
        if unexpected_full_grid:
            global_failures.append(
                f"full grid contains {len(unexpected_full_grid)} unexpected runs"
            )

    run_failures = [
        {"run": validation.key.display(), "failures": validation.failures}
        for validation in validations
        if validation.failures
    ]
    empirical_sigmas = [
        validation.empirical_noise_sigma
        for validation in validations
        if validation.empirical_noise_sigma is not None and not validation.failures
    ]
    allocation_counts = Counter(key.allocation for key in keys)
    report = {
        "valid": not global_failures and not run_failures,
        "sweep_dir": str(sweep_dir),
        "expected_runs": expected_runs,
        "summary_files_found": len(summary_paths),
        "validated_runs": len(validations) - len(run_failures),
        "allocation_counts": dict(sorted(allocation_counts.items())),
        "global_failures": global_failures,
        "run_failures": run_failures,
        "missing_full_grid": missing_full_grid,
        "unexpected_full_grid": unexpected_full_grid,
        "empirical_noise_sigma": (
            {
                "mean": float(np.mean(empirical_sigmas)),
                "min": float(np.min(empirical_sigmas)),
                "max": float(np.max(empirical_sigmas)),
            }
            if empirical_sigmas
            else None
        ),
    }
    return report


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir.expanduser().resolve()
    report_path = (
        args.report_path.expanduser().resolve()
        if args.report_path
        else sweep_dir / "validation_report.json"
    )
    report = validate_sweep(sweep_dir, args.expected_runs)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    status = "VALID" if report["valid"] else "INVALID"
    print(f"status={status}")
    print(f"summary_files_found={report['summary_files_found']}")
    print(f"validated_runs={report['validated_runs']}")
    print(f"report={report_path}")
    if not report["valid"]:
        for failure in report["global_failures"]:
            print(f"ERROR: {failure}", file=sys.stderr)
        for item in report["run_failures"][:20]:
            print(
                f"ERROR: {item['run']}: {'; '.join(item['failures'])}",
                file=sys.stderr,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
