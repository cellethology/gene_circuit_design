"""Simulated measurement records for retrospective active-learning runs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_EXPRESSION_SCORE_MODES = {"induced_over_basal", "and_score", "or_score"}
_SCORE_MODES = {"label", *_EXPRESSION_SCORE_MODES}


@dataclass(frozen=True)
class MeasurementSimulationConfig:
    """Configuration for simulated assay measurements."""

    enabled: bool = False
    replicates_per_construct: int = 1
    noise_sigma_log10_expression: float = 0.0
    expression_columns: tuple[str, ...] = ()
    score_mode: str = "label"
    basal_column: str | None = None
    induced_column: str | None = None
    single_input_a_column: str | None = None
    single_input_b_column: str | None = None
    dual_input_column: str | None = None
    confirmation_z: float = 1.96
    train_yvar_floor: float = 1e-12

    @classmethod
    def from_config(cls, value: Any | None) -> MeasurementSimulationConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value

        allowed = {field.name for field in fields(cls)}
        data = {
            key: _get_config_value(value, key)
            for key in allowed
            if _has_config_value(value, key)
        }
        if "expression_columns" in data and data["expression_columns"] is not None:
            data["expression_columns"] = tuple(data["expression_columns"])
        return cls(**data)

    def __post_init__(self) -> None:
        if self.replicates_per_construct < 1:
            raise ValueError("replicates_per_construct must be at least 1.")
        if self.noise_sigma_log10_expression < 0:
            raise ValueError("noise_sigma_log10_expression must be non-negative.")
        if self.confirmation_z < 0:
            raise ValueError("confirmation_z must be non-negative.")
        if self.train_yvar_floor < 0:
            raise ValueError("train_yvar_floor must be non-negative.")
        if self.score_mode not in _SCORE_MODES:
            raise ValueError(
                "score_mode must be one of: " + ", ".join(sorted(_SCORE_MODES))
            )


class MeasurementSimulator:
    """Generate and retain noisy replicate observations for selected constructs."""

    def __init__(
        self,
        *,
        sample_ids: np.ndarray | list[Any],
        true_labels: np.ndarray,
        metadata: pd.DataFrame | None,
        label_key: str,
        config: MeasurementSimulationConfig | None = None,
        random_seed: int = 0,
    ) -> None:
        self.sample_ids = np.asarray(sample_ids)
        self.true_labels = np.asarray(true_labels, dtype=float)
        self.metadata = (
            metadata.reset_index(drop=True) if metadata is not None else None
        )
        self.label_key = label_key
        self.config = config or MeasurementSimulationConfig()
        self.rng = np.random.default_rng(random_seed)

        n_samples = len(self.true_labels)
        self.observed_means = np.full(n_samples, np.nan, dtype=float)
        self.observed_vars = np.full(n_samples, np.nan, dtype=float)
        self.replicate_counts = np.zeros(n_samples, dtype=int)
        self.selected_rounds = np.full(n_samples, -1, dtype=int)
        self.measurement_rows: list[dict[str, Any]] = []
        self.training_observation_rows: list[dict[str, Any]] = []

        self._validate_metadata()

    def measure(self, indices: list[int] | np.ndarray, round_num: int) -> None:
        """Simulate replicate measurements for newly selected constructs."""
        for sample_index in indices:
            sample_index = int(sample_index)
            if self.replicate_counts[sample_index] > 0:
                continue
            if self.selected_rounds[sample_index] < 0:
                self.selected_rounds[sample_index] = int(round_num)

            replicate_scores = []
            for replicate_index in range(self.config.replicates_per_construct):
                score, row_values = self._simulate_one_measurement(sample_index)
                replicate_scores.append(score)
                self.measurement_rows.append(
                    {
                        "round": int(round_num),
                        "sample_index": sample_index,
                        "sample_id": self._sample_id(sample_index),
                        "replicate": replicate_index,
                        "replicates_per_construct": (
                            self.config.replicates_per_construct
                        ),
                        "noise_sigma_log10_expression": (
                            self.config.noise_sigma_log10_expression
                        ),
                        "score_mode": self.config.score_mode,
                        "true_score": float(self.true_labels[sample_index]),
                        "observed_score": float(score),
                        **row_values,
                    }
                )

            values = np.asarray(replicate_scores, dtype=float)
            self.observed_means[sample_index] = float(np.mean(values))
            self.replicate_counts[sample_index] = int(values.size)
            if values.size >= 2:
                self.observed_vars[sample_index] = float(np.var(values, ddof=1))

    def training_targets(self, indices: list[int] | np.ndarray) -> np.ndarray:
        """Return observed replicate means for measured training constructs."""
        values = self.observed_means[np.asarray(indices, dtype=int)]
        if np.any(~np.isfinite(values)):
            raise ValueError(
                "Training requested before all selected samples were measured."
            )
        return values

    def training_y_var(self, indices: list[int] | np.ndarray) -> np.ndarray | None:
        """Return replicate-estimated variance of each observed mean, if available."""
        pooled_var = self.pooled_within_construct_variance()
        if pooled_var is None:
            return None

        counts = self.replicate_counts[np.asarray(indices, dtype=int)]
        if np.any(counts < 1):
            raise ValueError("Training variance requested for unmeasured samples.")
        y_var = pooled_var / counts.astype(float)
        if self.config.train_yvar_floor > 0:
            y_var = np.maximum(y_var, self.config.train_yvar_floor)
        return y_var

    def record_training_observations(
        self,
        *,
        indices: list[int] | np.ndarray,
        training_round: int,
        train_y_var: np.ndarray | None,
    ) -> None:
        """Record the observations and variances used in one model fit."""
        index_array = np.asarray(indices, dtype=int)
        y_var_values = (
            np.full(len(index_array), np.nan, dtype=float)
            if train_y_var is None
            else np.asarray(train_y_var, dtype=float)
        )
        pooled_var = self.pooled_within_construct_variance()
        pooled_value = float(pooled_var) if pooled_var is not None else np.nan

        for sample_index, train_var in zip(index_array, y_var_values, strict=True):
            self.training_observation_rows.append(
                {
                    "training_round": int(training_round),
                    "sample_index": int(sample_index),
                    "sample_id": self._sample_id(int(sample_index)),
                    "selected_round": int(self.selected_rounds[int(sample_index)]),
                    "replicate_count": int(self.replicate_counts[int(sample_index)]),
                    "true_score": float(self.true_labels[int(sample_index)]),
                    "observed_score_mean": float(
                        self.observed_means[int(sample_index)]
                    ),
                    "observed_score_var": _finite_or_nan(
                        self.observed_vars[int(sample_index)]
                    ),
                    "pooled_observed_score_var": pooled_value,
                    "train_yvar": _finite_or_nan(train_var),
                }
            )

    def confirmed_true_top_indices(
        self, indices: list[int] | np.ndarray, top_p: float
    ) -> np.ndarray:
        """Return selected indices that are truly top and confirmed by observations."""
        threshold = self.top_threshold(top_p)
        confirmed = []
        pooled_var = self.pooled_within_construct_variance()

        for sample_index in np.asarray(indices, dtype=int):
            mean = self.observed_means[sample_index]
            if not np.isfinite(mean):
                continue
            true_top = self.true_labels[sample_index] >= threshold
            if self.replicate_counts[sample_index] <= 1:
                called_top = mean >= threshold
            else:
                var = pooled_var
                if var is None or not np.isfinite(var):
                    var = self.observed_vars[sample_index]
                if not np.isfinite(var):
                    var = 0.0
                se = float(np.sqrt(max(var, 0.0) / self.replicate_counts[sample_index]))
                called_top = mean - self.config.confirmation_z * se >= threshold
            if true_top and called_top:
                confirmed.append(int(sample_index))

        return np.asarray(confirmed, dtype=int)

    def top_threshold(self, top_p: float) -> float:
        """Return the true-label threshold for the top fraction."""
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        count = max(1, int(len(self.true_labels) * top_p))
        return float(np.sort(self.true_labels)[-count])

    def pooled_within_construct_variance(self) -> float | None:
        """Estimate shared assay variance from replicated constructs."""
        mask = (self.replicate_counts >= 2) & np.isfinite(self.observed_vars)
        if not np.any(mask):
            return None
        weights = self.replicate_counts[mask] - 1
        if np.sum(weights) <= 0:
            return None
        variance = np.sum(weights * self.observed_vars[mask]) / np.sum(weights)
        return float(max(variance, 0.0))

    def save_outputs(self, output_dir: Path) -> None:
        """Write replicate-level and training-observation records."""
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.measurement_rows).to_csv(
            output_dir / "simulated_measurements.csv",
            index=False,
        )
        pd.DataFrame(self.training_observation_rows).to_csv(
            output_dir / "training_observations.csv",
            index=False,
        )

    def _simulate_one_measurement(
        self, sample_index: int
    ) -> tuple[float, dict[str, Any]]:
        if self.config.score_mode == "induced_over_basal":
            return self._simulate_induced_over_basal(sample_index)
        if self.config.score_mode in {"and_score", "or_score"}:
            return self._simulate_multi_input_score(sample_index)
        return self._simulate_label(sample_index)

    def _simulate_label(self, sample_index: int) -> tuple[float, dict[str, Any]]:
        true_score = float(self.true_labels[sample_index])
        if self.config.noise_sigma_log10_expression == 0:
            observed_score = true_score
            log10_noise = 0.0
        elif true_score > 0:
            log10_noise = float(
                self.rng.normal(0.0, self.config.noise_sigma_log10_expression)
            )
            observed_score = true_score * 10**log10_noise
        else:
            log10_noise = float(
                self.rng.normal(0.0, self.config.noise_sigma_log10_expression)
            )
            observed_score = true_score + log10_noise
        return observed_score, {"score_log10_noise": log10_noise}

    def _simulate_induced_over_basal(
        self, sample_index: int
    ) -> tuple[float, dict[str, Any]]:
        basal_column, induced_column = self._two_state_columns()
        noisy_by_column, values = self._simulate_expression_values(sample_index)
        score = _safe_ratio(
            noisy_by_column[induced_column], noisy_by_column[basal_column]
        )
        return score, values

    def _simulate_multi_input_score(
        self, sample_index: int
    ) -> tuple[float, dict[str, Any]]:
        basal_column, input_a_column, input_b_column, dual_column = (
            self._multi_input_columns()
        )
        noisy_by_column, values = self._simulate_expression_values(sample_index)

        if self.config.score_mode == "and_score":
            denominator = max(
                noisy_by_column[basal_column],
                noisy_by_column[input_a_column],
                noisy_by_column[input_b_column],
            )
            score = _safe_ratio(noisy_by_column[dual_column], denominator)
        else:
            numerator = min(
                noisy_by_column[input_a_column],
                noisy_by_column[input_b_column],
                noisy_by_column[dual_column],
            )
            score = _safe_ratio(numerator, noisy_by_column[basal_column])
        return score, values

    def _simulate_expression_values(
        self, sample_index: int
    ) -> tuple[dict[str, float], dict[str, Any]]:
        if self.metadata is None:
            raise ValueError(
                f"metadata is required for {self.config.score_mode} simulation."
            )

        row = self.metadata.iloc[sample_index]
        values: dict[str, Any] = {}
        noisy_by_column: dict[str, float] = {}

        for column in self._expression_columns_to_simulate():
            true_expression = float(row[column])
            if true_expression < 0:
                raise ValueError(f"Expression column '{column}' must be non-negative.")
            log10_noise = (
                0.0
                if self.config.noise_sigma_log10_expression == 0
                else float(
                    self.rng.normal(0.0, self.config.noise_sigma_log10_expression)
                )
            )
            noisy_expression = true_expression * 10**log10_noise
            noisy_by_column[column] = noisy_expression
            values[f"true_expression_{column}"] = true_expression
            values[f"observed_expression_{column}"] = noisy_expression
            values[f"log10_noise_{column}"] = log10_noise

        return noisy_by_column, values

    def _validate_metadata(self) -> None:
        if self.config.score_mode not in _EXPRESSION_SCORE_MODES:
            return
        if self.metadata is None:
            raise ValueError(
                f"metadata is required for {self.config.score_mode} simulation."
            )
        self._score_columns()
        required = set(self._expression_columns_to_simulate())
        missing = sorted(required - set(self.metadata.columns))
        if missing:
            raise KeyError(f"Missing expression column(s): {missing}")

    def _score_columns(self) -> tuple[str, ...]:
        if self.config.score_mode == "induced_over_basal":
            return self._two_state_columns()
        if self.config.score_mode in {"and_score", "or_score"}:
            return self._multi_input_columns()
        return ()

    def _two_state_columns(self) -> tuple[str, str]:
        columns = self.config.expression_columns
        if self.config.basal_column is None and len(columns) < 1:
            raise ValueError(
                "induced_over_basal requires a basal_column or at least one "
                "expression column."
            )
        if self.config.induced_column is None and len(columns) < 2:
            raise ValueError(
                "induced_over_basal requires an induced_column or at least two "
                "expression columns."
            )
        return (
            self.config.basal_column or columns[0],
            self.config.induced_column or columns[1],
        )

    def _multi_input_columns(self) -> tuple[str, str, str, str]:
        columns = self.config.expression_columns
        named_columns = (
            self.config.basal_column,
            self.config.single_input_a_column,
            self.config.single_input_b_column,
            self.config.dual_input_column,
        )
        names = ("basal", "single_input_a", "single_input_b", "dual_input")
        resolved: list[str] = []
        for index, (name, configured) in enumerate(
            zip(names, named_columns, strict=True)
        ):
            if configured is not None:
                resolved.append(configured)
            elif len(columns) > index:
                resolved.append(columns[index])
            else:
                raise ValueError(
                    f"{self.config.score_mode} requires {name}_column or at least "
                    "four expression columns ordered as basal, single input A, "
                    "single input B, and dual input."
                )
        return resolved[0], resolved[1], resolved[2], resolved[3]

    def _expression_columns_to_simulate(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys((*self.config.expression_columns, *self._score_columns()))
        )

    def _sample_id(self, sample_index: int) -> Any:
        value = self.sample_ids[sample_index]
        if hasattr(value, "item"):
            return value.item()
        return value


def _has_config_value(config: Any, key: str) -> bool:
    if isinstance(config, dict):
        return key in config
    try:
        return hasattr(config, key)
    except Exception:
        return False


def _get_config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)


def _finite_or_nan(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else np.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.divide(np.float64(numerator), np.float64(denominator)))
