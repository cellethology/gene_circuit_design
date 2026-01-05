#!/usr/bin/env python3
"""
Generate summary_n.json files for partial runs based on results.csv.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SUMMARY_METRIC_RULES = {
    "auc_true": ("max_accumulate", "normalized_true"),
    "avg_top": ("top_mean", "n_top"),
    "overall_true": ("max_overall", "normalized_true"),
    "max_train_spearman": ("max_overall", "train_spearman"),
    "max_pool_spearman": ("max_overall", "pool_spearman"),
}


def _parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_selected_ids(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [item for item in text.split(",") if item.strip()]
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    for row in rows:
        try:
            row["_round"] = int(row.get("round", 0))
        except (TypeError, ValueError):
            row["_round"] = 0
    return sorted(rows, key=lambda r: r["_round"])


def _compute_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {name: float("nan") for name in SUMMARY_METRIC_RULES}

    columns = set(rows[0].keys())
    n_top_col = None
    if "n_top" in columns:
        n_top_col = "n_top"
    elif "n_selected_in_top" in columns:
        n_top_col = "n_selected_in_top"

    selected_counts = [
        len(_parse_selected_ids(row.get("selected_sample_ids"))) for row in rows
    ]
    cumulative_selected = int(np.sum(np.cumsum(selected_counts)))

    summary: dict[str, float] = {}
    for metric_name, (rule, metric_column) in SUMMARY_METRIC_RULES.items():
        column = metric_column
        if metric_column == "n_top" and n_top_col is not None:
            column = n_top_col
        if column not in columns:
            summary[metric_name] = float("nan")
            continue

        values = np.array([_parse_float(row.get(column)) for row in rows], dtype=float)
        if rule == "top_mean":
            if cumulative_selected <= 0:
                summary[metric_name] = 0.0
            else:
                cumulative_sum = np.cumsum(values)
                summary[metric_name] = (
                    float(np.sum(cumulative_sum)) / cumulative_selected
                )
        elif rule == "mean":
            summary[metric_name] = float(np.nanmean(values))
        elif rule == "max_accumulate":
            cumulative_max = np.maximum.accumulate(values)
            summary[metric_name] = float(np.sum(cumulative_max)) / len(values)
        elif rule == "max_overall":
            finite = values[np.isfinite(values)]
            summary[metric_name] = (
                float(np.max(finite)) if finite.size else float("nan")
            )
        else:
            raise ValueError(f"Unknown summary metric rule: {rule}")

    return summary


def _load_base_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _resolve_metrics_to_update(_: dict[str, Any]) -> list[str]:
    return list(SUMMARY_METRIC_RULES.keys())


def _iter_results(root: Path) -> Iterable[Path]:
    return root.rglob("results.csv")


def _write_summary(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create summary_n.json files for partial runs based on results.csv."
        )
    )
    parser.add_argument(
        "multirun_dir",
        type=Path,
        help="Path to the multirun folder (e.g. job_sub/multirun/2026-01-01).",
    )
    parser.add_argument(
        "--n",
        type=str,
        default=None,
        help="Comma-separated list of n values to generate (default: all rounds).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing summary_n.json files.",
    )
    args = parser.parse_args()

    multirun_dir = args.multirun_dir
    if not multirun_dir.exists():
        raise SystemExit(f"Multirun dir not found: {multirun_dir}")

    explicit_ns: list[int] | None = None
    if args.n:
        explicit_ns = [int(item) for item in args.n.split(",") if item.strip()]

    for results_path in _iter_results(multirun_dir):
        run_dir = results_path.parent
        rows = _load_rows(results_path)
        if not rows:
            continue

        base_summary = _load_base_summary(run_dir / "summary.json")
        metrics_to_update = _resolve_metrics_to_update(base_summary)

        max_rounds = len(rows)
        ns = explicit_ns or list(range(1, max_rounds + 1))
        for n in ns:
            if n < 1 or n > max_rounds:
                continue
            output_path = run_dir / f"summary_{n}.json"
            if output_path.exists() and not args.overwrite:
                continue
            partial_rows = rows[:n]
            computed = _compute_summary(partial_rows)
            summary = dict(base_summary)
            for key in metrics_to_update:
                summary[key] = computed.get(key, float("nan"))
            _write_summary(output_path, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
