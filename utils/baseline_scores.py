"""
Compute random baseline summary metrics across datasets.

Example:
    python utils/baseline_scores.py \\
        --datasets-yaml job_sub/datasets/datasets.yaml \\
        --output-csv results/baseline_scores.csv \\
        --num-experiments 1000 \\
        --num-rounds 10 \\
        --num-samples-per-round 12
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

DEFAULT_DATASETS_YAML = (
    Path(__file__).resolve().parents[1] / "job_sub" / "datasets" / "datasets.yaml"
)
DEFAULT_LABEL_KEY = "Fold Change (Induced/Basal)"
DEFAULT_OUTPUT_CSV = (
    Path(__file__).resolve().parents[1] / "results" / "baseline_scores.csv"
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    metadata_path: Path
    subset_ids_path: Path | None = None


def load_dataset_specs(dataset_yaml_path: Path) -> list[DatasetSpec]:
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml_path}")

    with dataset_yaml_path.open("r") as handle:
        payload = yaml.safe_load(handle) or {}

    datasets = payload.get("datasets") or []
    if not datasets:
        raise ValueError(f"No datasets found in {dataset_yaml_path}")

    specs: list[DatasetSpec] = []
    for entry in datasets:
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"Dataset entry missing name in {dataset_yaml_path}")

        metadata_raw = str(entry.get("metadata_path", "")).strip()
        if not metadata_raw:
            raise ValueError(
                f"Dataset '{name}' missing metadata_path in {dataset_yaml_path}"
            )
        metadata_path = Path(metadata_raw).expanduser()
        if not metadata_path.is_absolute():
            metadata_path = (dataset_yaml_path.parent / metadata_path).resolve()

        subset_raw = entry.get("subset_ids_path")
        subset_ids_path = None
        if subset_raw:
            subset_ids_path = Path(str(subset_raw)).expanduser()
            if not subset_ids_path.is_absolute():
                subset_ids_path = (dataset_yaml_path.parent / subset_ids_path).resolve()

        specs.append(
            DatasetSpec(
                name=name,
                metadata_path=metadata_path,
                subset_ids_path=subset_ids_path,
            )
        )

    return specs


def load_subset_ids(subset_ids_path: Path) -> np.ndarray:
    subset_ids = []
    for line in subset_ids_path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            subset_ids.append(int(text))
        except ValueError as exc:
            raise ValueError(
                f"Invalid sample id '{text}' in subset file {subset_ids_path}"
            ) from exc
    if not subset_ids:
        raise ValueError(f"Subset ids file {subset_ids_path} did not contain any ids.")
    return np.asarray(subset_ids, dtype=np.int64)


def load_label_array(
    metadata_path: Path,
    label_key: str,
    label_cache: dict[tuple[Path, str], np.ndarray],
) -> np.ndarray:
    cache_key = (metadata_path, label_key)
    if cache_key in label_cache:
        return label_cache[cache_key]

    try:
        df = pd.read_csv(metadata_path, usecols=[label_key])
    except ValueError as exc:
        raise ValueError(
            f"Label key '{label_key}' not found in {metadata_path}"
        ) from exc

    series = pd.to_numeric(df[label_key], errors="coerce")
    label_cache[cache_key] = series.to_numpy()
    return label_cache[cache_key]


def load_labels(
    dataset: DatasetSpec,
    label_key: str,
    label_cache: dict[tuple[Path, str], np.ndarray],
    subset_cache: dict[Path, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    labels = load_label_array(dataset.metadata_path, label_key, label_cache)
    sample_ids = np.arange(len(labels), dtype=np.int64)

    if dataset.subset_ids_path is not None:
        subset_ids_path = dataset.subset_ids_path
        if subset_ids_path not in subset_cache:
            subset_cache[subset_ids_path] = load_subset_ids(subset_ids_path)
        subset_ids = subset_cache[subset_ids_path]
        if np.any(subset_ids < 0) or subset_ids.max() >= len(labels):
            raise ValueError(
                f"Subset ids in {subset_ids_path} are out of bounds for "
                f"{dataset.metadata_path} (len={len(labels)})"
            )
        labels = labels[subset_ids]
        sample_ids = subset_ids

    finite_mask = np.isfinite(labels)
    labels = labels[finite_mask]
    sample_ids = sample_ids[finite_mask]
    if labels.size == 0:
        raise ValueError(
            f"No finite labels found for dataset '{dataset.name}' after filtering."
        )
    return labels, sample_ids


def build_top_mask(labels: np.ndarray, top_p: float) -> np.ndarray:
    num_top = max(1, int(len(labels) * top_p))
    if num_top >= len(labels):
        return np.ones(len(labels), dtype=bool)
    top_indices = np.argpartition(labels, -num_top)[-num_top:]
    top_mask = np.zeros(len(labels), dtype=bool)
    top_mask[top_indices] = True
    return top_mask


def draw_random_rounds(
    num_samples: int,
    num_rounds: int,
    num_samples_per_round: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_rounds <= 0 or num_samples_per_round <= 0:
        raise ValueError("num_rounds and num_samples_per_round must be > 0.")
    total_samples = num_rounds * num_samples_per_round
    if total_samples > num_samples:
        raise ValueError(
            "Cannot sample without replacement: requested samples exceed dataset size."
        )
    selections = rng.choice(num_samples, size=total_samples, replace=False)
    return selections.reshape(num_rounds, num_samples_per_round)


def compute_random_summary_metrics(
    labels: np.ndarray,
    top_mask: np.ndarray,
    max_label: float,
    num_rounds: int,
    num_samples_per_round: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rounds = draw_random_rounds(
        num_samples=len(labels),
        num_rounds=num_rounds,
        num_samples_per_round=num_samples_per_round,
        rng=rng,
    )
    round_labels = labels[rounds]
    normalized_true = round_labels.max(axis=1) / max_label
    n_top = top_mask[rounds].sum(axis=1).astype(np.float64)

    auc_true = float(np.sum(np.maximum.accumulate(normalized_true)) / num_rounds)
    overall_true = float(np.max(normalized_true))
    selected_per_round = np.full(num_rounds, num_samples_per_round, dtype=np.float64)
    cumulative_selected = float(np.sum(np.cumsum(selected_per_round)))
    avg_top = (
        float(np.sum(np.cumsum(n_top)) / cumulative_selected)
        if cumulative_selected > 0
        else 0.0
    )

    return {
        "auc_true": auc_true,
        "avg_top": avg_top,
        "overall_true": overall_true,
        "max_train_spearman": float("nan"),
        "max_pool_spearman": float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute random baseline metrics for each dataset in a YAML file."
    )
    parser.add_argument(
        "--datasets-yaml",
        default=str(DEFAULT_DATASETS_YAML),
        help="Path to the datasets YAML file.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to save aggregated results as CSV.",
    )
    parser.add_argument(
        "--label-key",
        default=DEFAULT_LABEL_KEY,
        help="Column name in the metadata CSV containing target labels.",
    )
    parser.add_argument("--num-experiments", type=int, default=10000)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--num-samples-per-round", type=int, default=12)
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.01,
        help="Top percentage used for avg_top (matches active learning defaults).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name to include (can be repeated).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_yaml_path = Path(args.datasets_yaml).expanduser()
    output_csv = Path(args.output_csv).expanduser()

    if args.num_experiments <= 0:
        raise ValueError("num_experiments must be > 0.")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top_p must be between 0 and 1.")

    datasets = load_dataset_specs(dataset_yaml_path)
    if args.dataset:
        requested = set(args.dataset)
        datasets = [dataset for dataset in datasets if dataset.name in requested]
        missing = requested - {dataset.name for dataset in datasets}
        if missing:
            raise ValueError(
                f"Requested datasets not found in {dataset_yaml_path}: {sorted(missing)}"
            )

    label_cache: dict[tuple[Path, str], np.ndarray] = {}
    subset_cache: dict[Path, np.ndarray] = {}
    experiment_seeds = np.arange(args.num_experiments, dtype=np.int64)

    random_states = []
    for dataset in tqdm(datasets):
        labels, _ = load_labels(dataset, args.label_key, label_cache, subset_cache)
        dataset_max_label = float(np.max(labels))
        top_mask = build_top_mask(labels, args.top_p)
        for seed_value in experiment_seeds:
            summary_metrics = compute_random_summary_metrics(
                labels=labels,
                top_mask=top_mask,
                max_label=dataset_max_label,
                num_rounds=args.num_rounds,
                num_samples_per_round=args.num_samples_per_round,
                seed=int(seed_value),
            )
            random_states.append(
                {
                    "dataset_name": dataset.name,
                    "query_strategy": "RANDOM",
                    "predictor": "NONE",
                    "initial_selection": "RANDOM",
                    "embedding_model": "NONE",
                    "feature_transforms": "NONE",
                    "target_transforms": "NONE",
                    "seed": int(seed_value),
                    "overall_true": summary_metrics["overall_true"],
                    "auc_true": summary_metrics["auc_true"],
                    "avg_top": summary_metrics["avg_top"],
                    "max_train_spearman": summary_metrics["max_train_spearman"],
                    "max_pool_spearman": summary_metrics["max_pool_spearman"],
                    "dataset_max_label": dataset_max_label,
                }
            )

    df = pd.DataFrame(random_states)
    if df.empty:
        raise ValueError("No datasets selected; nothing to write.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
