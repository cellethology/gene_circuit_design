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
) -> np.ndarray:
    labels = load_label_array(dataset.metadata_path, label_key, label_cache)

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

    labels = labels[np.isfinite(labels)]
    if labels.size == 0:
        raise ValueError(
            f"No finite labels found for dataset '{dataset.name}' after filtering."
        )
    return labels


def get_overall_true_auc_true(
    labels: np.ndarray,
    num_experiments: int,
    num_rounds: int = 10,
    num_samples_per_round: int = 12,
    rng: np.random.Generator | None = None,
    replace: bool = True,
    seeds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if num_rounds <= 0 or num_samples_per_round <= 0 or num_experiments <= 0:
        raise ValueError(
            "num_rounds, num_samples_per_round, and num_experiments must be > 0"
        )
    if seeds is not None and len(seeds) != num_experiments:
        raise ValueError("Seeds length must match num_experiments.")
    if seeds is None and rng is None:
        rng = np.random.default_rng()

    labels = np.asarray(labels)
    max_val = np.max(labels)
    if max_val <= 0:
        raise ValueError("Labels must have a positive max value for normalization.")

    normalized_labels = labels / max_val
    samples_per_experiment = num_rounds * num_samples_per_round
    if not replace and samples_per_experiment > len(normalized_labels):
        raise ValueError(
            "Cannot sample without replacement: requested samples exceed dataset size."
        )

    overall_true = np.empty(num_experiments, dtype=np.float64)
    auc_true = np.empty(num_experiments, dtype=np.float64)
    for i in range(num_experiments):
        sample_rng = rng if seeds is None else np.random.default_rng(int(seeds[i]))
        samples = sample_rng.choice(
            normalized_labels, size=samples_per_experiment, replace=replace
        )
        samples = samples.reshape(num_rounds, -1).max(axis=1)
        overall_true[i] = np.max(samples)
        auc_true[i] = np.sum(np.maximum.accumulate(samples)) / num_rounds
    return overall_true, auc_true


def generate_experiment_seeds(
    num_experiments: int,
    base_seed: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(base_seed)
    max_seed = np.iinfo(np.uint32).max
    return rng.integers(0, max_seed, size=num_experiments, dtype=np.uint32)


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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name to include (can be repeated).",
    )
    replacement_group = parser.add_mutually_exclusive_group()
    replacement_group.add_argument(
        "--with-replacement",
        dest="with_replacement",
        action="store_true",
        help="Sample with replacement within each experiment.",
    )
    replacement_group.add_argument(
        "--no-replacement",
        dest="with_replacement",
        action="store_false",
        help="Sample without replacement within each experiment (default).",
    )
    parser.set_defaults(with_replacement=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_yaml_path = Path(args.datasets_yaml).expanduser()
    output_csv = Path(args.output_csv).expanduser()

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
    experiment_seeds = generate_experiment_seeds(args.num_experiments, args.seed)

    random_states = []
    for dataset in tqdm(datasets):
        labels = load_labels(dataset, args.label_key, label_cache, subset_cache)
        overall_true, auc_true = get_overall_true_auc_true(
            labels=labels,
            num_experiments=args.num_experiments,
            num_rounds=args.num_rounds,
            num_samples_per_round=args.num_samples_per_round,
            replace=args.with_replacement,
            seeds=experiment_seeds,
        )
        for seed_value, overall_value, auc_value in zip(
            experiment_seeds, overall_true, auc_true
        ):
            random_states.append(
                {
                    "dataset_name": dataset.name,
                    "query_strategy": "RANDOM",
                    "predictor": "None",
                    "initial_selection": "RANDOM",
                    "embedding_model": "None",
                    "feature_transforms": "None",
                    "target_transforms": "None",
                    "seed": int(seed_value),
                    "overall_true": float(overall_value),
                    "auc_true": float(auc_value),
                }
            )

    df = pd.DataFrame(random_states)
    if df.empty:
        raise ValueError("No datasets selected; nothing to write.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
