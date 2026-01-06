"""
Aggregate summary.json files from Hydra sweeps into combined CSVs.

Run this script after SubmitIt jobs finish to generate combined_summaries.csv
for every dataset directory under a specific sweep timestamp directory.

Example:
    python job_sub/aggregate_summaries.py --sweep-dir job_sub/multirun/2025-12-30 \\
        --summary-names summary.json,summary_2.json,summary_5.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Set

import pandas as pd
from tqdm import tqdm

_MARKER_NAME = ".summaries_combined"


def combine_summaries(
    sweep_dir: Path,
    dataset_name: Optional[str] = None,
    summary_names: Optional[Sequence[str]] = None,
) -> dict[str, int]:
    """Combine summary files inside the sweep (per dataset) for multiple names."""
    search_dir = sweep_dir / dataset_name if dataset_name else sweep_dir
    if not search_dir.exists():
        search_dir = sweep_dir
    summary_names = summary_names or ["summary.json"]
    name_set = set(summary_names)
    summary_files = sorted(search_dir.rglob("*.json"))
    if not summary_files:
        return dict.fromkeys(summary_names, 0)

    rows_by_name = {name: [] for name in summary_names}
    for path in tqdm(summary_files, desc=f"Reading summaries for {dataset_name}"):
        if path.name not in name_set:
            continue
        try:
            data = json.loads(path.read_text())
            data["summary_path"] = str(path)
            rows_by_name[path.name].append(data)
        except json.JSONDecodeError:
            continue

    counts: dict[str, int] = {}
    for summary_name, rows in rows_by_name.items():
        if not rows:
            counts[summary_name] = 0
            continue
        df = pd.DataFrame(rows)
        safe_stem = Path(summary_name).name.replace("/", "_")
        safe_stem = Path(safe_stem).stem
        output_name = (
            "combined_summaries.csv"
            if summary_name == "summary.json"
            else f"combined_summaries.{safe_stem}.csv"
        )
        output_csv = search_dir / output_name
        df.to_csv(output_csv, index=False)
        counts[summary_name] = len(rows)
        print(f"Combined {len(rows)} summaries into {output_csv}")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help=(
            "Path to a sweep date directory (e.g., job_sub/multirun/2025-12-30) "
            "or a specific sweep timestamp directory "
            "(e.g., job_sub/multirun/2025-12-30/15-30-03)."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="NAME",
        help="Limit aggregation to specific dataset names. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--summary-names",
        dest="summary_names",
        default=None,
        help=(
            "Comma-separated list of summary filenames "
            "(e.g. summary.json,summary_2.json)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recombine even if a dataset sweep already has a marker file.",
    )
    return parser.parse_args()


def aggregate_summaries(
    sweep_dir: Optional[Path] = None,
    datasets: Optional[Sequence[str]] = None,
    summary_names: Optional[Sequence[str]] = None,
    force: bool = False,
) -> int:
    """Combine summary.json files for completed sweeps.

    Args:
        sweep_dir: Path to a sweep date directory or a specific sweep timestamp directory.
        datasets: Optional iterable of dataset names to aggregate. If None, defaults to dataset configs.
        summary_names: Summary filenames to aggregate.
        force: Recombine even if marker files exist.

    Returns:
        Number of dataset directories aggregated.
    """
    dataset_filters: Optional[Set[str]] = None
    if datasets:
        dataset_filters = {name.strip() for name in datasets if name}

    summary_names = summary_names or ["summary.json"]
    marker_names = {}
    for summary_name in summary_names:
        if summary_name == "summary.json":
            marker_names[summary_name] = _MARKER_NAME
        else:
            safe_name = summary_name.replace("/", "_")
            marker_names[summary_name] = f"{_MARKER_NAME}.{safe_name}"

    if sweep_dir is None:
        raise ValueError("sweep_dir is required.")
    sweep_dir = sweep_dir.expanduser().resolve()
    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}")
        return 0
    sweep_dirs = [
        path
        for path in sweep_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    time_dirs = [path for path in sweep_dirs if _is_time_dir(path.name)]
    if time_dirs:
        sweeps = set(time_dirs)
    else:
        sweeps = {sweep_dir}

    aggregated = 0
    for sweep_dir in sorted(sweeps):
        for dataset_dir in _iter_dataset_dirs(sweep_dir):
            dataset_name = dataset_dir.name
            if dataset_filters and dataset_name not in dataset_filters:
                continue
            pending = [
                name
                for name in summary_names
                if force or not (dataset_dir / marker_names[name]).exists()
            ]
            if not pending:
                continue

            counts = combine_summaries(
                sweep_dir, dataset_name=dataset_name, summary_names=pending
            )
            for summary_name in pending:
                marker_path = dataset_dir / marker_names[summary_name]
                if counts.get(summary_name, 0) == 0 and not force:
                    continue
                marker_path.write_text(
                    f"combined at {datetime.now().isoformat(timespec='seconds')}\n"
                )
                aggregated += 1

    if aggregated == 0:
        print("No new datasets required aggregation.")
    else:
        print(f"Aggregated summaries for {aggregated} dataset(s).")
    return aggregated


def _iter_dataset_dirs(sweep_dir: Path) -> List[Path]:
    """Return dataset directories under the given sweep timestamp directory."""
    if not sweep_dir.exists():
        return []
    dirs = [
        path
        for path in sweep_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    return dirs


def _is_time_dir(name: str) -> bool:
    parts = name.split("-")
    if len(parts) != 3:
        return False
    return all(part.isdigit() and len(part) == 2 for part in parts)


def main() -> None:
    args = parse_args()
    summary_names = None
    if args.summary_names:
        summary_names = [
            name.strip() for name in args.summary_names.split(",") if name.strip()
        ]
    aggregate_summaries(
        sweep_dir=args.sweep_dir,
        datasets=args.datasets,
        summary_names=summary_names,
        force=args.force,
    )


if __name__ == "__main__":
    main()
