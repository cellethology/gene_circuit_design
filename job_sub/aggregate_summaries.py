"""
Aggregate summary.json files from Hydra sweeps into combined CSVs.

Run this script after SubmitIt jobs finish to generate combined_summaries.csv
for every dataset directory under a specific sweep timestamp directory.
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
    summary_name: str = "summary.json",
) -> None:
    """Combine all summary files inside the sweep (per dataset)."""
    search_dir = sweep_dir / dataset_name if dataset_name else sweep_dir
    if not search_dir.exists():
        search_dir = sweep_dir
    summary_files = sorted(search_dir.rglob(summary_name))
    if not summary_files:
        return

    rows = []
    for path in tqdm(summary_files, desc=f"Reading {summary_name}"):
        try:
            data = json.loads(path.read_text())
            data["summary_path"] = str(path)
            rows.append(data)
        except json.JSONDecodeError:
            continue

    if not rows:
        return

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
    print(f"Combined {len(rows)} summaries into {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help=(
            "Path to a specific sweep timestamp directory "
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
        "--summary-name",
        action="append",
        dest="summary_names",
        default=["summary.json"],
        help=(
            "Filename to aggregate (default: summary.json). "
            "Repeat to aggregate multiple summary files."
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
        sweep_dir: Path to a specific sweep timestamp directory.
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
    sweeps = {sweep_dir}

    aggregated = 0
    for sweep_dir in sorted(sweeps):
        for dataset_dir in _iter_dataset_dirs(sweep_dir):
            dataset_name = dataset_dir.name
            if dataset_filters and dataset_name not in dataset_filters:
                continue
            for summary_name in summary_names:
                marker_path = dataset_dir / marker_names[summary_name]
                if marker_path.exists() and not force:
                    continue
                combine_summaries(
                    sweep_dir, dataset_name=dataset_name, summary_name=summary_name
                )
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


def main() -> None:
    args = parse_args()
    aggregate_summaries(
        sweep_dir=args.sweep_dir,
        datasets=args.datasets,
        summary_names=args.summary_names,
        force=args.force,
    )


if __name__ == "__main__":
    main()
