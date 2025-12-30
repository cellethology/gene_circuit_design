"""
Aggregate summary.json files from Hydra sweeps into combined CSVs.

Run this script after SubmitIt jobs finish to generate combined_summaries.csv
for every dataset directory under job_sub/multirun.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Set

from job_sub.utils.config_utils import ensure_resolvers, load_dataset_configs
from job_sub.utils.sweep_utils import combine_summaries, list_sweep_dirs

_SCRIPT_PATH = Path(__file__).resolve()
_DEFAULT_BASE = _SCRIPT_PATH.parent / "multirun"
_MARKER_NAME = ".summaries_combined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-base",
        type=Path,
        default=_DEFAULT_BASE,
        help="Root directory containing Hydra multirun outputs (default: job_sub/multirun).",
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        help=(
            "Optional path to a specific sweep timestamp directory "
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
        "--force",
        action="store_true",
        help="Recombine even if a dataset sweep already has a marker file.",
    )
    return parser.parse_args()


def aggregate_summaries(
    sweep_base: Path = _DEFAULT_BASE,
    sweep_dir: Optional[Path] = None,
    datasets: Optional[Sequence[str]] = None,
    force: bool = False,
) -> int:
    """Combine summary.json files for completed sweeps.

    Args:
        sweep_base: Root directory containing Hydra multirun outputs.
        sweep_dir: Optional path to a specific sweep timestamp directory.
        datasets: Optional iterable of dataset names to aggregate. If None, defaults to dataset configs.
        force: Recombine even if marker files exist.

    Returns:
        Number of dataset directories aggregated.
    """
    ensure_resolvers()
    dataset_filters: Optional[Set[str]] = None
    if datasets:
        dataset_filters = {name.strip() for name in datasets if name}
    elif datasets is None:
        # Default to dataset names from config to avoid accidental aggregation of temporary dirs.
        try:
            dataset_filters = {cfg.name for cfg in load_dataset_configs()}
        except Exception as exc:
            print(
                f"[WARN] Failed to load dataset configs ({exc}); "
                "aggregating all datasets found."
            )
            dataset_filters = None

    if sweep_dir is not None:
        sweep_dir = sweep_dir.expanduser().resolve()
        if not sweep_dir.exists():
            print(f"Sweep directory not found: {sweep_dir}")
            return 0
        sweeps = {sweep_dir}
    else:
        sweeps = list_sweep_dirs(sweep_base)
    if not sweeps:
        print(f"No sweeps found under {sweep_base}")
        return 0

    aggregated = 0
    for sweep_dir in sorted(sweeps):
        for dataset_dir in _iter_dataset_dirs(sweep_dir):
            dataset_name = dataset_dir.name
            if dataset_filters and dataset_name not in dataset_filters:
                continue
            marker_path = dataset_dir / _MARKER_NAME
            if marker_path.exists() and not force:
                continue
            combine_summaries(sweep_dir, dataset_name=dataset_name)
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
        sweep_base=args.sweep_base,
        sweep_dir=args.sweep_dir,
        datasets=args.datasets,
        force=args.force,
    )


if __name__ == "__main__":
    main()
