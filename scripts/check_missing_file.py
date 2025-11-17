#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

# You can customize/extend these aliases without touching the code below.
MODEL_ALIASES_DEFAULT: dict[str, Sequence[str]] = {
    "LINEAR": ("linear_regression", "linear"),
    "KNN": ("knn", "k_nn"),
    "RANDOM_FOREST": ("random_forest", "randomforest", "rf"),
    "XGBOOST": ("xg_boost", "xgboost", "xgb"),
}


@dataclass(frozen=True)
class Config:
    """Configuration for generating/checking expected result names."""

    output_dir: Path
    strategies: Sequence[str]
    seq_mod_methods: Sequence[str]
    regression_models: Sequence[str]
    seeds: Sequence[int]
    model_aliases: dict[str, Sequence[str]]


# ---------- naming helpers --------------------------------------------------


def to_lower_camel(snake_upper: str) -> str:
    """Convert SNAKE_CASE (possibly upper) to lowerCamelCase.

    Examples:
        HIGH_EXPRESSION -> highExpression
        RANDOM -> random
    """
    parts = snake_upper.lower().split("_")
    if not parts:
        return ""
    head, *tail = parts
    return "".join([head, *[p.capitalize() for p in tail]])


def build_stems_for_model(
    strategy: str, method: str, model: str, seed: int, aliases: Sequence[str]
) -> Sequence[str]:
    """Build all possible stems for (strategy, method, model, seed) across aliases.

    Pattern:
        {strategyCamel}_{methodLower}_{modelAlias}_seed_{seed}_results
    """
    base = f"{to_lower_camel(strategy)}_{method.lower()}"
    return [f"{base}_{alias}_seed_{seed}_results" for alias in aliases]


# ---------- existence checks ------------------------------------------------


def path_exists_loose(base: Path, stems: Sequence[str]) -> bool:
    """Return True if any given stem exists as a dir or file prefix in base."""
    for stem in stems:
        d = base / stem
        if d.exists():
            return True
        # accept files like <stem>.json, <stem>.pt, etc.
        if any(base.glob(f"{stem}*")):
            return True
    return False


# ---------- core ------------------------------------------------------------


def expected_items(
    cfg: Config,
) -> Iterable[tuple[tuple[str, str, str, int], Sequence[str]]]:
    """Yield ((strategy, method, model, seed), stems_for_aliases)."""
    for strategy, method, model, seed in product(
        cfg.strategies, cfg.seq_mod_methods, cfg.regression_models, cfg.seeds
    ):
        aliases = cfg.model_aliases.get(model, (model.lower(),))
        stems = build_stems_for_model(strategy, method, model, seed, aliases)
        yield (strategy, method, model, seed), stems


def find_missing(
    cfg: Config,
) -> list[tuple[str, tuple[str, str, str, int]]]:
    """Return list of (representative_stem, factors) that are missing.

    The representative stem is the first alias for that model, for reporting.
    """
    missing: list[tuple[str, tuple[str, str, str, int]]] = []
    for factors, stems in expected_items(cfg):
        if not path_exists_loose(cfg.output_dir, stems):
            # Use the first stem for a clean, canonical report.
            rep_stem = stems[0]
            missing.append((rep_stem, factors))
    return missing


# ---------- I/O -------------------------------------------------------------


def write_missing_csv(
    output_dir: Path, missing: Sequence[tuple[str, tuple[str, str, str, int]]]
) -> Path:
    """Write a CSV with missing entries; returns the CSV path."""
    csv_path = output_dir / "_missing_results.csv"
    lines = ["name,strategy,seq_mod_method,regression_model,seed"]
    for name, (strategy, method, model, seed) in missing:
        lines.append(f"{name},{strategy},{method},{model},{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with defaults matching your config + aliases."""
    parser = argparse.ArgumentParser(
        description="Check which experiment result files/folders are missing."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/big_experiment/onehot_pca_512"),
        help="Directory containing result files/folders.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["HIGH_EXPRESSION", "RANDOM"],
        help="Strategies, e.g., HIGH_EXPRESSION RANDOM.",
    )
    parser.add_argument(
        "--seq_mod_methods",
        nargs="+",
        default=["EMBEDDING"],
        help="Sequence modification methods, e.g., EMBEDDING.",
    )
    parser.add_argument(
        "--regression_models",
        nargs="+",
        default=["LINEAR", "KNN", "RANDOM_FOREST", "XGBOOST"],
        help="Model keys used for alias lookup.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(1, 31)),
        help="Seeds to check.",
    )
    # Optional: allow adding extra aliases from CLI like:
    # --alias XGBOOST:xg_boost,xgboost,xgb --alias KNN:knn,k_nn
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Add/override model aliases, format KEY:alias1,alias2",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Do not write _missing_results.csv.",
    )
    return parser.parse_args()


def parse_alias_overrides(items: Sequence[str]) -> dict[str, Sequence[str]]:
    """Parse --alias KEY:a,b,c items into a dict."""
    overrides: dict[str, Sequence[str]] = {}
    for itm in items:
        if ":" not in itm:
            continue
        key, aliases = itm.split(":", 1)
        aliases_list = tuple(a.strip() for a in aliases.split(",") if a.strip())
        if aliases_list:
            overrides[key.strip()] = aliases_list
    return overrides


def main() -> None:
    """Entry point."""
    args = parse_args()

    aliases = dict(MODEL_ALIASES_DEFAULT)
    aliases.update(parse_alias_overrides(args.alias))

    cfg = Config(
        output_dir=args.output_dir,
        strategies=args.strategies,
        seq_mod_methods=args.seq_mod_methods,
        regression_models=args.regression_models,
        seeds=args.seeds,
        model_aliases=aliases,
    )

    total = (
        len(cfg.strategies)
        * len(cfg.seq_mod_methods)
        * len(cfg.regression_models)
        * len(cfg.seeds)
    )
    missing = find_missing(cfg)

    print(f"Output directory: {cfg.output_dir}")
    print(f"Total expected combos: {total}")
    print(f"Found: {total - len(missing)}")
    print(f"Missing: {len(missing)}\n")

    if missing:
        print("Missing (representative stems):")
        for name, (strategy, method, model, seed) in missing:
            print(
                f"  {name} -> "
                f"(strategy={strategy}, method={method}, model={model}, seed={seed})"
            )
        if not args.no_csv:
            csv_path = write_missing_csv(cfg.output_dir, missing)
            print(f"\nWrote CSV: {csv_path}")
    else:
        print("✅ No missing results. All expected files/folders are present.")


if __name__ == "__main__":
    main()

#     python results/big_experiment/onehot_raw/check_missing_file.py \
#   --alias KNN:KNN_regression \
#   --alias XGBOOST:xg_boost \
#   --output_dir results/big_experiment/onehot_raw
