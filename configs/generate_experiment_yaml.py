#!/usr/bin/env python3
# generate_experiments_yaml.py
"""
Generate a YAML experiments config from a folder of *.safetensors files whose names
follow the pattern: "<PREFIX>-IDR-ZF_Aff_<a>-<b>-<c>_pca.safetensors".

Example filename -> YAML key:
  AD-IDR-ZF_Aff_1-2-1_pca.safetensors  ->  "AD-1-2-1"

Usage:
  python generate_experiment_yaml.py \
  --input-dir ./by_group_with_sequences_onehot_pca \
  --output-yaml ./experiments.yaml \
  --output-dir-root results/166k_2024_regulators

Optional overrides (see --help) let you change defaults like strategies, models, etc.

Python: 3.10+
python generate_experiment_yaml.py \
  --input-dir data_new/166k/sei/cis_grouped_safetensors \
  --output-yaml configs/cis_regulations/166k_cis_regulators_sei_experiments.yaml \
  --output-dir-root results/166k_2024_cis_regulators_auto_gen_1_30/ \
  --output-dir-second sei_pca --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

try:
    import yaml  # PyYAML
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc


# -----------------------------
# Data model & defaults
# -----------------------------
@dataclass(frozen=True)
class ExperimentDefaults:
    """Holds default fields to stamp into each experiment block."""

    strategies: Tuple[str, ...] = ("HIGH_EXPRESSION", "RANDOM")
    seq_mod_methods: Tuple[str, ...] = ("EMBEDDING",)
    target_val_key: str = "expressions"
    regression_models: Tuple[str, ...] = (
        "LINEAR",
        "KNN",
        "RANDOM_FOREST",
        "XGBOOST",
        "MLP",
    )
    seeds: Tuple[int, ...] = tuple(range(1, 21))
    initial_sample_size: int = 8
    batch_size: int = 8
    test_size: int = 30
    max_rounds: int = 20
    normalize_expression: bool = True
    no_test: bool = True
    normalize_input_output: bool = True

    def as_mapping(self) -> Mapping[str, object]:
        """Return a plain mapping that can be merged into each experiment entry."""
        return {
            "strategies": list(self.strategies),
            "seq_mod_methods": list(self.seq_mod_methods),
            "target_val_key": self.target_val_key,
            "regression_models": list(self.regression_models),
            "seeds": list(self.seeds),
            "initial_sample_size": self.initial_sample_size,
            "batch_size": self.batch_size,
            "test_size": self.test_size,
            "max_rounds": self.max_rounds,
            "normalize_expression": self.normalize_expression,
            "no_test": self.no_test,
            "normalize_input_output": self.normalize_input_output,
        }


# Original pattern: PREFIX-IDR-ZF_Aff_a-b-c(_pca).safetensors
FILE_REGEX_3NUM = re.compile(
    r"^(?P<prefix>[A-Z]+)-IDR-ZF_Aff_(?P<a>\\d+)-(?P<b>\\d+)-(?P<c>\\d+)"
    r"(?:_pca)?\\.safetensors$"
)
# New CIS-style pattern: PREFIX_a-b-c-d.safetensors
FILE_REGEX_4NUM = re.compile(
    r"^(?P<prefix>[A-Z]+)_(?P<nums>\d+(?:-\d+){3})\.safetensors$"
)


# -----------------------------
# Core logic
# -----------------------------
def discover_experiments(files: Iterable[Path]) -> List[Tuple[str, Path]]:
    """Parse filenames into (experiment_key, absolute_path) pairs.

    Args:
        files: Iterable of candidate paths.

    Returns:
        List of (key, absolute_file_path) sorted.

    Raises:
        ValueError: If a filename matches the general suffix but fails to parse.
    """
    parsed: List[Tuple[str, Path, Tuple]] = []
    for p in files:
        if not p.is_file():
            continue
        if p.suffix != ".safetensors":
            continue
        m3 = FILE_REGEX_3NUM.match(p.name)
        m4 = FILE_REGEX_4NUM.match(p.name)
        if m3:
            prefix = m3.group("prefix")
            a, b, c = (int(m3.group("a")), int(m3.group("b")), int(m3.group("c")))
            key = f"{prefix}-{a}-{b}-{c}"
            sortkey = (prefix, a, b, c)
            parsed.append((key, p.resolve(), sortkey))
            continue
        elif m4:
            prefix = m4.group("prefix")
            all_nums = m4.group("nums").split("-")
            a, b, c, d = (
                int(all_nums[0]),
                int(all_nums[1]),
                int(all_nums[2]),
                int(all_nums[3]),
            )
            key = f"{prefix}-{a}-{b}-{c}-{d}"
            sortkey = (prefix, a, b, c, d)
            parsed.append((key, p.resolve(), sortkey))
            continue
        else:
            # Ignore non-matching files silently (robustness).
            continue
    print("hello")
    # Deterministic sort by tuple length then numeric values
    parsed.sort(key=lambda x: x[2])
    return [(k, path) for (k, path, _sortkey) in parsed]


def build_yaml_structure(
    items: List[Tuple[str, Path]],
    defaults: ExperimentDefaults,
    output_dir_root: str | None = None,
    output_dir_second: str | None = None,
) -> Dict[str, Dict[str, object]]:
    """Construct the YAML mapping for the 'experiments' document.

    Args:
        items: (key, path) pairs from discover_experiments.
        defaults: Default fields.
        output_dir_root: Optional base folder for each output_dir; when provided,
            each experiment gets output_dir = f"{output_dir_root}/{key}"

    Returns:
        Mapping suitable for YAML dump: {"experiments": { key: {...}, ... }}
    """
    experiments: Dict[str, Dict[str, object]] = {}
    for key, fpath in items:
        exp_entry: Dict[str, object] = {
            "data_path": str(fpath),
            **defaults.as_mapping(),
        }
        if output_dir_root:
            exp_entry["output_dir"] = str(
                Path(output_dir_root) / key / output_dir_second
            )
        experiments[key] = exp_entry

    return {"experiments": experiments}


def dump_yaml(doc: Mapping[str, object], output: Path) -> None:
    """Write YAML with stable ordering and plain booleans."""
    # Default Dumper keeps insertion order; set sort_keys=False for readability.
    text = yaml.dump(doc, sort_keys=False, allow_unicode=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a folder of *.safetensors files named like "
            "<PREFIX>-IDR-ZF_Aff_<a>-<b>-<c>_pca.safetensors and generate a YAML "
            "experiments config matching your template."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *.safetensors files (e.g., ./by_group_with_sequences_onehot_pca).",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        required=True,
        help="Path to write the generated YAML file.",
    )
    parser.add_argument(
        "--output-dir-root",
        type=str,
        default="results/166k_2024_regulators",
        help="Base directory for each experiment's output_dir (default: %(default)s).",
    )
    # Optional overrides for defaults (power users)
    parser.add_argument(
        "--strategies",
        type=str,
        default="HIGH_EXPRESSION,RANDOM",
        help="Comma-separated strategies.",
    )
    parser.add_argument(
        "--seq-mod-methods",
        type=str,
        default="EMBEDDING",
        help="Comma-separated sequence modification methods.",
    )
    parser.add_argument(
        "--regression-models",
        type=str,
        default="LINEAR,RANDOM_FOREST,MLP",
        help="Comma-separated regression models.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(i) for i in range(1, 21)),
        help="Comma-separated integer seeds (default: 1..20).",
    )
    parser.add_argument(
        "--initial-sample-size", type=int, default=8, help="Initial sample size."
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--test-size", type=int, default=30, help="Test size.")
    parser.add_argument("--max-rounds", type=int, default=20, help="Max rounds.")
    parser.add_argument(
        "--no-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, include no_test: true (default: true). Use --no-test=false to disable.",
    )
    parser.add_argument(
        "--normalize-expression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="normalize_expression flag (default: true).",
    )
    parser.add_argument(
        "--normalize-input-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="normalize_input_output flag (default: true).",
    )
    parser.add_argument(
        "--output-dir-second",
        type=str,
        default="",
        help="Second directory for each experiment's output_dir (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _parse_comma_ints(s: str) -> Tuple[int, ...]:
    nums: List[int] = []
    for token in (t.strip() for t in s.split(",") if t.strip()):
        nums.append(int(token))
    return tuple(nums)


def _parse_comma_strs(s: str) -> Tuple[str, ...]:
    return tuple(t.strip() for t in s.split(",") if t.strip())


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    files = list(input_dir.glob("*.safetensors"))
    items = discover_experiments(files)

    defaults = ExperimentDefaults(
        strategies=_parse_comma_strs(args.strategies),
        seq_mod_methods=_parse_comma_strs(args.seq_mod_methods),
        target_val_key="expressions",
        regression_models=_parse_comma_strs(args.regression_models),
        seeds=_parse_comma_ints(args.seeds),
        initial_sample_size=args.initial_sample_size,
        batch_size=args.batch_size,
        test_size=args.test_size,
        max_rounds=args.max_rounds,
        normalize_expression=args.normalize_expression,
        no_test=args.no_test,
        normalize_input_output=args.normalize_input_output,
    )

    doc = build_yaml_structure(
        items,
        defaults,
        output_dir_root=args.output_dir_root,
        output_dir_second=args.output_dir_second,
    )
    dump_yaml(doc, args.output_yaml)

    print(f"[OK] Wrote {len(items)} experiments to {args.output_yaml}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
