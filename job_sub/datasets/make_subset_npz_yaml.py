"""
generate a YAML file automatically for running experiments over all *_indices.txt files in a subset directory

Usage:
python make_subset_npz_yaml.py --embedding-parent-dir /share/home/wangzitongLab/liuxuyin/gLMs_embeddings_retrieval/pca/subset_outputs/20260305_0926/ --output-yaml 166k_npz_per_subset.yaml --metadata-path /storage2/wangzitongLab/share/gene_circuit_design_data/data_new/Rai_2024_166k/166k_Library_CLASSIC_Data.csv

This will generate a YAML file like this:

"""

import argparse
from pathlib import Path

import yaml


def generate_subsets_yaml(
    embedding_parent_dir: Path,
    output_yaml: Path,
    metadata_path: str,
):
    """
    Generate a YAML config from *_indices.txt files.

    Args:
        embedding_parent_dir: parent directory containing subset directories
        output_yaml: path to write yaml
        metadata_path: shared metadata csv path
        embedding_dir: shared embedding directory
    """

    datasets = []

    for dir in sorted(embedding_parent_dir.iterdir()):
        name = dir.stem

        datasets.append(
            {
                "name": name,
                "metadata_path": str(metadata_path),
                "embedding_dir": str(dir.resolve()),
            }
        )

    doc = {"datasets": datasets}

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Generated {len(datasets)} datasets → {output_yaml}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a YAML config from *_indices.txt files."
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=None,
        help="Path to write YAML file (default: local_166k.yaml in script directory).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Path to the metadata CSV file.",
    )
    parser.add_argument(
        "--embedding-parent-dir",
        type=Path,
        required=True,
        help="Path to the embedding's parent directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_subsets_yaml(
        embedding_parent_dir=args.embedding_parent_dir,
        output_yaml=Path(args.output_yaml),
        metadata_path=str(args.metadata_path.expanduser().resolve()),
    )
