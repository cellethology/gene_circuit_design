"""
generate a YAML file automatically for running experiments over all *_indices.txt files in subset32/
the yaml file will be like this:
datasets:
  - name: "166k_AD_part1_indices"
    metadata_path: "/storage2/wangzitongLab/share/gene_circuit_design_data/data_new/Rai_2024_166k/166k_Library_CLASSIC_Data.csv"
    embedding_dir: "/storage2/wangzitongLab/share/gene_circuit_design_data/data_new/Rai_2024_166k"
    subset_ids_path: "/home/wangzitongLab/liuxuyin/gene_circuit_design/job_sub/datasets/subset32/166k_AD_part1_indices.txt"

"""

from pathlib import Path

import yaml


def generate_subsets_yaml(
    subset_dir: Path,
    output_yaml: Path,
    metadata_path: str,
    embedding_dir: str,
):
    """
    Generate a YAML config from *_indices.txt files.

    Args:
        subset_dir: directory containing *_indices.txt files
        output_yaml: path to write yaml
        metadata_path: shared metadata csv path
        embedding_dir: shared embedding directory
    """

    datasets = []

    for txt_file in sorted(subset_dir.glob("*_indices.txt")):
        name = txt_file.stem  # remove .txt

        datasets.append(
            {
                "name": name,
                "metadata_path": str(metadata_path),
                "embedding_dir": str(embedding_dir),
                "subset_ids_path": str(txt_file.resolve()),
            }
        )

    doc = {"datasets": datasets}

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w") as f:
        yaml.dump(doc, f, sort_keys=False)

    print(f"[OK] Generated {len(datasets)} datasets → {output_yaml}")


if __name__ == "__main__":
    subset_dir = Path(__file__).parent.resolve() / "166k_subsets"
    output_yaml = Path(__file__).parent.resolve() / "subsets39.yaml"
    metadata_path = "/storage2/wangzitongLab/share/gene_circuit_design_data/data_new/Rai_2024_166k/166k_Library_CLASSIC_Data.csv"
    embedding_dir = (
        "/storage2/wangzitongLab/share/gene_circuit_design_data/data_new/Rai_2024_166k"
    )

    generate_subsets_yaml(
        subset_dir=subset_dir,
        output_yaml=output_yaml,
        metadata_path=metadata_path,
        embedding_dir=embedding_dir,
    )
