from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd

# Load the GenBank file
record = SeqIO.read("384_Example_Plasmid_Map.gb", "genbank")


# Find the two flanking misc_features by label
def find_feature_by_label(type, label):
    for f in record.features:
        if f.type == type and f.qualifiers.get("label", [""])[0] == label:
            return f
    raise ValueError(f"Feature with label '{label}' not found.")


def substitute_features_in_region(
    record, start_feature, end_feature, substitutions=None
):
    """
    Substitute features in a specified region of a GenBank record.
    """
    if substitutions is None:
        substitutions = []

    start = int(start_feature.location.start)
    end = int(end_feature.location.end)
    region_seq = record.seq[start:end]

    # Build substitution map
    subs_dict = {label: Seq(new.upper()) for label, new in substitutions}

    # Get all features to substitute, ordered by position
    features_to_sub = [
        f
        for f in record.features
        if "label" in f.qualifiers
        and f.qualifiers["label"][0] in subs_dict
        and int(f.location.start) >= start
        and int(f.location.end) <= end
    ]
    features_to_sub.sort(key=lambda f: int(f.location.start))

    chunks = []
    cursor = 0

    for f in features_to_sub:
        label = f.qualifiers["label"][0]
        rel_start = int(f.location.start) - start
        rel_end = int(f.location.end) - start

        # Add unmodified chunk
        chunks.append(region_seq[cursor:rel_start])
        # Add substitution
        chunks.append(subs_dict[label])
        cursor = rel_end

    # Add remaining sequence
    chunks.append(region_seq[cursor:])

    return Seq("").join(chunks)

if __name__ == "__main__":
    components = pd.read_excel(
    "384_Library_Part_Sequences.xlsx", header=1, index_col=0
)
    # for all Part Type that is NaN, set it equal to the Part Type of the previous row
    components["Part Type"] = components["Part Type"].ffill()

    expressions = pd.read_csv("384_Library_CLASSIC_Data.csv", index_col=0)
    # check all rows of promoter, kozak, terminator combinations are unique
    assert len(expressions) == len(
        expressions.drop_duplicates(subset=["Promoter", "Kozaks", "Terminators"])
    ), "Duplicate combinations of promoter, kozak, and terminator found."

    # build a dictionary of lists of components using Part Code and Part Type in components

    part_dict = {}
    for index, row in components.iterrows():
        part_seq = row["Sequence"]
        part_type = row["Part Type"]
        if part_type not in part_dict:
            part_dict[part_type] = []
        part_dict[part_type].append(part_seq)   