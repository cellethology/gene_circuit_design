#!/usr/bin/env python3
"""
Inspect the contents of the enformer .pt file to understand its structure.
"""

import sys
from pathlib import Path

try:
    import torch

    # Path to the enformer .pt file
    pt_file = "/Users/LZL/Desktop/Westlake_Research/gene_circuit_design/data/166k_data/166k_rice/post_embeddings/all_embeddings_parallel_enformer.pt"

    if not Path(pt_file).exists():
        print(f"File not found: {pt_file}")
        sys.exit(1)

    print(f"Loading file: {pt_file}")

    # Load the file
    data = torch.load(pt_file, map_location="cpu")

    print(f"Data type: {type(data)}")

    if isinstance(data, dict):
        print(f"Dictionary keys: {list(data.keys())}")
        print("\nKey details:")
        for key, value in data.items():
            if hasattr(value, "shape"):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if key.lower() in ["expression", "expressions", "expr"]:
                    print(
                        f"    Expression data found! Min: {value.min():.3f}, Max: {value.max():.3f}"
                    )
            elif hasattr(value, "__len__"):
                print(f"  {key}: length={len(value)}, type={type(value)}")
                if key.lower() in ["expression", "expressions", "expr"]:
                    print(
                        f"    Expression data found! Min: {min(value):.3f}, Max: {max(value):.3f}"
                    )
            else:
                print(f"  {key}: {type(value)}")
    else:
        print(f"Data is not a dictionary: {type(data)}")
        if hasattr(data, "shape"):
            print(f"Shape: {data.shape}")

except ImportError:
    print("PyTorch not available. Cannot inspect .pt file.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)
