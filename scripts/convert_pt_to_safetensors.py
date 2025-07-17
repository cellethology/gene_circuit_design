#!/usr/bin/env python3
"""
Convert PyTorch .pt files to safetensors format.

This script converts PyTorch tensor files to safetensors format for better
compatibility and safety in the active learning pipeline.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_pt_to_safetensors(input_path: str, output_path: str = None):
    """
    Convert a PyTorch .pt file to safetensors format.

    Args:
        input_path: Path to input .pt file
        output_path: Path to output .safetensors file (optional)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_path.suffix == ".pt":
        raise ValueError(
            f"Input file must have .pt extension, got: {input_path.suffix}"
        )

    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.with_suffix(".safetensors")
    else:
        output_path = Path(output_path)

    print(f"Loading PyTorch file: {input_path}")

    # Load the PyTorch file
    try:
        data = torch.load(input_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading PyTorch file: {e}")
        return

    # Print information about the loaded data
    print(f"Loaded data type: {type(data)}")

    if isinstance(data, dict):
        print(f"Dictionary keys: {list(data.keys())}")
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)}")
    elif isinstance(data, torch.Tensor):
        print(f"Tensor shape: {data.shape} ({data.dtype})")
        # Convert single tensor to dict format
        data = {"tensor": data}
    else:
        print(f"Unexpected data type: {type(data)}")
        return

    # Convert all values to tensors if they aren't already
    tensor_dict = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            tensor_dict[key] = value
        elif key in ["variant_ids", "expressions", "embeddings"]:
            # Convert key data fields to tensors
            try:
                if isinstance(value, (list, tuple)):
                    tensor_dict[key] = torch.tensor(value)
                    print(
                        f"Converted {key} list/tuple to tensor: {tensor_dict[key].shape}"
                    )
                elif hasattr(value, "__iter__") and not isinstance(value, str):
                    # Try to convert other iterable types
                    tensor_dict[key] = torch.tensor(list(value))
                    print(
                        f"Converted {key} iterable to tensor: {tensor_dict[key].shape}"
                    )
                else:
                    print(f"Warning: Cannot convert {key} to tensor: {type(value)}")
            except Exception as e:
                print(f"Error converting {key} to tensor: {e}")
        else:
            # Try to convert other types to tensors
            try:
                if isinstance(value, (list, tuple)):
                    tensor_dict[key] = torch.tensor(value)
                    print(
                        f"Converted {key} list/tuple to tensor: {tensor_dict[key].shape}"
                    )
                elif hasattr(value, "__iter__") and not isinstance(value, str):
                    # Try to convert other iterable types
                    tensor_dict[key] = torch.tensor(list(value))
                    print(
                        f"Converted {key} iterable to tensor: {tensor_dict[key].shape}"
                    )
                else:
                    print(
                        f"Warning: Skipping non-convertible value for key '{key}': {type(value)}"
                    )
            except Exception as e:
                print(f"Warning: Could not convert {key} to tensor: {e}")

    # Note: Not adding dummy log_likelihoods - the experiment runner will automatically
    # skip LOG_LIKELIHOOD strategy if log likelihood data is not available

    if not tensor_dict:
        print("Error: No tensors found in the file")
        return

    print(f"Converting {len(tensor_dict)} tensors to safetensors format...")

    # Save as safetensors
    try:
        save_file(tensor_dict, output_path)
        print(f"Successfully converted to: {output_path}")

        # Print file sizes for comparison
        input_size = input_path.stat().st_size / (1024 * 1024)  # MB
        output_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"File size: {input_size:.1f} MB -> {output_size:.1f} MB")

    except Exception as e:
        print(f"Error saving safetensors file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pt files to safetensors format"
    )
    parser.add_argument("input", help="Input .pt file path")
    parser.add_argument(
        "-o", "--output", help="Output .safetensors file path (optional)"
    )

    args = parser.parse_args()

    convert_pt_to_safetensors(args.input, args.output)


if __name__ == "__main__":
    main()
