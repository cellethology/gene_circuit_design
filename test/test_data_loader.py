"""
Unit tests for DataLoader and related classes.

Tests data loading from various formats.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from experiments.core.data_loader import DataLoader, Dataset
from utils.sequence_utils import SequenceModificationMethod


class TestDataset:
    """Test cases for Dataset dataclass."""

    def test_dataset_creation(self):
        """Test creating a valid dataset."""
        sequences = ["ATGC", "CGTA"]
        sequence_labels = np.array([1.0, 2.0])
        log_likelihoods = np.array([-0.5, -0.3])
        embeddings = np.array([[1, 2], [3, 4]])
        variant_ids = np.array([1, 2])

        dataset = Dataset(
            sequences=sequences,
            sequence_labels=sequence_labels,
            log_likelihoods=log_likelihoods,
            embeddings=embeddings,
            variant_ids=variant_ids,
        )

        assert len(dataset.sequences) == 2
        assert dataset.sequence_labels.shape == (2,)
        assert dataset.embeddings.shape == (2, 2)
        assert dataset.variant_ids is not None

    def test_dataset_validation_length_mismatch(self):
        """Test dataset validation catches length mismatches."""
        sequences = ["ATGC", "CGTA"]
        sequence_labels = np.array([1.0, 2.0, 3.0])  # Wrong length
        log_likelihoods = np.array([-0.5, -0.3])

        with pytest.raises(ValueError, match="must have the same length"):
            Dataset(
                sequences=sequences,
                sequence_labels=sequence_labels,
                log_likelihoods=log_likelihoods,
                embeddings=None,
                variant_ids=None,
            )


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_load_csv_basic(self, tmp_path):
        """Test loading basic CSV file."""
        # Create test CSV
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame(
            {
                "Sequence": ["ATGC", "CGTA", "AAAA"],
                "Expression": [1.0, 2.0, 3.0],
            }
        )
        df.to_csv(csv_path, index=False)

        loader = DataLoader(
            data_path=str(csv_path),
            seq_mod_method=SequenceModificationMethod.EMBEDDING,
            normalize_input_output=False,
        )

        dataset = loader.load()

        assert len(dataset.sequences) == 3
        assert len(dataset.sequence_labels) == 3
        assert dataset.embeddings is None
        assert np.all(np.isnan(dataset.log_likelihoods))

    def test_load_csv_with_log_likelihood(self, tmp_path):
        """Test loading CSV with log likelihood."""
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame(
            {
                "Sequence": ["ATGC", "CGTA"],
                "Expression": [1.0, 2.0],
                "Log_Likelihood": [-0.5, -0.3],
            }
        )
        df.to_csv(csv_path, index=False)

        loader = DataLoader(
            data_path=str(csv_path),
            seq_mod_method=SequenceModificationMethod.EMBEDDING,
            normalize_input_output=False,
        )

        dataset = loader.load()

        assert len(dataset.sequences) == 2
        assert not np.all(np.isnan(dataset.log_likelihoods))
        assert dataset.log_likelihoods[0] == -0.5

    def test_load_safetensors_embeddings_format(self, tmp_path):
        """Test loading safetensors with embeddings format."""
        safetensors_path = tmp_path / "test_data.safetensors"

        # Create test data
        embeddings = torch.randn(5, 10)
        sequence_labels = torch.randn(5)
        log_likelihoods = torch.randn(5)

        save_file(
            {
                "embeddings": embeddings,
                "expressions": sequence_labels,
                "log_likelihoods": log_likelihoods,
            },
            str(safetensors_path),
        )

        loader = DataLoader(
            data_path=str(safetensors_path),
            normalize_input_output=False,
        )

        dataset = loader.load()

        assert len(dataset.sequences) == 5
        assert dataset.embeddings.shape == (5, 10)
        assert dataset.sequence_labels.shape == (5,)
        assert dataset.log_likelihoods.shape == (5,)

    def test_load_safetensors_pca_format(self, tmp_path):
        """Test loading safetensors with PCA format."""
        safetensors_path = tmp_path / "test_data.safetensors"

        # Create test data
        pca_components = torch.randn(5, 10)
        sequence_labels = torch.randn(5)
        log_likelihoods = torch.randn(5)
        variant_ids = torch.tensor([1, 2, 3, 4, 5])

        save_file(
            {
                "pca_components": pca_components,
                "expression": sequence_labels,
                "log_likelihood": log_likelihoods,
                "variant_ids": variant_ids,
            },
            str(safetensors_path),
        )

        loader = DataLoader(
            data_path=str(safetensors_path),
            normalize_input_output=False,
        )

        dataset = loader.load()

        assert len(dataset.sequences) == 5
        assert dataset.embeddings.shape == (5, 10)
        assert dataset.variant_ids is not None
        assert len(dataset.variant_ids) == 5

    def test_load_safetensors_missing_log_likelihood(self, tmp_path):
        """Test loading safetensors without log likelihood."""
        safetensors_path = tmp_path / "test_data.safetensors"

        embeddings = torch.randn(3, 5)
        sequence_labels = torch.randn(3)

        save_file(
            {
                "embeddings": embeddings,
                "expressions": sequence_labels,
            },
            str(safetensors_path),
        )

        loader = DataLoader(
            data_path=str(safetensors_path),
            normalize_input_output=False,
        )

        dataset = loader.load()

        assert len(dataset.sequences) == 3
        assert np.all(np.isnan(dataset.log_likelihoods))

    def test_load_safetensors_custom_target_key(self, tmp_path):
        """Test loading safetensors with custom target value key."""
        safetensors_path = tmp_path / "test_data.safetensors"

        embeddings = torch.randn(3, 5)
        custom_target = torch.randn(3)

        save_file(
            {
                "embeddings": embeddings,
                "custom_target": custom_target,
            },
            str(safetensors_path),
        )

        loader = DataLoader(
            data_path=str(safetensors_path),
            target_val_key="custom_target",
            normalize_input_output=False,
        )

        dataset = loader.load()

        assert len(dataset.sequence_labels) == 3
        np.testing.assert_array_almost_equal(dataset.sequence_labels, custom_target.numpy())

    def test_load_safetensors_missing_expression(self, tmp_path):
        """Test error when expression data is missing."""
        safetensors_path = tmp_path / "test_data.safetensors"

        embeddings = torch.randn(3, 5)

        save_file(
            {
                "embeddings": embeddings,
            },
            str(safetensors_path),
        )

        loader = DataLoader(
            data_path=str(safetensors_path),
            normalize_input_output=False,
        )

        with pytest.raises(ValueError, match="No expression data found"):
            loader.load()
