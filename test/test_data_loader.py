"""
Unit tests for DataLoader and related classes.

Tests data loading from various formats, data splitting, and normalization.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from experiments.core.data_loader import DataLoader, Dataset
from utils.config_loader import SelectionStrategy
from utils.sequence_utils import SequenceModificationMethod


class TestDataset:
    """Test cases for Dataset dataclass."""

    def test_dataset_creation(self):
        """Test creating a valid dataset."""
        sequences = ["ATGC", "CGTA"]
        expressions = np.array([1.0, 2.0])
        log_likelihoods = np.array([-0.5, -0.3])
        embeddings = np.array([[1, 2], [3, 4]])
        variant_ids = np.array([1, 2])

        dataset = Dataset(
            sequences=sequences,
            expressions=expressions,
            log_likelihoods=log_likelihoods,
            embeddings=embeddings,
            variant_ids=variant_ids,
        )

        assert len(dataset.sequences) == 2
        assert dataset.expressions.shape == (2,)
        assert dataset.embeddings.shape == (2, 2)
        assert dataset.variant_ids is not None

    def test_dataset_validation_length_mismatch(self):
        """Test dataset validation catches length mismatches."""
        sequences = ["ATGC", "CGTA"]
        expressions = np.array([1.0, 2.0, 3.0])  # Wrong length
        log_likelihoods = np.array([-0.5, -0.3])

        with pytest.raises(ValueError, match="must have the same length"):
            Dataset(
                sequences=sequences,
                expressions=expressions,
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
        assert len(dataset.expressions) == 3
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
        expressions = torch.randn(5)
        log_likelihoods = torch.randn(5)

        save_file(
            {
                "embeddings": embeddings,
                "expressions": expressions,
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
        assert dataset.expressions.shape == (5,)
        assert dataset.log_likelihoods.shape == (5,)

    def test_load_safetensors_pca_format(self, tmp_path):
        """Test loading safetensors with PCA format."""
        safetensors_path = tmp_path / "test_data.safetensors"

        # Create test data
        pca_components = torch.randn(5, 10)
        expressions = torch.randn(5)
        log_likelihoods = torch.randn(5)
        variant_ids = torch.tensor([1, 2, 3, 4, 5])

        save_file(
            {
                "pca_components": pca_components,
                "expression": expressions,
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
        expressions = torch.randn(3)

        save_file(
            {
                "embeddings": embeddings,
                "expressions": expressions,
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

        assert len(dataset.expressions) == 3
        np.testing.assert_array_almost_equal(dataset.expressions, custom_target.numpy())

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

    def test_normalize_data(self, tmp_path):
        """Test data normalization."""
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
            normalize_input_output=True,
        )

        dataset = loader.load()

        # Check that expressions are normalized (mean ~0, std ~1)
        assert abs(dataset.expressions.mean()) < 0.1
        assert abs(dataset.expressions.std() - 1.0) < 0.1

    def test_create_data_split_random(self):
        """Test creating data split with random initial selection."""
        # Create a simple dataset
        sequences = [f"seq_{i}" for i in range(20)]
        expressions = np.random.randn(20)
        log_likelihoods = np.random.randn(20)
        embeddings = np.random.randn(20, 10)

        dataset = Dataset(
            sequences=sequences,
            expressions=expressions,
            log_likelihoods=log_likelihoods,
            embeddings=embeddings,
            variant_ids=None,
        )

        loader = DataLoader(
            data_path="dummy",  # Not used for splitting
            normalize_input_output=False,
        )

        def encode_fn(indices):
            return embeddings[indices]

        data_split = loader.create_data_split(
            dataset=dataset,
            initial_sample_size=5,
            test_size=3,
            no_test=False,
            selection_strategy=SelectionStrategy.RANDOM,
            random_seed=42,
            encode_sequences_fn=encode_fn,
        )

        assert len(data_split.train_indices) == 5
        assert len(data_split.test_indices) == 3
        assert len(data_split.unlabeled_indices) == 12
        assert (
            len(data_split.train_indices)
            + len(data_split.test_indices)
            + len(data_split.unlabeled_indices)
            == 20
        )

    def test_create_data_split_no_test(self):
        """Test creating data split without test set."""
        sequences = [f"seq_{i}" for i in range(10)]
        expressions = np.random.randn(10)
        log_likelihoods = np.random.randn(10)
        embeddings = np.random.randn(10, 5)

        dataset = Dataset(
            sequences=sequences,
            expressions=expressions,
            log_likelihoods=log_likelihoods,
            embeddings=embeddings,
            variant_ids=None,
        )

        loader = DataLoader(
            data_path="dummy",
            normalize_input_output=False,
        )

        def encode_fn(indices):
            return embeddings[indices]

        data_split = loader.create_data_split(
            dataset=dataset,
            initial_sample_size=3,
            test_size=0,
            no_test=True,
            selection_strategy=SelectionStrategy.RANDOM,
            random_seed=42,
            encode_sequences_fn=encode_fn,
        )

        assert len(data_split.test_indices) == 0
        assert len(data_split.train_indices) == 3
        assert len(data_split.unlabeled_indices) == 7

    # TODO: ZELUN figure out why normalize_input_output=False is not working
    # Unsure why this only fail on pre-commit hook, but works locally
    # def test_create_data_split_kmeans(self):
    #     """Test creating data split with K-means initial selection."""
    #     sequences = [f"seq_{i}" for i in range(10)]
    #     expressions = np.random.randn(10)
    #     log_likelihoods = np.random.randn(10)
    #     embeddings = np.random.randn(10, 5)

    #     dataset = Dataset(
    #         sequences=sequences,
    #         expressions=expressions,
    #         log_likelihoods=log_likelihoods,
    #         embeddings=embeddings,
    #         variant_ids=None,
    #     )
    #     # TODO: ZELUN figure out why normalize_input_output=False is not working
    #     loader = DataLoader(
    #         data_path="dummy",
    #         normalize_input_output=True,
    #     )

    #     def encode_fn(indices):
    #         return embeddings[indices]

    #     data_split = loader.create_data_split(
    #         dataset=dataset,
    #         initial_sample_size=3,
    #         test_size=2,
    #         no_test=False,
    #         selection_strategy=SelectionStrategy.KMEANS_RANDOM,
    #         random_seed=42,
    #         encode_sequences_fn=encode_fn,
    #     )

    #     assert len(data_split.train_indices) == 3
    #     assert len(data_split.test_indices) == 2
    #     assert len(data_split.unlabeled_indices) == 5
