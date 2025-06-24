"""
Pytest-compatible tests for sequence utility functions.

This module contains comprehensive tests for DNA sequence processing,
one-hot encoding, and data loading functions.
"""

import os
import tempfile

import numpy as np
import pytest

# Import the functions to test - using proper package imports
from utils.sequence_utils import (
    calculate_sequence_statistics,
    flatten_one_hot_sequences,
    load_sequence_data,
    one_hot_encode_sequence,
    one_hot_encode_sequences,
)


class TestOneHotEncodeSequence:
    """Test cases for single sequence one-hot encoding."""

    def test_basic_sequence_encoding(self):
        """Test encoding of basic DNA sequence."""
        sequence = "ATGC"
        result = one_hot_encode_sequence(sequence)

        expected = np.array([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # T
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1]   # C
        ])

        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, expected)

    def test_empty_sequence(self):
        """Test encoding of empty sequence."""
        sequence = ""
        result = one_hot_encode_sequence(sequence)

        assert result.shape == (0, 4)

    def test_single_nucleotide(self):
        """Test encoding of single nucleotide."""
        sequence = "A"
        result = one_hot_encode_sequence(sequence)

        expected = np.array([[1, 0, 0, 0]])
        assert result.shape == (1, 4)
        np.testing.assert_array_equal(result, expected)

    def test_long_sequence(self):
        """Test encoding of longer sequence."""
        sequence = "ATGCATGCATGC"
        result = one_hot_encode_sequence(sequence)

        assert result.shape == (12, 4)
        # Check that each row sums to 1 (one-hot property)
        assert np.all(np.sum(result, axis=1) == 1)

    def test_invalid_nucleotide(self):
        """Test handling of invalid nucleotides."""
        sequence = "ATGCX"  # X is not a valid nucleotide

        with pytest.raises(ValueError, match="Invalid nucleotide"):
            one_hot_encode_sequence(sequence)

    def test_lowercase_sequence(self):
        """Test encoding of lowercase sequence."""
        sequence = "atgc"
        result = one_hot_encode_sequence(sequence)

        expected = np.array([
            [1, 0, 0, 0],  # a -> A
            [0, 1, 0, 0],  # t -> T
            [0, 0, 1, 0],  # g -> G
            [0, 0, 0, 1]   # c -> C
        ])

        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, expected)


class TestOneHotEncodeSequences:
    """Test cases for multiple sequence encoding."""

    def test_multiple_sequences_same_length(self):
        """Test encoding multiple sequences of same length."""
        sequences = ["ATGC", "GCTA", "TTAA"]
        result = one_hot_encode_sequences(sequences, "trim")

        assert len(result) == 3
        for encoded_seq in result:
            assert encoded_seq.shape == (4, 4)
            assert np.all(np.sum(encoded_seq, axis=1) == 1)

    def test_multiple_sequences_different_lengths(self):
        """Test encoding multiple sequences of different lengths."""
        sequences = ["AT", "GCTA", "TTAAGG"]
        result = one_hot_encode_sequences(sequences, "trim")

        assert len(result) == 3
        assert result[0].shape == (2, 4)
        assert result[1].shape == (4, 4)
        assert result[2].shape == (6, 4)

    def test_empty_sequence_list(self):
        """Test encoding empty sequence list."""
        sequences = []
        result = one_hot_encode_sequences(sequences, "trim")

        assert len(result) == 0

    def test_sequence_with_invalid_nucleotide(self):
        """Test handling sequences with invalid nucleotides."""
        sequences = ["ATGC", "GCTX", "TTAA"]  # Second sequence has invalid X

        with pytest.raises(ValueError):
            one_hot_encode_sequences(sequences)


class TestFlattenOneHotSequences:
    """Test cases for flattening one-hot encoded sequences."""

    def test_flatten_single_sequence(self):
        """Test flattening single encoded sequence."""
        # Create a simple 2x4 encoded sequence
        encoded_sequences = [np.array([[1, 0, 0, 0], [0, 1, 0, 0]])]  # AT
        result = flatten_one_hot_sequences(encoded_sequences)

        expected = np.array([[1, 0, 0, 0, 0, 1, 0, 0]])
        assert result.shape == (1, 8)
        np.testing.assert_array_equal(result, expected)

    def test_flatten_multiple_sequences_same_length(self):
        """Test flattening multiple sequences of same length."""
        encoded_sequences = [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),  # AT
            np.array([[0, 0, 1, 0], [0, 0, 0, 1]])   # GC
        ]
        result = flatten_one_hot_sequences(encoded_sequences)

        assert result.shape == (2, 8)
        expected_first = np.array([1, 0, 0, 0, 0, 1, 0, 0])
        expected_second = np.array([0, 0, 1, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(result[0], expected_first)
        np.testing.assert_array_equal(result[1], expected_second)

    def test_flatten_different_length_sequences_raises_error(self):
        """Test that flattening sequences of different lengths raises error."""
        encoded_sequences = [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),        # Length 2
            np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])  # Length 3
        ]

        with pytest.raises(ValueError, match="All sequences must have the same length"):
            flatten_one_hot_sequences(encoded_sequences)

    def test_flatten_empty_list(self):
        """Test flattening empty sequence list."""
        encoded_sequences = []
        result = flatten_one_hot_sequences(encoded_sequences)

        assert result.shape == (0, 0)


class TestLoadSequenceData:
    """Test cases for loading sequence data from CSV."""

    def test_load_valid_csv(self):
        """Test loading valid CSV file."""
        # Create temporary CSV file
        csv_data = """Variant_ID,Sequence,Expression,Promoter,Kozaks,Terminators
1,ATGCGTAC,1500.5,P1,K1,T1
2,GCATGCTA,2000.0,P2,K2,T2
3,TTAACCGG,800.25,P3,K3,T3"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            sequences, expressions = load_sequence_data(temp_path, "trim")

            assert len(sequences) == 3
            assert len(expressions) == 3
            assert sequences[0] == "ATGCGTAC"
            assert sequences[1] == "GCATGCTA"
            assert sequences[2] == "TTAACCGG"
            assert expressions[0] == 1500.5
            assert expressions[1] == 2000.0
            assert expressions[2] == 800.25
        finally:
            os.unlink(temp_path)

    def test_load_csv_with_missing_columns(self):
        """Test loading CSV with missing required columns."""
        csv_data = """Variant_ID,Sequence,Promoter
1,ATGC,P1
2,GCTA,P2"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                load_sequence_data(temp_path, "trim")
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_sequence_data("nonexistent_file.csv", "trim")

    def test_load_with_trimming(self):
        """Test loading with sequence trimming."""
        csv_data = """Variant_ID,Sequence,Expression
1,ATGCGTACAAAA,1500.5
2,GCATGCTATTTT,2000.0
3,TTAACCGGCCCC,800.25"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            sequences, expressions = load_sequence_data(temp_path, "trim")

            # All sequences should be trimmed to the length of the shortest (12)
            assert all(len(seq) == 12 for seq in sequences)
            assert sequences[0] == "ATGCGTACAAAA"
            assert sequences[1] == "GCATGCTATTTT"
            assert sequences[2] == "TTAACCGGCCCC"
        finally:
            os.unlink(temp_path)


class TestCalculateSequenceStatistics:
    """Test cases for sequence statistics calculation."""

    def test_basic_statistics(self):
        """Test basic sequence statistics."""
        sequences = ["ATGC", "GCTA", "TTAACCGG"]
        stats = calculate_sequence_statistics(sequences)

        assert stats['count'] == 3
        assert stats['min_length'] == 4
        assert stats['max_length'] == 8
        assert stats['mean_length'] == pytest.approx(5.33, rel=1e-2)
        assert stats['total_nucleotides'] == 16

    def test_single_sequence_statistics(self):
        """Test statistics for single sequence."""
        sequences = ["ATGCGTAC"]
        stats = calculate_sequence_statistics(sequences)

        assert stats['count'] == 1
        assert stats['min_length'] == 8
        assert stats['max_length'] == 8
        assert stats['mean_length'] == 8.0
        assert stats['total_nucleotides'] == 8

    def test_empty_sequence_list_statistics(self):
        """Test statistics for empty sequence list."""
        sequences = []
        stats = calculate_sequence_statistics(sequences)

        assert stats['count'] == 0
        assert stats['min_length'] == 0
        assert stats['max_length'] == 0
        assert stats['mean_length'] == 0.0
        assert stats['total_nucleotides'] == 0

    def test_sequences_with_varying_lengths(self):
        """Test statistics for sequences with varying lengths."""
        sequences = ["A", "AT", "ATG", "ATGC", "ATGCG"]
        stats = calculate_sequence_statistics(sequences)

        assert stats['count'] == 5
        assert stats['min_length'] == 1
        assert stats['max_length'] == 5
        assert stats['mean_length'] == 3.0
        assert stats['total_nucleotides'] == 15


# Integration tests
class TestSequenceProcessingIntegration:
    """Integration tests for the complete sequence processing pipeline."""

    def test_complete_pipeline(self):
        """Test complete pipeline from CSV loading to encoding."""
        csv_data = """Variant_ID,Sequence,Expression
1,ATGC,1500.5
2,GCTA,2000.0
3,TTAA,800.25"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            # Load data
            sequences, expressions = load_sequence_data(temp_path, "trim")

            # Calculate statistics
            stats = calculate_sequence_statistics(sequences)
            assert stats['count'] == 3

            # Encode sequences
            encoded = one_hot_encode_sequences(sequences, "trim")
            assert len(encoded) == 3

            # Flatten encoded sequences
            flattened = flatten_one_hot_sequences(encoded)
            assert flattened.shape == (3, 16)  # 3 sequences × 4 nucleotides × 4 positions

        finally:
            os.unlink(temp_path)


# Fixtures for shared test data
@pytest.fixture
def sample_sequences():
    """Fixture providing sample DNA sequences."""
    return ["ATGC", "GCTA", "TTAA", "CCGG"]


@pytest.fixture
def sample_csv_content():
    """Fixture providing sample CSV content."""
    return """Variant_ID,Sequence,Expression,Promoter,Kozaks,Terminators
1,ATGCGTAC,1500.5,P1,K1,T1
2,GCATGCTA,2000.0,P2,K2,T2
3,TTAACCGG,800.25,P3,K3,T3
4,CCGGTTAA,1200.75,P4,K4,T4"""


@pytest.fixture
def temp_csv_file(sample_csv_content):
    """Fixture providing temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


class TestWithFixtures:
    """Tests using pytest fixtures."""

    def test_encode_sample_sequences(self, sample_sequences):
        """Test encoding sample sequences using fixture."""
        encoded = one_hot_encode_sequences(sample_sequences, "trim")
        assert len(encoded) == 4
        for seq_encoded in encoded:
            assert seq_encoded.shape == (4, 4)

    def test_load_temp_csv(self, temp_csv_file):
        """Test loading temporary CSV file using fixture."""
        sequences, expressions = load_sequence_data(temp_csv_file, "trim")
        assert len(sequences) == 4
        assert len(expressions) == 4
        assert expressions[0] == 1500.5


# Pytest configuration and custom markers
pytestmark = pytest.mark.unit


def test_numpy_float32_dtype():
    """Test that all functions return np.float32 dtype for consistency."""
    sequence = "ATGC"
    encoded = one_hot_encode_sequence(sequence)
    assert encoded.dtype == np.float32

    sequences = ["ATGC", "GGCC"]
    multi_encoded = one_hot_encode_sequences(sequences, "trim")
    for encoded in multi_encoded:
        assert encoded.dtype == np.float32

    flattened = flatten_one_hot_sequences(multi_encoded)
    assert flattened.dtype == np.float32


if __name__ == "__main__":
    # Allow running tests directly with python
    pytest.main([__file__])
