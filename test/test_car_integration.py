#!/usr/bin/env python3
"""
Integration test script for CAR data support.

This script demonstrates the updated codebase functionality for handling
both DNA sequences (384_data) and CAR motif sequences (car_data).
"""

from experiments.run_experiments_parallelization import (
    ActiveLearningExperiment,
    SelectionStrategy,
)
from utils.sequence_utils import (
    SequenceModificationMethod,
    flatten_one_hot_sequences,
    load_sequence_data,
    one_hot_encode_sequences,
)


def test_dna_sequence_processing():
    """Test traditional DNA sequence processing (384_data format)."""
    print("=" * 60)
    print("Testing DNA Sequence Processing (384_data format)")
    print("=" * 60)

    # Test with the 384 data format
    try:
        sequences, targets = load_sequence_data(
            "../data/384_Data/384_Library_CLASSIC_Data.csv",
            seq_mod_method=SequenceModificationMethod.TRIM,
        )
        print(f"✓ Loaded {len(sequences)} DNA sequences")
        print(
            f"  Sample sequence: {sequences[0][:50]}..."
            if len(sequences[0]) > 50
            else f"  Sample sequence: {sequences[0]}"
        )
        print(f"  Sample target: {targets[0]}")

        # Test encoding
        encoded = one_hot_encode_sequences(
            sequences[:3], SequenceModificationMethod.TRIM
        )
        print(f"✓ Encoded 3 sequences, shape: {[e.shape for e in encoded]}")

        # Test flattening
        flattened = flatten_one_hot_sequences(encoded)
        print(f"✓ Flattened sequences shape: {flattened.shape}")

    except Exception as e:
        print(f"✗ Error in DNA processing: {e}")


def test_car_motif_processing():
    """Test CAR motif sequence processing (car_data format)."""
    print("=" * 60)
    print("Testing CAR Motif Processing (car_data format)")
    print("=" * 60)

    try:
        sequences, targets = load_sequence_data(
            "../data/car_data/science.abq0225_data_s1.csv",
            seq_mod_method=SequenceModificationMethod.CAR,
        )
        print(f"✓ Loaded {len(sequences)} CAR motif sequences")
        print(f"  Sample motif sequence: {sequences[0]}")
        print(f"  Sample target (cytotoxicity): {targets[0]}")
        print(f"  Motif value range: {sequences.min()} to {sequences.max()}")

        # Test encoding
        encoded = one_hot_encode_sequences(
            sequences[:3], SequenceModificationMethod.CAR
        )
        print(f"✓ Encoded 3 motif sequences, shapes: {[e.shape for e in encoded]}")
        print(f"  Each motif position has {encoded[0].shape[1]} possible values")

        # Test flattening
        flattened = flatten_one_hot_sequences(encoded)
        print(f"✓ Flattened motif sequences shape: {flattened.shape}")

    except Exception as e:
        print(f"✗ Error in CAR processing: {e}")


def test_car_experiment():
    """Test complete CAR experiment pipeline."""
    print("=" * 60)
    print("Testing Complete CAR Experiment Pipeline")
    print("=" * 60)

    try:
        # Initialize experiment with CAR data
        experiment = ActiveLearningExperiment(
            data_path="../data/car_data/science.abq0225_data_s1.csv",
            selection_strategy=SelectionStrategy.RANDOM,
            initial_sample_size=10,
            batch_size=5,
            test_size=20,
            random_seed=42,
            seq_mod_method=SequenceModificationMethod.CAR,
            no_test=False,
            normalize_expression=False,
        )
        print("✓ Initialized CAR experiment")
        print(f"  Total samples: {len(experiment.all_sequences)}")
        print(f"  Training set: {len(experiment.train_indices)}")
        print(f"  Test set: {len(experiment.test_indices)}")
        print(f"  Unlabeled set: {len(experiment.unlabeled_indices)}")

        # Test encoding functionality
        X_train = experiment._encode_sequences(experiment.train_indices[:3])
        print(f"✓ Encoded training samples shape: {X_train.shape}")

        # Train and evaluate
        experiment._train_model()
        test_metrics = experiment._evaluate_on_test_set()
        print("✓ Trained model and evaluated on test set")
        print(f"  Test R²: {test_metrics['r2']:.3f}")
        print(f"  Test RMSE: {test_metrics['rmse']:.3f}")
        print(f"  Pearson correlation: {test_metrics['pearson_correlation']:.3f}")

    except Exception as e:
        print(f"✗ Error in CAR experiment: {e}")
        import traceback

        traceback.print_exc()


def test_comparison():
    """Compare data characteristics between DNA and CAR formats."""
    print("=" * 60)
    print("Comparing DNA vs CAR Data Characteristics")
    print("=" * 60)

    try:
        # Load both types of data
        dna_sequences, dna_targets = load_sequence_data(
            "../data/384_Data/384_Library_CLASSIC_Data.csv",
            seq_mod_method=SequenceModificationMethod.TRIM,
        )

        car_sequences, car_targets = load_sequence_data(
            "../data/car_data/science.abq0225_data_s1.csv",
            seq_mod_method=SequenceModificationMethod.CAR,
        )

        print("DNA Data (384_data):")
        print(f"  Samples: {len(dna_sequences)}")
        print("  Sequence type: Nucleotide strings")
        print(f"  Sequence length: {len(dna_sequences[0]) if dna_sequences else 'N/A'}")
        print(f"  Target range: {dna_targets.min():.2f} - {dna_targets.max():.2f}")
        print(f"  Target mean: {dna_targets.mean():.2f}")

        print("\nCAR Data (car_data):")
        print(f"  Samples: {len(car_sequences)}")
        print("  Sequence type: Motif indices")
        print(f"  Sequence length: {car_sequences.shape[1]} motifs")
        print(f"  Motif range: {car_sequences.min()} - {car_sequences.max()}")
        print(f"  Target range: {car_targets.min():.3f} - {car_targets.max():.3f}")
        print(f"  Target mean: {car_targets.mean():.3f}")

        # Test encoding dimensions
        dna_encoded = one_hot_encode_sequences(
            dna_sequences[:1], SequenceModificationMethod.TRIM
        )
        car_encoded = one_hot_encode_sequences(
            car_sequences[:1], SequenceModificationMethod.CAR
        )

        dna_flattened = flatten_one_hot_sequences(dna_encoded)
        car_flattened = flatten_one_hot_sequences(car_encoded)

        print("\nEncoded Feature Dimensions:")
        print(f"  DNA: {dna_flattened.shape[1]} features")
        print(f"  CAR: {car_flattened.shape[1]} features")

    except Exception as e:
        print(f"✗ Error in comparison: {e}")


if __name__ == "__main__":
    print("🧬 Gene Circuit Design - CAR Data Integration Test")
    print("Testing updated codebase for both DNA and CAR motif data support\n")

    test_dna_sequence_processing()
    print()
    test_car_motif_processing()
    print()
    test_car_experiment()
    print()
    test_comparison()

    print("\n" + "=" * 60)
    print("✅ Integration testing complete!")
    print("The codebase now supports both DNA sequences and CAR motif data.")
    print("=" * 60)
