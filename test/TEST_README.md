# Unit Tests for Refactored Components

This directory contains comprehensive unit tests for the refactored active learning experiment components.

## Test Files

### `test_data_loader.py`
Tests for `DataLoader` and `Dataset` classes:
- Dataset creation and validation
- Loading from CSV files (with and without log likelihood)
- Loading from safetensors files (embeddings and PCA formats)
- Data normalization
- Creating train/test/unlabeled splits (random and K-means)
- Error handling for missing data

**Key Test Cases:**
- `test_load_csv_basic` - Basic CSV loading
- `test_load_safetensors_embeddings_format` - Safetensors with embeddings
- `test_load_safetensors_pca_format` - Safetensors with PCA components
- `test_normalize_data` - Data normalization
- `test_create_data_split_random` - Random data splitting
- `test_create_data_split_kmeans` - K-means data splitting

### `test_model_trainer.py`
Tests for `ModelTrainer` class:
- Model training with different model types
- Model evaluation on test sets
- Prediction functionality
- Complete train-evaluate workflows

**Key Test Cases:**
- `test_train_basic` - Basic training
- `test_evaluate_with_test_set` - Evaluation with metrics
- `test_evaluate_perfect_predictions` - Perfect prediction scenario
- `test_train_evaluate_workflow` - Complete workflow

### `test_metrics_calculator.py`
Tests for `MetricsCalculator` class:
- Round metrics calculation
- Cumulative metrics tracking
- Different selection strategies
- Top 10 ratio calculations

**Key Test Cases:**
- `test_calculate_round_metrics_basic` - Basic metrics calculation
- `test_update_cumulative_first_round` - First round cumulative
- `test_update_cumulative_multiple_rounds` - Multi-round cumulative
- `test_metrics_with_top_10_calculation` - Top 10% intersection

### `test_variant_tracker.py`
Tests for `VariantTracker` class:
- Tracking variants across rounds
- Handling variant IDs
- Sequence truncation
- NaN log likelihood handling

**Key Test Cases:**
- `test_track_round_basic` - Basic variant tracking
- `test_track_round_with_variant_ids` - With variant IDs
- `test_track_multiple_rounds` - Multi-round tracking
- `test_track_round_long_sequence_truncation` - Sequence truncation

### `test_result_manager.py`
Tests for `ResultManager` class:
- Saving results to CSV
- Saving custom metrics
- Saving selected variants
- Metadata column ordering

**Key Test Cases:**
- `test_save_results_basic` - Basic result saving
- `test_save_custom_metrics` - Custom metrics saving
- `test_save_selected_variants` - Variant saving
- `test_metadata_columns_order` - Column ordering

### `test_experiment_integration.py`
Integration tests for `ActiveLearningExperiment`:
- Complete experiment workflows
- Different data formats
- Different selection strategies
- Different regression models
- Backward compatibility

**Key Test Cases:**
- `test_experiment_initialization_safetensors` - Safetensors initialization
- `test_run_experiment_multiple_rounds` - Multi-round experiments
- `test_backward_compatibility_properties` - Backward compatibility
- `test_different_selection_strategies` - Different strategies
- `test_different_regression_models` - Different models

## Running Tests

### Run all tests:
```bash
pytest test/
```

### Run specific test file:
```bash
pytest test/test_data_loader.py
```

### Run with verbose output:
```bash
pytest test/ -v
```

### Run with coverage:
```bash
pytest test/ --cov=experiments.core --cov-report=html
```

### Run specific test class:
```bash
pytest test/test_data_loader.py::TestDataLoader
```

### Run specific test method:
```bash
pytest test/test_data_loader.py::TestDataLoader::test_load_csv_basic
```

## Test Coverage Goals

- **Unit Tests**: >90% coverage for each component
- **Integration Tests**: Cover all major workflows
- **Edge Cases**: Test error conditions and boundary cases

## Test Fixtures

Tests use `pytest` fixtures:
- `tmp_path` - Temporary directory for test files
- Custom fixtures for creating test data files

## Dependencies

Tests require:
- `pytest` - Testing framework
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `torch` - For safetensors file creation
- `safetensors` - For safetensors file handling
- `scikit-learn` - For model testing

## Notes

- All tests use temporary directories for file I/O
- Tests are isolated and can run in any order
- Mock data is generated deterministically using seeds
- Tests follow the existing codebase style

## Future Improvements

- Add performance benchmarks
- Add property-based testing (hypothesis)
- Add mutation testing
- Add test coverage reports to CI/CD
