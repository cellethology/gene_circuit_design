# Refactored ActiveLearningExperiment

This directory contains the refactored components that break down the monolithic `ActiveLearningExperiment` class into smaller, focused classes following the Single Responsibility Principle.

## Structure

```
experiments/core/
├── __init__.py              # Package exports
├── data_loader.py           # Data loading and splitting
├── model_trainer.py         # Model training and evaluation
├── metrics_calculator.py    # Custom metrics calculation
├── result_manager.py        # Result saving
├── variant_tracker.py       # Variant tracking
└── experiment.py            # Main orchestrator class
```

## Components

### DataLoader (`data_loader.py`)
- **Responsibility**: Load data from various formats (safetensors, CSV) and create train/test/unlabeled splits
- **Key Classes**:
  - `Dataset`: Data container with sequences, expressions, log_likelihoods, embeddings, variant_ids
  - `DataSplit`: Container for train/test/unlabeled indices
  - `DataLoader`: Main class for loading and splitting data

### ModelTrainer (`model_trainer.py`)
- **Responsibility**: Train models and evaluate performance
- **Key Methods**:
  - `train()`: Train the model on training data
  - `evaluate()`: Evaluate model on test set
  - `predict()`: Make predictions

### MetricsCalculator (`metrics_calculator.py`)
- **Responsibility**: Calculate and track custom metrics
- **Key Methods**:
  - `calculate_round_metrics()`: Calculate metrics for a round
  - `update_cumulative()`: Update cumulative metrics
  - `get_all_metrics()`: Get all calculated metrics

### VariantTracker (`variant_tracker.py`)
- **Responsibility**: Track selected variants across rounds
- **Key Methods**:
  - `track_round()`: Track variants selected in a round
  - `get_all_variants()`: Get all tracked variants

### ResultManager (`result_manager.py`)
- **Responsibility**: Save experiment results to files
- **Key Methods**:
  - `save_results()`: Save all results, custom metrics, and selected variants

### ActiveLearningExperiment (`experiment.py`)
- **Responsibility**: Orchestrate the active learning loop using composed components
- **Key Features**:
  - Uses composition instead of doing everything itself
  - Maintains backward compatibility through properties
  - Cleaner, more maintainable code

## Usage

The refactored `ActiveLearningExperiment` has the same interface as before:

```python
from experiments.core.experiment import ActiveLearningExperiment
from utils.config_loader import SelectionStrategy
from utils.model_loader import RegressionModelType
from utils.sequence_utils import SequenceModificationMethod

# Create experiment (same API as before)
experiment = ActiveLearningExperiment(
    data_path="data/sequences.safetensors",
    selection_strategy=SelectionStrategy.HIGH_EXPRESSION,
    regression_model=RegressionModelType.LINEAR,
    initial_sample_size=8,
    batch_size=8,
    random_seed=42,
    seq_mod_method=SequenceModificationMethod.EMBEDDING,
)

# Run experiment
results = experiment.run_experiment(max_rounds=30)

# Save results
experiment.save_results("output/results.csv")

# Access data (backward compatible)
print(f"Training set size: {len(experiment.train_indices)}")
print(f"Custom metrics: {experiment.custom_metrics}")
```

## Benefits

1. **Single Responsibility**: Each class has one clear purpose
2. **Testability**: Components can be tested independently
3. **Maintainability**: Easier to understand and modify
4. **Reusability**: Components can be reused in other contexts
5. **Backward Compatibility**: Existing code continues to work

## Migration

To migrate existing code:

1. **Option 1**: Use the refactored class directly (recommended)
   ```python
   from experiments.core.experiment import ActiveLearningExperiment
   ```

2. **Option 2**: Keep using the old class (it will be deprecated)
   ```python
   from experiments.run_experiments_parallelization import ActiveLearningExperiment
   ```

The refactored class maintains the same API, so no code changes are needed for basic usage.

## Testing

Each component can be tested independently:

```python
# Test DataLoader
from experiments.core.data_loader import DataLoader, Dataset

loader = DataLoader("data/sequences.safetensors")
dataset = loader.load()
assert len(dataset.sequences) > 0

# Test ModelTrainer
from experiments.core.model_trainer import ModelTrainer
from sklearn.linear_model import LinearRegression

trainer = ModelTrainer(LinearRegression())
trainer.train(X_train, y_train, train_indices)
metrics = trainer.evaluate(X_test, y_test)

# Test MetricsCalculator
from experiments.core.metrics_calculator import MetricsCalculator

calculator = MetricsCalculator(all_expressions)
round_metrics = calculator.calculate_round_metrics(selected_indices, strategy)
cumulative = calculator.update_cumulative(round_metrics)
```

## Future Improvements

- Add unit tests for each component
- Add integration tests for the full experiment
- Consider using dependency injection for better testability
- Add type hints for all public methods
- Add more comprehensive error handling
