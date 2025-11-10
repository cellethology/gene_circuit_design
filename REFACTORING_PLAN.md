# Code Refactoring Plan

This document outlines a comprehensive refactoring plan for the gene circuit design codebase, organized by priority and impact.

## Executive Summary

The codebase is functional but has several areas that need improvement:
1. **Code Duplication**: Two similar experiment runner files
2. **Large Classes**: `ActiveLearningExperiment` violates Single Responsibility Principle
3. **Complex Methods**: Several methods are too long and do multiple things
4. **Type Safety**: Missing or incomplete type annotations
5. **Error Handling**: Could be more robust with custom exceptions
6. **Code Organization**: Some utilities are misplaced

---

## Priority 1: Critical Refactoring (High Impact, High Value)

### 1.1 Consolidate Duplicate Experiment Runners

**Problem**: `run_experiments.py` and `run_experiments_parallelization.py` have ~80% code duplication.

**Solution**:
- Keep `run_experiments_parallelization.py` as the main file (it has parallelization)
- Extract common functionality into shared utility modules
- Make parallelization optional via a parameter
- Deprecate `run_experiments.py` after migration

**Files to modify**:
- `experiments/run_experiments_parallelization.py` - Add sequential mode option
- `experiments/run_experiments.py` - Mark as deprecated, redirect to parallel version
- Create `experiments/experiment_runner.py` - Core experiment logic

**Benefits**:
- Reduces maintenance burden
- Single source of truth
- Easier to add new features

---

### 1.2 Break Down `ActiveLearningExperiment` Class

**Problem**: The class is 700+ lines and handles:
- Data loading (multiple formats)
- Model training
- Selection strategies
- Metrics calculation
- Result saving
- Variant tracking

**Solution**: Apply Single Responsibility Principle:

```
ActiveLearningExperiment (orchestrator)
├── DataLoader (handles all data loading logic)
├── ModelTrainer (handles model training/evaluation)
├── MetricsCalculator (handles custom metrics)
├── ResultSaver (handles saving results)
└── VariantTracker (handles variant tracking)
```

**New Structure**:
```
experiments/
├── core/
│   ├── __init__.py
│   ├── experiment.py          # Main orchestrator
│   ├── data_loader.py          # Data loading logic
│   ├── model_trainer.py       # Model training/evaluation
│   ├── metrics_calculator.py  # Custom metrics
│   └── result_manager.py       # Saving results
```

**Benefits**:
- Each class has a single, clear responsibility
- Easier to test individual components
- Better code reusability
- Easier to understand and maintain

---

### 1.3 Extract Data Loading Logic

**Problem**: `_load_and_prepare_data()` is 200+ lines and handles:
- Safetensors files (multiple formats)
- CSV files
- Different data structures
- Normalization

**Solution**: Create a `DataLoader` class with strategy pattern:

```python
class DataLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Dataset:
        pass

class SafetensorsLoader(DataLoader):
    def load(self, path: str) -> Dataset:
        # Handle safetensors files

class CSVLoader(DataLoader):
    def load(self, path: str) -> Dataset:
        # Handle CSV files

class Dataset:
    sequences: List[str]
    expressions: np.ndarray
    log_likelihoods: np.ndarray
    embeddings: Optional[np.ndarray]
    variant_ids: Optional[np.ndarray]
```

**Benefits**:
- Clear separation of concerns
- Easy to add new data formats
- Better testability
- Type-safe data structures

---

## Priority 2: Important Improvements (Medium-High Impact)

### 2.1 Improve Type Annotations

**Problem**: Many functions lack complete type annotations, especially:
- `experiments/util.py` - `encode_sequences` return type could be more specific
- `utils/metrics.py` - Missing type hints
- Various `Any` types that could be more specific

**Solution**:
- Add comprehensive type annotations using `typing` module
- Create type aliases for common patterns:
  ```python
  from typing import TypeAlias

  Indices: TypeAlias = List[int]
  ExpressionArray: TypeAlias = np.ndarray
  ```
- Use `Protocol` for duck typing where appropriate
- Replace `Any` with specific types where possible

**Files to update**:
- `experiments/util.py`
- `utils/metrics.py`
- `utils/model_loader.py`
- `experiments/selection_strategies.py`

---

### 2.2 Enhance Error Handling

**Problem**: Generic exceptions and error messages make debugging difficult.

**Solution**: Create custom exception hierarchy:

```python
# utils/exceptions.py
class GeneCircuitDesignError(Exception):
    """Base exception for all gene circuit design errors."""
    pass

class DataLoadingError(GeneCircuitDesignError):
    """Raised when data loading fails."""
    pass

class InvalidDataFormatError(DataLoadingError):
    """Raised when data format is invalid."""
    pass

class MissingDataError(DataLoadingError):
    """Raised when required data is missing."""
    pass

class ModelTrainingError(GeneCircuitDesignError):
    """Raised when model training fails."""
    pass

class ExperimentConfigurationError(GeneCircuitDesignError):
    """Raised when experiment configuration is invalid."""
    pass
```

**Benefits**:
- More informative error messages
- Easier debugging
- Better error handling in calling code

---

### 2.3 Refactor Metrics Calculation

**Problem**: `_intermediate_evaluation_custom_metrics()` is complex and mixes concerns.

**Solution**: Create a `MetricsCalculator` class:

```python
class MetricsCalculator:
    def __init__(self, all_expressions: np.ndarray):
        self.all_expressions = all_expressions
        self.cumulative_metrics: List[Dict[str, float]] = []

    def calculate_round_metrics(
        self,
        selected_indices: List[int],
        predictions: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate metrics for a single round."""
        # Implementation

    def update_cumulative(self, round_metrics: Dict[str, float]) -> Dict[str, float]:
        """Update cumulative metrics."""
        # Implementation
```

**Benefits**:
- Clearer metric calculation logic
- Easier to add new metrics
- Better testability

---

### 2.4 Improve Configuration Management

**Problem**: Configuration conversion and validation could be more robust.

**Solution**:
- Use Pydantic models for configuration validation:
  ```python
  from pydantic import BaseModel, Field, validator

  class ExperimentConfig(BaseModel):
      data_path: Path
      strategies: List[SelectionStrategy]
      regression_models: List[RegressionModelType]
      seeds: List[int] = Field(..., min_items=1)
      initial_sample_size: int = Field(8, gt=0)
      batch_size: int = Field(8, gt=0)
      max_rounds: int = Field(20, gt=0)

      @validator('batch_size')
      def batch_size_positive(cls, v):
          if v <= 0:
              raise ValueError('batch_size must be positive')
          return v
  ```

**Benefits**:
- Automatic validation
- Better error messages
- Type safety
- IDE autocomplete support

---

## Priority 3: Code Quality Improvements (Medium Impact)

### 3.1 Extract Magic Numbers to Constants

**Problem**: Hardcoded values scattered throughout code:
- `1e-30` for numerical stability
- `0.1` for top 10% calculation
- Default values in multiple places

**Solution**: Create constants module:

```python
# utils/constants.py
EPSILON = 1e-30  # Small value for numerical stability
TOP_PERCENTILE = 0.1  # Top 10% for metric calculations
DEFAULT_INITIAL_SAMPLE_SIZE = 8
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_ROUNDS = 20
DEFAULT_TEST_SIZE = 50
```

---

### 3.2 Improve Logging Setup

**Problem**: Logging is configured at module level, making it hard to control.

**Solution**: Create a logging configuration module:

```python
# utils/logging_config.py
import logging
from pathlib import Path
from typing import Optional

def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Configure logging for the application."""
    # Implementation
```

**Benefits**:
- Centralized logging configuration
- Easier to change log levels
- Better for testing

---

### 3.3 Add Missing Docstrings

**Problem**: Some functions have minimal or missing docstrings.

**Solution**: Add comprehensive Google-style docstrings to:
- `utils/metrics.py` - All functions
- `experiments/util.py` - All functions
- `utils/model_loader.py` - `return_model` function

**Template**:
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of what the function does.

    More detailed description if needed. Explain the purpose,
    algorithm, or important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When and why this is raised
        TypeError: When and why this is raised

    Example:
        >>> result = function_name("example", 42)
        >>> print(result)
        "expected output"
    """
```

---

### 3.4 Simplify Complex Conditionals

**Problem**: Some methods have deeply nested conditionals (e.g., `_load_and_prepare_data`).

**Solution**: Use early returns and guard clauses:

```python
# Before
def method(self):
    if condition1:
        if condition2:
            if condition3:
                # do something
            else:
                # handle else
        else:
            # handle else
    else:
        # handle else

# After
def method(self):
    if not condition1:
        # handle early return
        return

    if not condition2:
        # handle early return
        return

    if not condition3:
        # handle early return
        return

    # do something
```

---

## Priority 4: Structural Improvements (Low-Medium Impact)

### 4.1 Create Data Transfer Objects (DTOs)

**Problem**: Data is passed around as dictionaries and lists, making it hard to track structure.

**Solution**: Create dataclasses for structured data:

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ExperimentResult:
    """Results from a single experiment round."""
    round: int
    strategy: str
    seq_mod_method: str
    seed: int
    train_size: int
    unlabeled_size: int
    rmse: Optional[float] = None
    r2: Optional[float] = None
    pearson_correlation: Optional[float] = None
    spearman_correlation: Optional[float] = None

@dataclass
class CustomMetrics:
    """Custom metrics for a round."""
    top_10_ratio_intersected_indices: float
    best_value_predictions_values: float
    normalized_predictions_predictions_values: float
    # ... etc
```

**Benefits**:
- Type safety
- IDE autocomplete
- Self-documenting code
- Easier to serialize/deserialize

---

### 4.2 Improve Selection Strategy Pattern

**Problem**: Selection strategies are well-designed, but could use better type hints.

**Solution**:
- Add `Protocol` for experiment interface
- Improve type annotations
- Add validation

```python
from typing import Protocol

class ExperimentProtocol(Protocol):
    """Protocol defining the interface an experiment must provide."""
    def _encode_sequences(self, indices: List[int]) -> np.ndarray:
        ...

    @property
    def model(self) -> Any:
        ...
```

---

### 4.3 Add Unit Tests for Critical Components

**Problem**: While tests exist, coverage could be improved for:
- Data loading logic
- Metrics calculation
- Selection strategies

**Solution**: Add comprehensive unit tests:
- Test each data loader independently
- Test metrics with known inputs/outputs
- Test selection strategies with mock data

---

## Implementation Strategy

### Phase 1: Foundation (Week 1-2)
1. Create custom exceptions
2. Extract constants
3. Improve type annotations in critical paths
4. Add missing docstrings

### Phase 2: Core Refactoring (Week 3-4)
1. Extract DataLoader class
2. Break down ActiveLearningExperiment
3. Create MetricsCalculator
4. Consolidate experiment runners

### Phase 3: Enhancement (Week 5-6)
1. Add Pydantic models for config
2. Create DTOs
3. Improve error handling throughout
4. Add comprehensive tests

### Phase 4: Polish (Week 7-8)
1. Code review
2. Performance optimization
3. Documentation updates
4. Migration guide

---

## Migration Guide

When refactoring, maintain backward compatibility where possible:

1. **Deprecation Warnings**: Add warnings for deprecated functions
2. **Adapter Pattern**: Create adapters for old interfaces
3. **Gradual Migration**: Migrate one module at a time
4. **Version Control**: Use feature branches for each refactoring

---

## Testing Strategy

After each refactoring:
1. Run existing tests to ensure no regressions
2. Add new tests for refactored components
3. Integration tests for end-to-end workflows
4. Performance benchmarks to ensure no degradation

---

## Metrics for Success

- **Code Duplication**: Reduce from ~80% to <10%
- **Class Size**: Average class size <200 lines
- **Method Complexity**: Average method <30 lines
- **Type Coverage**: >95% of functions fully typed
- **Test Coverage**: >90% for critical paths
- **Documentation**: 100% of public APIs documented

---

## Notes

- All refactoring should maintain existing functionality
- Prioritize backward compatibility
- Document breaking changes clearly
- Get team review before major structural changes
