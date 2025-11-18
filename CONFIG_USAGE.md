# Experiment Configuration System

The experiment runner now supports YAML-based configuration files for better organization and tracking of experiments.

## Quick Start

### List all available experiments:
```bash
python3 run_experiments.py --list-experiments
```

### Run a specific experiment:
```bash
python3 run_experiments.py --experiment car_experiments
```

### Preview what would run (dry-run):
```bash
python3 run_experiments.py --experiment car_experiments --dry-run
```

### Run all experiments in the config file:
```bash
python3 run_experiments.py --run-all
```

### Run all experiments (dry-run to see what would happen):
```bash
python3 run_experiments.py --run-all --dry-run
```

## Configuration File

Edit `configs/experiment_configs.yaml` to add/modify experiments.

### Available Experiments:
- `embeddings_basic` - Basic embedding experiments with log likelihood
- `embeddings_normalized` - Embedding experiments with normalization
- `embeddings_no_norm` - Embedding experiments without normalization
- `trim_pad_basic` - Trim and pad sequence experiments
- `car_experiments` - CAR-based sequence modification
- `current_active` - Current active experiment configuration

### Adding New Experiments:

```yaml
experiments:
  my_new_experiment:
    data_path: "/path/to/data.csv"
    strategies: ["HIGH_EXPRESSION", "RANDOM", "LOG_LIKELIHOOD"]
    seq_mod_methods: ["EMBEDDING"]
    seeds: [42, 123, 456]
    initial_sample_size: 8
    batch_size: 8
    max_rounds: 20
    normalize_expression: false
    output_dir: "results_my_experiment"
```
> **Note:** The pipeline no longer creates a dedicated test split. Any legacy `test_size`/`no_test`
> keys in old configuration files are ignored and can be safely removed.

### Available Options:

**Strategies:**
- `HIGH_EXPRESSION` - Select sequences with highest predicted expression
- `RANDOM` - Random selection
- `LOG_LIKELIHOOD` - Select based on log likelihood
- `UNCERTAINTY` - Select based on prediction uncertainty

**Sequence Modification Methods:**
- `TRIM` - Trim sequences to same length
- `PAD` - Pad sequences to same length
- `EMBEDDING` - Use embeddings
- `CAR` - CAR motif sequences

## Using in Code

```python
from utils.config_loader import get_experiment_config, run_experiment_from_config

# Load a specific experiment config
config = get_experiment_config("car_experiments")

# Run an experiment programmatically
results = run_experiment_from_config("car_experiments")
```

## Migration from Old System

Your old hardcoded configs have been converted to YAML format. Instead of:

```python
# Old way
config = {
    "data_path": "/path/to/data.csv",
    "strategies": [SelectionStrategy.HIGH_EXPRESSION, SelectionStrategy.RANDOM],
    # ... more config
}
```

Now use:

```bash
# New way
python3 run_experiments.py --experiment my_experiment_name
```

This makes it easier to:
- Track different experiment configurations
- Run experiments reproducibly
- Organize and compare results
- Share experiment setups
