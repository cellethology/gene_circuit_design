# Gene Circuit Design - Active Learning Project

This project implements active learning strategies for DNA sequence-expression prediction using machine learning models.

## Install Dependencies
```bash
# Clone the repository
git clone git@github.com:cellethology/gene_circuit_design.git
cd gene_circuit_design

# Install all dependencies
uv sync --python 3.9
```

## Quick Start - Test Run

For the fastest test to verify everything works:

```bash
# Create a minimal test config (2 rounds, 1 seed)
python run_and_plot.py --config configs/final_configs_test.yaml --experiment onehot_pca_experiment --max-workers 2
```

This will:
- Run the PCA experiment with 5 rounds and 2 seeds
- Use 2 parallel workers
- Generate plots automatically
- Complete in ~5-10 minutes

## Main Usage - Run and Plot Pipeline

### Basic Usage
```bash
# Run all experiments in a config file and generate plots
python run_and_plot.py --config configs/onehot.yaml

# Run specific experiment only
python run_and_plot.py --config configs/onehot.yaml --experiment onehot_pad

# Run with parallel processing (recommended)
python run_and_plot.py --config configs/onehot.yaml --max-workers 4
```

### Advanced Options
```bash
# Skip experiments, only generate plots for existing results
python run_and_plot.py --config configs/final_configs_test.yaml --skip-experiments

# Skip plots, only run experiments
python run_and_plot.py --config configs/final_configs_test.yaml --skip-plots

# Plot specific results directory
python run_and_plot.py --config configs/final_configs_test.yaml --results-dir results/final_test/onehot_pca
```

## Configuration Files

### Available Configs
- `configs/final_configs_test.yaml` - Main test configuration with PCA and embeddings experiments
- `configs/experiment_configs_166k_regressors.yaml` - Full 166k dataset experiments

### Config Structure
```yaml
experiments:
  experiment_name:
    data_path: "path/to/data.safetensors"
    strategies: ["HIGH_EXPRESSION", "RANDOM", "LOG_LIKELIHOOD"]
    regression_models: ["LINEAR", "KNN", "RANDOM_FOREST"]
    seeds: [42, 123]
    max_rounds: 5
    cores_per_process: 4
```

## Active Learning Strategies

1. **HIGH_EXPRESSION**: Selects sequences with highest predicted expression values
2. **RANDOM**: Random selection (baseline)
3. **LOG_LIKELIHOOD**: Selects sequences with highest zero-shot log likelihood values

## Regression Models

1. **LINEAR**: Linear regression
2. **KNN**: K-Nearest Neighbors regression
3. **RANDOM_FOREST**: Random forest regression

## Data Processing

### Sequence Formats
- **PCA**: Pre-computed PCA components from one-hot encoded sequences
- **Embeddings**: Pre-computed EVO embeddings for sequences

### Expression Filtering
```bash
# Filter high expression outliers (>750) and analyze distribution
python filter_high_expressions.py
```

## Development Commands

```bash
# Linting
ruff check .
ruff check --fix .

# Testing
pytest

# Manual experiment running (without plots)
python run_all_experiments.py --config configs/final_configs_test.yaml --max-workers 4

# Manual plot generation
python plotting/visualize_all_results.py
```

## Slurm Job Submission

For running experiments on a Slurm cluster, use the sequential_parallel_job_test.py script:

```bash
# Run experiments in parallel using Slurm
python job_sub/submitit/sequential_parallel_job_test.py \
    --config-files configs/enformer.yaml configs/another_config.yaml \
    --experiment-names enformer_template another_experiment \
    --timeout-min 30 \
    --slurm-cpus-per-task 2 \
    --slurm-mem-per-cpu 4GB
```

This script allows running multiple experiment configurations in parallel on a Slurm cluster. Each parameter combination is submitted as a separate Slurm job. The script handles job submission, monitoring, and result collection.

## Project Structure

```
├── configs/                    # Experiment configuration files
├── data/                      # Data files (safetensors format)
├── experiments/               # Main experiment scripts
├── plotting/                  # Plotting and visualization scripts
├── results/                   # Experiment results and plots
├── utils/                     # Utility functions
├── run_and_plot.py           # Main pipeline script
├── run_all_experiments.py    # Experiment runner
```

## Output

### Results
- Individual experiment results: `results/{experiment_name}/`
- Combined results: `results/{experiment_name}/combined_all_results.csv`
- Custom metrics: `results/{experiment_name}/combined_all_custom_metrics.csv`

### Plots
- Saved in `plots/` directory
- Includes learning curves, regressor comparisons, and custom metrics
- High-resolution PNG and PDF formats

## Troubleshooting

### Common Issues

1. **Permission errors**: Run on compute nodes without `uv` (scripts now use direct `python`)
2. **Memory issues**: Reduce `max_workers` or `cores_per_process` in config

### Performance Tips
- Use `--max-workers` for parallel processing
- Set `cores_per_process` in config for CPU-intensive models
- Use filtered data files to reduce memory usage
