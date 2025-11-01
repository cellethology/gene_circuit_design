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
## Data Directory
```bash
./data_new
├── 166k    
├── 1M
├── alcantar_2025
├── Angenent-Mari_2020
├── Angenent-Mari_2020_OFF
```

## Slurm Job Submission

For running experiments on a Slurm cluster, use the sequential_parallel_job_test.py script:

```bash
# Run experiments in parallel using Slurm
python job_sub/submitit/sequential_parallel_job_test.py \
    --config-files configs/feng_2023.yaml \
    --experiment-names evo2_pca \
    --timeout-min 30 \
    --slurm-cpus-per-task 2 \
    --slurm-mem-per-cpu 4GB
```
- **Do check the config file to update the experiment path for it to run properly**

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

## Running Experiments

### Configuration Files

Experiments are defined in YAML files located in the `configs/` directory. Each experiment configuration includes:

- **data_path**: Path to the data file (`.safetensors` or `.csv`)
- **strategies**: Selection strategies (e.g., `HIGH_EXPRESSION`, `RANDOM`, `KMEANS_RANDOM`)
- **seq_mod_methods**: Sequence modification methods (e.g., `EMBEDDING`, `TRIM`, `PAD`)
- **regression_models**: ML models to use (e.g., `LINEAR`, `KNN`, `RANDOM_FOREST`, `XGBOOST`, `MLP`)
- **seeds**: Random seeds for reproducibility
- **output_dir**: Where to save results

Example configuration structure:

```yaml
experiments:
  my_experiment:
    data_path: "data_new/166k/sei_embeddings_pca.safetensors"
    strategies: ["HIGH_EXPRESSION", "RANDOM"]
    seq_mod_methods: ["EMBEDDING"]
    regression_models: ["LINEAR", "KNN", "RANDOM_FOREST"]
    seeds: [1, 2, 3, 4, 5]
    initial_sample_size: 8
    batch_size: 8
    max_rounds: 20
    normalize_input_output: true
    output_dir: "results/my_experiment"
```

### Experiment Output

After running experiments, results are saved to the specified `output_dir` with the following structure:

```
results/{experiment_name}/
├── {strategy}_{method}_{regressor}_seed_{seed}_results.csv      # Individual seed results
├── {strategy}_{method}_{regressor}_seed_{seed}_custom_metrics.csv  # Custom metrics per seed
├── {strategy}_{method}_{regressor}_all_seeds_results.csv       # Combined results per strategy
├── combined_all_results.csv                                    # All experiments combined
└── combined_all_custom_metrics.csv                             # All custom metrics combined
```

## Generating Plots

The project includes several plotting scripts for visualizing experiment results:

### 1. Area Under Curve (AUC) Grid Plot

Generate heatmap plots showing AUC metrics across datasets, embeddings, and regressors:

```bash
python plotting/area_under_curve_grid_plot.py \
    --results-base-path results/auc_result
```

Options:
- `--results-base-path`: Base directory containing result files (default: `results/auc_result`)
- Output: Saves plots to `plots/AUC_results/`

### 2. Averaged Performance Analysis

Generate averaged performance analysis plots with baseline comparisons:

```bash
python plotting/averaged_performance_analysis.py \
    --results-base-path results/166k_2024_regulators_auto_gen \
    --output-dir plots/166k_2024_regulators_summary \
    --metric auc_normalized_pred
```

Options:
- `--results-base-path`: Base path for results (default: `results/166k_2024_regulators_auto_gen`)
- `--output-dir`: Output directory for plots (default: `plots/166k_2024_regulators_summary`)
- `--metric`: Metric to analyze (default: `auc_normalized_pred`)

All plots are saved in high-resolution PNG format (300 DPI) suitable for publication.

## Output Structure

### Results
- Individual experiment results: `results/{experiment_name}/`
- Combined results: `results/{experiment_name}/combined_all_results.csv`
- Custom metrics: `results/{experiment_name}/combined_all_custom_metrics.csv`
- Selected variants tracking: `results/{experiment_name}/*_selected_variants.csv`

### Plots
- Saved in `plots/` directory
- Includes learning curves, regressor comparisons, AUC heatmaps, and custom metrics
- High-resolution PNG formats (300 DPI)

## Troubleshooting

### Common Issues

1. **Memory issues**: Reduce `max_workers` or `cores_per_process` in config

### Performance Tips
- Use `--max-workers` for parallel processing
- Set `cores_per_process` in config for CPU-intensive models
- Use filtered data files to reduce memory usage
