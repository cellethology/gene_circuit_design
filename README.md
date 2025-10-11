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
- **Do check the config file to update the experiment path for it ro run properly**

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
