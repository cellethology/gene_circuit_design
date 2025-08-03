import os
import submitit
import json
from pathlib import Path

# 1. Import the run experiment function
# 2. Define the parameter combinations -> which is loaded from the config file
# 3. Submit the jobs
# 4. Collect the results
# 5. Combine the results into a single csv file
# 6. Call another function to plot the results

def run_experiment(params):
    """Run one experiment with given parameters"""
    learning_rate, batch_size, epochs = params
    
    # Simulate training (replace with your actual code)
    import time
    import random
    time.sleep(5)  # Simulate training time
    
    # Simulate some results
    accuracy = random.uniform(0.7, 0.95)
    loss = random.uniform(0.1, 0.5)
    
    results = {
        'learning_rate': learning_rate,
        'batch_size': batch_size, 
        'epochs': epochs,
        'accuracy': accuracy,
        'loss': loss
    }
    
    print(f"Experiment completed: lr={learning_rate}, bs={batch_size}, acc={accuracy:.3f}")
    return results

# Setup
executor = submitit.AutoExecutor(folder="logs_experiments")
executor.update_parameters(
    timeout_min=5,
    slurm_partition="wzt_20250411,intel-sc3",
    slurm_cpus_per_task=1,
    # Optional: email notifications
    slurm_mail_user="lizelun@westlake.edu.cn",
    slurm_mail_type="BEGIN,END,FAIL"
)

# Define parameter combinations
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
epochs = [10, 20]

# Create all combinations
experiment_params = []
for lr in learning_rates:
    for bs in batch_sizes:
        for ep in epochs:
            experiment_params.append((lr, bs, ep))

print(f"Running {len(experiment_params)} experiments...")

# Submit all experiments as array jobs
jobs = executor.map_array(run_experiment, experiment_params)

print(f"Submitted {len(jobs)} jobs")

# Collect results as they complete
results = []
for i, job in enumerate(jobs):
    try:
        result = job.result()
        results.append(result)
        print(f"Experiment {i+1}/{len(jobs)} done")
    except Exception as e:
        print(f"Experiment {i+1} failed: {e}")

# Save results
path_to_results = 'job_sub/submitit/results/experiment_results.json'
Path(path_to_results).parent.mkdir(parents=True, exist_ok=True)
with open(path_to_results, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Completed {len(results)}/{len(jobs)} experiments")
print("Results saved to experiment_results.json")