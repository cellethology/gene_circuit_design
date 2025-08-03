# 1. Import the run experiment function
from pathlib import Path
from experiments.run_experiments_parallelization import run_single_experiment
from utils.config_loader import get_experiment_config
import submitit
import logging
from utils.plotting import create_combined_results_from_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# only handle single experiment at a time
executor = submitit.AutoExecutor(folder="logs_experiments")
executor.update_parameters(
    timeout_min=10,
    slurm_partition="wzt_20250411,intel-sc3",
    slurm_cpus_per_task=1,
    slurm_mail_user="lizelun@westlake.edu.cn",
    slurm_mail_type="BEGIN,END,FAIL",
    slurm_qos="huge",
    slurm_mem_per_cpu="2GB",
)
# Define variables
config_file = "configs/enformer.yaml"
experiment_name = "enformer_template"
config = get_experiment_config(experiment_name=experiment_name, config_file=config_file)

# 2. Define the parameter combinations -> which is loaded from the config file
initial_sample_size = config["initial_sample_size"]
data_path = config["data_path"]
batch_size = config["batch_size"]
test_size = config["test_size"]
no_test = config["no_test"]
max_rounds = config["max_rounds"]
output_dir = config["output_dir"]
normalize_expression = config["normalize_expression"]

# I just need to convert config into a list of dictionaries
experiment_params = []
for strategy in config["strategies"]:
    for regression_model in config["regression_models"]:
        for seq_mod_method in config["seq_mod_methods"]:
            for seed in config["seeds"]:
                experiment_params.append({
                    "strategy": strategy,
                    "regression_model": regression_model,
                    "seq_mod_method": seq_mod_method,
                    "seed": seed,
                    "data_path": data_path,
                    "initial_sample_size": initial_sample_size,
                    "batch_size": batch_size,
                    "test_size": test_size,
                    "no_test": no_test,
                    "max_rounds": max_rounds,
                    "output_dir": output_dir,
                    "normalize_expression": normalize_expression,
                })

# print(experiment_params)

# 3. Submit the jobs
jobs = executor.map_array(run_single_experiment, experiment_params)

# 4. Collect the results
# Just a holder to wait for jobs to complete
results = [job.result() for job in jobs]

# it is okay if it failed here (this is only running locally)
# 5. Combine the results into a single csv file
create_combined_results_from_files(output_path=Path(output_dir))

# 6. Call another function to plot the results
# TODO