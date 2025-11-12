from experiments.run_experiments_parallelization import run_single_experiment
from omegaconf import DictConfig, OmegaConf
import hydra
import submitit


def build_experiment_params(cfg):
    '''
    Build the experiment parameters for the active learning experiments
    '''
    pass


@hydra.main(version_base=None, config_path="conf", config_name="test_config.yaml")
def run_experiments(cfg):
    print(OmegaConf.to_yaml(cfg))

    # Initialize the executor
    executor = submitit.AutoExecutor(folder=cfg.executor_folder)
    executor.update_parameters(cfg.slurm)

    # build the experiment parameters
    pipeline_params = build_experiment_params(cfg.pipeline_params)

    # submit the array job
    for data_path in cfg.data_paths:
        jobs = executor.map_array(run_single_experiment, data_path, pipeline_params)

    # wait for the jobs to complete
    results = [job.result() for job in jobs]
    print(results)

if __name__ == "__main__":
    run_experiments()