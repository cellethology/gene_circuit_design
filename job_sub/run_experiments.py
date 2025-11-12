import hydra
from omegaconf import OmegaConf

from experiments.run_experiments_parallelization import run_single_experiment


@hydra.main(version_base=None, config_path="conf", config_name="test_config.yaml")
def run_experiments(cfg):
    print(OmegaConf.to_yaml(cfg.pipeline_params))
    run_single_experiment(data_path=cfg.data_paths, pipeline_param=cfg.pipeline_params)


if __name__ == "__main__":
    run_experiments()
