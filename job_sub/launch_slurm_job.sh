#!/bin/bash
#SBATCH -p intel-sc3
#SBATCH -t 24:00:00          # longer than the entire sweep
#SBATCH -c 1
#SBATCH --mem=2G             # tiny; the job just submits children
#SBATCH -J al_sweep

cd /home/wangzitongLab/wangzitong/gene_circuit_design
source .venv/bin/activate
python job_sub/run_config.py -m
