#!/bin/bash
#SBATCH --time 1-00:00:00
##SBATCH --nodes 1
#SBATCH --partition maxgpu
#SBATCH --job-name SFtraining
##SBATCH --constraint="GPUx1&A100"
#SBATCH --output /data/dust/user/mmozzani/pion-clouds/slurm_logs/sf/TrainSF-%j.out
# bash 
# source ~/.bashrc 

module load maxwell mamba
. mamba-init
# mamba activate NewEnv
mamba activate customConda

cd /data/dust/user/mmozzani/pion-clouds

python scripts/shower_flow_train.py
# python distillation.py
# python timing.py

exit