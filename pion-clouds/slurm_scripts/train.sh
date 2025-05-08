#!/bin/bash
#SBATCH --time 7-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --partition maxgpu
#SBATCH --job-name CCtraining

#SBATCH --constraint="GPUx1&A100"
##SBATCH --constraint="GPUx1&A100-PCIE-80GB"
##SBATCH --constraint=(A100|V100)&GPUx1

## for multi-gpu
##SBATCH --constraint=A100&GPUx4
##SBATCH --constraint=GPUx4

##SBATCH --nodelist=max-wng054
##SBATCH --exclude=max-wng055
#SBATCH --output /data/dust/user/mmozzani/pion-clouds/slurm_logs/training/Train-%j.out
# source ~/.zshrc 
##SBATCH --mail-type END 
##SBATCH --mail-user martina.mozzanica@desy.de

# mamba activate V100-torch
module load maxwell mamba
mamba init
mamba activate NewEnv
cd /data/dust/user/mmozzani/pion-clouds/

echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
export type="hcal" # options: ["hcal", "ecal"]'

PARAMS=(
      --type $type 
)
echo $type
 
python scripts/training.py --type $type 
# python scripts/training_CD.py --type $type 

exit