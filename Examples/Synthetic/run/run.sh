#!/bin/sh

#SBATCH -A JSATS
#SBATCH -t 24:0:0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1

module load python/miniconda25.5.1
module load cuda/12.9
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
source ~/.bashrc
conda activate py39


# Supports: sbatch run.sh <group_id> <group_size>

# In array mode, group_id/group_size come from SLURM.
GROUP_ID="$1"
GROUP_SIZE="$2"
SCRIPT="$3"


srun python "$SCRIPT" --group-id "$GROUP_ID" --group-size "$GROUP_SIZE"




conda deactivate
