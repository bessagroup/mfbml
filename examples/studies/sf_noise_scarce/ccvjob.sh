#!/bin/bash
#SBATCH --job-name=gpr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00


# Load the Conda module (adjust the path if necessary)
module load anaconda/2023.03-1
source /gpfs/runtime/opt/anaconda/2023.03-1/etc/profile.d/conda.sh
conda activate mfpml_env

# activate abaqus 
# source /gpfs/runtime/opt/abaqus/2021.1/abaqus
# module load abaqus/2021


# get to the directory 
echo "Master process running on $(hostname)"
echo "Directory is $(pwd)"
echo "Starting execution at $(date)"
echo "Current PATH is $PATH" 


python main.py --jobid=${SLURM_ARRAY_TASK_ID}
