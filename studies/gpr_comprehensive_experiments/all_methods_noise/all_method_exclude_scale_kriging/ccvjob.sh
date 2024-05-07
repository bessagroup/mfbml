#!/usr/bin/bash
#SBATCH --job-name=cohesive_parameters
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=200:00:00
#SBATCH -p batch
#SBATCH --account=mbessa-condo 


# activate abaqus 
# source /gpfs/runtime/opt/abaqus/2021.1/abaqus
# module load abaqus/2021.1
# module load intel-oneapi-compilers


# Load the Conda module (adjust the path if necessary)

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate mfpml_env



# get to the directory 
echo "Master process running on $(hostname)"
echo "Directory is $(pwd)"
echo "Starting execution at $(date)"
echo "Current PATH is $PATH" 


python main.py --jobid=${SLURM_ARRAY_TASK_ID}
