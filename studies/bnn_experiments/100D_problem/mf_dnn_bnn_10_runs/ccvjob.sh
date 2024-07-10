#!/usr/bin/bash
#SBATCH --job-name=mf_dnn_bnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=400:00:00
#SBATCH -p batch
#SBATCH --account=mbessa-condo 

# Load the Conda module (adjust the path if necessary)

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate mfpml_env



# get to the directory 
echo "Master process running on $(hostname)"
echo "Directory is $(pwd)"
echo "Starting execution at $(date)"
echo "Current PATH is $PATH" 


python main.py
