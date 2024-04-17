#!/bin/bash

#SBATCH --job-name="Midterm Project"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4GB

# Load the required modules
module load Python
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your Python environment if needed
source ~/DSF/midterm/bin/activate

# Run your Python script
srun python3 job.py
