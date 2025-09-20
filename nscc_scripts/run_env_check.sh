#!/bin/bash

# --- PBS Directives ---
#PBS -N env_check_final
#PBS -q normal
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -j oe


#Run this script to make sure the actual compute node can interact with your userspace 
# and can call the python libraries

# --- Your commands start here ---

echo "--- Job Started on $(hostname) at $(date) ---"

#cd to the current working directory first
cd ${PBS_O_WORKDIR}
echo "Current Working Directory: $(pwd)"

# 2. Load the base anaconda module
module load pytorch

# 5. Run the environment check script
echo "Running environment check script..."
python env_check.py
echo "--- Job Finished at $(date) ---"1~

