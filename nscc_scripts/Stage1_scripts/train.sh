#!/bin/bash

# --- PBS Directives for Full Run ---
#PBS -N triage_train_full
#PBS -q normal
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -j oe

# --- Your commands start here ---

#In my experience this training run took about 29 min to finish and took up about 35 credits from the account

echo "--- Full Training Job Started on $(hostname) at $(date) ---"
cd ${PBS_O_WORKDIR}
echo "Current directory: $(pwd)"

# 1. Set up environment
echo "Purging and loading modules..."
module load pytorch

# 2. Run the full training script
echo "Running full training script..."
python train_triage_full.py
echo "--- Job Finished at $(date) ---"
