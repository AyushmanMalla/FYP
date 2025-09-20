#!/bin/bash

# --- PBS Directives ---
#PBS -N triage_smoke_test
#PBS -q normal
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -j oe

# --- Your commands start here ---

echo "--- Job Started on $(hostname) at $(date) ---"
cd ${PBS_O_WORKDIR}
echo "Current directory: $(pwd)"

# 1. Set up environment
echo "Purging and loading modules..."
module load pytorch


# 2. Run the smoke test script
echo "Running smoke test training script..."
python train_triage_smoke_test.py
echo "--- Job Finished at $(date) ---"
