#!/bin/bash

#PBS -N consensus_training
#PBS -q normal
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -j oe

echo "--- Consensus Training Job Started on $(hostname) at $(date) ---"
cd ${PBS_O_WORKDIR}

# 1. Set up environment
module load pytorch

# 2. Define the scratch path
SCRATCH_PATH="${PBS_O_WORKDIR}/dataset" # ⚠️ UPDATE THIS

# 3. Run the final training script
python generate_triage_preds.py --base_dir $SCRATCH_PATH

echo "--- Job Finished at $(date) ---"
