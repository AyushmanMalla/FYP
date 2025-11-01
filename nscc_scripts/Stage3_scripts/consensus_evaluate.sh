#!/bin/bash

#PBS -N evaluate_consensus
#PBS -q normal
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -j oe

echo "--- Evaluation of Consensus Model Job Started on $(hostname) at $(date) ---"
cd ${PBS_O_WORKDIR}

# 1. Set up environment
module load pytorch

# 2. Define paths
SCRATCH_PATH="${PBS_O_WORKDIR}/dataset" # ⚠️ UPDATE THIS


# 3. Run the evaluation script
python evaluate_consensus.py --base_dir $SCRATCH_PATH 

echo "--- Job Finished at $(date) ---"
