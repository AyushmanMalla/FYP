#!/bin/bash

#PBS -N consensus_dataprep
#PBS -q normal 
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -j oe

echo "--- Consensus Dataprep Job Started on $(hostname) at $(date) ---"
cd ${PBS_O_WORKDIR}

# 1. Set up environment
module load pytorch

# 2. Define paths
SCRATCH_PATH="${PBS_O_WORKDIR}/dataset" # ⚠️ UPDATE THIS
IMAGE_PATH="$SCRATCH_PATH/CXR_ALL_FLAT" # ⚠️ UPDATE THIS
TRIAGE_MODEL_PATH="$SCRATCH_PATH/triage_model_best.pth" # ⚠️ UPDATE THIS

# 3. Run the data generation script
python metaDataset_Creator.py \
    --base_dir $SCRATCH_PATH \
    --image_dir $IMAGE_PATH \
    --triage_model_path $TRIAGE_MODEL_PATH

echo "--- Job Finished at $(date) ---"