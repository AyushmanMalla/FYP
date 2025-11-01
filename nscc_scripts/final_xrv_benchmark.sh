#!/bin/bash

#PBS -N evaluate_xrv
#PBS -q normal
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -j oe

echo "--- Final X-ray vision Benchmark Job Started on $(hostname) at $(date) ---"
cd ${PBS_O_WORKDIR}

# 1. Set up environment
module load pytorch

# 2. Define paths
SCRATCH_PATH="${PBS_O_WORKDIR}/dataset" # ⚠️ UPDATE THIS


# 3. Run the evaluation script
python xrv_benchmark_final.py \
    --test_csv $SCRATCH_PATH/triage_test_set.csv \
    --image_dir $SCRATCH_PATH/CXR_ALL_FLAT \
    --output_csv $SCRATCH_PATH/torchxrayvision_results.csv \
    --model_weights densenet121-res224-all

echo "--- Job Finished at $(date) ---"
