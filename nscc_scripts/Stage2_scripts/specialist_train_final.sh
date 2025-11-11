#!/bin/bash

# --- PBS Directives for Job Array ---
#PBS -N specialist_training
#PBS -q normal
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -j oe

# ## JOB ARRAY DIRECTIVE: This will create 5 jobs, with indices 1 through 5 ##
#PBS -J 1-5

# --- Your commands start here ---

echo "--- Job Array Task $PBS_ARRAY_INDEX Started on $(hostname) at $(date) ---"
cd ${PBS_O_WORKDIR}

# 1. Set up environment
echo "Purging and loading modules..."
module load pytorch

# 2. Define arrays for pathologies and architectures
#    The order MUST match your final architectural plan
pathologies=( 'Atelectasis' 'Cardiomegaly' 'Effusion' 'Infiltration' 'Pneumonia' )
architectures=( 'resnet50' 'resnet50' 'resnet50' 'densenet121' 'densenet121' )

# 3. Get the parameters for this specific job array task
#    PBS_ARRAY_INDEX starts at 1, so we subtract 1 for 0-indexed bash arrays
INDEX=$(($PBS_ARRAY_INDEX - 1))
CURRENT_PATHOLOGY=${pathologies[$INDEX]}
CURRENT_ARCH=${architectures[$INDEX]}
SCRATCH_PATH="${PBS_O_WORKDIR}/dataset"

echo "Task details:"
echo "  Pathology: $CURRENT_PATHOLOGY"
echo "  Architecture: $CURRENT_ARCH"
echo "  Base Directory: $SCRATCH_PATH"

# 4. Run the training script with the correct parameters
python train_specialist_final.py \
    --pathology $CURRENT_PATHOLOGY \
    --architecture $CURRENT_ARCH \
    --base_dir $SCRATCH_PATH

echo "--- Job Array Task $PBS_ARRAY_INDEX Finished at $(date) ---"
