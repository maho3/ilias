#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=4:00:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out

source ~/.bashrc
conda activate ili

cd /u/maho3/git/ili-at-scale

# ---- USER CONFIG ----
MODEL_DIR="/path/to/output/models"
DEVICE="cpu"

# Loader kwargs (passed as +loader.key=value)
LOADER_ARGS="+loader.data_path=/path/to/data +loader.kmin=0.0 +loader.kmax=0.4"

# ---- RUN ----
python -m ilias.preprocess \
    model_dir=$MODEL_DIR \
    device=$DEVICE \
    $LOADER_ARGS
