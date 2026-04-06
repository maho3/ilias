#!/bin/bash
#SBATCH --job-name=validate
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --account=bdne-delta-cpu
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out

source ~/.bashrc
conda activate ili

cd /u/maho3/git/ili-at-scale

# ---- USER CONFIG ----
MODEL_DIR="/path/to/output/models"
DEVICE="cpu"

# ---- RUN ----
python -m ili_at_scale.validate \
    model_dir=$MODEL_DIR \
    device=$DEVICE
