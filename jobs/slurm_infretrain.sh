#!/bin/bash
#SBATCH --job-name=retrain
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=3:00:00
#SBATCH --partition=cpu
#SBATCH --account=bdne-delta-cpu
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out

source ~/.bashrc
conda activate ili

sleep $SLURM_ARRAY_TASK_ID  # stagger start

cd /u/maho3/git/ili-at-scale

# ---- USER CONFIG ----
MODEL_DIR="/path/to/output/models"
DEVICE="cpu"

# ---- RUN ----
# Each array task retrains one of the top nets from the Optuna study.
python -m ilias.train \
    model_dir=$MODEL_DIR \
    device=$DEVICE \
    embedding_net=fun \
    net=niall2 \
    retrain=True \
    net_index=$SLURM_ARRAY_TASK_ID
