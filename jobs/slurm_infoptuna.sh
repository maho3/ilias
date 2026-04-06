#!/bin/bash
#SBATCH --job-name=optuna
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
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
# Each array task runs n_trials Optuna trials in parallel via the shared
# SQLite study. Adjust --array and n_trials to control total search budget.
python -m ili_at_scale.optuna \
    model_dir=$MODEL_DIR \
    device=$DEVICE \
    embedding_net=fun \
    net=niall2
