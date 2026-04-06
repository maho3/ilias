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
conda activate cmass

sleep $SLURM_ARRAY_TASK_ID  # stagger start

# ---- PATHS (edit these) ----
REPO_DIR=/u/maho3/git/ili-at-scale
MODEL_DIR=/path/to/output/models

cd $REPO_DIR

export TQDM_DISABLE=0

# ---- Matches original slurm_infoptuna.sh ----
# Original: nbody=abacuslike, sim=fastpm_recnoise_temploglinear,
#           tracer=galaxy, embedding_net=fun, net=niall2,
#           loglinear_start_idx=30, include_noise=False, include_hod=False
python -m ilias.optuna \
    model_dir=$MODEL_DIR \
    device=cpu \
    embedding_net=fun \
    net=niall2
