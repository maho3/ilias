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
conda activate cmass

sleep $SLURM_ARRAY_TASK_ID  # stagger start

# ---- PATHS (edit these) ----
REPO_DIR=/u/maho3/git/ili-at-scale
MODEL_DIR=/path/to/output/models

cd $REPO_DIR

# ---- Matches original slurm_infretrain.sh ----
# Original: nbody=abacuslike, sim=fastpm_recnoise_tempOms8,
#           tracer=galaxy, embedding_net=fun, net=niall2,
#           subselect_cosmo=[0,4], include_noise=False, include_hod=False
python -m ilias.train \
    model_dir=$MODEL_DIR \
    device=cpu \
    embedding_net=fun \
    net=niall2 \
    retrain=True \
    net_index=$SLURM_ARRAY_TASK_ID
