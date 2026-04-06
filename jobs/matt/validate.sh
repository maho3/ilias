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
conda activate cmass

# ---- PATHS (edit these) ----
REPO_DIR=/u/maho3/git/ili-at-scale
MODEL_DIR=/path/to/output/models

cd $REPO_DIR

# ---- Matches original slurm_infvalid.sh ----
# Original: nbody=abacuslike, sim=fastpm_recnoise_tempOms8,
#           tracer=galaxy, embedding_net=fun, net=niall2,
#           subselect_cosmo=[0,4], include_noise=False, include_hod=False
python -m ilias.validate \
    model_dir=$MODEL_DIR \
    device=cpu
