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
conda activate cmass

# ---- PATHS (edit these) ----
REPO_DIR=/u/maho3/git/ilias
MODEL_DIR=/work/hdd/bdne/maho3/temp_models
DATA_DIR=/work/hdd/bdne/maho3/cmass-ili/quijote/nbody/L1000-N128  # parent of 0/, 1/, 2/, ...

cd $REPO_DIR

# ---- Matches original slurm_infpre.sh ----
# Original: nbody=quijote, sim=nbody, tracer=galaxy, Nmax=4000,
#           val_frac=0, test_frac=1, include_noise=True, include_hod=False
python -m ilias.preprocess \
    model_dir=$MODEL_DIR \
    device=cpu \
    +loader.data_path=$DATA_DIR \
    +loader.tracer=galaxy \
    +loader.a=0.66667 \
    +loader.Nmax=200 \
    +loader.summary='[nbar,zPk0]' \
    +loader.include_hod=False \
    +loader.include_noise=True \
    +loader.correct_shot=True
