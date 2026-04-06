# Matt's ltu-cmass migration guide

This directory contains job scripts that replicate the existing ltu-cmass
inference pipeline using ilias.

## Setup

### 1. Install

```bash
cd ~/git/ili-at-scale
pip install -e ~/git/ltu-ili
pip install -e .
```

### 2. Configure the loader

Copy `loaders_matt.py` from this directory into the module:

```bash
cp jobs/matt/loaders_matt.py ilias/loaders.py
```

This loader replicates the data loading logic from `cmass/infer/loaders.py`
and `cmass/infer/preprocess.py`. It reads h5 diagnostic files from a
simulation suite directory and produces preprocessed summary statistics.

### 3. Configure the prior

Copy `priors_matt.py` from this directory into the module:

```bash
cp jobs/matt/priors_matt.py ilias/priors.py
```

This provides the Quijote cosmology prior and supports optional HOD/noise
prior augmentation, matching the original ltu-cmass behavior.

### 4. Set paths

Edit the job scripts in this directory to set:
- `REPO_DIR`: path to your ili-at-scale clone
- `MODEL_DIR`: where to save outputs
- `DATA_DIR`: path to the simulation suite (the parent directory containing
  numbered simulation folders like `0/`, `1/`, etc.)

## Job scripts

These mirror the four original ltu-cmass SLURM scripts:

| Script | Original | What it does |
|--------|----------|-------------|
| `preprocess.sh` | `slurm_infpre.sh` | Load Quijote galaxy summaries, preprocess, split |
| `optuna.sh` | `slurm_infoptuna.sh` | Optuna HP search with funnel net + niall2 config |
| `retrain.sh` | `slurm_infretrain.sh` | Retrain top nets from Optuna study |
| `validate.sh` | `slurm_infvalid.sh` | Validate ensemble on test set |

## Key differences from ltu-cmass

- **No `nbody`/`sim`/`tracer` config** — the loader handles data paths directly
- **`model_dir` is explicit** — you set it directly instead of it being
  constructed from `wdir/suite/sim/models/tracer/summary`
- **Single experiment per run** — instead of iterating over a list of
  experiments, each run handles one configuration. The batch scripts show
  how to sweep over kmin/kmax if needed.
- **Loader kwargs** — dataset-specific settings (tracer, Nmax, summary names,
  kmin/kmax, etc.) go under `+loader.*` on the command line

## Running the old pipeline exactly

The provided scripts replicate the `abacuslike/fastpm_recnoise_tempOms8`
configuration with `embedding_net=fun`, `net=niall2`, `subselect_cosmo=[0,4]`.
Adjust `DATA_DIR` and `MODEL_DIR` in each script to match your paths.

The preprocessing script defaults to the Quijote galaxy setup with `Nmax=4000`.
If you need different settings, just change the `+loader.*` overrides.
