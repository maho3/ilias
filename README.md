# ili-at-scale

Scalable simulation-based inference (SBI) training, hyperparameter optimization,
and validation using [ltu-ili](https://github.com/maho3/ltu-ili), designed for
SLURM-based HPC systems.

## Setup

```bash
pip install -e ~/git/ltu-ili
pip install -e .
```

## Workflow

1. **Edit `ili_at_scale/loaders.py`** — implement `load_data()` for your dataset
2. **Edit `ili_at_scale/priors.py`** — implement `build_prior()` for your prior
3. **Edit `conf/config.yaml`** — set `model_dir` and loader kwargs

Then run the pipeline:

```bash
# 1. Preprocess: load data, split train/val/test, init Optuna
python -m ili_at_scale.preprocess model_dir=/path/to/output

# 2. Hyperparameter search (run many in parallel via SLURM)
python -m ili_at_scale.optuna model_dir=/path/to/output

# 3. Retrain top nets on full train+val split
python -m ili_at_scale.train model_dir=/path/to/output retrain=True net_index=0

# 4. Validate ensemble
python -m ili_at_scale.validate model_dir=/path/to/output
```

## SLURM

Example batch scripts for NCSA Delta are in `jobs/`. Adapt for your cluster.

## Configuration

All configuration is via Hydra. Override any key on the command line:

```bash
python -m ili_at_scale.optuna model_dir=/my/path embedding_net=fun net=niall2
```

Network architecture search spaces are defined in `conf/net/`. Copy
`conf/net/_template.yaml` to create your own.
