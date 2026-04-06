# ili-at-scale

Scalable simulation-based inference (SBI) training, hyperparameter optimization,
and validation using [ltu-ili](https://github.com/maho3/ltu-ili), designed for
SLURM-based HPC systems.

## Setup

```bash
cd ~/git/ltu-ili && git checkout ltu
pip install -e ~/git/ltu-ili
pip install -e .
```

**Important:** Use the `ltu` branch of `ltu-ili`.

## Workflow

1. **Edit `ilias/loaders.py`** — implement `load_data()` for your dataset
2. **Edit `ilias/priors.py`** — implement `build_prior()` for your prior
3. **Edit `conf/config.yaml`** — set `model_dir` and loader kwargs

Then run the four-stage pipeline:

### Stage 1: Preprocess

```bash
python -m ilias.preprocess model_dir=/path/to/output
```

Calls your `load_data()` function to load summary statistics (e.g. power
spectra, bispectra) and their corresponding physical parameters (e.g.
cosmological parameters). The loaded data is then split into train/validation/test
sets by simulation ID (so no simulation appears in more than one split,
preventing data leakage). Optionally applies PCA dimensionality reduction
to the summaries. Saves the splits as `.npy` arrays and initializes an
Optuna SQLite study database for the next stage.

### Stage 2: Hyperparameter optimization

```bash
python -m ilias.optuna model_dir=/path/to/output
```

Searches for the best neural density estimator (NDE) architecture and
training hyperparameters using Optuna. Each trial samples a configuration
from the hyperprior (defined in `conf/net/*.yaml`) — this includes the
normalizing flow architecture (e.g. NSF), the number of transforms, hidden
features, learning rate, batch size, weight decay, and embedding network
parameters. The trial trains an NDE on the training split, evaluates it by
computing the mean log-posterior probability on the test split, and records
the result in the shared SQLite study. Supports optional K-fold
cross-validation (splitting by simulation ID across folds) for more robust
evaluation. Multiple SLURM array tasks can run trials in parallel against
the same study via the `constant_liar` sampler.

### Stage 3: Retrain

```bash
python -m ilias.train model_dir=/path/to/output retrain=True net_index=0
```

After the Optuna search, selects the top-N trial configurations by test
log-probability and retrains each on the original train+validation split
(the full non-test data). This is necessary when cross-validation was used
in Stage 2, since those models were trained on CV folds rather than the
full training set. Each retrained model is saved as a `posterior.pkl` file
under `nets/net-{trial_id}/`. Use `net_index` with SLURM arrays to
parallelize across the top nets.

### Stage 4: Validate

```bash
python -m ilias.validate model_dir=/path/to/output
```

Loads the top-N retrained posteriors and combines them into a softmax-weighted
ensemble (weighted by test log-probability). Runs diagnostic metrics on the
test set: plots a single example posterior, computes posterior coverage
(fraction of true parameters falling within credible intervals), generates
TARP (Tests of Accuracy with Random Points) calibration curves, and saves
the ensemble posterior as `posterior.pkl`. Also produces Optuna diagnostic
plots (optimization history, hyperparameter importance, parameter slices).

## SLURM

Example batch scripts for NCSA Delta are in `jobs/`. Adapt for your cluster.

## Configuration

All configuration is via Hydra. Override any key on the command line:

```bash
python -m ilias.optuna model_dir=/my/path embedding_net=fun net=niall2
```

Network architecture search spaces are defined in `conf/net/`. Copy
`conf/net/_template.yaml` to create your own.
