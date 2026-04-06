"""
Hyperparameter optimization via Optuna.

Supports both standard training and cross-validation strategies.

Usage:
    python -m ili_at_scale.optuna model_dir=/path/to/output
"""

import os
import time
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from os.path import join
import logging
import numpy as np

from sklearn.model_selection import GroupShuffleSplit

from .hyperparameters import sample_hyperparameters_optuna
from .preprocess import setup_optuna
from .train import (load_preprocessed_data,
                    run_training, run_training_with_precompression,
                    evaluate_posterior)
from .tools import timing_decorator, clean_up


def objective(trial, cfg,
              x_train, theta_train, x_val, theta_val, x_test, theta_test):
    """Standard objective: train once, evaluate on test set."""
    mcfg = sample_hyperparameters_optuna(
        trial, cfg.net, cfg.embedding_net)

    start = time.time()
    train_fn = run_training_with_precompression if cfg.precompress else run_training
    posterior, histories = train_fn(
        x_train=x_train, theta_train=theta_train,
        x_val=x_val, theta_val=theta_val,
        out_dir=None, cfg=cfg, mcfg=mcfg)
    elapsed = time.time() - start

    trial.set_user_attr("timing", elapsed)
    trial.set_user_attr("mcfg", OmegaConf.to_container(mcfg, resolve=True))

    return evaluate_posterior(posterior, x_test, theta_test)


def objective_cval(trial, cfg,
                   x_train, theta_train, x_val, theta_val, x_test, theta_test,
                   n_splits, ids_train, ids_val, ids_test):
    """Cross-validation objective: train on K folds, average test scores."""
    x_all = np.vstack((x_train, x_val, x_test))
    theta_all = np.vstack((theta_train, theta_val, theta_test))
    ids_all = np.concatenate((ids_train, ids_val, ids_test))

    mcfg = sample_hyperparameters_optuna(
        trial, cfg.net, cfg.embedding_net)

    trial.set_user_attr("mcfg", OmegaConf.to_container(mcfg, resolve=True))

    gss = GroupShuffleSplit(
        n_splits=n_splits, test_size=cfg.test_frac,
        random_state=9)
    scores = np.zeros(n_splits)
    timings = np.zeros(n_splits)

    for K, (train_valid_idx, test_idx) in enumerate(
            gss.split(x_all, theta_all, ids_all)):
        logging.info(f'Cross-validation fold {K+1}/{n_splits}...')

        start = time.time()
        x_tv = x_all[train_valid_idx]
        theta_tv = theta_all[train_valid_idx]
        ids_tv = ids_all[train_valid_idx]

        x_test_fold = x_all[test_idx]
        theta_test_fold = theta_all[test_idx]

        # Inner train/val split
        cv_val_frac = cfg.val_frac / (1 - cfg.test_frac)
        gss_inner = GroupShuffleSplit(
            n_splits=1, test_size=cv_val_frac, random_state=1)
        train_idx, val_idx = next(gss_inner.split(x_tv, theta_tv, ids_tv))

        train_fn = run_training_with_precompression if cfg.precompress else run_training
        posterior, _ = train_fn(
            x_train=x_tv[train_idx], theta_train=theta_tv[train_idx],
            x_val=x_tv[val_idx], theta_val=theta_tv[val_idx],
            out_dir=None, cfg=cfg, mcfg=mcfg)

        scores[K] = evaluate_posterior(posterior, x_test_fold, theta_test_fold)
        timings[K] = time.time() - start

    trial.set_user_attr("timing_splits", timings.tolist())
    trial.set_user_attr("log_prob_splits", scores.tolist())

    return scores.mean()


def run_optuna(cfg):
    """Run Optuna hyperparameter optimization."""
    model_dir = cfg.model_dir

    (x_train, theta_train, ids_train,
     x_val, theta_val, ids_val,
     x_test, theta_test, ids_test) = load_preprocessed_data(model_dir)

    logging.info(f'Split: {len(x_train)} train, {len(x_val)} val, '
                 f'{len(x_test)} test')

    if cfg.cross_val:
        logging.info('Using cross-validation objective.')
        cv_args = (cfg.n_splits, ids_train, ids_val, ids_test)
        obj_fn = objective_cval
    else:
        logging.info('Using standard objective.')
        cv_args = ()
        obj_fn = objective

    study = setup_optuna(model_dir, 'study', cfg.n_startup_trials)
    study.optimize(
        lambda trial: obj_fn(
            trial, cfg,
            x_train, theta_train, x_val, theta_val,
            x_test, theta_test, *cv_args),
        n_trials=cfg.n_trials,
        n_jobs=1,
        timeout=60*60*24,
        show_progress_bar=False,
        gc_after_trial=True
    )


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))
    run_optuna(cfg)


if __name__ == "__main__":
    main()
