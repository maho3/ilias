"""
Preprocesses data for training.

Loads data via the user-defined loader, performs train/val/test splits,
optionally applies PCA, and initializes an Optuna study.

Usage:
    python -m ili_at_scale.preprocess model_dir=/path/to/output
"""

import os
import numpy as np
import logging
from os.path import join, isfile
import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

from .tools import timing_decorator, clean_up
from .loaders import load_data


def split_train_val_test(x, theta, ids, val_frac, test_frac, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x, theta, ids = map(np.array, [x, theta, ids])

    unique_ids = np.unique(ids)
    np.random.shuffle(unique_ids)
    s1 = int(val_frac * len(unique_ids))
    s2 = int(test_frac * len(unique_ids))
    ui_val = unique_ids[:s1]
    ui_test = unique_ids[s1:s1+s2]
    ui_train = unique_ids[s1+s2:]

    train_mask = np.isin(ids, ui_train)
    val_mask = np.isin(ids, ui_val)
    test_mask = np.isin(ids, ui_test)

    return (
        (x[train_mask], x[val_mask], x[test_mask]),
        (theta[train_mask], theta[val_mask], theta[test_mask]),
        (ids[train_mask], ids[val_mask], ids[test_mask]),
    )


def setup_optuna(exp_path, name, n_startup_trials):
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,
        multivariate=True,
        constant_liar=True,
    )
    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        storage='sqlite:///' + join(exp_path, 'optuna_study.db'),
        study_name=name,
        load_if_exists=True
    )
    return study


def run_preprocessing(summaries, parameters, sim_ids, cfg):
    model_dir = cfg.model_dir

    logging.info(f'Data shape: {summaries.shape[0]} samples, '
                 f'{summaries.shape[1]} features, '
                 f'{parameters.shape[1]} parameters')

    # Split train/val/test
    ((x_train, x_val, x_test),
     (theta_train, theta_val, theta_test),
     (ids_train, ids_val, ids_test)) = split_train_val_test(
        summaries, parameters, sim_ids,
        cfg.val_frac, cfg.test_frac, cfg.seed)

    logging.info(f'Split: {len(x_train)} train, '
                 f'{len(x_val)} val, {len(x_test)} test')

    os.makedirs(model_dir, exist_ok=True)

    # Optional PCA compression
    if cfg.get('pca_features') is not None and cfg.pca_features > 0:
        logging.info(f"PCA compression to {cfg.pca_features} features")
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        pca = PCA(n_components=cfg.pca_features)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_val = pca.transform(x_val)
        x_test = pca.transform(x_test)
        joblib.dump((scaler, pca), join(model_dir, 'pca.pkl'))

    # Save data
    logging.info(f'Saving to {model_dir}')
    with open(join(model_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    np.save(join(model_dir, 'x_train.npy'), x_train)
    np.save(join(model_dir, 'x_val.npy'), x_val)
    np.save(join(model_dir, 'x_test.npy'), x_test)
    np.save(join(model_dir, 'theta_train.npy'), theta_train)
    np.save(join(model_dir, 'theta_val.npy'), theta_val)
    np.save(join(model_dir, 'theta_test.npy'), theta_test)
    np.save(join(model_dir, 'ids_train.npy'), ids_train)
    np.save(join(model_dir, 'ids_val.npy'), ids_val)
    np.save(join(model_dir, 'ids_test.npy'), ids_test)

    # Initialize Optuna study
    if not isfile(join(model_dir, 'optuna_study.db')):
        _ = setup_optuna(model_dir, 'study', cfg.n_startup_trials)


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    summaries, parameters, sim_ids = load_data(cfg)
    run_preprocessing(summaries, parameters, sim_ids, cfg)


if __name__ == "__main__":
    main()
