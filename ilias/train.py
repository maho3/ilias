"""
Trains posterior inference models using ltu-ili.

Supports different inference backends (lampe, sbi), various embedding
architectures, and retraining from Optuna HP studies.

Usage:
    python -m ilias.train model_dir=/path/to/output
    python -m ilias.train model_dir=/path/to/output retrain=True
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
import yaml
import time
import optuna

from .tools import timing_decorator, clean_up, select_top_trials, prepare_loader
from .hyperparameters import sample_hyperparameters_randomly
from .priors import build_prior
from .architectures import CNN, MultiHeadEmbedding, FunnelNetwork, MultiHeadFunnel

import ili
from ili.dataloaders import TorchLoader
from ili.inference import InferenceRunner
from ili.embedding import FCN

import matplotlib.pyplot as plt


def _train_runner(loader, prior, nets, train_args, out_dir,
                  backend, engine, device, verbose=False):
    """Helper function to run training."""
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    runner = InferenceRunner.load(
        backend=backend,
        engine=engine,
        prior=prior,
        nets=nets,
        device=device,
        train_args=train_args,
        out_dir=out_dir
    )

    posterior, histories = runner(loader=loader, verbose=verbose)
    return posterior, histories


def build_embedding(mcfg, x_train, start_idx=None):
    """Build the embedding network from model config."""
    if mcfg.embedding_net == 'fcn':
        if mcfg.fcn_depth == 0:
            return nn.Identity()
        return FCN(n_hidden=[mcfg.fcn_width]*mcfg.fcn_depth, act_fn='ReLU')
    elif mcfg.embedding_net == 'cnn':
        return CNN(
            out_channels=[mcfg.out_channels] * mcfg.cnn_depth,
            kernel_size=mcfg.kernel_size, act_fn='ReLU')
    elif mcfg.embedding_net == 'mhe':
        in_features = np.diff(start_idx).tolist()
        return MultiHeadEmbedding(
            start_idx=start_idx,
            in_features=in_features,
            out_features=[mcfg.out_features] * len(in_features),
            hidden_layers=[[mcfg.hidden_width]*mcfg.hidden_depth] * len(in_features),
            act_fn='ReLU', dropout=mcfg.dropout)
    elif mcfg.embedding_net == 'fun':
        return FunnelNetwork(
            in_features=x_train.shape[-1],
            out_features=mcfg.out_features,
            hidden_depth=mcfg.hidden_depth,
            act_fn='ReLU', dropout=mcfg.dropout,
            linear_dim=mcfg.get('linear_dim'),
            bypass=mcfg.get('bypass', False))
    elif mcfg.embedding_net == 'mhf':
        in_features = np.diff(start_idx).tolist()
        linear_dims = [mcfg.linear_dim] * len(in_features) if 'linear_dim' in mcfg else None
        return MultiHeadFunnel(
            start_idx=start_idx,
            in_features=in_features,
            out_features=[mcfg.out_features] * len(in_features),
            hidden_depth=[mcfg.hidden_depth] * len(in_features),
            act_fn='ReLU', dropout=mcfg.dropout,
            linear_dims=linear_dims)
    else:
        raise ValueError(f"Unknown embedding net: {mcfg.embedding_net}")


def build_train_args(cfg, mcfg):
    """Build training arguments dict, preferring mcfg over cfg defaults."""
    def _get(key):
        if key in mcfg:
            return mcfg[key]
        return cfg.get(key)

    return {
        'learning_rate': _get('learning_rate'),
        'stop_after_epochs': cfg.stop_after_epochs,
        'validation_fraction': cfg.val_frac,
        'weight_decay': _get('weight_decay'),
        'lr_decay_factor': _get('lr_decay_factor'),
        'lr_patience': _get('lr_patience'),
        'ema_decay': cfg.get('ema_decay', 0.9),
        'validation_smoothing_method': cfg.get(
            'validation_smoothing_method', 'none').lower(),
        'early_stopping': _get('early_stopping'),
        'noise_percent': _get('noise_percent'),
        'lr_scheduler': _get('lr_scheduler'),
        'max_epochs': _get('max_epochs'),
    }


def run_training(
    x_train, theta_train, x_val, theta_val, out_dir,
    cfg, mcfg, start_idx=None,
):
    """Train a neural network to learn the posterior distribution."""
    verbose = cfg.get('verbose', False)
    if verbose:
        logging.info(f'Using network architecture: {mcfg}')

    prior = build_prior(cfg, theta_train)
    embedding = build_embedding(mcfg, x_train, start_idx)

    # Instantiate NDE
    if cfg.backend == 'lampe':
        net_loader = ili.utils.load_nde_lampe
        extra_kwargs = {}
    elif cfg.backend == 'sbi':
        net_loader = ili.utils.load_nde_sbi
        extra_kwargs = {'engine': cfg.engine}
    else:
        raise NotImplementedError(f"Backend '{cfg.backend}' not supported.")

    kwargs = {k: v for k, v in mcfg.items() if k in [
        'model', 'hidden_features', 'num_transforms', 'num_components']}
    nets = [net_loader(**kwargs, **extra_kwargs, embedding_net=embedding)]

    train_args = build_train_args(cfg, mcfg)
    batch_size = mcfg.batch_size if 'batch_size' in mcfg else cfg.batch_size

    train_loader = prepare_loader(
        x_train, theta_train, device=cfg.device,
        batch_size=batch_size, shuffle=True)
    val_loader = prepare_loader(
        x_val, theta_val, device=cfg.device,
        batch_size=batch_size, shuffle=False)
    loader = TorchLoader(train_loader, val_loader)

    posterior, histories = _train_runner(
        loader=loader, prior=prior, nets=nets,
        train_args=train_args, out_dir=out_dir,
        backend=cfg.backend, engine=cfg.engine,
        device=cfg.device, verbose=verbose)

    return posterior, histories


def run_training_with_precompression(
    x_train, theta_train, x_val, theta_val, out_dir,
    cfg, mcfg, start_idx=None,
):
    """Train with a pre-compression layer (FCN only)."""
    verbose = cfg.get('verbose', False)
    if cfg.embedding_net != 'fcn':
        raise ValueError('Precompression only supported for FCN embedding_net.')

    prior = build_prior(cfg, theta_train)
    train_args = build_train_args(cfg, mcfg)
    batch_size = mcfg.batch_size if 'batch_size' in mcfg else cfg.batch_size

    train_loader = prepare_loader(
        x_train, theta_train, device=cfg.device,
        batch_size=batch_size, shuffle=True)
    val_loader = prepare_loader(
        x_val, theta_val, device=cfg.device,
        batch_size=batch_size, shuffle=False)
    loader = TorchLoader(train_loader, val_loader)

    net_loader = ili.utils.load_nde_lampe

    # Train pre-compression network
    logging.info('Training pre-compression network...')
    nets = [net_loader(model='mdn', hidden_features=mcfg.fcn_width,
                       hidden_depth=mcfg.fcn_depth, num_components=4)]
    posterior, _ = _train_runner(
        loader=loader, prior=prior, nets=nets,
        train_args=train_args, out_dir=out_dir,
        backend=cfg.backend, engine=cfg.engine,
        device=cfg.device, verbose=verbose)

    embedding = posterior.posteriors[0].nde.flow.hyper
    for param in embedding.parameters():
        param.requires_grad = False

    # Train final network with frozen pre-compression
    logging.info('Training final network with pre-compression...')
    kwargs = {k: v for k, v in mcfg.items() if k in [
        'model', 'hidden_features', 'num_transforms', 'num_components']}
    nets = [net_loader(**kwargs, embedding_net=embedding)]
    posterior, histories = _train_runner(
        loader=loader, prior=prior, nets=nets,
        train_args=train_args, out_dir=out_dir,
        backend=cfg.backend, engine=cfg.engine,
        device=cfg.device, verbose=verbose)

    return posterior, histories


def plot_training_history(histories, out_dir):
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, h in enumerate(histories):
        ax.plot(h['validation_log_probs'], label=f'Net {i}', lw=1)
    ax.set(xlabel='Epoch', ylabel='Validation log prob')
    ax.legend()
    f.savefig(join(out_dir, 'loss.jpg'), dpi=100, bbox_inches='tight')
    plt.close(f)


def evaluate_posterior(posterior, x, theta):
    log_prob = posterior.log_prob(theta=theta, x=x)
    return log_prob.mean()


def load_preprocessed_data(model_dir):
    """Load preprocessed data from model_dir."""
    try:
        x_train = np.load(join(model_dir, 'x_train.npy'))
        theta_train = np.load(join(model_dir, 'theta_train.npy'))
        ids_train = np.load(join(model_dir, 'ids_train.npy'))
        x_val = np.load(join(model_dir, 'x_val.npy'))
        theta_val = np.load(join(model_dir, 'theta_val.npy'))
        ids_val = np.load(join(model_dir, 'ids_val.npy'))
        x_test = np.load(join(model_dir, 'x_test.npy'))
        theta_test = np.load(join(model_dir, 'theta_test.npy'))
        ids_test = np.load(join(model_dir, 'ids_test.npy'))
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Could not find preprocessed data in {model_dir}. '
            'Run ilias.preprocess first.'
        )

    return (x_train, theta_train, ids_train,
            x_val, theta_val, ids_val,
            x_test, theta_test, ids_test)


def run_experiment(cfg):
    """Run a single training experiment."""
    model_dir = cfg.model_dir

    (x_train, theta_train, ids_train,
     x_val, theta_val, ids_val,
     x_test, theta_test, ids_test) = load_preprocessed_data(model_dir)

    logging.info(f'Split: {len(x_train)} train, {len(x_val)} val, '
                 f'{len(x_test)} test')

    out_dir = join(model_dir, 'nets', f'net-{cfg.net_index}')
    logging.info(f'Saving models to {out_dir}')

    start = time.time()
    posterior, histories = run_training(
        x_train, theta_train, x_val, theta_val,
        out_dir=out_dir, cfg=cfg, mcfg=cfg.net)
    elapsed = time.time() - start

    with open(join(out_dir, 'timing.txt'), 'w') as f:
        f.write(f'{elapsed:.3f}')
    with open(join(out_dir, 'model_config.yaml'), 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg.net, resolve=True), f)

    plot_training_history(histories, out_dir)

    log_prob_test = evaluate_posterior(posterior, x_test, theta_test)
    with open(join(out_dir, 'log_prob_test.txt'), 'w') as f:
        f.write(f'{log_prob_test}\n')


def select_nets_retrain(model_dir, Nnets):
    """Select top nets from HP study for retraining."""
    optunafile = join(model_dir, 'optuna_study.db')
    storage = f"sqlite:///{optunafile}"
    study = optuna.load_study(storage=storage, study_name='study')

    top_trials = select_top_trials(study, Nnets)
    logging.info(f'Selected {len(top_trials)} nets.')

    trial_numbers = [t.number for t in top_trials]
    mcfgs = [t.user_attrs['mcfg'] for t in top_trials]
    return trial_numbers, mcfgs


def run_retraining(cfg):
    """Retrain top nets from Optuna study on the full train+val split."""
    model_dir = cfg.model_dir

    if not cfg.cross_val:
        raise ValueError(
            'There is no reason to retrain without cross-validation.')

    Nnets = cfg.Nnets
    train_fn = run_training_with_precompression if cfg.precompress else run_training

    trial_numbers, net_configs = select_nets_retrain(model_dir, Nnets)

    # If net_index is set, select only that net
    if cfg.get('net_index') is not None:
        net_index = cfg.net_index
        if net_index < len(trial_numbers):
            logging.info(f"Selecting net index {net_index} from top "
                         f"{len(trial_numbers)} models.")
            trial_numbers = [trial_numbers[net_index]]
            net_configs = [net_configs[net_index]]
        else:
            logging.warning(f"net_index {net_index} out of bounds. Exiting.")
            return

    (x_train, theta_train, ids_train,
     x_val, theta_val, ids_val,
     x_test, theta_test, ids_test) = load_preprocessed_data(model_dir)

    logging.info(f'Split: {len(x_train)} train, {len(x_val)} val, '
                 f'{len(x_test)} test')

    for trial_number, config in zip(trial_numbers, net_configs):
        out_dir = join(model_dir, "nets", f"net-{trial_number}")
        os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(join(out_dir, 'posterior.pkl')):
            logging.info(f"Net-{trial_number} already trained. Skipping.")
            continue

        mcfg = OmegaConf.create(config)

        start = time.time()
        posterior, histories = train_fn(
            x_train, theta_train, x_val, theta_val,
            out_dir=out_dir, cfg=cfg, mcfg=mcfg)
        elapsed = time.time() - start

        with open(join(out_dir, 'timing.txt'), 'w') as f:
            f.write(f'{elapsed:.3f}')
        with open(join(out_dir, 'model_config.yaml'), 'w') as f:
            yaml.dump(OmegaConf.to_container(mcfg, resolve=True), f)

        plot_training_history(histories, out_dir)

        log_prob_test = evaluate_posterior(posterior, x_test, theta_test)
        with open(join(out_dir, 'log_prob_test.txt'), 'w') as f:
            f.write(f'{log_prob_test}\n')


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    if not cfg.get('retrain', False):
        cfg.net = sample_hyperparameters_randomly(
            hyperprior=cfg.net,
            embedding_net=cfg.embedding_net,
            seed=cfg.net_index
        )
        run_experiment(cfg)
    else:
        run_retraining(cfg)


if __name__ == "__main__":
    main()
