"""
Validates trained posterior inference models.

Loads an ensemble of trained models, runs posterior coverage diagnostics,
and optionally cleans up poorly performing models.

Usage:
    python -m ilias.validate model_dir=/path/to/output
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pickle
import scipy
import shutil
import optuna
import optuna.visualization.matplotlib as vis
from matplotlib import pyplot as plt

from .tools import timing_decorator, clean_up, select_top_trials, load_posterior

from ili.utils.ndes_pt import LampeEnsemble
from ili.validation import PlotSinglePosterior, PosteriorCoverage


def run_validation(posterior, x, theta, out_dir, names=None):
    logging.info('Running validation...')

    with open(join(out_dir, 'posterior.pkl'), "wb") as handle:
        pickle.dump(posterior, handle)

    # Single posterior plot
    logging.info('Plotting single posterior...')
    xobs, thetaobs = x[0], theta[0]
    metric = PlotSinglePosterior(
        num_samples=1000, sample_method='direct',
        labels=names, out_dir=out_dir
    )
    metric(posterior, x_obs=xobs, theta_fid=thetaobs.to('cpu'))

    # Posterior coverage
    logging.info('Running posterior coverage...')
    metric = PosteriorCoverage(
        num_samples=2000, sample_method='direct',
        labels=names,
        plot_list=["coverage", "histogram", "predictions", "tarp", "logprob"],
        out_dir=out_dir,
        save_samples=True
    )
    metric(posterior, x, theta.to('cpu'))


def plot_optuna_diagnostics(study, out_dir):
    ax = vis.plot_optimization_history(study)
    fig = ax.get_figure()
    fig.savefig(join(out_dir, 'optuna_history.png'), bbox_inches='tight')
    plt.close(fig)

    axs = vis.plot_slice(study)
    fig = axs[0].get_figure()
    fig.savefig(join(out_dir, 'optuna_hyperparam_dependence.png'), bbox_inches='tight')
    plt.close(fig)

    ax = vis.plot_param_importances(study)
    fig = ax.get_figure()
    fig.savefig(join(out_dir, 'optuna_param_importance.png'), bbox_inches='tight')
    plt.close(fig)

    ax = vis.plot_timeline(study)
    fig = ax.get_figure()
    fig.savefig(join(out_dir, 'optuna_timeline.png'), bbox_inches='tight')
    plt.close(fig)


def load_ensemble(model_dir, Nnets, weighted=True, plot=True, clean=False):
    """Load an ensemble of posteriors from an Optuna study."""
    optunafile = join(model_dir, 'optuna_study.db')
    storage = f"sqlite:///{optunafile}"
    study = optuna.load_study(storage=storage, study_name='study')

    top_trials = select_top_trials(study, Nnets)
    logging.info(f'Selected {len(top_trials)} nets.')

    if plot:
        plot_optuna_diagnostics(study, model_dir)

    ensemble_list = []
    valid_trials = []
    for t in top_trials:
        model_path = join(model_dir, 'nets',
                          f'net-{t.number}', 'posterior.pkl')
        if not os.path.exists(model_path):
            logging.warning(f"Model not found, skipping: {model_path}")
            continue
        pi = load_posterior(model_path, 'cpu')
        ensemble_list.append(pi.posteriors[0])
        valid_trials.append(t)
    top_trials = valid_trials

    if not top_trials:
        raise RuntimeError("No valid models found to form an ensemble.")

    if clean:
        all_net_dirs = os.listdir(join(model_dir, "nets"))
        top_net_numbers = {str(t.number) for t in top_trials}
        for n in all_net_dirs:
            if n.startswith('net-'):
                net_number = n.split('net-')[-1]
                if net_number not in top_net_numbers:
                    shutil.rmtree(join(model_dir, 'nets', n))

    if weighted:
        ensemble_logprobs = [t.value for t in top_trials]
        weights = scipy.special.softmax(ensemble_logprobs)
        weights = torch.Tensor(weights)
    else:
        weights = torch.ones(len(top_trials)) / len(top_trials)

    ensemble = LampeEnsemble(
        posteriors=ensemble_list,
        weights=weights
    )
    return ensemble


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    model_dir = cfg.model_dir

    # Load test data
    if cfg.get('testing') and cfg.testing.get('path'):
        logging.info(f'Loading external test data from {cfg.testing.path}')
        x_test = np.load(join(cfg.testing.path, 'x_test.npy'))
        theta_test = np.load(join(cfg.testing.path, 'theta_test.npy'))
        out_path = join(model_dir, 'testing', cfg.testing.name)
        os.makedirs(out_path, exist_ok=True)
    else:
        logging.info(f'Loading test data from {model_dir}')
        x_test = np.load(join(model_dir, 'x_test.npy'))
        theta_test = np.load(join(model_dir, 'theta_test.npy'))
        out_path = model_dir

    logging.info(f'Testing on {len(x_test)} examples')

    # Load trained posterior ensemble
    posterior_ensemble = load_ensemble(
        model_dir, cfg.Nnets,
        clean=cfg.get('clean_models', False))

    # Run validation
    x_test = torch.Tensor(x_test).to(cfg.device)
    theta_test = torch.Tensor(theta_test).to(cfg.device)

    param_names = cfg.get('param_names', None)
    if param_names is not None:
        param_names = list(param_names)

    run_validation(posterior_ensemble, x_test, theta_test,
                   out_path, names=param_names)


if __name__ == "__main__":
    main()
