"""
Prior construction for ltu-cmass data.

Supports the Quijote cosmology prior with optional HOD and noise prior
augmentation, matching the original ltu-cmass behavior.
"""

import os
from os.path import join
import numpy as np
import logging
from omegaconf import OmegaConf
import ili.utils


def build_prior(cfg, theta_train):
    """
    Construct the prior distribution.

    Supports:
        - 'uniform': data-driven uniform prior from theta min/max
        - 'quijote': fixed Quijote cosmology prior, optionally extended
          with HOD and noise priors loaded from model_dir

    Prior name is set via cfg.prior (default: 'uniform').
    """
    prior_name = cfg.get('prior', 'uniform')
    device = cfg.get('device', 'cpu')

    if prior_name.lower() == 'uniform':
        return ili.utils.Uniform(
            low=theta_train.min(axis=0),
            high=theta_train.max(axis=0),
            device=device)

    elif prior_name.lower() == 'quijote':
        prior_lims = np.array([
            (0.1, 0.5),    # Omega_m
            (0.03, 0.07),  # Omega_b
            (0.5, 0.9),    # h
            (0.8, 1.2),    # n_s
            (0.6, 1.0),    # sigma8
        ])

        subselect = cfg.get('subselect_cosmo', None)
        if subselect is not None:
            prior_lims = prior_lims[subselect]

        # Check for HOD prior
        model_dir = cfg.model_dir
        hodprior_path = join(model_dir, 'hodprior.csv')
        if os.path.exists(hodprior_path):
            hodprior = np.genfromtxt(hodprior_path, delimiter=',', dtype=object)
            hod_lims = hodprior[:, 2:4].astype(float)
            prior_lims = np.vstack([prior_lims, hod_lims])

        # Check for noise prior
        noiseprior_path = join(model_dir, 'noiseprior.yaml')
        if os.path.exists(noiseprior_path):
            noiseprior = OmegaConf.load(noiseprior_path)
            if noiseprior.dist == 'Uniform':
                low, high = noiseprior.params.a, noiseprior.params.b
            elif noiseprior.dist == 'Reciprocal':
                low, high = noiseprior.params.a, noiseprior.params.b
            elif noiseprior.dist == 'Exponential':
                low, high = 0, np.inf
            elif noiseprior.dist == 'Fixed':
                low, high = -np.inf, np.inf
            else:
                raise NotImplementedError(
                    f'Noise prior {noiseprior.dist} not implemented.')
            noise_lims = np.array([[low, high]] * 2)
            prior_lims = np.vstack([prior_lims, noise_lims])

        return ili.utils.Uniform(
            low=prior_lims[:, 0],
            high=prior_lims[:, 1],
            device=device)

    else:
        raise NotImplementedError(
            f"Prior '{prior_name}' not implemented. "
            "Edit priors.py to add your own.")
