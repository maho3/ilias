"""
Prior construction for ili-at-scale.

This file is USER-EDITABLE. Modify `build_prior` to define the prior
distribution for your inference problem.

The default implementation provides a uniform prior based on the
range of the training parameters. You can replace this with any
prior supported by ltu-ili (e.g., ili.utils.Uniform, IndependentNormal).
"""

import numpy as np
import ili.utils


def build_prior(cfg, theta_train):
    """
    Construct the prior distribution for inference.

    Args:
        cfg: Full Hydra config.
        theta_train: np.ndarray of shape (N, n_params) — training parameters,
            useful for setting prior bounds from data.

    Returns:
        prior: an ili-compatible prior object (e.g., ili.utils.Uniform)
    """
    prior_name = cfg.get('prior', 'uniform')
    device = cfg.get('device', 'cpu')

    if prior_name.lower() == 'uniform':
        prior = ili.utils.Uniform(
            low=theta_train.min(axis=0),
            high=theta_train.max(axis=0),
            device=device)
    else:
        raise NotImplementedError(
            f"Prior '{prior_name}' not implemented. "
            "Edit priors.py to add your own prior."
        )

    return prior


# =============================================================================
# Example: Quijote cosmology prior (from ltu-cmass)
#
# def build_prior(cfg, theta_train):
#     device = cfg.get('device', 'cpu')
#     prior_name = cfg.get('prior', 'uniform')
#
#     if prior_name.lower() == 'quijote':
#         prior_lims = np.array([
#             (0.1, 0.5),    # Omega_m
#             (0.03, 0.07),  # Omega_b
#             (0.5, 0.9),    # h
#             (0.8, 1.2),    # n_s
#             (0.6, 1.0),    # sigma8
#         ])
#         subselect = cfg.get('subselect_cosmo', None)
#         if subselect is not None:
#             prior_lims = prior_lims[subselect]
#
#         # Optionally append HOD prior bounds from hodprior.csv
#         # hodprior = np.genfromtxt('hodprior.csv', delimiter=',', dtype=object)
#         # hod_lims = hodprior[:, 2:4].astype(float)
#         # prior_lims = np.vstack([prior_lims, hod_lims])
#
#         prior = ili.utils.Uniform(
#             low=prior_lims[:, 0],
#             high=prior_lims[:, 1],
#             device=device)
#     elif prior_name.lower() == 'uniform':
#         prior = ili.utils.Uniform(
#             low=theta_train.min(axis=0),
#             high=theta_train.max(axis=0),
#             device=device)
#     else:
#         raise NotImplementedError(f"Prior '{prior_name}' not implemented.")
#
#     return prior
# =============================================================================
