"""
Data loading for ili-at-scale.

This file is USER-EDITABLE. Modify `load_data` to load your own dataset.
The function signature must remain the same.

Below is both a minimal skeleton and the full ltu-cmass example loader.
Uncomment/modify as needed for your use case.
"""

import numpy as np
import logging
from omegaconf import DictConfig


def load_data(cfg: DictConfig):
    """
    Load summaries and parameters for preprocessing.

    This function is called by preprocess.py. Users should modify this
    function to load their own data.

    Args:
        cfg: Full Hydra config. Access loader-specific kwargs via cfg.loader.*

    Returns:
        summaries: np.ndarray of shape (N, n_features) — summary statistics
        parameters: np.ndarray of shape (N, n_params) — target parameters
        sim_ids: np.ndarray of shape (N,) — unique IDs per simulation,
            used to split train/val/test without data leakage
    """
    raise NotImplementedError(
        "You must implement load_data() in loaders.py for your dataset. "
        "See the example below or the ltu-cmass loader for reference."
    )

    # Example skeleton:
    #
    # data_path = cfg.loader.data_path
    # summaries = np.load(f"{data_path}/summaries.npy")
    # parameters = np.load(f"{data_path}/parameters.npy")
    # sim_ids = np.load(f"{data_path}/sim_ids.npy")
    # return summaries, parameters, sim_ids


# =============================================================================
# Example: ltu-cmass loader (for reference)
#
# This is the original loader from ltu-cmass, adapted for ili-at-scale.
# It loads power spectra and bispectra from h5 diagnostic files produced
# by ltu-cmass simulations. You can use this as a starting point.
#
# Required cfg.loader keys:
#   data_path: path to the suite directory containing simulation folders
#   Nmax: max number of simulations to load (-1 for all)
#   tracer: 'halo', 'galaxy', 'ngc_lightcone', etc.
#   a: scale factor (e.g. 1.0)
#   summary: list of summary statistic names (e.g. ['zPk0', 'zPk2'])
#   kmin: float
#   kmax: float
#   include_hod: bool
#   include_noise: bool
#   subselect_cosmo: list of int indices or null
#   correct_shot: bool
#   loglinear_start_idx: int or null
#
# Dependencies:
#   - h5py
#   - For HOD prior construction, see ltu-cmass/cmass/bias/tools/hod.py
#
# def load_data(cfg):
#     import os
#     from os.path import join
#     import h5py
#     import multiprocessing
#     from tqdm import tqdm
#     from omegaconf import OmegaConf
#     from .tools import log2_avg
#
#     loader_cfg = cfg.loader
#     suite_path = loader_cfg.data_path
#     tracer = loader_cfg.tracer
#     Nmax = loader_cfg.Nmax
#     a = loader_cfg.get('a', None)
#     kmin = loader_cfg.kmin
#     kmax = loader_cfg.kmax
#     include_hod = loader_cfg.get('include_hod', False)
#     include_noise = loader_cfg.get('include_noise', False)
#     subselect_cosmo = loader_cfg.get('subselect_cosmo', None)
#     correct_shot = loader_cfg.get('correct_shot', True)
#     loglinear_start_idx = loader_cfg.get('loglinear_start_idx', None)
#     summary_names = list(loader_cfg.summary)
#
#     # --- Helper functions (from ltu-cmass/cmass/infer/loaders.py) ---
#
#     def get_cosmo(source_path):
#         try:
#             cfg_local = OmegaConf.load(join(source_path, 'config.yaml'))
#             return np.array(cfg_local.nbody.cosmo)
#         except Exception as e:
#             logging.warning(f"Error loading cosmo params: {e}")
#             return None
#
#     def closest_a(lst, a):
#         lst = np.array([float(el) for el in lst])
#         if len(lst) == 0:
#             return 0
#         return lst[np.abs(lst - a).argmin()]
#
#     def signed_log(x, base=10):
#         return np.sign(x) * np.log1p(np.abs(x)) / np.log(base)
#
#     def load_Pk(diag_file, a):
#         if not os.path.exists(diag_file):
#             return {}
#         summ = {}
#         try:
#             with h5py.File(diag_file, 'r') as f:
#                 a = closest_a(f.keys(), a)
#                 a = f'{a:.6f}'
#                 for stat in ['Pk', 'zPk']:
#                     if stat in f[a]:
#                         for i in range(3):
#                             summ[stat+str(2*i)] = {
#                                 'k': f[a][stat+'_k3D'][:],
#                                 'value': f[a][stat][:, i],
#                                 'nbar': f[a].attrs['nbar']}
#         except (OSError, KeyError):
#             return {}
#         return summ
#
#     def preprocess_Pk(data, kmin, kmax, norm=None,
#                       correct_shot=False, loglinear_start_idx=None):
#         k = np.atleast_2d(data[0]['k'])
#         mask = np.all((kmin <= k) & (k <= kmax), axis=0)
#         X = np.array([x['value'][mask] for x in data])
#         if norm is None:
#             if correct_shot:
#                 nbar = np.array([x['nbar'] for x in data]).reshape(-1, 1)
#                 X -= 1./nbar
#             X = signed_log(X)
#         else:
#             k_norm = np.atleast_2d(norm[0]['k'])
#             mask_norm = np.all((kmin <= k_norm) & (k_norm <= kmax), axis=0)
#             Xnorm = np.array([x['value'][mask_norm] for x in norm])
#             X /= Xnorm
#         if loglinear_start_idx is not None:
#             X = np.apply_along_axis(log2_avg, 1, X, s=loglinear_start_idx)
#         return np.nan_to_num(X, nan=0.0).reshape(len(X), -1)
#
#     # --- Main loading logic ---
#
#     simpaths = sorted(os.listdir(suite_path), key=lambda x: int(x))
#     if Nmax >= 0:
#         simpaths = simpaths[:Nmax]
#
#     all_summ, all_params, all_ids = {}, {}, {}
#     for lhid in tqdm(simpaths, desc='Loading simulations'):
#         source = join(suite_path, lhid)
#         diagpath = join(source, 'diag')
#         if tracer == 'galaxy':
#             diagpath = join(diagpath, 'galaxies')
#         filelist = ['halos.h5'] if tracer == 'halo' else os.listdir(diagpath)
#
#         for fname in filelist:
#             diagfile = join(diagpath, fname)
#             summ = load_Pk(diagfile, a)
#             if not summ:
#                 continue
#             params = get_cosmo(source)
#             if params is None:
#                 continue
#             if subselect_cosmo is not None:
#                 params = params[subselect_cosmo]
#             if include_hod:
#                 with h5py.File(diagfile, 'r') as f:
#                     hod = f.attrs['HOD_params'][:]
#                 params = np.concatenate([params, hod])
#
#             for key in summ:
#                 all_summ.setdefault(key, []).append(summ[key])
#                 all_params.setdefault(key, []).append(params)
#                 all_ids.setdefault(key, []).append(lhid)
#
#     # Process and concatenate summaries
#     xs = []
#     ref_key = summary_names[0]
#     for name in summary_names:
#         base = name
#         if name in ['nbar', 'nz']:
#             continue
#         if 'Pk' in name:
#             norm_key = base[:-1] + '0'
#             x = preprocess_Pk(
#                 all_summ[base], kmin, kmax,
#                 norm=None if '0' in base else all_summ.get(norm_key),
#                 correct_shot=correct_shot,
#                 loglinear_start_idx=loglinear_start_idx)
#             xs.append(x)
#             ref_key = base
#
#     summaries = np.concatenate(xs, axis=-1)
#     parameters = np.array(all_params[ref_key])
#     sim_ids = np.array(all_ids[ref_key])
#
#     return summaries, parameters, sim_ids
# =============================================================================
