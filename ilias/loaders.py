"""
Data loader for ltu-cmass simulation data.

This loader replicates the data loading pipeline from cmass/infer/loaders.py
and cmass/infer/preprocess.py. It reads h5 diagnostic files from a simulation
suite and returns preprocessed summary statistics ready for training.

Required cfg.loader keys:
    data_path: str — path to suite directory (contains numbered sim folders)
    tracer: str — 'halo', 'galaxy', 'ngc_lightcone', 'sgc_lightcone', etc.
    Nmax: int — max simulations to load (-1 for all)
    a: float — scale factor (e.g. 1.0 for z=0, ~0.667 for z=0.5)
    summary: list[str] — summary statistic names, e.g. ['nbar', 'zPk0', 'zPk2']
    kmin: float — minimum k
    kmax: float — maximum k
    include_hod: bool — whether to include HOD parameters in theta
    include_noise: bool — whether to include noise parameters in theta
    subselect_cosmo: list[int] or null — indices of cosmo params to keep
    correct_shot: bool — whether to subtract shot noise from Pk monopole
    loglinear_start_idx: int or null — starting index for log-linear binning
"""

import os
from os.path import join
import logging

import h5py
import numpy as np
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .tools import log2_avg


# ---- Low-level h5 loading ----

def _closest_a(lst, a):
    lst = np.array([float(el) for el in lst])
    if len(lst) == 0:
        return 0
    return lst[np.abs(lst - a).argmin()]


def _get_cosmo(source_path):
    try:
        cfg = OmegaConf.load(join(source_path, 'config.yaml'))
        return np.array(cfg.nbody.cosmo)
    except Exception as e:
        logging.warning(f"Error loading cosmo params from {source_path}: {e}")
        return None


def _load_Pk(diag_file, a):
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
            a_val = _closest_a(f.keys(), a)
            a_str = f'{a_val:.6f}'
            for stat in ['Pk', 'zPk']:
                if stat in f[a_str]:
                    for i in range(3):
                        summ[stat + str(2 * i)] = {
                            'k': f[a_str][stat + '_k3D'][:],
                            'value': f[a_str][stat][:, i],
                            'nbar': f[a_str].attrs['nbar'],
                            'log10nbar': f[a_str].attrs['log10nbar'],
                            'a_loaded': a_str,
                        }
    except (OSError, KeyError):
        return {}
    return summ


def _load_lc_Pk(diag_file):
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
            for i in range(3):
                summ['Pk' + str(2 * i)] = {
                    'k': f['Pk_k3D'][:],
                    'value': f['Pk'][:, i],
                    'nbar': f.attrs['nbar'],
                    'log10nbar': f.attrs['log10nbar'],
                    'nz': f['nz'][:],
                    'nz_bins': f['nz_bins'][:],
                }
    except (OSError, KeyError):
        return {}
    return summ


def _load_Bk(diag_file, a):
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
            a_val = _closest_a(f.keys(), a)
            a_str = f'{a_val:.6f}'
            for stat in ['Bk', 'Qk', 'zBk', 'zQk']:
                if stat in f[a_str]:
                    for i in range(2):
                        if i >= f[a_str][stat].shape[0]:
                            continue
                        summ[stat + str(2 * i)] = {
                            'k': f[a_str]['Bk_k123'][:],
                            'value': f[a_str][stat][i, :],
                            'nbar': f[a_str].attrs['nbar'],
                            'log10nbar': f[a_str].attrs['log10nbar'],
                            'a_loaded': a_str,
                        }
    except (OSError, KeyError):
        return {}
    return summ


def _load_lc_Bk(diag_file):
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
            for stat in ['Bk', 'Qk']:
                if stat in f:
                    summ[stat + '0'] = {
                        'k': f['Bk_k123'][:],
                        'value': f[stat][0, :],
                        'nbar': f.attrs['nbar'],
                        'log10nbar': f.attrs['log10nbar'],
                        'nz': f['nz'][:],
                        'nz_bins': f['nz_bins'][:],
                    }
    except (OSError, KeyError):
        return {}
    return summ


# ---- Preprocessing helpers ----

def _signed_log(x, base=10):
    return np.sign(x) * np.log1p(np.abs(x)) / np.log(base)


def _is_in_kminmax(k, kmin, kmax):
    k = np.atleast_2d(k)
    return np.all((kmin <= k) & (k <= kmax), axis=0)


def _filter_Pk(X, kmin, kmax):
    return np.array(
        [x['value'][_is_in_kminmax(x['k'], kmin, kmax)] for x in X])


def _get_nbar(data):
    return np.array([x['nbar'] for x in data]).reshape(-1, 1)


def _get_log10nbar(data):
    return np.repeat(
        np.array([x['log10nbar'] for x in data]).reshape(-1, 1),
        10, axis=-1)


def _get_log10nz(data):
    num_bins = 10
    bin_edges = np.linspace(0.4, 0.7, num_bins + 1)
    binned_nz = np.empty((len(data), num_bins))
    for i, entry in enumerate(data):
        nz_values = entry['nz']
        nz_bin_edges = entry['nz_bins']
        nz_bin_centers = 0.5 * (nz_bin_edges[:-1] + nz_bin_edges[1:])
        coarse_bin_indices = np.digitize(nz_bin_centers, bin_edges) - 1
        for j in range(num_bins):
            binned_nz[i, j] = np.sum(nz_values[coarse_bin_indices == j])
    num_repeat = 5
    binned_nz = np.repeat(binned_nz, num_repeat, axis=-1)
    return np.where(binned_nz == 0, -1, np.log10(binned_nz))


def preprocess_Pk(data, kmin, kmax, norm=None, correct_shot=False,
                  loglinear_start_idx=None):
    X = _filter_Pk(data, kmin, kmax)
    if norm is None:
        if correct_shot:
            X -= 1. / _get_nbar(data)
        X = _signed_log(X)
    else:
        Xnorm = _filter_Pk(norm, kmin, kmax)
        X /= Xnorm
    if loglinear_start_idx is not None:
        X = np.apply_along_axis(log2_avg, 1, X, s=loglinear_start_idx)
    return np.nan_to_num(X, nan=0.0).reshape(len(X), -1)


def _is_valid_triangle(k):
    k1, k2, k3 = k
    return (k1 < k2 + k3) & (k2 < k1 + k3) & (k3 < k1 + k2)


def _is_equilateral(k):
    k1, k2, k3 = k
    return np.isclose(k1, k2) & np.isclose(k2, k3)


def _is_squeezed(k):
    k1, k2, k3 = k
    return np.isclose(k1, k2) & (k3 < k2)


def _is_isoceles(k):
    k1, k2, k3 = k
    return np.isclose(k1, k2) | np.isclose(k2, k3) | np.isclose(k1, k3)


def _is_subsampled(k):
    return np.arange(len(k[0])) % 5 == 0


def _filter_Bk(X, kmin, kmax, equilateral=False, squeezed=False,
               subsampled=False, isoceles=False):
    k123 = np.array([x['k'] for x in X])
    X_vals = np.array([x['value'] for x in X])
    k123 = k123[0]
    mask = _is_in_kminmax(k123, kmin, kmax)
    if equilateral:
        mask &= _is_equilateral(k123)
    elif squeezed:
        mask &= _is_squeezed(k123)
    elif subsampled:
        mask &= _is_subsampled(k123)
    elif isoceles:
        mask &= _is_isoceles(k123)
    else:
        mask &= _is_valid_triangle(k123)
    return X_vals[:, mask]


def preprocess_Bk(data, kmin, kmax, norm=None, mode=None,
                  correct_shot=False):
    X = _filter_Bk(
        data, kmin, kmax,
        equilateral=(mode == 'Eq'),
        squeezed=(mode == 'Sq'),
        subsampled=(mode == 'Ss'),
        isoceles=(mode == 'Is'))
    if norm is None:
        X = _signed_log(X)
    else:
        Xnorm = _filter_Bk(
            norm, kmin, kmax,
            equilateral=(mode == 'Eq'),
            squeezed=(mode == 'Sq'),
            subsampled=(mode == 'Ss'),
            isoceles=(mode == 'Is'))
        X /= Xnorm
    return np.nan_to_num(X, nan=0.0).reshape(len(X), -1)


# ---- Single simulation loader ----

def _load_single_sim(sourcepath, tracer, a=None,
                     include_hod=False, include_noise=False,
                     subselect_cosmo=None):
    diagpath = join(sourcepath, 'diag')
    if tracer == 'galaxy':
        diagpath = join(diagpath, 'galaxies')
    elif 'lightcone' in tracer:
        diagpath = join(diagpath, f'{tracer}')
    if not os.path.isdir(diagpath):
        return [], []

    filelist = ['halos.h5'] if tracer == 'halo' else os.listdir(diagpath)
    summlist, paramlist = [], []

    for fname in filelist:
        diagfile = join(diagpath, fname)
        summ = {}
        if 'lightcone' in tracer:
            summ.update(_load_lc_Pk(diagfile))
            summ.update(_load_lc_Bk(diagfile))
        else:
            summ.update(_load_Pk(diagfile, a))
            summ.update(_load_Bk(diagfile, a))
        if not summ:
            continue

        params = _get_cosmo(sourcepath)
        if params is None:
            continue
        if subselect_cosmo is not None:
            params = params[subselect_cosmo]
        if tracer != 'halo' and include_hod:
            with h5py.File(diagfile, 'r') as f:
                hod = f.attrs['HOD_params'][:]
            params = np.concatenate([params, hod])
        if include_noise:
            with h5py.File(diagfile, 'r') as f:
                if 'noise_radial' in f.attrs:
                    noise = [f.attrs['noise_radial'],
                             f.attrs['noise_transverse']]
                else:
                    g = f[list(f.keys())[0]]
                    noise = [g.attrs['noise_radial'],
                             g.attrs['noise_transverse']]
            params = np.concatenate([params, np.array(noise)])

        summlist.append(summ)
        paramlist.append(params)

    return summlist, paramlist


# ---- Main entry point ----

def load_data(cfg: DictConfig):
    """
    Load and preprocess ltu-cmass simulation summaries.

    Returns:
        summaries: np.ndarray (N, n_features)
        parameters: np.ndarray (N, n_params)
        sim_ids: np.ndarray (N,)
    """
    lcfg = cfg.loader
    suite_path = lcfg.data_path
    tracer = lcfg.tracer
    Nmax = lcfg.get('Nmax', -1)
    a = lcfg.get('a', None)
    summary_names = list(lcfg.summary)
    kmin = lcfg.kmin
    kmax = lcfg.kmax
    include_hod = lcfg.get('include_hod', False)
    include_noise = lcfg.get('include_noise', False)
    subselect_cosmo = lcfg.get('subselect_cosmo', None)
    correct_shot = lcfg.get('correct_shot', True)
    loglinear_start_idx = lcfg.get('loglinear_start_idx', None)

    # List simulation directories
    simpaths = [d for d in os.listdir(suite_path)
                if os.path.isdir(join(suite_path, d))]
    simpaths.sort(key=lambda x: int(x))
    if Nmax >= 0:
        simpaths = simpaths[:Nmax]

    logging.info(f'Loading {tracer} summaries from {suite_path} '
                 f'({len(simpaths)} sims)')

    # Load all simulations
    all_summ = defaultdict(list)
    all_params = defaultdict(list)
    all_ids = defaultdict(list)

    for lhid in tqdm(simpaths, desc='Loading'):
        sourcepath = join(suite_path, lhid)
        summs, params = _load_single_sim(
            sourcepath, tracer, a=a,
            include_hod=include_hod, include_noise=include_noise,
            subselect_cosmo=subselect_cosmo)
        for summ, param in zip(summs, params):
            for key in summ:
                all_summ[key].append(summ[key])
                all_params[key].append(param)
                all_ids[key].append(lhid)

    # Preprocess and concatenate requested summaries
    xs = []
    ref_key = None

    for name in summary_names:
        if name in ['nbar', 'nz']:
            continue

        base = name
        mode = None
        for tag in ['Eq', 'Sq', 'Ss', 'Is']:
            if tag in name:
                base = name.replace(tag, '')
                mode = tag
                break

        if base not in all_summ or len(all_summ[base]) == 0:
            logging.warning(f'No data for {name}. Skipping.')
            continue

        if ref_key is None:
            ref_key = base

        if 'Pk' in name:
            norm_key = base[:-1] + '0'
            x = preprocess_Pk(
                all_summ[base], kmin, kmax,
                norm=None if '0' in base else all_summ.get(norm_key),
                correct_shot=correct_shot,
                loglinear_start_idx=loglinear_start_idx)
        elif 'Bk' in name or 'Qk' in name:
            norm_key = base[:-1] + '0'
            x = preprocess_Bk(
                all_summ[base], kmin, kmax,
                norm=None if '0' in base else all_summ.get(norm_key),
                mode=mode)
        else:
            raise NotImplementedError(f"Unknown summary type: {name}")
        xs.append(x)

    if 'nz' in summary_names and 'Pk0' in all_summ:
        xs.append(_get_log10nz(all_summ['Pk0']))
    if 'nbar' in summary_names and 'Pk0' in all_summ:
        xs.append(_get_log10nbar(all_summ['Pk0']))

    if not xs:
        raise ValueError("No summaries were loaded. Check your summary names "
                         "and data_path.")

    summaries = np.concatenate(xs, axis=-1)
    parameters = np.array(all_params[ref_key])
    sim_ids = np.array(all_ids[ref_key])

    logging.info(f'Loaded {len(summaries)} samples with '
                 f'{summaries.shape[1]} features and '
                 f'{parameters.shape[1]} parameters')

    return summaries, parameters, sim_ids
