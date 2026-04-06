import io
import logging
import pickle
import shutil
import functools
import time

import numpy as np
import torch
import optuna
from torch.utils.data import TensorDataset, DataLoader


def timing_decorator(func):
    """Logs execution time of the decorated function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f'{func.__name__} completed in {elapsed:.1f}s')
        return result
    return wrapper


def clean_up(hydra_module):
    """Decorator to clean up Hydra output directories after execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            try:
                hydra_cfg = hydra_module.core.hydra_config.HydraConfig.get()
                output_dir = hydra_cfg.runtime.output_dir
                if output_dir is not None:
                    shutil.rmtree(output_dir, ignore_errors=True)
            except Exception:
                pass
            return result
        return wrapper
    return decorator


def prepare_loader(x, theta, device='cpu', **kwargs):
    x = torch.Tensor(x).to(device)
    theta = torch.Tensor(theta).to(device)
    dataset = TensorDataset(x, theta)
    loader = DataLoader(dataset, **kwargs)
    return loader


class CPU_Unpickler(pickle.Unpickler):
    """Unpickles a torch model saved on GPU to CPU."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_posterior(modelpath, device):
    """Load a posterior from a model file."""
    with open(modelpath, 'rb') as f:
        ensemble = CPU_Unpickler(f).load()
    ensemble = ensemble.to(device)
    for p in ensemble.posteriors:
        p.to(device)
    return ensemble


def select_top_trials(study, n_nets):
    """Select the top N nets from an optuna study."""
    trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    if len(trials) == 0:
        raise ValueError('No completed trials found in the study.')

    trials = sorted(trials, key=lambda t: t.value, reverse=True)
    return trials[:n_nets]


def log2_avg(A, s=0):
    A = np.asarray(A)
    if len(A) <= s:
        return A
    idx = s + (1 << np.arange((len(A) - s).bit_length())) - 1
    idx = np.r_[np.arange(s), idx] if s > 0 else idx
    return np.add.reduceat(A, idx) / np.diff(np.append(idx, len(A)))
