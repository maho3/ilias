import numpy as np
from omegaconf import DictConfig, OmegaConf


def _get_or_sample_optuna(trial, name, value, sample_func, **kwargs):
    """Helper to sample a value for optuna or return it if it's fixed."""
    is_list_like = hasattr(value, '__len__') and not isinstance(value, str)
    if not is_list_like:
        return value

    if sample_func == 'categorical':
        return trial.suggest_categorical(name, value)
    elif sample_func == 'int':
        return trial.suggest_int(name, *value, **kwargs)
    elif sample_func == 'float':
        return trial.suggest_float(name, *value, **kwargs)
    raise ValueError(f"Unknown sample function: {sample_func}")


def _get_or_sample_random(value, sample_logic):
    """Helper to sample a value randomly or return it if it's fixed."""
    is_list_like = hasattr(value, '__len__') and not isinstance(value, str)
    if not is_list_like:
        return value

    if sample_logic == 'choice':
        return np.random.choice(value)
    elif sample_logic == 'randint':
        return np.random.randint(value[0], value[1] + 1)
    elif sample_logic == 'loguniform':
        return np.exp(np.random.uniform(np.log(value[0]), np.log(value[1])))
    elif sample_logic == 'uniform':
        return np.random.uniform(value[0], value[1])
    raise ValueError(f"Unknown sample logic: {sample_logic}")


def _sample_shared(trial_or_none, hp, is_optuna):
    """Sample shared hyperparameters (common to all embedding nets)."""
    mcfg = {}

    if is_optuna:
        s = lambda name, val, func, **kw: _get_or_sample_optuna(trial_or_none, name, val, func, **kw)
        mcfg['model'] = s("model", hp.model, 'categorical')
        mcfg['hidden_features'] = s("hidden_features", hp.hidden_features, 'int', log=True)
        mcfg['num_transforms'] = s("num_transforms", hp.num_transforms, 'int')
        log2_batch_size = s("log2_batch_size", hp.log2_batch_size, 'int')
        mcfg['batch_size'] = int(2**log2_batch_size)
        mcfg['learning_rate'] = s("learning_rate", hp.learning_rate, 'float', log=True)
        mcfg['weight_decay'] = s("weight_decay", hp.weight_decay, 'float', log=True)
        mcfg['lr_patience'] = s("lr_patience", hp.lr_patience, 'int')
        mcfg['lr_decay_factor'] = s("lr_decay_factor", hp.lr_decay_factor, 'float', log=True)
        mcfg['early_stopping'] = s("early_stopping", hp.early_stopping, 'categorical')
        mcfg['noise_percent'] = s("noise_percent", hp.noise_percent, 'float', log=True)
        mcfg['lr_scheduler'] = s("lr_scheduler", hp.lr_scheduler, 'categorical')
        mcfg['max_epochs'] = s("max_epochs", hp.max_epochs, 'int', log=True)
        mcfg['dropout'] = s("dropout", hp.dropout, 'float')
    else:
        s = lambda val, logic: _get_or_sample_random(val, logic)
        mcfg['model'] = s(hp.model, 'choice')
        mcfg['hidden_features'] = int(s(hp.hidden_features, 'loguniform'))
        mcfg['num_transforms'] = s(hp.num_transforms, 'randint')
        mcfg['batch_size'] = int(2**s(hp.log2_batch_size, 'randint'))
        mcfg['learning_rate'] = s(hp.learning_rate, 'loguniform')
        mcfg['weight_decay'] = s(hp.weight_decay, 'loguniform')
        mcfg['lr_patience'] = s(hp.lr_patience, 'randint')
        mcfg['lr_decay_factor'] = s(hp.lr_decay_factor, 'loguniform')
        mcfg['early_stopping'] = s(hp.early_stopping, 'choice')
        mcfg['noise_percent'] = s(hp.noise_percent, 'loguniform')
        mcfg['lr_scheduler'] = s(hp.lr_scheduler, 'choice')
        mcfg['max_epochs'] = int(s(hp.max_epochs, 'loguniform'))
        mcfg['dropout'] = s(hp.dropout, 'uniform')

    return mcfg


def _sample_embedding(trial_or_none, hp_emb, embedding_net, is_optuna):
    """Sample embedding-net-specific hyperparameters."""
    mcfg = {}

    if is_optuna:
        s = lambda name, val, func, **kw: _get_or_sample_optuna(trial_or_none, name, val, func, **kw)
    else:
        s = lambda name, val, func, **kw: _get_or_sample_random(val, {
            'int': 'randint', 'float': 'loguniform', 'categorical': 'choice'
        }.get(func, func))

    if embedding_net == 'fcn':
        mcfg['fcn_depth'] = s('fcn_depth', hp_emb.fcn_depth, 'int')
        v = s('fcn_width', hp_emb.fcn_width, 'int', log=True) if is_optuna else int(
            _get_or_sample_random(hp_emb.fcn_width, 'loguniform'))
        mcfg['fcn_width'] = v
    elif embedding_net == 'cnn':
        mcfg['cnn_depth'] = s('cnn_depth', hp_emb.cnn_depth, 'int')
        v = s('out_channels', hp_emb.out_channels, 'int', log=True) if is_optuna else int(
            _get_or_sample_random(hp_emb.out_channels, 'loguniform'))
        mcfg['out_channels'] = v
        mcfg['kernel_size'] = s('kernel_size', hp_emb.kernel_size, 'int') if is_optuna else _get_or_sample_random(
            hp_emb.kernel_size, 'randint')
    elif embedding_net == 'mhe':
        mcfg['hidden_depth'] = s('hidden_depth', hp_emb.hidden_depth, 'int') if is_optuna else _get_or_sample_random(
            hp_emb.hidden_depth, 'randint')
        v = s('hidden_width', hp_emb.hidden_width, 'int', log=True) if is_optuna else int(
            _get_or_sample_random(hp_emb.hidden_width, 'loguniform'))
        mcfg['hidden_width'] = v
        v = s('out_features', hp_emb.out_features, 'int', log=True) if is_optuna else int(
            _get_or_sample_random(hp_emb.out_features, 'loguniform'))
        mcfg['out_features'] = v
    elif embedding_net in ['fun', 'mhf']:
        mcfg['hidden_depth'] = s('hidden_depth', hp_emb.hidden_depth, 'int') if is_optuna else _get_or_sample_random(
            hp_emb.hidden_depth, 'randint')
        v = s('out_features', hp_emb.out_features, 'int', log=True) if is_optuna else int(
            _get_or_sample_random(hp_emb.out_features, 'loguniform'))
        mcfg['out_features'] = v
        if is_optuna:
            mcfg['linear_dim'] = s('linear_dim', hp_emb.linear_dim, 'int', log=True)
            mcfg['bypass'] = s('bypass', hp_emb.bypass, 'categorical')
        else:
            linear_dim_value = _get_or_sample_random(hp_emb.linear_dim, 'loguniform')
            mcfg['linear_dim'] = int(linear_dim_value) if linear_dim_value is not None else None
            mcfg['bypass'] = _get_or_sample_random(hp_emb.bypass, 'choice')
    else:
        raise ValueError(f"Unknown embedding net: {embedding_net}")

    return mcfg


def sample_hyperparameters_optuna(
    trial: "optuna.trial.Trial",
    hyperprior: DictConfig,
    embedding_net: str
) -> DictConfig:
    """Sample hyperparameters from the hyperprior for Optuna and return a model config."""
    mcfg = {"embedding_net": embedding_net}
    mcfg.update(_sample_shared(trial, hyperprior.shared, is_optuna=True))
    mcfg.update(_sample_embedding(trial, hyperprior[embedding_net], embedding_net, is_optuna=True))
    return OmegaConf.create(mcfg)


def sample_hyperparameters_randomly(
    hyperprior: DictConfig,
    embedding_net: str,
    seed: int = None
) -> DictConfig:
    """Randomly sample hyperparameters from the hyperprior and return a model config."""
    if seed is not None:
        np.random.seed(seed)

    mcfg = {"embedding_net": embedding_net}
    mcfg.update(_sample_shared(None, hyperprior.shared, is_optuna=False))
    mcfg.update(_sample_embedding(None, hyperprior[embedding_net], embedding_net, is_optuna=False))

    # typecasting for OmegaConf
    for k, v in mcfg.items():
        if isinstance(v, np.int64):
            mcfg[k] = int(v)
        elif isinstance(v, np.float64):
            mcfg[k] = float(v)
        elif isinstance(v, np.str_):
            mcfg[k] = str(v)

    return OmegaConf.create(mcfg)
