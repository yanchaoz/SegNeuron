"""Fail-fast validation for the public legacy supervised-training configuration."""

from collections.abc import Mapping


class ConfigError(ValueError):
    """Raised when a legacy configuration cannot start safely."""


def _get(config, dotted_path):
    value = config
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError("Missing required configuration field: %s" % dotted_path)
        value = value[key]
    return value


def _is_placeholder(value):
    text = str(value).strip()
    return not text or "***" in text


def validate_config(config):
    """Validate startup requirements without changing legacy numeric behavior."""
    if not isinstance(config, Mapping):
        raise ConfigError("The YAML root must be a mapping.")

    required = (
        "NAME",
        "MODEL.model_type",
        "MODEL.pre_train",
        "MODEL.continue_train",
        "TRAIN.loss_func",
        "TRAIN.total_iters",
        "TRAIN.base_lr",
        "TRAIN.end_lr",
        "TRAIN.batch_size",
        "TRAIN.num_workers",
        "TRAIN.if_cuda",
        "TRAIN.random_seed",
        "TRAIN.freq_mix_prob",
        "TRAIN.spa_mix_prob",
        "DATA.data_folder",
    )
    for field in required:
        _get(config, field)

    model_type = _get(config, "MODEL.model_type")
    if model_type not in ("superhuman", "mala"):
        raise ConfigError("MODEL.model_type must be 'superhuman' or 'mala'.")

    # Preserve the published loss-name mapping. It is counterintuitive but may
    # be part of the behavior used to create historical checkpoints.
    if _get(config, "TRAIN.loss_func") not in ("MSELoss", "BCELoss"):
        raise ConfigError("TRAIN.loss_func must be 'MSELoss' or 'BCELoss'.")

    for field in ("TRAIN.total_iters", "TRAIN.batch_size"):
        if int(_get(config, field)) <= 0:
            raise ConfigError("%s must be greater than zero." % field)
    if int(_get(config, "TRAIN.num_workers")) < 0:
        raise ConfigError("TRAIN.num_workers cannot be negative.")
    for field in ("TRAIN.freq_mix_prob", "TRAIN.spa_mix_prob"):
        value = float(_get(config, field))
        if not 0.0 <= value <= 1.0:
            raise ConfigError("%s must be between zero and one." % field)

    if _is_placeholder(_get(config, "DATA.data_folder")):
        raise ConfigError("Set DATA.data_folder to the legacy supervised dataset root.")
    if _get(config, "MODEL.pre_train") and _is_placeholder(_get(config, "MODEL.pretrain_path")):
        raise ConfigError(
            "MODEL.pre_train is enabled; set MODEL.pretrain_path or disable it explicitly."
        )
    if _get(config, "MODEL.continue_train") and _is_placeholder(
        _get(config, "MODEL.continue_path")
    ):
        raise ConfigError("MODEL.continue_train is enabled; set MODEL.continue_path.")
    if _get(config, "TRAIN.resume"):
        _get(config, "TRAIN.model_name")

    return config
