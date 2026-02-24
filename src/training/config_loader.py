import os
from dataclasses import fields
from typing import Dict, Tuple, Any

import yaml

try:
    from .trainer import TrainingConfig
    from ..model.tormented_bert_frankestein import UltraConfig
except ImportError:
    from training.trainer import TrainingConfig
    from model.tormented_bert_frankestein import UltraConfig


def _field_names(cls) -> set:
    return {field.name for field in fields(cls)}


def load_training_config(path: str) -> Tuple[str, UltraConfig, TrainingConfig, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    model_data = data.get("model", {}) or {}
    training_data = data.get("training", {}) or {}

    model_config = UltraConfig(**model_data)

    optimizer_data = training_data.get("optimizer")
    if not isinstance(optimizer_data, dict):
        raise ValueError(
            "Missing required 'training.optimizer' object in config. "
            "Legacy top-level optimizer fields were removed."
        )

    optimizer_class = optimizer_data.get("optimizer_class")
    if not isinstance(optimizer_class, str) or not optimizer_class.strip():
        raise ValueError("training.optimizer.optimizer_class must be a non-empty string")

    optimizer_parameters = optimizer_data.get("parameters", {})
    if not isinstance(optimizer_parameters, dict):
        raise ValueError("training.optimizer.parameters must be an object")

    training_fields = _field_names(TrainingConfig)
    training_kwargs = {k: v for k, v in training_data.items() if k in training_fields}
    training_kwargs["optimizer_class"] = optimizer_class.strip().lower()
    training_kwargs["optimizer_parameters"] = optimizer_parameters
    training_config = TrainingConfig(**training_kwargs)

    runtime_keys = set(training_data.keys()) - training_fields - {"optimizer"}
    training_runtime = {k: training_data[k] for k in runtime_keys}

    model_class = data.get("model_class", "frankenstein")
    return model_class, model_config, training_config, training_runtime


def list_config_paths(config_dir: str) -> Dict[str, str]:
    configs = {}
    if not os.path.isdir(config_dir):
        return configs

    for filename in sorted(os.listdir(config_dir)):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            name = os.path.splitext(filename)[0]
            configs[name] = os.path.join(config_dir, filename)
    return configs
