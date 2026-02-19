import os
from dataclasses import fields
from typing import Dict, Tuple, Any

import yaml

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

    training_fields = _field_names(TrainingConfig)
    training_kwargs = {k: v for k, v in training_data.items() if k in training_fields}
    training_config = TrainingConfig(**training_kwargs)

    runtime_keys = set(training_data.keys()) - training_fields
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
