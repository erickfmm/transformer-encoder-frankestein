from __future__ import annotations

from typing import Any, Dict, List

from torch.optim import Optimizer

from .adamw import AdamWOptimizer
from .adafactor import Adafactor
from .adan import Adan
from .ademamix import AdEMAMix
from .adopt import ADOPT
from .base import (
    GROUP_NAMES,
    ensure_no_unknown_parameters,
    extract_prefixed_parameters,
    parse_group_value,
    to_betas,
    to_float,
    with_named_groups,
)
from .cautious_adamw import CautiousAdamW
from .galore_adamw import GaLoreAdamW
from .lamb import LAMB
from .lion import Lion
from .mars_adamw import MARSAdamW
from .muon import Muon
from .prodigy import Prodigy
from .radam import RAdamOptimizer
from .schedulefree_adamw import ScheduleFreeAdamW
from .sgd_momentum import SGDMomentum
from .shampoo import Shampoo
from .sophia import Sophia
from .soap import SOAP
from .turbo_muon import TurboMuon

OPTIMIZER_REGISTRY = {
    "sgd_momentum": SGDMomentum,
    "adamw": AdamWOptimizer,
    "adafactor": Adafactor,
    "galore_adamw": GaLoreAdamW,
    "prodigy": Prodigy,
    "lion": Lion,
    "sophia": Sophia,
    "muon": Muon,
    "turbo_muon": TurboMuon,
    "radam": RAdamOptimizer,
    "adan": Adan,
    "adopt": ADOPT,
    "ademamix": AdEMAMix,
    "mars_adamw": MARSAdamW,
    "cautious_adamw": CautiousAdamW,
    "lamb": LAMB,
    "schedulefree_adamw": ScheduleFreeAdamW,
    "shampoo": Shampoo,
    "soap": SOAP,
}

_COMMON_PER_GROUP_KEYS = {
    f"lr_{group}" for group in GROUP_NAMES
} | {
    f"wd_{group}" for group in GROUP_NAMES
} | {
    f"betas_{group}" for group in GROUP_NAMES
} | {
    f"eps_{group}" for group in GROUP_NAMES
}

OPTIMIZER_ALLOWED_KEYS = {
    "sgd_momentum": _COMMON_PER_GROUP_KEYS | {"momentum", "nesterov"},
    "adamw": _COMMON_PER_GROUP_KEYS,
    "adafactor": _COMMON_PER_GROUP_KEYS | {"beta2_decay", "clip_threshold", "eps1", "eps2"},
    "galore_adamw": _COMMON_PER_GROUP_KEYS | {"rank", "update_proj_gap"},
    "prodigy": _COMMON_PER_GROUP_KEYS | {"d_coef"},
    "lion": _COMMON_PER_GROUP_KEYS,
    "sophia": _COMMON_PER_GROUP_KEYS | {"rho", "update_k"},
    "muon": _COMMON_PER_GROUP_KEYS | {"momentum", "nesterov", "ns_steps", "ns_eps"},
    "turbo_muon": _COMMON_PER_GROUP_KEYS | {"momentum", "nesterov", "ns_steps", "ns_eps"},
    "radam": _COMMON_PER_GROUP_KEYS,
    "adan": _COMMON_PER_GROUP_KEYS,
    "adopt": _COMMON_PER_GROUP_KEYS,
    "ademamix": _COMMON_PER_GROUP_KEYS,
    "mars_adamw": _COMMON_PER_GROUP_KEYS,
    "cautious_adamw": _COMMON_PER_GROUP_KEYS | {"cautious_clip"},
    "lamb": _COMMON_PER_GROUP_KEYS,
    "schedulefree_adamw": _COMMON_PER_GROUP_KEYS,
    "shampoo": _COMMON_PER_GROUP_KEYS,
    "soap": _COMMON_PER_GROUP_KEYS,
}


def _build_param_groups_for_optimizer(
    optimizer_name: str,
    base_param_groups: List[Dict[str, Any]],
    scoped_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for group in with_named_groups(base_param_groups):
        group_name = str(group["name"])
        converted = {
            "params": group["params"],
            "lr": to_float(parse_group_value(scoped_params, "lr", group_name, group.get("lr", 1e-3)), 1e-3),
            "weight_decay": to_float(
                parse_group_value(scoped_params, "wd", group_name, group.get("weight_decay", 0.0)),
                0.0,
            ),
        }

        if optimizer_name not in {"sgd_momentum"}:
            converted["betas"] = to_betas(parse_group_value(scoped_params, "betas", group_name, group.get("betas", (0.9, 0.95))))
            converted["eps"] = to_float(parse_group_value(scoped_params, "eps", group_name, group.get("eps", 1e-8)), 1e-8)

        converted["name"] = group_name
        result.append(converted)
    return result


def build_optimizer(
    optimizer_class: str,
    param_groups: List[Dict[str, Any]],
    parameters: Dict[str, Any],
) -> Optimizer:
    optimizer_name = str(optimizer_class or "").strip().lower()
    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer_class '{optimizer_class}'. "
            f"Supported values: {sorted(OPTIMIZER_REGISTRY)}"
        )

    scoped = extract_prefixed_parameters(optimizer_name, parameters or {})
    ensure_no_unknown_parameters(
        optimizer_name,
        scoped,
        OPTIMIZER_ALLOWED_KEYS[optimizer_name],
    )

    converted_groups = _build_param_groups_for_optimizer(optimizer_name, param_groups, scoped)
    optimizer_cls = OPTIMIZER_REGISTRY[optimizer_name]

    if optimizer_name == "sgd_momentum":
        momentum = to_float(scoped.get("momentum"), 0.9)
        nesterov = bool(scoped.get("nesterov", False))
        return optimizer_cls(converted_groups, momentum=momentum, nesterov=nesterov)

    if optimizer_name == "cautious_adamw":
        cautious_clip = to_float(scoped.get("cautious_clip"), 1.0)
        return optimizer_cls(converted_groups, cautious_clip=cautious_clip)

    if optimizer_name == "adafactor":
        beta2_decay = to_float(scoped.get("beta2_decay"), 0.8)
        clip_threshold = to_float(scoped.get("clip_threshold"), 1.0)
        eps1 = to_float(scoped.get("eps1"), 1e-30)
        eps2 = to_float(scoped.get("eps2"), 1e-3)
        return optimizer_cls(converted_groups, beta2_decay=beta2_decay, clip_threshold=clip_threshold, eps=(eps1, eps2))

    if optimizer_name == "galore_adamw":
        rank = int(to_float(scoped.get("rank"), 128))
        update_proj_gap = int(to_float(scoped.get("update_proj_gap"), 200))
        return optimizer_cls(converted_groups, rank=rank, update_proj_gap=update_proj_gap)

    if optimizer_name == "prodigy":
        d_coef = to_float(scoped.get("d_coef"), 0.8)
        return optimizer_cls(converted_groups, d_coef=d_coef)

    if optimizer_name == "sophia":
        rho = to_float(scoped.get("rho"), 0.04)
        update_k = int(to_float(scoped.get("update_k"), 10))
        return optimizer_cls(converted_groups, rho=rho, update_k=update_k)

    if optimizer_name in {"muon", "turbo_muon"}:
        momentum = to_float(scoped.get("momentum"), 0.95)
        nesterov = bool(scoped.get("nesterov", True))
        ns_steps = int(to_float(scoped.get("ns_steps"), 5 if optimizer_name == "muon" else 4))
        ns_eps = to_float(scoped.get("ns_eps"), 1e-7)
        return optimizer_cls(
            converted_groups,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
        )

    return optimizer_cls(converted_groups)
