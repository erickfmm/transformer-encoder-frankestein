from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

GROUP_NAMES: Tuple[str, ...] = (
    "embeddings",
    "norms",
    "ode",
    "retnet",
    "mamba",
    "attention",
    "other",
)


def to_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def to_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def to_betas(value: Any, default: Tuple[float, float] = (0.9, 0.95)) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return default
    return default


def extract_prefixed_parameters(prefix: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    expected_prefix = f"{prefix}-"
    for key, value in (parameters or {}).items():
        if key.startswith(expected_prefix):
            result[key[len(expected_prefix):]] = value
    return result


def ensure_no_unknown_parameters(
    optimizer_name: str,
    scoped_params: Dict[str, Any],
    allowed_keys: Iterable[str],
) -> None:
    allowed = set(allowed_keys)
    unknown = sorted(k for k in scoped_params if k not in allowed)
    if unknown:
        raise ValueError(
            f"Unknown parameters for optimizer '{optimizer_name}': {unknown}. "
            f"Allowed keys: {sorted(allowed)}"
        )


def parse_group_value(scoped_params: Dict[str, Any], key_stem: str, group_name: str, default: Any) -> Any:
    key = f"{key_stem}_{group_name}"
    if key in scoped_params:
        return scoped_params[key]
    return default


def with_named_groups(param_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    named_groups: List[Dict[str, Any]] = []
    for group in param_groups:
        copied = dict(group)
        if "name" not in copied:
            copied["name"] = "other"
        named_groups.append(copied)
    return named_groups
