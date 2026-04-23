"""Shared value validators used across radar configuration modules."""

from __future__ import annotations

import math
from typing import Any, Iterable


def finite_float(name: str, value: Any, *, prefix: str = "Field") -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix} '{name}' must be a finite float.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{prefix} '{name}' must be a finite float.")
    return parsed


def non_negative_float(name: str, value: Any, *, prefix: str = "Field") -> float:
    parsed = finite_float(name, value, prefix=prefix)
    if parsed < 0.0:
        raise ValueError(f"{prefix} '{name}' must be non-negative.")
    return parsed


def positive_float(name: str, value: Any, *, prefix: str = "Field") -> float:
    parsed = finite_float(name, value, prefix=prefix)
    if parsed <= 0.0:
        raise ValueError(f"{prefix} '{name}' must be positive.")
    return parsed


def positive_int(name: str, value: Any, *, prefix: str = "Field") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{prefix} '{name}' must be a positive int.")
    return value


def optional_seed(value: Any, *, name: str = "seed", prefix: str = "Field") -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{prefix} '{name}' must be a non-negative int.")
    return value


def require_keys(config: dict[str, Any], keys: Iterable[str], *, label: str = "config") -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{label} is missing required keys: {joined}")
