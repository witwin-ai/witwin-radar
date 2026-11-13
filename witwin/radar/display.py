"""Compact text forms of the published records.

A result or a processing product holds tensors of millions of elements, and the
dataclass default ``repr`` prints every one of them. The form here prints what
a reader needs to recognise the record: each tensor as its dtype, shape and
device, each long tuple as its ends and its length, and each nested record by
its class name. It reads only host metadata, so it never synchronises the
device.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

#: A tuple longer than this prints its first and last entries and its length.
_TUPLE_HEAD = 6


def describe_value(value: Any) -> str:
    """One field's compact text: a tensor's dtype, shape and device, never its elements."""

    if isinstance(value, torch.Tensor):
        dtype = str(value.dtype).removeprefix("torch.")
        shape = ", ".join(str(int(size)) for size in value.shape)
        flags = " grad" if value.requires_grad else ""
        return f"{dtype}[{shape}] {value.device}{flags}"
    if isinstance(value, np.ndarray):
        shape = ", ".join(str(int(size)) for size in value.shape)
        return f"ndarray {value.dtype}[{shape}]"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        items = list(value)
        if len(items) > _TUPLE_HEAD:
            head = ", ".join(describe_value(item) for item in items[:2])
            return f"({head}, ..., {describe_value(items[-1])}) [{len(items)}]"
        inner = ", ".join(describe_value(item) for item in items)
        return f"({inner},)" if len(items) == 1 else f"({inner})"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return f"{type(value).__name__}(...)"
    if value is None or isinstance(value, (bool, int, str)):
        return repr(value)
    return f"<{type(value).__name__}>"


def compact_repr(record: Any) -> str:
    """``Name(field=value, ...)`` over a dataclass's ``repr`` fields, in the compact form."""

    fields = (field for field in dataclasses.fields(record) if field.repr and not field.name.startswith("_"))
    parts = ", ".join(f"{field.name}={describe_value(getattr(record, field.name))}" for field in fields)
    return f"{type(record).__name__}({parts})"


def summary_repr(name: str, **fields: Any) -> str:
    """``name(key=value, ...)`` for a record that chooses which members to show."""

    parts = ", ".join(f"{key}={describe_value(value)}" for key, value in fields.items())
    return f"{name}({parts})"


__all__ = ["compact_repr", "describe_value", "summary_repr"]
