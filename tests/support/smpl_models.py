"""Where this checkout keeps its SMPL model pickles, or ``None``."""

from __future__ import annotations

import pathlib

import witwin.radar.smpl as smpl_module


def smpl_model_root() -> str | None:
    """The first candidate directory that holds a ``*.pkl`` model.

    Two candidates: the package default, and a git worktree's sibling. A
    worktree sits one level deeper than the checkout the default path is
    written against, so the models live beside the main checkout.
    """

    candidates = (
        pathlib.Path(smpl_module._default_smpl_model_root()),
        pathlib.Path(__file__).resolve().parents[4] / "radar" / "models" / "smpl_models",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.pkl")):
            return str(candidate)
    return None


__all__ = ["smpl_model_root"]
