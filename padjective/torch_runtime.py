"""Torch runtime helpers that keep CUDA probing quiet on CPU-only hosts."""

from __future__ import annotations

import warnings

import torch


_CUDA_INIT_WARNING_RE = r"CUDA initialization: .*"


def cuda_is_available() -> bool:
    """Return whether CUDA can be used without surfacing probe warnings."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_CUDA_INIT_WARNING_RE,
            category=UserWarning,
        )
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False


def select_torch_device() -> torch.device:
    """Choose CUDA when available, otherwise fall back to CPU."""

    return torch.device("cuda" if cuda_is_available() else "cpu")
