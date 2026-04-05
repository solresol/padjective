"""Tests for torch runtime helpers."""

from __future__ import annotations

import warnings

from padjective import torch_runtime


def test_select_torch_device_suppresses_noisy_cuda_probe_warning(monkeypatch):
    """CPU fallback should not leak PyTorch's CUDA probe warning."""

    def fake_is_available():
        warnings.warn(
            "CUDA initialization: Unexpected error from cudaGetDeviceCount().",
            UserWarning,
        )
        return False

    monkeypatch.setattr(torch_runtime.torch.cuda, "is_available", fake_is_available)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        device = torch_runtime.select_torch_device()

    assert device.type == "cpu"
    assert caught == []
