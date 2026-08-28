from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from padjective.paper_revision_neural_scaling import (
    NeuralWidthSummary,
    fit_neural_scaling,
    render_neural_scaling_figure,
    validate_neural_width_summaries,
    write_neural_scaling_tex,
)


def _summary(hidden: int, params: float, loss: float) -> NeuralWidthSummary:
    return NeuralWidthSummary(
        hidden_units=hidden,
        fold_count=5,
        mean_params=params,
        mean_loss=loss,
        loss_sd=loss / 20,
        mean_iterations=100.0,
        min_iterations=90,
        max_iterations=110,
        all_converged=True,
    )


def test_log_log_scaling_fit_recovers_power_law() -> None:
    summaries = [
        _summary(4, 100.0, 0.4),
        _summary(8, 1_000.0, 0.2),
        _summary(16, 10_000.0, 0.1),
    ]

    fit = fit_neural_scaling(summaries, log_parameters=True)

    assert fit.slope == pytest.approx(-0.30103)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.x_transform == "log10(parameters)"


def test_scaling_fit_rejects_unconverged_width() -> None:
    summaries = [
        _summary(4, 100.0, 0.4),
        _summary(8, 1_000.0, 0.2),
        replace(_summary(16, 10_000.0, 0.1), all_converged=False),
    ]

    with pytest.raises(ValueError, match="did not converge"):
        validate_neural_width_summaries(summaries)


def test_render_and_tex_output_include_scaling_results(tmp_path: Path) -> None:
    summaries = [
        _summary(4, 10_000.0, 0.16),
        _summary(8, 20_000.0, 0.12),
        _summary(16, 40_000.0, 0.10),
        _summary(32, 80_000.0, 0.08),
    ]
    figure_path = tmp_path / "neural-scaling.eps"
    tex_path = tmp_path / "neural-scaling.tex"

    log_log_fit = render_neural_scaling_figure(summaries, figure_path)
    semi_log_fit = fit_neural_scaling(summaries, log_parameters=False)
    write_neural_scaling_tex(
        tex_path,
        log_log_fit=log_log_fit,
        semi_log_fit=semi_log_fit,
    )

    assert figure_path.read_text(encoding="latin-1").startswith("%!PS-Adobe")
    tex = tex_path.read_text(encoding="utf-8")
    assert "\\PadNeuralScalingSlope" in tex
    assert "\\PadNeuralSemiLogPValue" in tex
