from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from padjective.paper_revision_neural_scaling import (
    NeuralWidthSummary,
    build_extrapolation_check,
    fit_neural_active_support_scaling,
    fit_neural_scaling,
    render_figure_one_with_neural_widths,
    render_neural_scaling_figure,
    validate_neural_width_summaries,
    write_neural_scaling_tex,
)


def _summary(
    hidden: int,
    params: float,
    loss: float,
    *,
    active_support: float | None = None,
) -> NeuralWidthSummary:
    return NeuralWidthSummary(
        hidden_units=hidden,
        fold_count=5,
        mean_params=params,
        mean_active_support=(
            float(active_support) if active_support is not None else params / 10
        ),
        mean_loss=loss,
        loss_sd=loss / 20,
        mean_iterations=100.0,
        min_iterations=90,
        max_iterations=110,
        all_converged=True,
    )


def _figure_one_bundle() -> dict[str, object]:
    return {
        "models": {
            "rows": [
                {
                    "model_key": "dummy",
                    "model_label": "Dummy Baseline",
                    "short_label": "Dummy",
                    "mean_scoring_ops": 1.0,
                    "mean_padic_loss": 0.560442,
                    "color": "#94a3b8",
                    "marker": "X",
                },
                {
                    "model_key": "umllr",
                    "model_label": "Importance-Optimised p-adic Linear Regression",
                    "short_label": "Importance-Optimised",
                    "mean_scoring_ops": 1.110177,
                    "mean_padic_loss": 0.263237,
                    "color": "#0b6ce3",
                    "marker": "o",
                },
                {
                    "model_key": "ulr",
                    "model_label": "Logistic Regression with L1",
                    "short_label": "L1 Logit",
                    "mean_scoring_ops": 375.990451,
                    "mean_padic_loss": 0.085839,
                    "color": "#8b5cf6",
                    "marker": "D",
                },
                {
                    "model_key": "levelwise",
                    "model_label": "Level-wise Logistic Regression",
                    "short_label": "Level-wise",
                    "mean_scoring_ops": 199.985029,
                    "mean_padic_loss": 0.120450,
                    "color": "#f97316",
                    "marker": "^",
                },
                {
                    "model_key": "dt",
                    "model_label": "Decision Tree",
                    "short_label": "Decision Tree",
                    "mean_scoring_ops": 258.352052,
                    "mean_padic_loss": 0.110314,
                    "color": "#14b8a6",
                    "marker": "h",
                },
                {
                    "model_key": "unn",
                    "model_label": "Unconstrained Neural Network with L2",
                    "short_label": "Unconstr. NN",
                    "mean_scoring_ops": 4_766.06,
                    "mean_padic_loss": 0.114660,
                    "color": "#ec4899",
                    "marker": "p",
                },
                {
                    "model_key": "zubarev",
                    "model_label": "Zubarev",
                    "short_label": "Zubarev",
                    "mean_scoring_ops": 1.54,
                    "mean_padic_loss": 0.333,
                    "color": "#7c3aed",
                    "marker": "v",
                },
            ]
        }
    }


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
    active_support_fit = fit_neural_active_support_scaling(summaries)
    write_neural_scaling_tex(
        tex_path,
        log_log_fit=log_log_fit,
        semi_log_fit=semi_log_fit,
        active_support_fit=active_support_fit,
    )

    assert figure_path.read_text(encoding="latin-1").startswith("%!PS-Adobe")
    tex = tex_path.read_text(encoding="utf-8")
    assert "\\PadNeuralScalingSlope" in tex
    assert "\\PadNeuralSemiLogPValue" in tex
    assert "\\PadNeuralActiveSlope" in tex


def test_combined_figure_one_uses_width_12_as_cross_model_neural_point(
    tmp_path: Path,
) -> None:
    summaries = [
        _summary(4, 11_977, 0.150690, active_support=1_829.35),
        _summary(8, 23_593, 0.112713, active_support=3_297.70),
        _summary(12, 35_209, 0.099097, active_support=4_766.06),
        _summary(24, 70_057, 0.082141, active_support=9_171.11),
        _summary(48, 139_753, 0.081112, active_support=17_981.22),
    ]
    bundle = {
        "models": {
            "rows": [
                {
                    "model_key": "dummy",
                    "model_label": "Dummy Baseline",
                    "short_label": "Dummy",
                    "mean_scoring_ops": 1.0,
                    "mean_padic_loss": 0.560442,
                    "color": "#94a3b8",
                    "marker": "X",
                },
                {
                    "model_key": "umllr",
                    "model_label": "Importance-Optimised p-adic Linear Regression",
                    "short_label": "Importance-Optimised",
                    "mean_scoring_ops": 1.110177,
                    "mean_padic_loss": 0.263237,
                    "color": "#0b6ce3",
                    "marker": "o",
                },
                {
                    "model_key": "ulr",
                    "model_label": "Logistic Regression with L1",
                    "short_label": "L1 Logit",
                    "mean_scoring_ops": 375.990451,
                    "mean_padic_loss": 0.085839,
                    "color": "#8b5cf6",
                    "marker": "D",
                },
                {
                    "model_key": "levelwise",
                    "model_label": "Level-wise Logistic Regression",
                    "short_label": "Level-wise",
                    "mean_scoring_ops": 199.985029,
                    "mean_padic_loss": 0.120450,
                    "color": "#f97316",
                    "marker": "^",
                },
                {
                    "model_key": "dt",
                    "model_label": "Decision Tree",
                    "short_label": "Decision Tree",
                    "mean_scoring_ops": 258.352052,
                    "mean_padic_loss": 0.110314,
                    "color": "#14b8a6",
                    "marker": "h",
                },
                {
                    "model_key": "unn",
                    "model_label": "Unconstrained Neural Network with L2",
                    "short_label": "Unconstr. NN",
                    "mean_scoring_ops": 4_766.06,
                    "mean_padic_loss": 0.114660,
                    "color": "#ec4899",
                    "marker": "p",
                },
                {
                    "model_key": "zubarev",
                    "model_label": "Zubarev",
                    "short_label": "Zubarev",
                    "mean_scoring_ops": 1.54,
                    "mean_padic_loss": 0.333,
                    "color": "#7c3aed",
                    "marker": "v",
                },
            ]
        }
    }
    figure_path = tmp_path / "figure-one.eps"

    (
        cross_model_fit,
        neural_fit,
        all_neural_fit,
        extrapolation_check,
    ) = render_figure_one_with_neural_widths(
        bundle,
        summaries,
        figure_path,
    )

    assert figure_path.read_text(encoding="latin-1").startswith("%!PS-Adobe")
    assert cross_model_fit["slope"] == pytest.approx(-0.1916, abs=0.0002)
    assert neural_fit.slope == pytest.approx(-0.2719, abs=0.0002)
    assert all_neural_fit == neural_fit
    assert extrapolation_check is None


def test_combined_figure_one_holds_out_width_2000(tmp_path: Path) -> None:
    summaries = [
        _summary(4, 11_977, 0.150690, active_support=1_829.35),
        _summary(8, 23_593, 0.112713, active_support=3_297.70),
        _summary(12, 35_209, 0.099097, active_support=4_766.06),
        _summary(24, 70_057, 0.082141, active_support=9_171.11),
        _summary(48, 139_753, 0.081112, active_support=17_981.22),
        _summary(2000, 5_808_361, 0.075387561, active_support=734_536.922),
    ]
    figure_path = tmp_path / "figure-one-check.eps"

    cross_model_fit, neural_fit, all_neural_fit, check = (
        render_figure_one_with_neural_widths(
            _figure_one_bundle(),
            summaries,
            figure_path,
            fit_max_hidden_units=48,
        )
    )

    assert figure_path.read_text(encoding="latin-1").startswith("%!PS-Adobe")
    assert cross_model_fit["slope"] == pytest.approx(-0.1916, abs=0.0002)
    assert neural_fit.slope == pytest.approx(-0.2719, abs=0.0002)
    assert all_neural_fit.slope == pytest.approx(-0.0908, abs=0.0002)
    assert check is not None
    assert check.predicted_neural_loss == pytest.approx(0.02696, abs=0.00002)
    assert check.predicted_cross_model_loss == pytest.approx(0.02700, abs=0.00002)
    assert check.observed_to_neural_ratio == pytest.approx(2.80, abs=0.01)


def test_extrapolation_check_can_be_written_to_tex(tmp_path: Path) -> None:
    fitted = [
        _summary(4, 11_977, 0.150690, active_support=1_829.35),
        _summary(8, 23_593, 0.112713, active_support=3_297.70),
        _summary(12, 35_209, 0.099097, active_support=4_766.06),
        _summary(24, 70_057, 0.082141, active_support=9_171.11),
        _summary(48, 139_753, 0.081112, active_support=17_981.22),
    ]
    held_out = _summary(
        2000,
        5_808_361,
        0.075387561,
        active_support=734_536.922,
    )
    active_fit = fit_neural_active_support_scaling(fitted)
    all_active_fit = fit_neural_active_support_scaling([*fitted, held_out])
    check = build_extrapolation_check(
        held_out,
        neural_fit=active_fit,
        cross_model_fit={"slope": -0.191615, "intercept": -0.444648},
    )
    tex_path = tmp_path / "extrapolation-check.tex"

    write_neural_scaling_tex(
        tex_path,
        log_log_fit=fit_neural_scaling(fitted, log_parameters=True),
        semi_log_fit=fit_neural_scaling(fitted, log_parameters=False),
        active_support_fit=active_fit,
        all_active_support_fit=all_active_fit,
        extrapolation_check=check,
    )

    tex = tex_path.read_text(encoding="utf-8")
    assert "\\newcommand{\\PadNeuralAllActiveSlope}{-0.0908}" in tex
    assert "\\newcommand{\\PadNeuralCheckWidth}{2000}" in tex
    assert "\\newcommand{\\PadNeuralCheckParams}{5{,}808{,}361}" in tex
    assert "\\newcommand{\\PadNeuralCheckPredictionRatio}{2.80}" in tex
