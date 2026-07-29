"""Build and render benchmark bundles for the website, notebook, and paper."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from . import db, data_access
from .benchmark_runtime import (
    build_snapshot_benchmark_bundle,
    load_snapshot_tables,
)


MODEL_ORDER = (
    "dummy",
    "umllr",
    "pclr",
    "ulr",
    "pcnn",
    "dt",
    "levelwise",
    "unn",
    "zubarev",
)

TRAINED_PARAMS_LABEL = "Trained params"
ACTIVE_PARAMS_LABEL = "Avg active params / classification"

ABLATION_STRATEGY_METADATA: dict[str, dict[str, str]] = {
    "battle_elo": {
        "headline": "Pairwise battle ranking",
        "description": (
            "Ranks tags by fold-local Elo scores estimated from tag battles, while "
            "excluding the holdout fold from the ranking fit."
        ),
    },
    "frequency": {
        "headline": "Most common tags first",
        "description": "Ranks tags by how often they appear in the training products.",
    },
    "mean_title_position": {
        "headline": "Average title position",
        "description": (
            "Ranks tags by their average recorded title position in the training "
            "products."
        ),
    },
    "taxonomy_association": {
        "headline": "Taxonomy-peaked tags first",
        "description": (
            "For each tag, measure the share of its training occurrences that land "
            "in its single most common taxonomy. Tags with the strongest one-taxonomy "
            "association are scored first."
        ),
    },
    "random": {
        "headline": "Seeded random control",
        "description": (
            "Uses a seeded random shuffle of the training tag vocabulary as a control "
            "condition."
        ),
    },
}


def _json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=False) + "\n"


def load_bundle(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_rows_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(bundle.get("models", {}).get("rows", []))
    if frame.empty:
        return frame
    frame["order"] = frame["model_key"].map(
        {key: idx for idx, key in enumerate(MODEL_ORDER)}
    )
    return frame.sort_values(["order", "model_label"]).drop(columns=["order"]).reset_index(drop=True)


def ablation_runs_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(bundle.get("ablation", {}).get("runs", []))


def ablation_strategy_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(bundle.get("ablation", {}).get("strategy_rows", []))
    if frame.empty:
        return frame
    return frame.sort_values(["mean_padic_loss", "tag_order_strategy"]).reset_index(drop=True)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def _format_percent_tex(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}\\%"


def _format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _format_intish(value: float | None) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return f"{int(round(numeric)):,}"
    return f"{numeric:.1f}"


def render_model_comparison_html(bundle: dict[str, Any]) -> str:
    rows = []
    for row in model_rows_frame(bundle).to_dict("records"):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_label']))}</td>"
            f"<td>{_format_float(row.get('params'), 1)}</td>"
            f"<td>{_format_float(row.get('mean_padic_loss'), 6)}</td>"
            f"<td>{_format_percent(row.get('mean_exact_accuracy'))}</td>"
            f"<td>{_format_percent(row.get('mean_prefix2_accuracy'))}</td>"
            f"<td>{_format_float(row.get('mean_scoring_ops'), 2)}</td>"
            "</tr>"
        )
    return (
        "<table class=\"benchmark-table\">"
        "<thead><tr>"
        "<th>Model</th>"
        f"<th title=\"Average non-zero parameter count in the fitted model across folds.\">{TRAINED_PARAMS_LABEL}</th>"
        "<th>Mean p-adic loss</th>"
        "<th>Exact acc.</th><th>Prefix-2 acc.</th>"
        f"<th title=\"Mean number of active parameters or scoring decisions touched while classifying one product.\">{ACTIVE_PARAMS_LABEL}</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_ablation_html(bundle: dict[str, Any]) -> str:
    rows = []
    for row in ablation_strategy_frame(bundle).to_dict("records"):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['tag_order_strategy']))}</td>"
            f"<td>{_format_float(row.get('mean_padic_loss'), 6)}</td>"
            f"<td>{_format_float(row.get('loss_delta_vs_baseline'), 6)}</td>"
            f"<td>{row.get('wins_vs_baseline', 0)}/{row.get('comparisons_vs_baseline', 0)}</td>"
            f"<td>{_format_percent(row.get('mean_exact_accuracy'))}</td>"
            f"<td>{_format_percent(row.get('mean_prefix2_accuracy'))}</td>"
            f"<td>{_format_float(row.get('mean_scoring_ops'), 2)}</td>"
            "</tr>"
        )
    return (
        "<table class=\"benchmark-table\">"
        "<thead><tr>"
        "<th>Strategy</th><th>Mean p-adic loss</th><th>Δ vs battle_elo</th>"
        "<th>Fold wins</th><th>Exact acc.</th><th>Prefix-2 acc.</th>"
        f"<th title=\"Mean number of active coefficients touched while classifying one product.\">{ACTIVE_PARAMS_LABEL}</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_ablation_strategy_guide_html(bundle: dict[str, Any]) -> str:
    frame = ablation_strategy_frame(bundle)
    if frame.empty:
        return ""

    cards = []
    for strategy in frame["tag_order_strategy"].tolist():
        metadata = ABLATION_STRATEGY_METADATA.get(
            str(strategy),
            {
                "headline": "Strategy",
                "description": "Strategy metadata has not been documented yet.",
            },
        )
        cards.append(
            "<article class=\"benchmark-card benchmark-strategy-card\">"
            f"<h2><code>{html.escape(str(strategy))}</code></h2>"
            f"<p class=\"benchmark-strategy-headline\">{html.escape(metadata['headline'])}</p>"
            f"<p>{html.escape(metadata['description'])}</p>"
            "</article>"
        )

    return (
        "<div class=\"benchmark-methods\">"
        "<h2>Ordering methods</h2>"
        "<p>The ablation keeps the greedy p-adic regressor fixed and changes only "
        "the tag ordering heuristic used before coefficient fitting.</p>"
        "<div class=\"benchmark-grid benchmark-strategy-grid\">"
        + "".join(cards)
        + "</div></div>"
    )


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def render_benchmark_numbers_tex(bundle: dict[str, Any]) -> str:
    snapshot = bundle.get("snapshot", {})
    narrative = bundle.get("narrative", {})
    lines = [
        "% Generated by padjective.benchmark_bundle",
        "\\providecommand{\\PadBenchSnapshotLabel}{" + _latex_escape(str(snapshot.get("label", ""))) + "}",
        "\\providecommand{\\PadBenchSnapshotName}{" + _latex_escape(str(snapshot.get("snapshot_name", ""))) + "}",
        "\\providecommand{\\PadBenchSnapshotCutoff}{" + _latex_escape(str(snapshot.get("as_of") or "")) + "}",
        "\\providecommand{\\PadBenchFilteredProducts}{" + f"{int(snapshot.get('product_count_filtered', 0)):,}" + "}",
        "\\providecommand{\\PadBenchFilteredTags}{" + f"{int(snapshot.get('tag_count_filtered', 0)):,}" + "}",
        "\\providecommand{\\PadBenchFilteredTaxonomies}{" + f"{int(snapshot.get('taxonomy_count_filtered', 0)):,}" + "}",
        "\\providecommand{\\PadBenchPrimeBase}{" + str(int(snapshot.get("prime_base", 0))) + "}",
        "\\providecommand{\\PadBenchBattleCount}{" + f"{int(snapshot.get('battle_count', 0)):,}" + "}",
        "\\providecommand{\\PadBenchBestAblationStrategy}{" + _latex_escape(str(narrative.get("best_ablation_strategy") or "")) + "}",
        "\\providecommand{\\PadBenchBestAblationMeanLoss}{" + _format_float(narrative.get("best_ablation_mean_padic_loss"), 6) + "}",
        "\\providecommand{\\PadBenchBestAblationDeltaVsBattle}{" + _format_float(narrative.get("best_ablation_delta_vs_baseline"), 6) + "}",
        "\\providecommand{\\PadBenchBattleMeanLoss}{" + _format_float(narrative.get("battle_elo_mean_padic_loss"), 6) + "}",
        "\\providecommand{\\PadBenchUMLLRMeanLoss}{" + _format_float(narrative.get("umllr_mean_padic_loss"), 6) + "}",
        "\\providecommand{\\PadBenchUMLLRParams}{" + _format_intish(narrative.get("umllr_mean_params")) + "}",
        "\\providecommand{\\PadBenchUMLLRScoringOps}{" + _format_float(narrative.get("umllr_mean_scoring_ops"), 2) + "}",
        "\\providecommand{\\PadBenchLevelwiseScoringOps}{" + _format_float(narrative.get("levelwise_mean_scoring_ops"), 2) + "}",
    ]
    return "\n".join(lines) + "\n"


def render_model_comparison_tex(bundle: dict[str, Any]) -> str:
    lines = [
        "% Generated by padjective.benchmark_bundle",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "\\makecell{\\textbf{Model}} & \\makecell{\\textbf{Non-zero}\\\\\\textbf{params.}} & \\makecell{\\textbf{Avg active}\\\\\\textbf{params.}} & \\makecell{\\textbf{Mean $p$-adic}\\\\\\textbf{loss}} \\\\ % Avg active params.",
        "\\midrule",
    ]
    for row in model_rows_frame(bundle).to_dict("records"):
        lines.append(
            f"{_latex_escape(str(row['model_label']))} & "
            f"{_format_float(row.get('params'), 1)} & "
            f"{_format_float(row.get('mean_scoring_ops'), 2)} & "
            f"{_format_float(row.get('mean_padic_loss'), 6)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def render_umllr_ablation_tex(bundle: dict[str, Any]) -> str:
    lines = [
        "% Generated by padjective.benchmark_bundle",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Strategy} & \\textbf{Mean loss} & \\textbf{Fold SD} & $\\Delta$ vs battle & \\textbf{Wins} & \\textbf{Exact acc.} & \\textbf{Prefix-2 acc.} \\\\",
        "\\midrule",
    ]
    for row in ablation_strategy_frame(bundle).to_dict("records"):
        lines.append(
            f"{_latex_escape(str(row['tag_order_strategy']))} & "
            f"{_format_float(row.get('mean_padic_loss'), 6)} & "
            f"{_format_float(row.get('loss_std'), 6)} & "
            f"{_format_float(row.get('loss_delta_vs_baseline'), 6)} & "
            f"{int(row.get('wins_vs_baseline', 0))}/{int(row.get('comparisons_vs_baseline', 0))} & "
            f"{_format_percent_tex(row.get('mean_exact_accuracy'))} & "
            f"{_format_percent_tex(row.get('mean_prefix2_accuracy'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def write_bundle_outputs(bundle: dict[str, Any], out_dir: str | Path) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark.json").write_text(_json_dump(bundle), encoding="utf-8")

    model_frame = model_rows_frame(bundle).copy()
    if "folds" in model_frame.columns:
        model_frame = model_frame.drop(columns=["folds"])
    model_frame.to_csv(root / "model_comparison.csv", index=False)

    run_frame = ablation_strategy_frame(bundle).copy()
    if "folds" in run_frame.columns:
        run_frame = run_frame.drop(columns=["folds"])
    run_frame.to_csv(root / "umllr_ablation.csv", index=False)

    fragments_dir = root / "fragments"
    fragments_dir.mkdir(parents=True, exist_ok=True)
    (fragments_dir / "model_comparison.html").write_text(
        render_model_comparison_html(bundle),
        encoding="utf-8",
    )
    (fragments_dir / "umllr_ablation.html").write_text(
        render_ablation_html(bundle),
        encoding="utf-8",
    )


def write_paper_tex_outputs(bundle: dict[str, Any], out_dir: str | Path) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark_numbers.tex").write_text(
        render_benchmark_numbers_tex(bundle),
        encoding="utf-8",
    )
    (root / "model_comparison_table.tex").write_text(
        render_model_comparison_tex(bundle),
        encoding="utf-8",
    )
    (root / "umllr_ablation_table.tex").write_text(
        render_umllr_ablation_tex(bundle),
        encoding="utf-8",
    )


def _build_live_model_row(
    *,
    model_key: str,
    model_label: str,
    short_label: str,
    color: str,
    marker: str,
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    fold_rows = []
    for row in fold_results:
        fold_rows.append(
            {
                "cv_fold": int(row["cv_fold"]),
                "padic_loss_mean": float(row.get("padic_loss_mean") or row.get("mean_loss") or row.get("loss") or 0.0),
                "accuracy": float(row["test_accuracy"]) if row.get("test_accuracy") is not None else float(row.get("accuracy") or row.get("exact_accuracy") or 0.0),
                "exact_accuracy": float(row.get("exact_accuracy") if row.get("exact_accuracy") is not None else row.get("test_accuracy") or row.get("accuracy") or 0.0),
                "prefix1_accuracy": float(row["prefix1_accuracy"]) if row.get("prefix1_accuracy") is not None else None,
                "prefix2_accuracy": float(row["prefix2_accuracy"]) if row.get("prefix2_accuracy") is not None else None,
                "mean_shared_prefix_depth": float(row["mean_shared_prefix_depth"]) if row.get("mean_shared_prefix_depth") is not None else None,
                "mean_scoring_ops": float(row["mean_scoring_ops"]) if row.get("mean_scoring_ops") is not None else None,
                "num_params": float(
                    row.get("num_params")
                    or row.get("num_nonzero_params")
                    or row.get("effective_params")
                    or row.get("num_nonzero_coefficients")
                    or 0.0
                ),
                "num_train_samples": int(row.get("num_train_samples") or 0),
                "num_test_samples": int(row.get("num_test_samples") or 0),
            }
        )
    return {
        "model_key": model_key,
        "model_label": model_label,
        "short_label": short_label,
        "color": color,
        "marker": marker,
        "params": sum(row["num_params"] for row in fold_rows) / len(fold_rows) if fold_rows else 0.0,
        "mean_padic_loss": sum(row["padic_loss_mean"] for row in fold_rows) / len(fold_rows) if fold_rows else 0.0,
        "mean_accuracy": _mean([row.get("accuracy") for row in fold_rows]),
        "mean_exact_accuracy": _mean([row.get("exact_accuracy") for row in fold_rows]),
        "mean_prefix1_accuracy": _mean([row.get("prefix1_accuracy") for row in fold_rows]),
        "mean_prefix2_accuracy": _mean([row.get("prefix2_accuracy") for row in fold_rows]),
        "mean_shared_prefix_depth": _mean([row.get("mean_shared_prefix_depth") for row in fold_rows]),
        "mean_scoring_ops": _mean([row.get("mean_scoring_ops") for row in fold_rows]),
        "loss_std": _std([row.get("padic_loss_mean") for row in fold_rows]),
        "folds": fold_rows,
    }


def _mean(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def _std(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    series = pd.Series(numeric, dtype=float)
    return float(series.std(ddof=0))


def _add_parsimony_columns(
    model_rows: list[dict[str, Any]],
    *,
    num_taxonomies: int,
) -> list[dict[str, Any]]:
    slope = -0.1
    intercept = 0.0
    taxonomy_coefficient = 0.3
    taxonomy_reference = 1000.0
    safe_taxonomies = max(float(num_taxonomies), 1.0)
    taxonomy_adjustment = taxonomy_coefficient * math.log10(
        safe_taxonomies / taxonomy_reference
    )
    for row in model_rows:
        params = max(float(row.get("params") or 0.0), 1.0)
        loss = max(float(row.get("mean_padic_loss") or 0.0), 1e-12)
        log10_params = math.log10(params)
        log10_loss = math.log10(loss)
        baseline_log10_loss = slope * log10_params + intercept + taxonomy_adjustment
        row["log10_params"] = log10_params
        row["log10_loss"] = log10_loss
        row["baseline_log10_loss"] = baseline_log10_loss
        row["parsimony_score"] = baseline_log10_loss - log10_loss
    return model_rows


def build_live_benchmark_bundle(
    *,
    dsn: str | None,
    schema: str,
    product_table: str,
    snapshot_ref: str | None = None,
    snapshot_schema: str = "padjective",
    ablation_snapshot_ref: str | None = "live",
) -> dict[str, Any]:
    from . import build_site

    conn = db.get_connection(dsn)
    try:
        dataset = data_access.build_feature_dataset(
            conn,
            product_table=product_table,
            require_taxonomy=True,
            min_tag_count=5,
            min_samples_per_taxonomy=5,
            snapshot_ref=snapshot_ref,
            snapshot_schema=snapshot_schema,
        )
        dummy_results = build_site._load_dummy_results(conn, schema=schema) or {"metrics": []}
        umllr_results = build_site._load_umllr_results(conn, schema=schema) or {"metrics": []}
        ablation = build_site._load_umllr_order_ablation_results(
            conn,
            schema=schema,
            snapshot_ref=ablation_snapshot_ref,
        ) or {"runs": [], "random_summary": None}
        pclr = build_site._load_taxonomy_pclr_fold_results(conn, schema=schema) or []
        pcnn = build_site._load_taxonomy_pcnn_fold_results(conn, schema=schema) or []
        ulr = build_site._load_taxonomy_ulr_fold_results(conn, schema=schema) or []
        levelwise = build_site._load_taxonomy_levelwise_fold_results(conn, schema=schema) or []
        unn = build_site._load_taxonomy_unn_fold_results(conn, schema=schema) or []
        dt = build_site._load_taxonomy_dt_fold_results(conn, schema=schema) or []
        zubarev = build_site._load_zubarev_fold_results(
            conn,
            schema=schema,
            initialization_method="umllr",
            mahler_degree=0,
        ) or []
    finally:
        conn.close()

    model_rows = [
        _build_live_model_row(
            model_key="dummy",
            model_label="Dummy Baseline",
            short_label="Dummy",
            color="#94a3b8",
            marker="X",
            fold_results=[
                {
                    "cv_fold": row["cv_fold"],
                    "padic_loss_mean": row.get("loss"),
                    "accuracy": row.get("accuracy"),
                    "num_params": 1.0,
                }
                for row in dummy_results.get("metrics", [])
            ],
        ),
        _build_live_model_row(
            model_key="umllr",
            model_label="Importance-Optimised p-adic Linear Regression",
            short_label="Importance-Optimised",
            color="#0b6ce3",
            marker="o",
            fold_results=umllr_results.get("metrics", []),
        ),
        _build_live_model_row(
            model_key="pclr",
            model_label="Parameter-constrained Logistic Regression",
            short_label="Constr. Logit",
            color="#2563eb",
            marker="s",
            fold_results=pclr,
        ),
        _build_live_model_row(
            model_key="ulr",
            model_label="Unconstrained Logistic Regression with L1",
            short_label="L1 Logit",
            color="#8b5cf6",
            marker="D",
            fold_results=ulr,
        ),
        _build_live_model_row(
            model_key="pcnn",
            model_label="Parameter-constrained Neural Network",
            short_label="Constr. NN",
            color="#16a34a",
            marker="P",
            fold_results=pcnn,
        ),
        _build_live_model_row(
            model_key="dt",
            model_label="Decision Tree",
            short_label="Decision Tree",
            color="#14b8a6",
            marker="h",
            fold_results=dt,
        ),
        _build_live_model_row(
            model_key="levelwise",
            model_label="Level-wise Logistic Regression",
            short_label="Level-wise",
            color="#f97316",
            marker="^",
            fold_results=levelwise,
        ),
        _build_live_model_row(
            model_key="unn",
            model_label="Unconstrained Neural Network with L1",
            short_label="Unconstr. NN",
            color="#ec4899",
            marker="p",
            fold_results=unn,
        ),
        _build_live_model_row(
            model_key="zubarev",
            model_label="Zubarev (greedy init.)",
            short_label="Zubarev",
            color="#7c3aed",
            marker="v",
            fold_results=zubarev,
        ),
    ]
    model_rows = _add_parsimony_columns(
        model_rows,
        num_taxonomies=int(dataset.taxonomy_count),
    )

    strategy_rows = []
    baseline_mean = None
    baseline_folds: dict[int, float] = {}
    for run_row in ablation.get("runs", []):
        if run_row["tag_order_strategy"] == "battle_elo" and run_row.get("tag_order_seed") is None:
            baseline_mean = float(run_row["mean_loss"])
            baseline_folds = {
                int(fold["cv_fold"]): float(fold["mean_loss"])
                for fold in run_row.get("folds", [])
                if fold.get("mean_loss") is not None
            }
            break
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run_row in ablation.get("runs", []):
        grouped.setdefault(str(run_row["tag_order_strategy"]), []).append(run_row)
    for strategy, rows in grouped.items():
        comparable_losses: list[tuple[float, float]] = []
        exact_accuracies: list[float] = []
        prefix2_accuracies: list[float] = []
        scoring_ops_values: list[float] = []
        for run_row in rows:
            for fold in run_row.get("folds", []):
                fold_index = int(fold["cv_fold"])
                fold_loss = fold.get("mean_loss")
                baseline_loss = baseline_folds.get(fold_index)
                if fold_loss is not None and baseline_loss is not None:
                    comparable_losses.append((float(fold_loss), float(baseline_loss)))
                if fold.get("exact_accuracy") is not None:
                    exact_accuracies.append(float(fold["exact_accuracy"]))
                if fold.get("prefix2_accuracy") is not None:
                    prefix2_accuracies.append(float(fold["prefix2_accuracy"]))
                if fold.get("mean_scoring_ops") is not None:
                    scoring_ops_values.append(float(fold["mean_scoring_ops"]))
        wins_vs_baseline = sum(1 for fold_loss, baseline_loss in comparable_losses if fold_loss < baseline_loss)
        comparisons_vs_baseline = len(comparable_losses)
        if strategy == "random":
            mean_loss = ablation.get("random_summary", {}).get("mean_loss")
            strategy_rows.append(
                {
                    "tag_order_strategy": strategy,
                    "run_key": "random",
                    "mean_padic_loss": mean_loss,
                    "loss_std": _std([pair[0] for pair in comparable_losses]),
                    "loss_delta_vs_baseline": (mean_loss - baseline_mean) if mean_loss is not None and baseline_mean is not None else None,
                    "mean_exact_accuracy": _mean(exact_accuracies),
                    "mean_prefix2_accuracy": _mean(prefix2_accuracies),
                    "mean_scoring_ops": _mean(scoring_ops_values),
                    "wins_vs_baseline": wins_vs_baseline,
                    "comparisons_vs_baseline": comparisons_vs_baseline,
                    "folds": [],
                }
            )
        else:
            row = rows[0]
            strategy_rows.append(
                {
                    "tag_order_strategy": strategy,
                    "run_key": row["run_key"],
                    "mean_padic_loss": row.get("mean_loss"),
                    "loss_std": _std(
                        [fold.get("mean_loss") for fold in row.get("folds", [])]
                    ),
                    "loss_delta_vs_baseline": (row.get("mean_loss") - baseline_mean) if row.get("mean_loss") is not None and baseline_mean is not None else None,
                    "mean_exact_accuracy": _mean(exact_accuracies),
                    "mean_prefix2_accuracy": _mean(prefix2_accuracies),
                    "mean_scoring_ops": _mean(scoring_ops_values),
                    "wins_vs_baseline": wins_vs_baseline,
                    "comparisons_vs_baseline": comparisons_vs_baseline,
                    "folds": row.get("folds", []),
                }
            )
    strategy_rows = sorted(
        strategy_rows,
        key=lambda row: (float("inf") if row.get("mean_padic_loss") is None else float(row["mean_padic_loss"]), row["tag_order_strategy"]),
    )

    return {
        "bundle_version": 1,
        "source": "live_db",
        "snapshot": {
            "label": snapshot_ref or "live",
            "snapshot_name": snapshot_ref or "live",
            "as_of": None,
            "product_count_all": int(dataset.product_count),
            "product_count_filtered": int(dataset.product_count),
            "tag_count_all": int(len(dataset.feature_names) + len(dataset.discarded_tags)),
            "tag_count_filtered": int(len(dataset.feature_names)),
            "taxonomy_count_all": int(dataset.taxonomy_count),
            "taxonomy_count_filtered": int(dataset.taxonomy_count),
            "prime_base": int(umllr_results.get("metrics", [{}])[0].get("prime_base") or 0) if umllr_results.get("metrics") else 0,
            "max_digit": int(umllr_results.get("metrics", [{}])[0].get("max_digit") or 0) if umllr_results.get("metrics") else 0,
            "battle_count": None,
            "folds": [int(row["cv_fold"]) for row in umllr_results.get("metrics", [])],
        },
        "models": {"rows": model_rows},
        "ablation": {
            "baseline_strategy": "battle_elo",
            "baseline_mean_padic_loss": baseline_mean,
            "runs": ablation.get("runs", []),
            "strategy_rows": strategy_rows,
            "random_summary": ablation.get("random_summary"),
            "best_strategy": strategy_rows[0]["tag_order_strategy"] if strategy_rows else None,
            "best_mean_padic_loss": strategy_rows[0].get("mean_padic_loss") if strategy_rows else None,
            "best_delta_vs_baseline": strategy_rows[0].get("loss_delta_vs_baseline") if strategy_rows else None,
        },
        "narrative": {
            "best_ablation_strategy": strategy_rows[0]["tag_order_strategy"] if strategy_rows else None,
            "best_ablation_mean_padic_loss": strategy_rows[0].get("mean_padic_loss") if strategy_rows else None,
            "best_ablation_delta_vs_baseline": strategy_rows[0].get("loss_delta_vs_baseline") if strategy_rows else None,
            "battle_elo_mean_padic_loss": baseline_mean,
            "umllr_mean_padic_loss": next((row["mean_padic_loss"] for row in model_rows if row["model_key"] == "umllr"), None),
            "umllr_mean_params": next((row["params"] for row in model_rows if row["model_key"] == "umllr"), None),
            "umllr_mean_scoring_ops": next((row["mean_scoring_ops"] for row in model_rows if row["model_key"] == "umllr"), None),
            "levelwise_mean_scoring_ops": next((row["mean_scoring_ops"] for row in model_rows if row["model_key"] == "levelwise"), None),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSON/CSV/HTML/TeX benchmark bundles."
    )
    parser.add_argument("--snapshot-dir", type=Path, help="Local exported snapshot directory to analyse.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for benchmark artifacts.")
    parser.add_argument("--dsn", help="Postgres DSN for live bundle generation.")
    parser.add_argument("--schema", default="padjective", help="Schema for live model tables.")
    parser.add_argument("--product-table", default="cantbuymelove.product", help="Qualified product table.")
    parser.add_argument("--snapshot-ref", help="Optional snapshot ref for live dataset stats.")
    parser.add_argument("--snapshot-schema", default="padjective", help="Schema containing benchmark snapshot tables.")
    parser.add_argument("--ablation-snapshot-ref", default="live", help="Snapshot label for live ablation rows.")
    parser.add_argument("--paper-tex-dir", type=Path, help="Optional output dir for TeX includes.")
    args = parser.parse_args()

    if args.snapshot_dir:
        bundle = build_snapshot_benchmark_bundle(
            load_snapshot_tables(args.snapshot_dir),
        )
    else:
        bundle = build_live_benchmark_bundle(
            dsn=args.dsn,
            schema=args.schema,
            product_table=args.product_table,
            snapshot_ref=args.snapshot_ref,
            snapshot_schema=args.snapshot_schema,
            ablation_snapshot_ref=args.ablation_snapshot_ref,
        )

    write_bundle_outputs(bundle, args.out_dir)
    if args.paper_tex_dir:
        write_paper_tex_outputs(bundle, args.paper_tex_dir)


if __name__ == "__main__":
    main()
