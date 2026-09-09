"""Analyse and plot the predeclared ensemble scaling follow-up (aggregate data)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

from .paper_ensemble_scaling_batch import SIZES

TARGETS = {"NN-2000": 0.0755591194166452, "L1 logistic": 0.08583859907596855}
LABELS = {"dummy": "Dummy", "umllr": "Greedy p-adic", "ulr": "L1 logistic",
          "dt": "Decision tree", "levelwise": "Level-wise logistic", "unn": "NN-2000"}


def power_curve(m, a, b):
    return a*np.asarray(m, dtype=float)**(-b)


def floor_curve(m, c, a, b):
    return c+power_curve(m, a, b)


def crossing(parameters, target):
    """Conditional continuous member count; None means no finite crossing."""
    c, a, b = parameters["floor"], parameters["amplitude"], parameters["exponent"]
    if c >= target:
        return None
    if a == 0 or c+a <= target:
        return 1.0
    log_m = math.log(a/(target-c))/b
    return math.exp(log_m) if log_m < 700 else None


def fit_curve(sizes, losses, with_floor):
    x, y = np.asarray(sizes, dtype=float), np.asarray(losses, dtype=float)
    assert len(x) == len(y) and len(x) >= (3 if with_floor else 2)
    assert np.all(x >= 1) and np.all(y > 0)
    function = floor_curve if with_floor else power_curve
    fits = []
    for b in (.15, .5, 1.0):
        initial = [y[-1]*.8, max(y[0]-y[-1]*.8, .001), b] if with_floor else [y[0], b]
        lower = [0, 0, 1e-8] if with_floor else [0, 1e-8]
        parameters, _ = curve_fit(function, x, y, p0=initial, bounds=(lower, np.inf),
                                  maxfev=30000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        predicted = function(x, *parameters)
        fits.append((float(np.square(y-predicted).sum()), parameters, predicted))
    sse, p, predicted = min(fits, key=lambda r: r[0])
    c, a, b = p if with_floor else (0, *p)
    result = dict(form="floor_plus_power" if with_floor else "power_to_zero", floor=float(c),
        amplitude=float(a), exponent=float(b), sse=sse, rmse=math.sqrt(sse/len(x)),
        fit_sizes=x.tolist(), observed_loss=y.tolist(), predicted_loss=predicted.tolist(),
        residuals=(y-predicted).tolist())
    result["target_crossings"] = {name: crossing(result, target) for name, target in TARGETS.items()}
    return result


def evaluate(fit, sizes):
    return floor_curve(sizes, fit["floor"], fit["amplitude"], fit["exponent"])


def ols(points):
    x, y = np.log10([r["active"] for r in points]), np.log10([r["loss"] for r in points])
    result = linregress(x, y)
    return dict(n=len(points), slope=float(result.slope), intercept=float(result.intercept),
        r_squared=float(result.rvalue**2), p_value=float(result.pvalue),
        slope_stderr=float(result.stderr), inference="nominal configuration-level OLS; one shared dataset")


def references(aggregate, baseline_directory):
    paths = [baseline_directory/"validation-single-thread-models.json", baseline_directory/"validated-neural-2000.json"]
    bundle, neural = [json.loads(p.read_text()) for p in paths]
    wanted = {"dummy", "umllr", "ulr", "dt", "levelwise"}
    selected = [r for r in bundle["models"]["rows"] if r["model_key"] in wanted]
    assert len(selected) == 5 and len(neural) == 5 and {r["cv_fold"] for r in neural} == set(range(5))
    points = [dict(key=r["model_key"], label=LABELS[r["model_key"]], active=r["mean_scoring_ops"],
        loss=r["mean_padic_loss"], stored=r["params"], exact_accuracy=r["mean_exact_accuracy"],
        fold_losses=[v["padic_loss_mean"] for v in sorted(r["folds"], key=lambda v:v["cv_fold"])]) for r in selected]
    assert all(r["hidden_units"] == 2000 for r in neural)
    params = statistics.fmean(r["num_params"] for r in neural)
    active = params-2000*(aggregate["feature_count"]-aggregate["mean_input_active_features"])
    points.append(dict(key="unn", label=LABELS["unn"], active=active, stored=params,
        loss=statistics.fmean(r["mean_loss"] for r in neural),
        exact_accuracy=statistics.fmean(r["exact_accuracy"] for r in neural),
        fold_losses=[r["mean_loss"] for r in sorted(neural, key=lambda r:r["cv_fold"])],
        source_note="Original width-2000 reference, 12 BLAS threads; not the single-thread diagnostic"))
    assert abs(points[-1]["loss"]-TARGETS["NN-2000"]) < 1e-14
    fit = ols(points)
    assert abs(fit["slope"]-(-.1300)) < .0001 and abs(fit["r_squared"]-.696) < .001
    provenance = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    return points, provenance


def analyse(aggregate, points):
    summaries, rows = aggregate["summaries"], aggregate["ensembles"]
    primary = [next(r for r in summaries if r["roster"] == 0 and r["members"] == size) for size in SIZES]
    losses = np.array([r["mean"]["mean_padic_loss"] for r in primary])
    fold_losses = np.array([[next(r["metrics"]["mean_padic_loss"] for r in rows
        if r["fold"] == fold and r["roster"] == 0 and r["members"] == size) for size in SIZES] for fold in range(5)])
    assert np.allclose(losses, fold_losses.mean(axis=0), atol=1e-14, rtol=0)
    permutation_curves = np.array([[next(r["mean"]["mean_padic_loss"] for r in summaries
        if r["roster"] == roster and r["members"] == size) for size in SIZES] for roster in range(1,21)])
    curves = []
    for name, values in (("primary", losses), ("alternative_roster_mean", permutation_curves.mean(axis=0))):
        for with_floor in (False, True):
            full = fit_curve(SIZES, values, with_floor)
            check = fit_curve(SIZES[:-1], values[:-1], with_floor)
            early = fit_curve(SIZES[:3], values[:3], with_floor)
            check["forecast_243"] = float(evaluate(check, [243])[0])
            check["observed_243"] = float(values[-1])
            check["forecast_error_observed_minus_predicted"] = float(values[-1]-check["forecast_243"])
            curves.append(dict(series=name, full=full, through_135_check=check, early_1_3_9=early))
    leave_one_fold_out = []
    for omitted in range(5):
        values = np.delete(fold_losses, omitted, axis=0).mean(axis=0)
        for with_floor in (False, True):
            leave_one_fold_out.append(dict(omitted_fold=omitted, fit=fit_curve(SIZES, values, with_floor)))
    frozen = ols(points)
    overlays, all_size_refits = [], []
    for convention in ("mean_member_terms_consulted", "broader_scoring_proxy"):
        all_points = []
        for row in primary:
            point = dict(active=row[convention], loss=row["mean"]["mean_padic_loss"])
            predicted = 10**frozen["intercept"]*point["active"]**frozen["slope"]
            overlays.append(dict(convention=convention, members=row["members"], **point,
                frozen_line_predicted_loss=predicted, observed_to_line_ratio=point["loss"]/predicted,
                log10_residual=math.log10(point["loss"]/predicted), original_plus_one=ols(points+[point])))
            all_points.append(point)
        refit = ols(points+all_points)
        refit["inference"] = "Mechanical fit only: nested ensemble sizes are dependent, so this p-value is not an independent-observation significance test"
        all_size_refits.append(dict(convention=convention, fit=refit))
    comparisons = []
    for point in points:
        delta = fold_losses[:, -1]-point["fold_losses"]
        comparisons.append(dict(comparator=point["label"], target_loss=point["loss"],
            largest_ensemble_loss=float(losses[-1]), largest_to_comparator_ratio=float(losses[-1]/point["loss"]),
            fold_differences=delta.tolist(), lower_loss_folds=int((delta < 0).sum()), tied_folds=int((delta == 0).sum())))
    changes = []
    for i in range(1, len(SIZES)):
        delta = fold_losses[:,i]-fold_losses[:,i-1]
        changes.append(dict(from_members=SIZES[i-1], to_members=SIZES[i],
            loss_change=float(losses[i]-losses[i-1]), fold_changes=delta.tolist(), improved_folds=int((delta < 0).sum())))
    return dict(primary=primary, primary_fold_losses=fold_losses.tolist(), alternative_roster_mean=permutation_curves.mean(axis=0).tolist(),
        alternative_roster_min=permutation_curves.min(axis=0).tolist(), alternative_roster_max=permutation_curves.max(axis=0).tolist(),
        alternative_roster_curves=permutation_curves.tolist(), changes=changes, curves=curves,
        leave_one_fold_out=leave_one_fold_out, reference_points=points, original_ols=frozen,
        overlays=overlays, all_size_refits=all_size_refits, largest_comparisons=comparisons,
        interpretation_limits=["Exploratory extension on the same five folds; training folds overlap",
            "Roster ranges condition on one fitted bank, and collapse at 243 by construction; they are not confidence intervals",
            "Crossings are conditional extrapolations, not established achievable performance",
            "Broader counters mix coefficient consultations with prefix work; neither axis is a runtime measurement",
            "The original six-model roster is unchanged; matched-budget models remain excluded"])


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=.6)
    ax.set_axisbelow(True)


def plot_results(result, directory):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10,
                         "savefig.dpi": 180, "font.family": "DejaVu Sans"})
    blue, grey = "#1768ac", "#666e78"
    x = np.array(SIZES)
    y = np.array([r["mean"]["mean_padic_loss"] for r in result["primary"]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.suptitle("Larger random-coordinate ensembles: measured gains and extrapolation", x=.07, ha="left", fontsize=15)
    fig.text(.07, .91, "6,693 products · five frozen folds · lower p-adic loss is better · all 243 members fitted per fold", color=grey)
    ax = axes[0]
    for i, fold in enumerate(result["primary_fold_losses"]):
        ax.plot(x, fold, color="#c1c7ce", lw=.8, label="Individual folds" if i == 0 else None)
    ax.fill_between(x, result["alternative_roster_min"], result["alternative_roster_max"],
        color=blue, alpha=.13, label="20 alternative rosters: min–max")
    ax.plot(x, result["alternative_roster_mean"], color=blue, lw=1, ls=":", label="Alternative-roster mean")
    ax.plot(x, y, "o-", color=blue, lw=2, ms=4, label="Fixed primary roster")
    for name, target in TARGETS.items():
        ax.axhline(target, color=grey, ls="--" if name == "NN-2000" else ":", lw=1)
        ax.annotate(f"{name} {target:.4f}", (1.08, target), xytext=(0, 4), textcoords="offset points", color=grey, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(x, [str(v) for v in x], rotation=45)
    ax.set(xlabel="Ensemble members (log scale)", ylabel="Mean held-out p-adic loss", ylim=(.055, .34), title="Observed sizes")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax = axes[1]
    fits = [r["full"] for r in result["curves"] if r["series"] == "primary"]
    future = np.geomspace(243, 100000, 200)
    for fit, color, label in zip(fits, [grey, blue], ["Power law forced to zero", "Power law with fitted floor"]):
        ax.plot(future, evaluate(fit, future), ls="--", color=color, lw=1.6, label=label)
        if fit["floor"]:
            ax.axhline(fit["floor"], color=color, lw=.8, alpha=.45)
    ax.scatter([243], [y[-1]], color=blue, s=35, zorder=3, label="Measured 243-member loss")
    ax.axhline(TARGETS["NN-2000"], color="#333333", ls=":", lw=1, label="NN-2000 target")
    ax.set_xscale("log")
    ax.set(xlabel="Members beyond the measured bank (log scale)", ylabel="Mean held-out p-adic loss", ylim=(0, max(.16, y[-1]*1.2)), title="Conditional forecasts — not measured")
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    for ax in axes:
        style_axes(ax)
    fig.text(.07, .02, "Roster band is conditional on this bank, not a confidence interval; its spread is zero at 243 because membership is identical.", fontsize=8, color=grey)
    fig.subplots_adjust(left=.07, right=.98, bottom=.20, top=.80, wspace=.27)
    for extension in ("png", "svg"):
        fig.savefig(directory/f"ensemble-size-loss.{extension}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    fig.suptitle("Ensembles against the original six-model log–log relationship", x=.07, ha="left", fontsize=15)
    fig.text(.07, .91, "Frozen original regression; ensemble sizes are one dependent model family, not independent replications", color=grey)
    for ax, convention, title in zip(axes, ("mean_member_terms_consulted", "broader_scoring_proxy"),
            ("Member coefficient consultations only", "Members + defaults + consensus-prefix proxy")):
        points = result["reference_points"]
        ax.scatter([r["active"] for r in points], [r["loss"] for r in points], color=grey, s=30, zorder=3)
        offsets = {"dummy": (8, 0), "umllr": (7, -4), "ulr": (-4, -15), "dt": (5, 1),
                   "levelwise": (-90, 9), "unn": (-56, -14)}
        for point in points:
            ax.annotate(point["label"], (point["active"], point["loss"]), xytext=offsets[point["key"]],
                        textcoords="offset points", fontsize=8, color=grey)
        fit = result["original_ols"]
        line_x = np.geomspace(.8, 2e6, 300)
        ax.plot(line_x, 10**fit["intercept"]*line_x**fit["slope"], color=grey, ls="--", lw=1.2, label="Original six-model line")
        rows = [r for r in result["overlays"] if r["convention"] == convention]
        ax.plot([r["active"] for r in rows], [r["loss"] for r in rows], "o-", color=blue, ms=4, lw=1.5, label="Ensemble trajectory")
        for row in rows:
            if row["members"] in (1,9,27,243):
                shift = (5, 7) if convention == "mean_member_terms_consulted" else (6, 4 if row["members"] == 1 else -5)
                ax.annotate(str(row["members"]), (row["active"],row["loss"]), xytext=shift, textcoords="offset points", color=blue, fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlabel="Mean per-prediction count (log scale)", title=title, xlim=(.7, 2e6), ylim=(.045,.7))
        ax.set_yticks([.05,.075,.1,.15,.2,.3,.5], [".05",".075",".10",".15",".20",".30",".50"])
        style_axes(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    axes[0].set_ylabel("Mean held-out p-adic loss (log scale)")
    fig.text(.07, .045, "Numbers label member counts. Right-hand counts mix heterogeneous operations; neither panel measures runtime.", fontsize=8, color=grey)
    fig.text(.07, .02, f"Original: slope {result['original_ols']['slope']:.4f}, R² {result['original_ols']['r_squared']:.3f}, nominal p = {result['original_ols']['p_value']:.3f}. No matched-budget rows added.", fontsize=8, color=grey)
    fig.subplots_adjust(left=.07, right=.98, bottom=.18, top=.80, wspace=.10)
    for extension in ("png", "svg"):
        fig.savefig(directory/f"ensemble-log-log.{extension}")
    plt.close(fig)


def persist_analysis(conn, result):
    """Save this new analysis without touching any earlier experiment report."""
    from psycopg.types.json import Jsonb
    with conn.cursor() as cur:
        cur.execute("SELECT report FROM padjective.paper_ensemble_scaling_metric_audit WHERE batch_id=%s", (result["batch_id"],))
        audit = cur.fetchone()
        assert audit and audit[0]["status"] == "passed" and audit[0]["aggregate_sha256"] == result["aggregate_sha256"]
        cur.execute("SET LOCAL default_tablespace='pg_default'")
        cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_ensemble_scaling_analysis (
            batch_id UUID PRIMARY KEY, analysed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            report JSONB NOT NULL) TABLESPACE pg_default""")
        cur.execute("INSERT INTO padjective.paper_ensemble_scaling_analysis (batch_id,report) VALUES (%s,%s)",
                    (result["batch_id"], Jsonb(result)))
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline-directory", type=Path, default=Path("build/paper-submission-2026-09-06/hf-stage/submission/2026-09-06"))
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--persist", action="store_true", help="Persist the newly computed analysis to Shopify Postgres")
    args = parser.parse_args()
    aggregate = json.loads(args.results.read_text())
    assert aggregate["validation"]["status"] == "passed" and aggregate["validation"]["members"] == 1215
    points, provenance = references(aggregate, args.baseline_directory)
    result = analyse(aggregate, points)
    result.update(batch_id=aggregate["batch_id"], snapshot_id=aggregate["snapshot_id"],
        snapshot_digest=aggregate["snapshot_digest"], source_provenance=provenance,
        aggregate_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(),
        analysis_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    plot_results(result, args.output_directory)
    if args.persist:
        from . import db
        with db.get_connection() as conn:
            persist_analysis(conn, result)
    print(json.dumps({key:result[key] for key in ("original_ols", "primary", "curves", "overlays")}, indent=2))


if __name__ == "__main__":
    main()
