"""Plot fixed-prediction loss versus Q from the matched PostgreSQL aggregates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter


EXPERIMENT = "q-loss-curves-2026-11-02"
FOLD_SIZES = [1300, 1337, 1348, 1372, 1336]
# The main comparison roster, retaining both 243-member feature fractions.
STYLES = {
    "dummy": ("Dummy control", "black", "X", ":"),
    "taxonomy_association": ("p-adic greedy", "black", "P", "-"),
    "ensemble_0.75_243": ("p-adic ensemble: 243, 75% features", "black", "^", "-."),
    "ensemble_1_243": ("p-adic ensemble: 243, all features", "black", "v", "--"),
    "unn": ("Neural network, width 2,000", "black", "o", "-"),
    "ulr": ("L1 logistic regression", "black", "s", "--"),
    "dt": ("Decision tree", "black", "D", "-."),
    "levelwise": ("Level-wise logistic regression", "black", "h", ":"),
}
CAPTION = (
    "Mean held-out $d_{71,Q}$ loss versus $Q$ on logarithmic axes. "
    "Fixed predictions across five frozen paper folds (6,693 products). "
    "Encoding base stays 71. Only the scoring weight $Q$ changes. "
    "Markers show the calculated $Q$ values. "
    "Lines connect scored points; no power law is fitted"
)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def build_data(native, classical):
    for key in ("snapshot_id", "snapshot_digest", "products", "prime", "q_values"):
        if native[key] != classical[key]:
            raise ValueError(f"Source reports disagree on {key}")
    if native["prime"] != 71:
        raise ValueError("The scoring valuation base must remain 71")
    if native["products"] != sum(FOLD_SIZES):
        raise ValueError("Reported product count differs from the frozen fold sizes")
    grid = sorted(native["q_values"].items(), key=lambda item: item[1])
    if len({v for _, v in grid}) != len(grid) or any(not math.isfinite(q) or q <= 1 for _, q in grid):
        raise ValueError("Q values must be distinct finite numbers greater than one")
    selected = {s["configuration"]: s for s in native["series"] if s["training_q"] == "71"}
    selected.update({s["configuration"]: s for s in classical["series"]})
    curves = []
    for key, (label, *_style) in STYLES.items():
        s = selected[key]
        if [f["n"] for f in s["folds"]] != FOLD_SIZES:
            raise ValueError("Fold populations differ from the frozen paper benchmark")
        if s["training_q"] not in ("71", "classification_objective_unchanged"):
            raise ValueError("A fixed-prediction curve cannot mix retrained models")
        points = []
        for q_label, q in grid:
            values = [f["losses"][q_label] for f in s["folds"]]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("Losses must be finite values between zero and one")
            loss, sd = mean(values), stdev(values)
            if abs(loss - s["mean_losses"][q_label]) > 1e-12 or abs(sd - s["fold_sd"][q_label]) > 1e-12:
                raise ValueError("Stored aggregates disagree with the five fold scores")
            if loss - sd <= 0:
                raise ValueError("The logarithmic SD band requires a positive lower bound")
            points.append(dict(q_label=q_label, q=q, mean_loss=loss, sample_fold_sd=sd,
                               log10_q=math.log10(q), log10_mean_loss=math.log10(loss),
                               log10_loss_minus_sd=math.log10(loss - sd),
                               log10_loss_plus_sd=math.log10(loss + sd), fold_losses=values))
        if any(a["mean_loss"] < b["mean_loss"] - 1e-12 for a, b in zip(points, points[1:])):
            raise ValueError("Fixed-prediction loss cannot increase as Q increases")
        curves.append(dict(configuration=key, label=label, training_q=s["training_q"], points=points))
    return dict(
        experiment_key=EXPERIMENT, snapshot_id=native["snapshot_id"],
        snapshot_digest=native["snapshot_digest"], products=native["products"],
        fold_sizes=FOLD_SIZES, valuation_prime=71, q_values=dict(grid),
        horizontal_axis="Q, shown on a logarithmic scale",
        vertical_axis="equal-weight mean of fold mean d_71,Q losses, shown on a logarithmic scale",
        metric="d_71,Q(a,b) = 0 for equality, otherwise Q ** (-v_71(a-b))",
        uncertainty="plus/minus one sample fold SD in loss units, transformed at its endpoints; not a confidence interval",
        training="All predictions held fixed across Q. Native fits trained at Q=71; classical objectives unchanged.",
        sources=[dict(experiment_key=r["experiment_key"], canonical_sha256=canonical_sha256(r))
                 for r in (native, classical)],
        curves=curves,
    )


def render(data, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10.8, 7.1))
    fig.subplots_adjust(left=.10, right=.97, top=.90, bottom=.29)
    for curve in data["curves"]:
        label, color, marker, line = STYLES[curve["configuration"]]
        ps = curve["points"]
        x = [p["q"] for p in ps]
        y = [p["mean_loss"] for p in ps]
        marker_face = "white" if curve["configuration"] in {"unn", "ulr", "dt", "levelwise"} else color
        ax.plot(x, y, color=color, marker=marker, linestyle=line, linewidth=1.5,
                markersize=5, markerfacecolor=marker_face, markeredgewidth=1.1,
                label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    grid = list(data["q_values"].values())
    ax.xaxis.set_major_locator(FixedLocator(grid))
    ax.set_xticklabels(["√2" if label == "sqrt2" else label for label in data["q_values"]])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([.075, .1, .15, .2, .3, .5, .75])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(1.3, 82)
    ax.set_ylim(.065, .88)
    ax.set_xlabel("Depth weight Q (log scale)", labelpad=10)
    ax.set_ylabel(r"Mean $d_{71,Q}$ loss (log scale)", labelpad=10)
    ax.grid(axis="y", which="major", color=".88", linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.21), ncol=2,
              frameon=False, fontsize=10, columnspacing=2.8, handlelength=3.2)
    fig.suptitle("Loss sensitivity to the hierarchy depth weight", y=.965, fontsize=17)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"q-loss-log-log.{suffix}", dpi=180)
    plt.close(fig)


def write_readme(data, output):
    lines = ["# Fixed-prediction Q sensitivity", "",
             "Both axes are logarithmic: Q versus the equal-weight mean held-out d_71,Q loss. "
             "The taxonomy encoding and valuation remain base 71.", "",
             "The older appendix already reports Q = 2, 3, 5, 10 and 71 for greedy and ensemble "
             "models. The matched 2 November results add sqrt(2) and the dummy/classical "
             "models on all six Q values. No new training or rescoring run was needed for this figure.", "",
             "The curves use the original Q=71 native fits and unchanged classical objectives "
             "throughout. The separate native retraining experiments at sqrt(2) and 2 are not "
             "mixed into these curves. Markers identify scored values; connecting segments "
             "guide the eye rather than assert a power law. Sample fold SDs are reported in the table below.", "",
             "The figure uses black lines with distinct markers and dash patterns. "
             "Experimental context is in the separate caption, without "
             "a subtitle or explanatory footer inside the figure.", "",
             "## Caption", "", CAPTION, "",
             "`caption.md` contains the caption text; `caption.tex` provides the LaTeX caption command.", "",
             "## Scores", "",
             "| Model | " + " | ".join(f"Q={'√2' if q == 'sqrt2' else q}" for q in data["q_values"]) + " |",
             "|---|" + "---:|" * len(data["q_values"])]
    for curve in data["curves"]:
        values = " | ".join(f"{p['mean_loss']:.3f} ± {p['sample_fold_sd']:.3f}" for p in curve["points"])
        lines.append(f"| {curve['label']} | {values} |")
    lines += ["", "## Evidence and reproduction", "",
              "`data.json` contains all 48 plotted points, their five fold scores, log10 "
              "coordinates, SD interval endpoints and source hashes. Source aggregate files:", "",
              "- `../low-q-sensitivity-2026-11-02/results.json` (native fits).",
              "- `../low-q-classical-2026-11-02/results.json` (dummy and classical fits).", "",
              "All curves use the same 6,693 products and fold sizes 1300, 1337, 1348, 1372, 1336. "
              "No product-level records or predictions are included here. The source experiments "
              "and their prediction checks are documented in their own directories.", "",
              "`validation.json` records independent checks of all 48 plotted points and 15 "
              "overlapping score cells from the older appendix, the source/PostgreSQL checks, "
              "unit tests, and visual checks of the PNG and grayscale-rendered PDF. The plotted "
              "data and validation are also stored in `padjective.paper_low_q_reports` under "
              f"`{EXPERIMENT}` and `{EXPERIMENT}-validation`.", "",
              "```sh", "uv run python -m padjective.q_loss_curves \\",
              "  --native docs/low-q-sensitivity-2026-11-02/results.json \\",
              "  --classical docs/low-q-classical-2026-11-02/results.json \\",
              "  --output docs/q-loss-curves-2026-11-02", "```", ""]
    (output / "README.md").write_text("\n".join(lines))
    (output / "caption.md").write_text(CAPTION + "\n")
    (output / "caption.tex").write_text("\\caption{" + CAPTION.replace("6,693", "6{,}693") + "}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = build_data(json.loads(args.native.read_text()), json.loads(args.classical.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data.json").write_text(json.dumps(data, indent=2) + "\n")
    render(data, args.output)
    write_readme(data, args.output)


if __name__ == "__main__":
    main()
