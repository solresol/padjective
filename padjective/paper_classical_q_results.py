"""Combine aggregate classical scores with the completed native low-Q study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .low_q_results import LABELS as NATIVE_LABELS


LABELS = dict(NATIVE_LABELS, dummy="Dummy control", ulr="L1 logistic regression",
              dt="Decision tree", levelwise="Level-wise logistic regression",
              unn="Neural network (width 2,000)")
DISPLAY = ("unn", "ulr", "dt", "levelwise", "ensemble_0.75_243",
           "ensemble_1_243", "taxonomy_association", "dummy")


def combine(native, classical):
    for key in ("snapshot_id", "snapshot_digest", "products", "prime", "q_values"):
        if native[key] != classical[key]:
            raise ValueError(f"Reports disagree on {key}")
    index = {(s["training_q"], s["configuration"]): s for s in native["series"]}
    baseline = {s["configuration"]: s for s in classical["series"]}
    native_configs = {k for t, k in index if t == "71"}
    if len(native_configs) != 11 or set(baseline) != {"dummy", "ulr", "dt", "levelwise", "unn"}:
        raise ValueError("Incomplete model roster")
    settings = [("original", "71", "71"), ("fixed_sqrt2", "71", "sqrt2"),
                ("retrained_sqrt2", "sqrt2", "sqrt2"),
                ("fixed_q2", "71", "2"), ("retrained_q2", "2", "2")]
    comparisons = {}
    for name, training_q, scoring_q in settings:
        selected = {k: index[training_q, k] for k in native_configs} | baseline
        records = []
        for rank, key in enumerate(sorted(selected, key=lambda k: selected[k]["mean_losses"][scoring_q]), 1):
            s = selected[key]
            if [f["n"] for f in s["folds"]] != [1300, 1337, 1348, 1372, 1336]:
                raise ValueError("Fold populations differ")
            records.append(dict(
                rank=rank, configuration=key, label=LABELS[key],
                training_q=s["training_q"], scoring_q=scoring_q,
                mean_loss=s["mean_losses"][scoring_q], fold_sd=s["fold_sd"][scoring_q],
                fold_losses=[f["losses"][scoring_q] for f in s["folds"]],
                exact_accuracy=s["exact_accuracy"], root_accuracy=s["root_accuracy"],
            ))
        comparisons[name] = records
    return dict(
        native_experiment=native["experiment_key"], classical_experiment=classical["experiment_key"],
        snapshot_id=native["snapshot_id"], snapshot_digest=native["snapshot_digest"],
        products=native["products"], prime=71, comparisons=comparisons,
        interpretation="Classical training objectives unchanged. Fixed comparisons change only Q in "
                       "evaluation. Retrained comparisons also use Q-specific native fits and consensus. "
                       "Ranks use fold-mean point estimates; no significance claim.",
    )


def render(report, output):
    names = ("original", "fixed_sqrt2", "retrained_sqrt2")
    columns = [{r["configuration"]: r for r in report["comparisons"][name]} for name in names]
    # Show the main cross-family comparison; retain all 16 configurations in
    # rankings.json and README.md. Ranks in this figure refer to these eight.
    ranks = [{k: i + 1 for i, k in enumerate(sorted(DISPLAY, key=lambda k: c[k]["mean_loss"]))}
             for c in columns]
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    fig.subplots_adjust(left=.32, right=.96, top=.79, bottom=.19)
    for i, key in enumerate(DISPLAY):
        ys = [c[key] for c in ranks]
        ax.plot([0, 1, 2], ys, color="black" if key in ("levelwise", "ensemble_0.75_243", "ensemble_1_243") else ".6",
                marker=["o", "s", "^", "D", "v", "P", "X", "*"][i], linewidth=1.4, markersize=6)
        ax.text(-.07, ys[0], LABELS[key], ha="right", va="center", fontsize=10)
    ax.set_xticks([0, 1, 2], ["Original fits\nscore Q = 71", "Original fits\nscore Q = √2",
                            "Native fits retrained at √2\nall models scored at √2"])
    ax.set_yticks(range(1, 9))
    ax.yaxis.tick_right()
    ax.set_ylim(8.45, .55)
    ax.set_xlim(-.015, 2.12)
    ax.grid(axis="y", color=".9", linewidth=.7)
    for edge in ("top", "left", "bottom"):
        ax.spines[edge].set_visible(False)
    fig.suptitle("Does a lower depth weight change the cross-family ranking?", fontsize=15, x=.51, y=.96)
    fig.text(.5, .875, "Eight main configurations · five frozen folds · lower mean loss ranks higher",
             ha="center", fontsize=10)
    fig.text(.02, .055, "Classical objectives stay unchanged. Positions use fold-mean point estimates; small gaps are uncertain.\n"
             "The complete 16-configuration comparison, including Q = 2, is in the accompanying results.", fontsize=9)
    fig.savefig(output / "ranking.png", dpi=180)
    plt.close(fig)


def write_readme(report, output):
    comparisons = report["comparisons"]
    lookups = {name: {r["configuration"]: r for r in rows} for name, rows in comparisons.items()}
    lines = ["# Classical models scored with the same low Q", "",
             "Five fixed paper folds, 6,693 products. Lower loss is better. Classical training "
             "objectives are unchanged. Their predictions were recovered with the archived "
             "settings and environment, checking each fold against its original Q=71 loss and exact accuracy.", "",
             "## Main comparison", "",
             "Low-Q columns below use the Q-specific retrained native models from the preceding "
             "study; classical fits stay unchanged. These are fold means ± sample fold SDs. "
             "Exact accuracy refers to the sqrt(2) comparison.", "",
             "| Model | Q=71 | Q=√2 | Q=2 | Exact accuracy |", "|---|---:|---:|---:|---:|"]
    for key in DISPLAY:
        cells = [lookups[n][key] for n in ("original", "retrained_sqrt2", "retrained_q2")]
        losses = " | ".join(f"{c['mean_loss']:.3f} ± {c['fold_sd']:.3f}" for c in cells)
        lines.append(f"| {LABELS[key]} | {losses} | {cells[1]['exact_accuracy']:.1%} |")
    lines += ["", "## Complete rankings", "",
              "Point-estimate ranks are descriptive, without a significance claim. The same "
              "folds have already been used for other experiments. Fixed-fit rows isolate "
              "rescoring; retrained rows additionally change native fitting and consensus.", ""]
    for name, rows in comparisons.items():
        lines += [f"### {name}", "", "| Rank | Model | Mean loss | Fold SD | Exact accuracy |",
                  "|---:|---|---:|---:|---:|"]
        lines += [f"| {r['rank']} | {r['label']} | {r['mean_loss']:.6f} | {r['fold_sd']:.6f} | {r['exact_accuracy']:.2%} |"
                  for r in rows]
        lines.append("")
    lines += ["## Evidence", "",
              "- `results.json`: classical fold scores, histograms, prediction hashes and provenance; no product rows.",
              "- `rankings.json`: all five comparison arms with paired fold losses.",
              "- `reference.json`: frozen environment and archived fold aggregates, with source SHA-256 hashes.",
              "- `protocol.md`: fixed training and evaluation settings.",
              "- `validation.json`: independent PostgreSQL reconstruction and export checks.",
              "- `../low-q-sensitivity-2026-11-02/results.json`: unchanged native comparison.", "",
              "Private recovered predictions are in `padjective.paper_classical_q_runs`; the aggregate "
              "classical report is in `padjective.paper_low_q_reports`. No manuscript text was changed.", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = combine(json.loads(args.native.read_text()), json.loads(args.classical.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rankings.json").write_text(json.dumps(report, indent=2) + "\n")
    render(report, args.output)
    write_readme(report, args.output)


if __name__ == "__main__":
    main()
