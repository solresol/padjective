"""Apply the paper's minimum-valuation medoid to archived low-q coefficients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from . import db, low_q_linear
from . import paper_low_q_experiment as refit
from . import paper_matched_store_folds as matched
from . import paper_sum_oracle as archive
from . import paper_valuation_ensembles as valuation
from .paper_valuation_selector import EXPERIMENT as ONE_PASS_REFERENCE
from .randomised_linear import apply_default, consensus_predictions, linear_predictions
from .subspace_voting import feature_mask

EXPERIMENT = "paper-low-q-valuation-2026-11-09"
WEIGHTS = ("71", "sqrt2", "2")
FRACTIONS = (.75, 1.)
MEMBERS = 243
P, E = valuation.P, valuation.E
METRICS = ("loss_q2", "root_accuracy", "exact_accuracy")


def saved_fits(conn, kind, family, fold, fraction):
    rows = conn.execute("""
        SELECT member_index, evidence - 'history' - 'certificate' - 'scores'
        FROM padjective.paper_low_q_models
        WHERE experiment_key=%s AND q_label=%s AND family=%s
          AND cv_fold=%s AND fraction=%s AND status=%s
    """, (refit.EXPERIMENT, kind, family, fold, fraction,
          "coordinate_optimum" if family == "coordinate" else "one_pass_complete"))
    fits = dict(rows)
    if any(item["snapshot_digest"] != refit.SNAPSHOT_DIGEST for item in fits.values()):
        raise ValueError("Saved coefficients use a different snapshot")
    return fits


def check_sum(targets, predictions, control):
    if (refit.fingerprint(predictions.tolist()) != control["prediction_sha256"]
            or abs(matched.SCORE(targets, predictions.tolist())["loss_q2"]
                   - control["losses"]["2"]) > 1e-12):
        raise ValueError("Sum failed to reproduce the archived prediction")


def one_pass(conn, dataset, fold, context, kind, reference, selector_reference):
    fit = saved_fits(conn, kind, "greedy_taxonomy_association", fold, 1.)[-1]
    weights = fit["coefficients"]
    if kind == "71":
        by_tag = {term["tag"]: term["coefficient"] for term in weights}
        weights = [by_tag.get(tag, 0) for tag in dataset.feature_names]
    test, targets = context["x_test"], context["y_test"]
    summed = apply_default(np.asarray(test @ weights).ravel(), fit["default"])
    if summed.tolist() != fit["predictions"]:
        raise ValueError("One-pass sum differs from archived predictions")
    selected = valuation.select_rows(test, weights, fit["default"])
    result = {key: matched.SCORE(targets, pred.tolist())
              for key, pred in (("sum", summed), ("medoid", selected))}
    if abs(result["sum"]["loss_q2"] - reference["losses"]["2"]) > 1e-12:
        raise ValueError("One-pass sum differs from the archived score")
    if kind == "71" and result["medoid"] != selector_reference:
        raise ValueError("One-pass medoid differs from the published control")
    return result


def repeated(conn, fold, context, fraction, kind, reference):
    if kind == "71":
        sum_score, selected_score = conn.execute("""
            SELECT evidence #> '{variants,sum,prefixes,-1}',
                   evidence #> '{variants,selector,prefixes,-1}'
            FROM padjective.paper_store_held_out_runs
            WHERE experiment_key=%s AND fold=%s AND model=%s
        """, (valuation.EXPERIMENT, fold, f"product-repeated-{fraction:g}")).fetchone()
        if (abs(sum_score["loss_q2"] - reference["losses"]["2"]) > 1e-12
                or sum_score["prediction_sha256"] != reference["prediction_sha256"]):
            raise ValueError("Original ensemble scores differ across archives")
        return dict(sum=sum_score, medoid=selected_score)

    members, = conn.execute("""
        SELECT evidence->'member_indices' FROM padjective.paper_subspace_results
        WHERE batch_id=%s AND cv_fold=%s AND fraction=%s AND roster=0
          AND members=%s AND rule='raw_valid_medoid'
    """, (refit.SOURCE_BATCH, fold, fraction, MEMBERS)).fetchone()
    fits = saved_fits(conn, kind, "coordinate", fold, fraction)
    if len(members) != MEMBERS or len(fits) != MEMBERS or set(members) != fits.keys():
        raise ValueError("Incomplete ensemble roster")

    test = context["x_test"]
    supported = np.flatnonzero(context["x_train"].getnnz(axis=0))
    sums, medoids = [], []
    for member in members:
        fit = fits[member]
        seed = refit.BANK_SEEDS[member]
        mask = (np.arange(test.shape[1]) if fraction == 1. else
                feature_mask(supported, fraction, fold, seed))
        if fit["seed"] != seed + fold or fit["mask_sha256"] != refit.fingerprint(mask.tolist()):
            raise ValueError("Member mask or seed differs from the fitting archive")
        masked = test[:, mask]
        weights, default = fit["coefficients"], fit["default"]
        summed = apply_default(linear_predictions(masked, weights, P, E), default)
        if summed.tolist() != fit["predictions"]:
            raise ValueError("Member sum differs from the fitting archive")
        sums.append(summed)
        medoids.append(valuation.select_rows(masked, weights, default))

    sums, medoids = np.asarray(sums).T, np.asarray(medoids).T
    candidates, targets = context["y_train"], context["y_test"]
    matched_sum = low_q_linear.consensus_predictions(sums, candidates, kind=kind)
    check_sum(targets, matched_sum, reference)
    result = {}
    for rule, bank in (("sum", sums), ("medoid", medoids)):
        predictions, _ = consensus_predictions(bank, candidates, p=P, precision=E)
        result[rule] = matched.SCORE(targets, predictions.tolist())
    result["sum_with_fit_matched_consensus"] = matched.SCORE(targets, matched_sum.tolist())
    return result


def summarise(folds):
    return {rule: dict(
        **{metric: mean(fold[rule][metric] for fold in folds) for metric in METRICS},
        loss_q2_sd=stdev(fold[rule]["loss_q2"] for fold in folds),
    ) for rule in folds[0]}


def audit(conn):
    dataset = refit._load_paper_dataset(conn, snapshot_ref=refit.PAPER_SNAPSHOT, schema="padjective")
    reference = archive.archived_report(conn, refit.EXPERIMENT)
    selector = archive.archived_report(conn, ONE_PASS_REFERENCE)
    if (refit._snapshot_digest(dataset) != refit.SNAPSHOT_DIGEST
            or reference["snapshot_digest"] != refit.SNAPSHOT_DIGEST
            or selector["product_fold_digest"] != refit.SNAPSHOT_DIGEST):
        raise ValueError("Archived controls differ from the frozen product folds")
    baseline = {(r["configuration"], r["training_q"]): r["folds"] for r in reference["series"]}
    contexts = matched.contexts(dataset)
    series = []
    for kind in WEIGHTS:
        model = "taxonomy_association"
        folds = [one_pass(conn, dataset, f, c, kind, baseline[model, kind][f],
                          selector["splits"]["product"]["folds"][f]["selected"])
                 for f, c in contexts.items()]
        series.append(dict(model=model, training_q=kind, scores=summarise(folds), folds=folds))
        for fraction in FRACTIONS:
            model = f"ensemble_{fraction:g}_{MEMBERS}"
            folds = [repeated(conn, f, c, fraction, kind, baseline[model, kind][f])
                     for f, c in contexts.items()]
            series.append(dict(model=model, training_q=kind, scores=summarise(folds), folds=folds))
            print(f"Scored {model} trained at {kind}", flush=True)
    return dict(
        experiment_key=EXPERIMENT, snapshot_id=refit.PAPER_SNAPSHOT,
        snapshot_digest=refit.SNAPSHOT_DIGEST, fitting_experiment=refit.EXPERIMENT,
        references=dict(one_pass=ONE_PASS_REFERENCE, ensembles=valuation.EXPERIMENT),
        valuation_prime=P, evaluation_q=2, medoid_q=P, consensus_q=P,
        protocol="Replay saved fits and defaults on the product folds. Hold the medoid "
                 "and consensus weights fixed at 71 across fitting weights. For low-q "
                 "ensembles, reproduce the published sum using fit-matched consensus as a control.",
        series=series,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with db.get_connection() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        report = audit(conn)
    archive.save_report(report)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
