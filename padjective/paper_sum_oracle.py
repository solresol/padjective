"""Audit the one-pass taxonomy-association sum against its best active term."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
import json
from math import fsum
from pathlib import Path
from statistics import mean

from psycopg.types.json import Jsonb

from . import benchmark_runtime as runtime, db
from . import paper_matched_store_folds as matched
from .paper_q2_primary import EXPERIMENT as PRODUCT_REFERENCE
from .paper_revision_experiments import q_weighted_distance
from .q_loss_rescore import rescore

EXPERIMENT = "paper-sum-oracle-2026-11-09"
ORDER = "taxonomy_association"
DISTANCE = partial(q_weighted_distance, prime_base=matched.PRIME, q=2)


def fold_report(dataset, fold, reference_scores, score_predictions):
    fit = runtime.umllr_run_fold(fold, dataset.records, [], matched.PRIME, tag_order_strategy=ORDER)
    held_out = [r for r in dataset.records if r.cv_fold == fold]
    predicted = [p.predicted_value for p in fit.predictions]
    scores = score_predictions([r.encoded_path for r in held_out], predicted)
    scores["first_disagreement_counts"] = {
        str(depth): count for depth, count in scores["first_disagreement_counts"].items()
    }
    if scores != reference_scores:
        raise ValueError(f"One-pass predictions do not reproduce the archived fold {fold}")

    coefficients = {c.tag: c.coefficient for c in fit.coefficients if c.coefficient}
    active_counts, outcomes = Counter(), Counter()
    sum_losses, best_losses = [], []
    cancelled = 0
    for record, prediction in zip(held_out, fit.predictions, strict=True):
        active = [coefficients[tag] for tag in record.tags if tag in coefficients]
        total = sum(active)
        if (record.product_id, record.encoded_path, total or fit.default_prediction) != (
            prediction.product_id, prediction.true_value, prediction.predicted_value
        ):
            raise ValueError("Could not reconstruct the original one-pass prediction")
        active_counts[len(active)] += 1
        if len(active) < 2:
            continue
        cancelled += total == 0
        sum_loss = DISTANCE(record.encoded_path, total)
        best_loss = min(DISTANCE(record.encoded_path, value) for value in active)
        sum_losses.append(sum_loss)
        best_losses.append(best_loss)
        outcome = "beats" if sum_loss < best_loss else "loses" if sum_loss > best_loss else "ties"
        outcomes[outcome] += 1
    multi = len(sum_losses)
    if not multi:
        raise ValueError(f"No multi-term products in fold {fold}")
    sum_total, best_total = fsum(sum_losses), fsum(best_losses)
    return dict(fold=fold, products=len(held_out), multi_term=multi, cancelled_multi_term=cancelled,
                active_count_histogram={str(k): v for k, v in sorted(active_counts.items())},
                outcomes={k: outcomes[k] for k in ("beats", "ties", "loses")},
                sum_loss_total=sum_total, best_single_loss_total=best_total,
                sum_mean_loss=sum_total / multi, best_single_mean_loss=best_total / multi,
                sum_minus_single_mean_loss=(sum_total - best_total) / multi,
                full_fold_loss_difference=(sum_total - best_total) / len(held_out),
                original_prediction_sha256=scores["prediction_sha256"])


def summarise(folds):
    products = sum(r["products"] for r in folds)
    multi = sum(r["multi_term"] for r in folds)
    outcomes = {k: sum(r["outcomes"][k] for r in folds) for k in ("beats", "ties", "loses")}
    sum_total = fsum(r["sum_loss_total"] for r in folds)
    best_total = fsum(r["best_single_loss_total"] for r in folds)
    return dict(products=products, multi_term=multi, multi_term_share=multi / products,
                cancelled_multi_term=sum(r["cancelled_multi_term"] for r in folds),
                outcomes=outcomes, outcome_shares={k: v / multi for k, v in outcomes.items()},
                pooled_sum_mean_loss=sum_total / multi, pooled_best_single_mean_loss=best_total / multi,
                pooled_sum_minus_single_mean_loss=(sum_total - best_total) / multi,
                equal_fold_sum_mean_loss=mean(r["sum_mean_loss"] for r in folds),
                equal_fold_best_single_mean_loss=mean(r["best_single_mean_loss"] for r in folds),
                equal_fold_sum_minus_single_mean_loss=mean(r["sum_minus_single_mean_loss"] for r in folds),
                equal_fold_full_benchmark_loss_difference=mean(r["full_fold_loss_difference"] for r in folds),
                folds=folds)


def archived_report(conn, key):
    row = conn.execute("SELECT report FROM padjective.paper_low_q_reports WHERE experiment_key=%s",
                       (key,)).fetchone()
    if row is None:
        raise ValueError(f"Missing archived report {key}")
    return row[0]


def run(source_commit):
    with db.get_connection() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        product = matched._load_paper_dataset(conn, snapshot_ref=matched.PAPER_SNAPSHOT, schema="padjective")
        store, _, provenance = matched.load_data(conn)
        product_reference = archived_report(conn, PRODUCT_REFERENCE)
        store_reference = archived_report(conn, matched.EXPERIMENT)
    if (matched._snapshot_digest(product) != product_reference["snapshot_digest"]
            or any(store_reference["manifest"][k] != v for k, v in provenance.items())):
        raise ValueError("The benchmark or store assignments differ from the archived experiments")
    product_scores = {r["fold"]: r["scores"] for r in product_reference["greedy"] if r["strategy"] == ORDER}
    store_scores = {r["fold"]: r["fits"][0]["scores"] for model in store_reference["models"]
                    if model["model"] == ORDER for r in model["folds"]}
    product_scoring = partial(rescore, q_values={"2": 2., "71": 71.})
    folds = dict(
        product=[fold_report(product, f, product_scores[f], product_scoring) for f in matched.FOLDS],
        store=[fold_report(store, f, store_scores[f], matched.SCORE) for f in matched.FOLDS],
    )
    report = dict(experiment_key=EXPERIMENT, status="validated", source_commit=source_commit, order=ORDER,
                  valuation_prime=matched.PRIME, evaluation_q=2, fitting_q=matched.PRIME,
                  references=dict(product=PRODUCT_REFERENCE, store=matched.EXPERIMENT),
                  protocol="Compare raw sums with the best raw nonzero coefficient on products with two or more "
                           "active terms. Full-benchmark differences leave other predictions unchanged.",
                  **provenance, splits={name: summarise(rows) for name, rows in folds.items()})
    with db.get_connection() as conn:
        conn.execute("""
            INSERT INTO padjective.paper_low_q_reports(experiment_key,report) VALUES (%s,%s)
            ON CONFLICT (experiment_key) DO NOTHING
        """, (EXPERIMENT, Jsonb(report)))
        if archived_report(conn, EXPERIMENT) != report:
            raise ValueError("Oracle report failed Postgres readback")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.source_commit), indent=2) + "\n")


if __name__ == "__main__":
    main()
