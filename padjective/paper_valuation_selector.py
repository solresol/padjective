"""Test selection by lowest p-adic valuation on the frozen one-pass fits."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from math import fsum
from pathlib import Path
from statistics import mean

from . import paper_matched_store_folds as matched
from . import paper_sum_oracle as oracle
from .paper_revision_examples import valuation

EXPERIMENT = "paper-valuation-selector-2026-11-09"
AUGMENTATION = "paper-valuation-augmentation-2026-11-09"


def select(active, prime):
    """Choose a medoid of the lowest-valuation terms, retaining multiplicities."""
    valued = [(valuation(value, prime_base=prime), value) for value in active]
    lowest = min(depth for depth, _ in valued)
    tied = [value for depth, value in valued if depth == lowest]
    selected = min(set(tied), key=lambda candidate: (
        sum(Fraction(1, prime**valuation(candidate - value, prime_base=prime))
            for value in tied if value != candidate),
        candidate,
    ))
    return selected, len(tied)


def fold_report(dataset, fold, reference_scores, score_predictions):
    default, original_scores, rows = oracle.term_rows(dataset, fold, reference_scores, score_predictions)
    targets, selected = [], []
    multi_selected_losses, multi_original_losses = [], []
    tied_valuation = oracle_matches = 0
    for target, active, original in rows:
        prediction, tied = select(active, matched.PRIME) if active else (default, 0)
        targets.append(target)
        selected.append(prediction)
        if len(active) < 2:
            continue
        tied_valuation += tied > 1
        selected_loss = oracle.DISTANCE(target, prediction)
        multi_selected_losses.append(selected_loss)
        multi_original_losses.append(oracle.DISTANCE(target, original))
        oracle_matches += selected_loss == min(oracle.DISTANCE(target, value) for value in active)

    # The product and store archives use different score schemas.
    original_loss = (original_scores["losses"]["2"] if "losses" in original_scores
                     else original_scores["loss_q2"])
    comparison = oracle.compare_losses(multi_selected_losses, multi_original_losses, len(rows))
    return dict(fold=fold, products=len(rows), multi_term=len(multi_selected_losses),
                multi_term_tied_valuation=tied_valuation, multi_term_oracle_loss_matches=oracle_matches,
                original=dict(loss_q2=original_loss, root_accuracy=original_scores["root_accuracy"],
                              exact_accuracy=original_scores["exact_accuracy"],
                              prediction_sha256=original_scores["prediction_sha256"]),
                selected=matched.SCORE(targets, selected),
                multi_term_outcomes=comparison["outcomes"],
                multi_term_selected_loss_total=comparison["sum_loss_total"],
                multi_term_original_loss_total=comparison["best_single_loss_total"])


def summarise(folds):
    products = sum(row["products"] for row in folds)
    multi = sum(row["multi_term"] for row in folds)
    outcomes = Counter()
    for row in folds:
        outcomes.update(row["multi_term_outcomes"])
    metrics = ("loss_q2", "root_accuracy", "exact_accuracy")
    return dict(products=products, multi_term=multi, multi_term_share=multi / products,
                multi_term_tied_valuation=sum(r["multi_term_tied_valuation"] for r in folds),
                multi_term_oracle_loss_matches=sum(r["multi_term_oracle_loss_matches"] for r in folds),
                multi_term_outcomes=dict(outcomes),
                multi_term_selected_mean_loss=sum(r["multi_term_selected_loss_total"] for r in folds) / multi,
                multi_term_original_mean_loss=sum(r["multi_term_original_loss_total"] for r in folds) / multi,
                original={key: mean(r["original"][key] for r in folds) for key in metrics},
                selected={key: mean(r["selected"][key] for r in folds) for key in metrics}, folds=folds)


def run(source_commit):
    provenance, inputs = oracle.load_splits()
    splits = {
        name: summarise([fold_report(dataset, f, scores[f], score) for f in matched.FOLDS])
        for name, (dataset, scores, score) in inputs.items()
    }
    return oracle.save_report(dict(
        experiment_key=EXPERIMENT, status="validated", source_commit=source_commit, order=oracle.ORDER,
        valuation_prime=matched.PRIME, evaluation_q=2, fitting_q=matched.PRIME, medoid_q=matched.PRIME,
        references=dict(product=oracle.PRODUCT_REFERENCE, store=matched.EXPERIMENT),
        protocol="Among nonzero fitted terms, keep those of minimum valuation and select their exact "
                 "p-adic medoid from the retained terms, breaking ties by signed integer order. "
                 "The original fold default applies to products with no active term. Scores use raw "
                 "integer outputs and the original equal-fold averaging.",
        **provenance, splits=splits))


def append_higher(active, selected, prime):
    """Add each active term divisible by the next power above the selected valuation."""
    modulus = prime**(valuation(selected, prime_base=prime) + 1)
    higher = [value for value in active if value % modulus == 0]
    return selected + sum(higher), len(higher)


def augmentation_fold(dataset, fold, reference_scores, scoring, archived):
    default, original, rows = oracle.term_rows(dataset, fold, reference_scores, scoring)
    targets, selected, augmented = [], [], []
    before, after = [], []
    changed = 0
    for target, active, _ in rows:
        prediction, _ = select(active, matched.PRIME) if active else (default, 0)
        extended, count = append_higher(active, prediction, matched.PRIME) if active else (prediction, 0)
        targets.append(target)
        selected.append(prediction)
        augmented.append(extended)
        if count:
            before.append(oracle.DISTANCE(target, prediction))
            after.append(oracle.DISTANCE(target, extended))
            changed += extended != prediction

    selector = matched.SCORE(targets, selected)
    result = matched.SCORE(targets, augmented)
    if selector != archived["selected"] or original["prediction_sha256"] != archived["original"]["prediction_sha256"]:
        raise ValueError("Selector or original sum differs from the previous audit")
    if any((a - b) % matched.PRIME for a, b in zip(selected, augmented, strict=True)):
        raise ValueError("Higher terms changed a root digit")
    outcomes = Counter("beats" if a < b else "loses" if a > b else "ties"
                       for a, b in zip(after, before, strict=True))
    return dict(fold=fold, products=len(rows), eligible=len(before), changed=changed,
                outcomes={k: outcomes[k] for k in ("beats", "ties", "loses")},
                eligible_selector_loss_total=fsum(before), eligible_augmented_loss_total=fsum(after),
                original=archived["original"], selector=selector, augmented=result)


def summarise_augmentation(folds):
    products = sum(r["products"] for r in folds)
    eligible = sum(r["eligible"] for r in folds)
    metrics = ("loss_q2", "root_accuracy", "exact_accuracy")
    return dict(products=products, eligible=eligible, eligible_share=eligible / products,
                changed=sum(r["changed"] for r in folds),
                outcomes={k: sum(r["outcomes"][k] for r in folds) for k in ("beats", "ties", "loses")},
                eligible_selector_mean_loss=fsum(r["eligible_selector_loss_total"] for r in folds) / eligible
                if eligible else None,
                eligible_augmented_mean_loss=fsum(r["eligible_augmented_loss_total"] for r in folds) / eligible
                if eligible else None,
                **{key: {metric: mean(r[key][metric] for r in folds) for metric in metrics}
                   for key in ("original", "selector", "augmented")}, folds=folds)


def run_augmentation(source_commit):
    provenance, inputs = oracle.load_splits()
    with oracle.db.get_connection() as conn:
        conn.read_only = True
        selector = oracle.archived_report(conn, EXPERIMENT)
    if any(selector[k] != v for k, v in provenance.items()):
        raise ValueError("The selector audit used different data")
    splits = {}
    for name, (dataset, scores, scoring) in inputs.items():
        archived = {r["fold"]: r for r in selector["splits"][name]["folds"]}
        splits[name] = summarise_augmentation([
            augmentation_fold(dataset, f, scores[f], scoring, archived[f]) for f in matched.FOLDS
        ])
    return oracle.save_report(dict(
        experiment_key=AUGMENTATION, status="validated", source_commit=source_commit, order=oracle.ORDER,
        valuation_prime=matched.PRIME, evaluation_q=2, fitting_q=matched.PRIME, medoid_q=matched.PRIME,
        references=dict(product=oracle.PRODUCT_REFERENCE, store=matched.EXPERIMENT, selector=EXPERIMENT),
        protocol="Select the lowest-valuation medoid, then add every active coefficient with strictly "
                 "higher valuation once per tag. Keep the original default for products with no active terms.",
        **provenance, splits=splits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--add-higher", action="store_true", help="add the higher-valuation terms after selection")
    args = parser.parse_args()
    report = run_augmentation(args.source_commit) if args.add_higher else run(args.source_commit)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
