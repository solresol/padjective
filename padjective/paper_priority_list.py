"""Fit priority lists, a max-aggregated alternative to summed tag coefficients.

A priority list predicts with one coefficient: that of the highest-priority
tag present. For a fixed tag order the best coefficient of each tag is the
medoid of the labels of the products it heads, so one sweep is exact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import heapq
import json
from math import fsum
from pathlib import Path
from statistics import mean, stdev

from . import benchmark_runtime as runtime, db
from . import paper_matched_store_folds as matched
from . import paper_sum_oracle as oracle
from . import paper_valuation_orders as valuation_orders
from . import paper_valuation_selector as selector
from .paper_store_held_out_greedy import ORDERS
from .subspace_voting import corrected_test, holm

EXPERIMENT = "paper-priority-list-2026-11-23-v2"
FIT_Q = (71, 2)
SMOOTHING = (0, 1, 4)
DEPTH_CAP = 16
BEST_STATIC = "taxonomy_association"
VARIANTS = ("cell", "cell_backfill", "pooled_priority")
METRICS = ("loss_q2", "loss_q71", "root_accuracy", "exact_accuracy", "coverage",
           "covered_loss_q2", "default_loss_q2")


def medoid_cost(counts, *, prime, q):
    """Medoid of a multiset of nonnegative codes and its summed distance scaled by q**DEPTH_CAP.

    Two codes first differ at digit k exactly when they agree modulo prime**k
    but not modulo prime**(k+1), so residue counts replace pairwise distances.
    Ties go to the smallest integer.
    """
    if not isinstance(q, int) or q <= 1:
        raise ValueError("Exact medoids need an integer q greater than one")
    if not counts:
        raise ValueError("A medoid needs at least one value")
    if min(counts) < 0:
        raise ValueError("Codes must be nonnegative")
    levels, modulus, top = [], 1, max(counts)
    while modulus <= top:
        modulus *= prime
        level = Counter()
        for value, multiplicity in counts.items():
            level[value % modulus] += multiplicity
        levels.append((modulus, level, q ** (DEPTH_CAP - len(levels))))
    if len(levels) > DEPTH_CAP:
        raise ValueError("Codes exceed the supported depth")
    total = sum(counts.values())
    best = best_cost = None
    for candidate in sorted(counts):
        cost, sharing = 0, total
        for modulus, level, weight in levels:
            deeper = level[candidate % modulus]
            cost += (sharing - deeper) * weight
            sharing = deeper
        if best_cost is None or cost < best_cost:
            best, best_cost = candidate, cost
    return best, best_cost


def medoid(values, *, prime, q):
    """Exact minimiser of the summed ``q``-weighted distance; smallest integer on ties."""
    return medoid_cost(Counter(int(value) for value in values), prime=prime, q=q)[0]


def head(tags, rank, coefficients):
    """The highest-priority tag that has a coefficient, or None."""
    return min((tag for tag in tags if tag in coefficients), key=rank.__getitem__, default=None)


def fit_pooled(training, *, prime, q):
    """Order-free values: every tag takes the medoid of all of its products."""
    by_tag = defaultdict(list)
    for record in training:
        for tag in dict.fromkeys(record.tags):
            by_tag[tag].append(record.encoded_path)
    values = {tag: medoid(labels, prime=prime, q=q) for tag, labels in by_tag.items()}
    return values, medoid([record.encoded_path for record in training], prime=prime, q=q)


def fit_cells(training, order, *, prime, q):
    """Exact fit for a fixed order: each tag takes the medoid of the products it heads."""
    rank = {tag: position for position, tag in enumerate(order)}
    cells = defaultdict(list)
    for record in training:
        tag = head(record.tags, rank, rank)
        if tag is not None:
            cells[tag].append(record.encoded_path)
    return rank, {tag: medoid(labels, prime=prime, q=q) for tag, labels in cells.items()}


def greedy_order(training, fallback, *, prime, q, smoothing):
    """Learn an order by repeatedly taking the purest tag on the products not yet headed.

    Purity is the mean distance of a tag's unheaded labels to their medoid,
    shrunk towards the constant predictor's mean distance by ``smoothing``
    pseudo-products. Ties go to the larger tag, then the tag name. Tags that
    head nothing follow in ``fallback`` order.
    """
    labels = [record.encoded_path for record in training]
    tags_of = [list(dict.fromkeys(record.tags)) for record in training]
    members = defaultdict(set)
    for index, tags in enumerate(tags_of):
        for tag in tags:
            members[tag].add(index)
    counts = {tag: Counter(labels[i] for i in rows) for tag, rows in members.items()}
    prior = Fraction(medoid_cost(Counter(labels), prime=prime, q=q)[1], len(labels))

    def entry(tag):
        size = len(members[tag])
        cost = medoid_cost(counts[tag], prime=prime, q=q)[1]
        return (cost + smoothing * prior) / (size + smoothing), -size, tag

    current = {tag: entry(tag) for tag in members}
    heap = list(current.values())
    heapq.heapify(heap)
    order = []
    while heap:
        candidate = heapq.heappop(heap)
        tag = candidate[2]
        if current.get(tag) != candidate:
            continue
        del current[tag]
        order.append(tag)
        touched = set()
        for index in members.pop(tag):
            for other in tags_of[index]:
                if other in members:
                    members[other].discard(index)
                    counts[other][labels[index]] -= 1
                    if not counts[other][labels[index]]:
                        del counts[other][labels[index]]
                    touched.add(other)
        for other in touched:
            if members[other]:
                current[other] = entry(other)
                heapq.heappush(heap, current[other])
            else:
                del members[other], current[other]
    chosen = set(order)
    return order + [tag for tag in fallback if tag not in chosen]


def predict_priority(records, rank, coefficients, default):
    predictions, covered = [], []
    for record in records:
        tag = head(record.tags, rank, coefficients)
        predictions.append(default if tag is None else coefficients[tag])
        covered.append(tag is not None)
    return predictions, covered


def predict_vote(records, coefficients, default, *, prime):
    predictions, covered = [], []
    for record in records:
        active = [coefficients[tag] for tag in dict.fromkeys(record.tags) if tag in coefficients]
        predictions.append(selector.select(active, prime)[0] if active else default)
        covered.append(bool(active))
    return predictions, covered


def scored(targets, predictions, covered):
    row = matched.SCORE(targets, predictions)
    n, used = row["n"], sum(covered)
    losses = [oracle.DISTANCE(target, value) for target, value in zip(targets, predictions, strict=True)]
    row.update(
        loss_q71=fsum(count * 71.0 ** -int(depth)
                      for depth, count in row["first_disagreement_counts"].items()) / n,
        coverage=used / n,
        covered_loss_q2=fsum(v for v, c in zip(losses, covered) if c) / used if used else None,
        default_loss_q2=fsum(v for v, c in zip(losses, covered) if not c) / (n - used) if n - used else None)
    return row


def fill(training, order, pooled_values, *, prime, q):
    """The three ways of filling a priority list for one order."""
    rank, cells = fit_cells(training, order, prime=prime, q=q)
    kept = {tag: pooled_values[tag] for tag in rank if tag in pooled_values}
    return rank, dict(cell=cells, cell_backfill={**kept, **cells}, pooled_priority=kept)


def list_rows(training, testing, order, pooled_values, default, *, prime, q):
    """Score the three ways of filling a priority list for one order."""
    targets = [record.encoded_path for record in testing]
    rank, coefficients = fill(training, order, pooled_values, prime=prime, q=q)
    fitted = matched.SCORE([record.encoded_path for record in training],
                           predict_priority(training, rank, coefficients["cell"], default)[0])
    return {variant: dict(
        stored=len(values),
        train_loss_q2=None if variant == "pooled_priority" else fitted["loss_q2"],
        **scored(targets, *predict_priority(testing, rank, values, default)))
        for variant, values in coefficients.items()}


def average(rows, extra=()):
    out = {}
    for metric in (*METRICS, *extra):
        values = [row[metric] for row in rows if row.get(metric) is not None]
        out[metric] = mean(values) if values else None
    return out


def summarise(rows, extra=()):
    """Average seeds within each fold, then weight the folds equally."""
    folds = [dict(fold=fold, **average([r for r in rows if r["fold"] == fold], extra))
             for fold in matched.FOLDS]
    return dict(**average(folds, extra), loss_q2_sd=stdev(r["loss_q2"] for r in folds), folds=folds)


def paired(candidates, baselines, sizes):
    """Corrected paired tests on fold losses, Holm-adjusted within the family."""
    tests = {key: corrected_test(
        [a["loss_q2"] - b["loss_q2"] for a, b in
         zip(candidates[key]["folds"], baselines[key]["folds"], strict=True)], sizes)
        for key in candidates}
    for key, adjusted in zip(tests, holm([t["p_value"] for t in tests.values()]), strict=True):
        tests[key]["holm_p_value"] = adjusted
    return tests


def run_split(dataset, scoring, references, archived):
    reference = defaultdict(lambda: defaultdict(list))
    lists = {q: defaultdict(lambda: defaultdict(list)) for q in FIT_Q}
    votes = {q: [] for q in FIT_Q}
    constants = {q: [] for q in FIT_Q}
    sizes = []
    for fold in matched.FOLDS:
        training = [r for r in dataset.records if r.cv_fold != fold]
        testing = [r for r in dataset.records if r.cv_fold == fold]
        targets = [r.encoded_path for r in testing]
        sizes.append(len(testing))
        battles = runtime._derive_battles(training)
        pooled = {q: fit_pooled(training, prime=matched.PRIME, q=q) for q in FIT_Q}
        for q, (values, default) in pooled.items():
            votes[q].append(dict(fold=fold, stored=len(values), **scored(
                targets, *predict_vote(testing, values, default, prime=matched.PRIME))))
            constants[q].append(dict(fold=fold, **scored(
                targets, [default] * len(targets), [False] * len(targets))))
        for order, seed in ORDERS:
            one_pass_default, original, terms = oracle.term_rows(
                dataset, fold, references[fold, order, seed], scoring,
                strategy=order, seed=seed, battles=battles)
            active = [bool(term[1]) for term in terms]
            medoids = [selector.select(term[1], matched.PRIME)[0] if term[1] else one_pass_default
                       for term in terms]
            frozen = next(r for r in archived[order]["folds"][fold]["seeds"] if r["seed"] == seed)
            if (frozen["selected"] != matched.SCORE(targets, medoids)
                    or frozen["original"]["prediction_sha256"] != original["prediction_sha256"]):
                raise ValueError(f"Archived {order} scores did not reproduce on fold {fold}")
            reference[order]["sum"].append(dict(
                fold=fold, seed=seed, **scored(targets, [term[2] for term in terms], active)))
            reference[order]["medoid"].append(dict(fold=fold, seed=seed, **scored(targets, medoids, active)))

            tag_order = runtime._tag_order(training, battles, fold, strategy=order, seed=seed)
            for q, (values, default) in pooled.items():
                rows = list_rows(training, testing, tag_order, values, default, prime=matched.PRIME, q=q)
                for variant, row in rows.items():
                    lists[q][order][variant].append(dict(fold=fold, seed=seed, **row))

        fallback = runtime._tag_order(training, battles, fold, strategy=BEST_STATIC)
        for q, (values, default) in pooled.items():
            for smoothing in SMOOTHING:
                tag_order = greedy_order(training, fallback, prime=matched.PRIME, q=q, smoothing=smoothing)
                rows = list_rows(training, testing, tag_order, values, default, prime=matched.PRIME, q=q)
                for variant, row in rows.items():
                    lists[q][f"greedy_m{smoothing}"][variant].append(dict(fold=fold, seed=None, **row))

    static = list(dict.fromkeys(order for order, _ in ORDERS))
    learned = [f"greedy_m{smoothing}" for smoothing in SMOOTHING]
    refs = {order: {kind: summarise(reference[order][kind]) for kind in ("sum", "medoid")}
            for order in static}
    fits = {}
    for q in FIT_Q:
        by_order = {order: {variant: summarise(lists[q][order][variant], ("stored", "train_loss_q2"))
                            for variant in VARIANTS} for order in (*static, *learned)}

        def pick(variant, orders):
            return {order: by_order[order][variant] for order in orders}

        vote = summarise(votes[q], ("stored",))
        medoids = {order: refs[order]["medoid"] for order in static}
        versus_best = {f"{order}:{variant}": by_order[order][variant]
                       for order in learned for variant in ("cell", "cell_backfill")}
        fits[str(q)] = dict(
            orders=by_order, pooled_vote=vote, constant=summarise(constants[q]),
            tests=dict(
                cell_minus_medoid=paired(pick("cell", static), medoids, sizes),
                cell_minus_sum=paired(pick("cell", static), {o: refs[o]["sum"] for o in static}, sizes),
                pooled_priority_minus_medoid=paired(pick("pooled_priority", static), medoids, sizes),
                pooled_vote_minus_medoid=paired({o: vote for o in static}, medoids, sizes),
                cell_minus_pooled_priority=paired(
                    pick("cell", static), pick("pooled_priority", static), sizes),
                backfill_minus_cell=paired(
                    pick("cell_backfill", by_order), pick("cell", by_order), sizes),
                backfill_minus_pooled_priority=paired(
                    pick("cell_backfill", by_order), pick("pooled_priority", by_order), sizes),
                learned_minus_best_static=paired(
                    versus_best, {key: by_order[BEST_STATIC][key.split(":")[1]] for key in versus_best},
                    sizes)))
    return dict(fold_sizes=sizes, reference=refs, fits=fits)


def run(source_commit, *, archive=False):
    provenance, inputs = oracle.load_splits()
    with db.get_connection() as conn:
        conn.read_only = True
        references, _ = valuation_orders.controls(conn, provenance)
        archived = oracle.archived_report(conn, valuation_orders.EXPERIMENT)["splits"]
    splits = {name: run_split(dataset, scoring, references[name], archived[name])
              for name, (dataset, _, scoring) in inputs.items()}
    report = dict(
        experiment_key=EXPERIMENT, status="exploratory", source_commit=source_commit,
        valuation_prime=matched.PRIME, evaluation_q=2, fitting_q=list(FIT_Q),
        greedy_smoothing=list(SMOOTHING),
        references=dict(product=oracle.PRODUCT_REFERENCE, store=matched.EXPERIMENT,
                        orders=valuation_orders.EXPERIMENT),
        protocol="For each tag order, assign every fitting product to its highest-priority tag and "
                 "give each tag the exact q-weighted medoid of the labels it heads, breaking ties by "
                 "integer order. Predict the coefficient of the highest-priority tag that has one, "
                 "else the medoid of all fitting labels. The pooled variants give each tag the "
                 "medoid of all of its fitting products and then select by the same order or by the "
                 "native 71-adic medoid of the active values. Backfill keeps the cell coefficients "
                 "and gives the pooled value to tags that head nothing. Greedy orders repeatedly "
                 "take the tag with the lowest mean q-weighted distance to its medoid on the "
                 "products not yet headed, shrunk towards the constant predictor by the stated "
                 "number of pseudo-products, with ties to the larger tag; tags that head nothing "
                 "follow in taxonomy-association order. The constant row always predicts the "
                 "default. The sum and medoid references reuse the frozen one-pass fits and must "
                 "reproduce their archived scores. Average random seeds within each fold, then "
                 "weight all five folds equally.",
        **provenance, splits=splits)
    return oracle.save_report(report) if archive else report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true",
                        help="also insert the report into padjective.paper_low_q_reports")
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.source_commit, archive=args.archive), indent=2) + "\n")


if __name__ == "__main__":
    main()
