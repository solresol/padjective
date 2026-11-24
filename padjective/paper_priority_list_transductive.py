"""Pass priority-list values to unseen tags through a held-out store's own products.

A held-out store arrives with all of its products and their tags but no
labels. A product with no fitted tag would take the global default. Here it
borrows a prediction through the tags it shares with the store's other
products. Held-out tags and store membership are used; held-out labels are not,
except by the one diagnostic row that says so.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
from statistics import mean, median

from . import benchmark_runtime as runtime, db, live_subspace as live
from . import paper_matched_store_folds as matched
from . import paper_priority_list as plist
from . import paper_sum_oracle as oracle
from . import paper_valuation_selector as selector

EXPERIMENT = "paper-priority-list-transductive-2026-11-23"
FIT_Q = 71
ORDER = "taxonomy_association"
FILLS = ("cell_backfill", "pooled_priority")
BASE = "base"
# Diagnostic rows that read held-out labels. They are not predictors.
ORACLES = ("oracle_store_constant", "oracle_covered_labels", "oracle_tags", "oracle_leave_one_out")
EXTRA = ("uncovered_loss_q2", "covered_products_loss_q2", "reached_share", "reached_loss_q2",
         "reached_default_loss_q2", "changed_share")
CONTRASTS = (("tags", "tags_one_round"), ("tags_vote", "tags"), ("tags_then_store", "store_default"),
             ("tags_then_store", "tags"), ("tags_fold_scope", "tags"), ("local_first", "tags_one_round"),
             ("store_consensus", "store_default"))


def pick(active, choice, prime):
    """One value from a product's valued tags: the tightest tag's, or the medoid of them all."""
    if choice == "vote":
        return selector.select([entry[3] for entry in active], prime)[0]
    return min(active)[3]


def spread(tag_lists, predictions, covered, *, prime, q, rounds=None, choice="tightest"):
    """Pass predictions to products with no fitted tag through the tags they share.

    ``tag_lists`` holds each product's tags that have no fitted coefficient.
    Each round gives every tag without a value the medoid of the current
    predictions of the reached products that carry it, then predicts each
    unreached product that carries a valued tag. The tightest tag has the
    lowest mean distance from those predictions to their medoid, with ties to
    the tag on more products and then the tag name. Tag values are never
    revised, so round k reaches the products k steps from a covered one.

    Returns the predictions, the round in which each product was reached (0 for
    covered products, None for unreached ones) and the tag entries.
    """
    values = list(predictions)
    reached = [0 if flag else None for flag in covered]
    local, step = {}, 0
    while rounds is None or step < rounds:
        step += 1
        seen = defaultdict(Counter)
        for tags, value, when in zip(tag_lists, values, reached, strict=True):
            if when is not None:
                for tag in set(tags):
                    if tag not in local:
                        seen[tag][value] += 1
        for tag, counts in seen.items():
            value, cost = plist.medoid_cost(counts, prime=prime, q=q)
            support = sum(counts.values())
            local[tag] = (Fraction(cost, support), -support, tag, value)
        fresh = False
        for index, tags in enumerate(tag_lists):
            if reached[index] is None:
                active = [local[tag] for tag in set(tags) if tag in local]
                if active:
                    values[index], reached[index], fresh = pick(active, choice, prime), step, True
        if not fresh:
            break
    return values, reached, local


def smooth(tag_lists, predictions, covered, *, prime, q):
    """One round in which the store's own tags outrank the fitted list.

    Uncovered products are predicted as in one round of ``spread``. A covered
    product that carries an unfitted tag takes the value of its tightest one,
    which is its own prediction whenever that tag's products all agree.
    """
    values, reached, local = spread(tag_lists, predictions, covered, prime=prime, q=q, rounds=1)
    for index, tags in enumerate(tag_lists):
        if covered[index] and tags:
            values[index] = min(local[tag] for tag in tags)[3]
    return values, reached


def store_variants(tag_lists, base, covered, targets, *, prime, q):
    """Predictions of every variant for one store, each with a flag for products off the default."""
    n = len(base)
    anchors = [value for value, flag in zip(base, covered, strict=True) if flag]
    prior = plist.medoid(anchors, prime=prime, q=q) if anchors else None

    def then_store(values, flags):
        if prior is None:
            return list(values), list(flags)
        return [value if flag else prior for value, flag in zip(values, flags)], [True] * n

    out = {BASE: (list(base), list(covered))}
    out["store_default"] = then_store(*out[BASE])
    out["store_consensus"] = out[BASE] if prior is None else ([prior] * n, [True] * n)
    for name, options in (("tags_one_round", dict(rounds=1)), ("tags", {}),
                          ("tags_vote", dict(choice="vote"))):
        values, when, _ = spread(tag_lists, base, covered, prime=prime, q=q, **options)
        out[name] = values, [step is not None for step in when]
    out["tags_then_store"] = then_store(*out["tags"])
    values, when = smooth(tag_lists, base, covered, prime=prime, q=q)
    out["local_first"] = values, [step is not None for step in when]

    # Diagnostics from held-out labels. The first three change only the uncovered products.
    missed = [target for target, flag in zip(targets, covered, strict=True) if not flag]
    best = plist.medoid(missed, prime=prime, q=2) if missed else None
    out["oracle_store_constant"] = (
        [value if flag else best for value, flag in zip(base, covered)], [True] * n)
    truth = [target for target, flag in zip(targets, covered) if flag]
    known = plist.medoid(truth, prime=prime, q=2) if truth else None
    out["oracle_covered_labels"] = out[BASE] if known is None else (
        [value if flag else known for value, flag in zip(base, covered)], [True] * n)
    values, when, _ = spread(
        tag_lists, [target if flag else value for target, value, flag in zip(targets, base, covered)],
        covered, prime=prime, q=q)
    out["oracle_tags"] = ([old if flag else value for old, value, flag in zip(base, values, covered)],
                          [step is not None for step in when])
    counts = Counter(targets)
    others = []
    for target in targets:
        rest = counts - Counter([target])
        others.append(plist.medoid_cost(rest, prime=prime, q=2)[0] if rest else None)
    out["oracle_leave_one_out"] = (
        [value if other is None else other for value, other in zip(base, others)],
        [flag or other is not None for flag, other in zip(covered, others)])
    return out


def store_groups(testing, stores):
    groups = defaultdict(list)
    for index, record in enumerate(testing):
        groups[stores[record.product_key]].append(index)
    return groups


def unfitted_tags(testing, fitted):
    return [sorted({tag for tag in record.tags if tag not in fitted}) for record in testing]


def part(losses, rows):
    return mean(losses[i] for i in rows) if rows else None


def row(targets, predictions, reached, base, covered):
    """Scores, with the products the fitted list leaves on the default reported separately."""
    out = plist.scored(targets, predictions, reached)
    after = [oracle.DISTANCE(target, value) for target, value in zip(targets, predictions, strict=True)]
    before = [oracle.DISTANCE(target, value) for target, value in zip(targets, base, strict=True)]
    uncovered = [i for i, flag in enumerate(covered) if not flag]
    gained = [i for i in uncovered if reached[i]]
    changed = sum(flag and value != old for flag, value, old in zip(covered, predictions, base))
    out.update(
        uncovered_loss_q2=part(after, uncovered),
        covered_products_loss_q2=part(after, [i for i, flag in enumerate(covered) if flag]),
        reached_share=len(gained) / len(uncovered) if uncovered else None,
        reached_loss_q2=part(after, gained), reached_default_loss_q2=part(before, gained),
        changed_share=changed / sum(covered) if any(covered) else None)
    return out


def fold_rows(testing, stores, base, covered, fitted, *, prime=matched.PRIME, q=FIT_Q):
    """Score every variant on one held-out fold for one way of filling the list."""
    targets = [record.encoded_path for record in testing]
    tags = unfitted_tags(testing, fitted)
    predictions = defaultdict(lambda: [None] * len(testing))
    reached = defaultdict(lambda: [None] * len(testing))
    for rows in store_groups(testing, stores).values():
        variants = store_variants(*([column[i] for i in rows] for column in (tags, base, covered, targets)),
                                  prime=prime, q=q)
        for name, (values, flags) in variants.items():
            for i, value, flag in zip(rows, values, flags, strict=True):
                predictions[name][i], reached[name][i] = value, flag
    values, when, _ = spread(tags, base, covered, prime=prime, q=q)
    predictions["tags_fold_scope"], reached["tags_fold_scope"] = values, [step is not None for step in when]
    return {name: row(targets, predictions[name], reached[name], base, covered) for name in predictions}


def facts(testing, stores, covered, fitted, *, prime=matched.PRIME, q=FIT_Q):
    """How much of a held-out fold the store's own tags can reach."""
    tags = unfitted_tags(testing, fitted)
    groups = store_groups(testing, stores)
    uncovered = [i for i, flag in enumerate(covered) if not flag]
    anchored = {store for store, rows in groups.items() if any(covered[i] for i in rows)}
    carriers = defaultdict(set)
    for index, record in enumerate(testing):
        for tag in tags[index]:
            carriers[tag].add(stores[record.product_key])
    when = [None] * len(testing)
    for rows in groups.values():
        steps = spread([tags[i] for i in rows], [0] * len(rows), [covered[i] for i in rows],
                       prime=prime, q=q)[1]
        for i, step in zip(rows, steps, strict=True):
            when[i] = step

    def share(test):
        return sum(test(i) for i in uncovered) / len(uncovered) if uncovered else None

    return dict(
        stores=len(groups), median_store_products=median(map(len, groups.values())),
        mean_store_products=len(testing) / len(groups),
        single_product_stores=sum(len(rows) == 1 for rows in groups.values()),
        stores_with_uncovered_product=len({stores[testing[i].product_key] for i in uncovered}),
        stores_without_covered_product=len(groups) - len(anchored),
        unfitted_tags=len(carriers),
        unfitted_tags_in_several_stores=sum(len(found) > 1 for found in carriers.values()),
        uncovered_share=len(uncovered) / len(testing),
        uncovered_without_tags=share(lambda i: not testing[i].tags),
        uncovered_in_store_without_covered_product=share(
            lambda i: stores[testing[i].product_key] not in anchored),
        reached_in_round_1=share(lambda i: when[i] == 1),
        reached_in_later_rounds=share(lambda i: when[i] is not None and when[i] > 1),
        rounds=max((step for step in when if step is not None), default=0))


def fold_job(dataset, stores, fold):
    training = [r for r in dataset.records if r.cv_fold != fold]
    testing = [r for r in dataset.records if r.cv_fold == fold]
    pooled, default = plist.fit_pooled(training, prime=matched.PRIME, q=FIT_Q)
    order = runtime._tag_order(training, [], fold, strategy=ORDER)
    rank, coefficients = plist.fill(training, order, pooled, prime=matched.PRIME, q=FIT_Q)
    if len({frozenset(coefficients[name]) for name in FILLS}) != 1:
        raise ValueError("The fills must give coefficients to the same tags")
    rows = {}
    for name in FILLS:
        base, covered = plist.predict_priority(testing, rank, coefficients[name], default)
        rows[name] = fold_rows(testing, stores, base, covered, coefficients[name])
    return dict(fold=fold, n=len(testing), rows=rows,
                facts=facts(testing, stores, covered, coefficients[FILLS[0]]))


def collect(results):
    results = sorted(results, key=lambda r: r["fold"])
    sizes = [r["n"] for r in results]
    variants = list(results[0]["rows"][FILLS[0]])
    fits = {fill: {name: plist.summarise(
        [dict(fold=r["fold"], **r["rows"][fill][name]) for r in results], EXTRA)
        for name in variants} for fill in FILLS}

    def family(pairs):
        keys = [(fill, a, b) for fill in FILLS for a, b in pairs]
        return plist.paired({f"{a}_minus_{b}|{fill}": fits[fill][a] for fill, a, b in keys},
                            {f"{a}_minus_{b}|{fill}": fits[fill][b] for fill, a, b in keys}, sizes)

    return dict(
        fold_sizes=sizes, fits=fits,
        tests=dict(
            minus_base=family([(name, BASE) for name in variants if name not in (BASE, *ORACLES)]),
            contrasts=family(CONTRASTS)),
        facts={key: mean(r["facts"][key] for r in results) for key in results[0]["facts"]})


def run(source_commit, *, archive=False):
    provenance, inputs = oracle.load_splits()
    dataset = inputs["store"][0]
    with db.get_connection() as conn:
        conn.read_only = True
        stores = matched.store_membership(conn, {r.product_key for r in dataset.records})
    assignment = [(r.product_key, stores[r.product_key]) for r in dataset.records]
    if live.digest(assignment) != provenance["store_assignment_sha256"]:
        raise ValueError("Store membership differs from the archived store folds")
    report = dict(
        experiment_key=EXPERIMENT, status="exploratory", source_commit=source_commit,
        valuation_prime=matched.PRIME, evaluation_q=2, fitting_q=FIT_Q, order=ORDER,
        references=dict(store=matched.EXPERIMENT),
        protocol="Store holdouts only. A taxonomy-association priority list is fitted on the other "
                 "stores with backfilled cell medoids and with pooled values, and predicts every "
                 "held-out product; the base rows stop there. The other rows also use the tags and "
                 "store membership of the held-out products, never their labels. Within one store, "
                 "each tag with no fitted coefficient takes the medoid of the predictions of the "
                 "products that carry it, and a product with no fitted tag takes the value of its "
                 "tightest valued tag: lowest mean distance from those predictions to their "
                 "medoid, ties to the tag on more products, then the tag name. One round uses only "
                 "products the fitted list covers; the full version repeats from the newly "
                 "predicted products until nothing changes. The vote row takes the 71-adic medoid "
                 "of a product's valued tags instead. The store default gives remaining products "
                 "the medoid of the predictions on the store's covered products. The fold-scope "
                 "row pools all held-out stores of a fold. The local-first row runs one round and "
                 "also lets a covered product take the value of its tightest unfitted tag. The "
                 "store-consensus row gives that store default to every product of the store. "
                 "Four oracle rows are diagnostics that read held-out labels and are not "
                 "predictors: the best single code for each store's uncovered products; the store "
                 "default and the full tag passing computed from the true labels of the store's "
                 "covered products; and, for every product, the medoid of the labels of the "
                 "store's other products. Folds have equal weight.",
        **provenance, **collect([fold_job(dataset, stores, fold) for fold in matched.FOLDS]))
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
