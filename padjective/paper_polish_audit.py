"""Verify tree shape, path digits and merchant-category coverage for SHIMP."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from psycopg.types.json import Jsonb
from sklearn import __version__ as sklearn_version
from sklearn.tree import DecisionTreeClassifier

from . import benchmark_runtime, db
from .paper_published_methods import PAPER_SNAPSHOT
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_sum_oracle import save_report
from .paper_tree_ensemble_tradeoff import encode, fingerprint, load_context, tree_counts

EXPERIMENT = "paper-polish-audit-2026-11-10"
TREE_BATCH = "95b8a6bf-28b9-4a17-9ee6-58759d149ff5"
CONFIG = dict(family="tree", class_weight=None, seed=42, sweep="leaves", max_leaf_nodes=512)


def tree_shape(conn, fold, context):
    original, = conn.execute("""
        SELECT evidence FROM padjective.paper_tree_tradeoff_runs
        WHERE batch_id=%s AND cv_fold=%s AND configuration=%s
    """, (TREE_BATCH, fold, Jsonb(CONFIG))).fetchone()
    model = DecisionTreeClassifier(max_leaf_nodes=512, random_state=42)
    model.fit(context["x_train"], context["labels_train"])
    pred = encode(model.predict(context["x_test"]), context["codes"])
    counts, _, _ = tree_counts(model, context["x_test"])
    if counts != original["counts"] or fingerprint(pred.tolist()) != original["prediction_sha256"]:
        raise ValueError("Tree does not reproduce the archived fit")

    tree = model.tree_
    left, right = tree.children_left, tree.children_right
    parents = {child: node for node, pair in enumerate(zip(left, right, strict=True))
               for child in pair if child != -1}
    node = int(np.argmax(tree.compute_node_depths()))
    offshoot_leaves = absent_branches = rare_present = 0
    while node in parents:
        parent = parents[node]
        if tree.threshold[parent] != 0.5:
            raise ValueError("Expected a split on a binary tag")
        absent = left[parent] == node
        sibling = right[parent] if absent else left[parent]
        offshoot_leaves += left[sibling] == -1
        absent_branches += absent
        rare_present += tree.n_node_samples[right[parent]] * 10 <= tree.n_node_samples[parent]
        node = parent
    return dict(fold=fold, depth=counts["max_depth"], leaves=counts["leaves"],
                leaf_offshoots=int(offshoot_leaves), absent_tag_branches=int(absent_branches),
                splits_with_at_most_ten_percent_present=int(rare_present),
                model_sha256=counts["model_sha256"], prediction_sha256=original["prediction_sha256"])


def path_digits(conn):
    paths = [
        benchmark_runtime.parse_taxonomy_digits(path)
        for path, in conn.execute("""
            SELECT DISTINCT taxonomy_path FROM padjective.product_taxonomy_bench_products
            WHERE snapshot_id=%s
        """, (PAPER_SNAPSHOT,))
    ]
    children = defaultdict(set)
    for path in paths:
        for i, digit in enumerate(path):
            children[path[:i]].add(digit)
    return dict(paths=len(paths), smallest_digit=min(min(digits) for digits in children.values()),
                largest_digit=max(max(digits) for digits in children.values()),
                maximum_observed_out_degree=max(map(len, children.values())))


def category_sample(conn):
    rows = conn.execute("""
        WITH sampled AS MATERIALIZED (
            SELECT run_name, product_detail->'product' AS product
            FROM public.product_details TABLESAMPLE SYSTEM (0.003) REPEATABLE (42)
        )
        SELECT run_name, count(*), count(*) FILTER (WHERE jsonb_typeof(product)='object'),
               count(*) FILTER (WHERE product ? 'product_type'),
               count(*) FILTER (WHERE product ?| array['category','product_category',
                   'standardized_product_type','taxonomy','category_id','taxonomy_id',
                   'productCategory','standardizedProductType'])
        FROM sampled GROUP BY run_name ORDER BY run_name
    """)
    return [dict(run=run, sampled_rows=n, wrapped_products=wrapped, product_type_key=types,
                 standard_category_keys=categories)
            for run, n, wrapped, types, categories in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contexts = load_context()
    with db.get_connection() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        conn.execute("SET LOCAL statement_timeout='60s'")
        report = dict(
            experiment_key=EXPERIMENT, snapshot_id=PAPER_SNAPSHOT, snapshot_digest=SNAPSHOT_DIGEST,
            tree_batch=TREE_BATCH, configuration=CONFIG, scikit_learn=sklearn_version,
            trees=[tree_shape(conn, fold, context) for fold, context in contexts.items()],
            encoding=path_digits(conn), catalogue_key_sample=category_sample(conn),
            catalogue_sample_protocol="PostgreSQL system block sample, 0.003% with seed 42. "
                                      "Key coverage cannot establish merchant category assignment.",
        )
    save_report(report)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
