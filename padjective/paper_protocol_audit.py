"""Audit product folds and title coverage for the frozen SHIMP snapshot."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import warnings

from sklearn.model_selection import KFold, StratifiedKFold

from . import benchmark_runtime as runtime, db
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_sum_oracle import save_report
from .product_hash import canonicalize_product_url, hash_product_url

EXPERIMENT = "paper-protocol-audit-2026-11-09"
REPLAY_CUTOFF = datetime.fromisoformat("2026-02-25T20:49:55.183887+00:00")


def replay_folds(conn, stored_folds, cutoff):
    """Replay both splitters on the dated, pre-eligibility source join."""
    keys, paths = [], []
    for url, domain, handle, path in conn.execute("""
        SELECT p.product_url, p.myshopify_domain, p.product_handle, t.taxonomy_path
        FROM cantbuymelove.product p
        JOIN cantbuymelove.product_taxonomy pt ON pt.product_id=p.id
        JOIN cantbuymelove.taxonomy t ON t.taxonomy_id=pt.taxonomy_id
        WHERE p.product_title IS NOT NULL AND NULLIF(t.taxonomy_path, '') IS NOT NULL
          AND pt.classified_at <= %s
        ORDER BY p.id
    """, (cutoff,)):
        url = canonicalize_product_url(url, myshopify_domain=domain, product_handle=handle)
        keys.append(hash_product_url(url) if url else None)
        paths.append(path)

    source_counts = Counter(paths)
    matches = {}
    for splitter in (KFold, StratifiedKFold):
        matching_keys = set()
        with warnings.catch_warnings():
            # Compare the stratified counterfactual even when the original guard disallows it.
            warnings.filterwarnings("ignore", message="The least populated class in y")
            for fold, (_, indices) in enumerate(
                splitter(n_splits=5, shuffle=True, random_state=42).split(keys, paths)
            ):
                matching_keys.update(
                    keys[i] for i in indices if stored_folds.get(keys[i]) == fold
                )
        matches[splitter.__name__] = len(matching_keys)

    return dict(
        classification_cutoff=cutoff.isoformat(),
        source_rows=len(keys),
        source_paths=len(source_counts),
        minimum_source_path_count=min(source_counts.values()),
        source_paths_with_fewer_than_five=sum(n < 5 for n in source_counts.values()),
        matched_product_keys=len(stored_folds.keys() & set(keys)),
        stored_folds_reproduced=matches,
    )


def audit(conn):
    dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema="padjective")
    if _snapshot_digest(dataset) != SNAPSHOT_DIGEST:
        raise ValueError("The frozen paper snapshot does not match")
    capture, cutoff, builder, min_tags, min_samples = conn.execute("""
        SELECT created_at, as_of, code_version, min_tag_count, min_samples_per_taxonomy
        FROM padjective.product_taxonomy_bench_snapshots WHERE snapshot_id=%s
    """, (PAPER_SNAPSHOT,)).fetchone()
    records = dataset.records
    positions = {
        (r.product_key, tag, part, offset)
        for r in records for tag, part, offset in r.title_tag_positions
    }
    stored_positions = set(conn.execute("""
        SELECT product_id_hash, tag_id, title_part, title_position
        FROM padjective.product_taxonomy_bench_product_tags
        WHERE snapshot_id=%s AND in_title
    """, (PAPER_SNAPSHOT,)))
    if positions != stored_positions:
        raise ValueError("The runtime positions and stored title flags disagree")
    tag_counts = Counter(tag for r in records for tag in r.tags)
    path_depths = {r.encoded_path: r.taxonomy_depth for r in records}
    stored_folds = {r.product_key: r.cv_fold for r in records}
    battles = runtime._derive_battles(records)
    matched_tags = {tag for _, tag, _, _ in positions}
    products_with_pairs = sum(
        any(n >= 2 for n in Counter(part for _, part, _ in r.title_tag_positions).values())
        for r in records
    )
    return dict(
        experiment_key=EXPERIMENT,
        snapshot_id=PAPER_SNAPSHOT,
        snapshot_digest=SNAPSHOT_DIGEST,
        captured_at=capture.isoformat(),
        source_cutoff=cutoff.isoformat(),
        snapshot_git_head=builder,
        first_committed_snapshot_builder="f214fd441986e90247f757a83d36e240523df800",
        minimum_source_tag_count=min_tags,
        minimum_source_taxonomy_count=min_samples,
        minimum_frozen_tag_count=min(tag_counts.values()),
        products=len(records),
        taxonomy_paths=len(path_depths),
        product_fold_sizes=[sum(r.cv_fold == f for r in records) for f in range(5)],
        training_taxonomy_counts=[
            len({r.encoded_path for r in records if r.cv_fold != f}) for f in range(5)
        ],
        fold_replay=dict(
            cutoff_selection="The reconstructed source cutoff maximised agreement with the "
                             "stored folds among nearby classification timestamps. The builder "
                             "itself assigned folds before applying the snapshot cutoff.",
            capture_timestamp=replay_folds(conn, stored_folds, capture),
            reconstructed_source=replay_folds(conn, stored_folds, REPLAY_CUTOFF),
        ),
        title_coverage=dict(
            product_tag_pairs=sum(tag_counts.values()),
            matching_product_tag_pairs=len(positions),
            distinct_tags=len(tag_counts),
            distinct_matching_tags=len(matched_tags),
            products_with_title_match=len({key for key, _, _, _ in positions}),
            products_with_same_part_pair=products_with_pairs,
            elo_battles=len(battles),
        ),
        encoding=dict(
            prime=dataset.prime_base,
            maximum_digit=dataset.max_digit,
            maximum_path_depth=max(path_depths.values()),
            paths_by_depth={str(depth): count for depth, count in
                            sorted(Counter(path_depths.values()).items())},
            products_by_depth={str(depth): count for depth, count in
                               sorted(Counter(r.taxonomy_depth for r in records).items())},
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with db.get_connection() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        report = audit(conn)
    save_report(report)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
