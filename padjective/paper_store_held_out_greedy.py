"""Run every paper one-pass order on the frozen store folds and audit the labels."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
from statistics import mean, stdev

from psycopg.types.json import Jsonb

from . import benchmark_runtime as runtime, data_access, db, live_subspace as live, tagbattle
from . import paper_store_held_out as headline
from .product_hash import canonicalize_product_url, hash_product_url
from .product_taxonomy_bench import title_part_positions
from .subspace_voting import corrected_test, holm

EXPERIMENT = "paper-store-held-out-greedy-orders-2026-11-09"
ORDERS = [(s, None) for s in runtime.DEFAULT_ABLATION_STRATEGIES]
ORDERS += [("random", seed) for seed in runtime.DEFAULT_RANDOM_SEEDS]


def source_evidence(conn, rows, manifest):
    """Recover title positions from matching dated source rows, without exporting titles."""
    by_key = {r["product_key"]: r for r in rows}
    paper = dict((key, (label, path)) for key, label, path in conn.execute("""
        SELECT product_id_hash,taxonomy_id,taxonomy_path FROM padjective.product_taxonomy_bench_products
        WHERE snapshot_id=%s
    """, (live.PAPER_SNAPSHOT,)))
    titles = conn.execute("""
        SELECT c.product_key,p.myshopify_domain,p.product_handle,p.product_url,
               p.updated_at,p.product_title,pd.product_detail->'product'->>'title',
               pd.product_detail->'product'->>'tags',pd.when_fetched<=%s
        FROM padjective.live_subspace_products c
        JOIN cantbuymelove.product p ON p.id=(c.evidence->>'source_product_id')::bigint
        JOIN public.product_details pd USING(myshopify_domain,run_name,product_handle)
        WHERE c.cohort_id=%s ORDER BY c.product_key
    """, (datetime.fromisoformat(manifest["captured_at"]), manifest["cohort_id"]))
    positions = {}
    overlap = Counter()
    for key, domain, handle, url, updated, title, detail_title, raw_tags, fetched_in_time in titles:
        row = by_key[key]
        if (key in positions or not fetched_in_time or title is None or title != detail_title
                or updated != datetime.fromisoformat(row["source_updated_at"])
                or live.digest([domain.strip().lower(), handle.strip()]) != key
                or sorted(tagbattle.filter_nested_tags(data_access.parse_tags(raw_tags))) != row["tags"]):
            raise ValueError("Title or tags cannot be aligned with the frozen cohort")
        positions[key] = tuple((tag, part, pos) for part, matches in
                               title_part_positions(title, row["tags"]).items()
                               for tag, pos in sorted(matches.items()))
        url_key = hash_product_url(canonicalize_product_url(url, myshopify_domain=domain, product_handle=handle))
        if row["paper_overlap"]:
            overlap["marked_overlap"] += 1
        if url_key in paper:
            overlap["url_matched"] += 1
            label, path = paper[url_key]
            overlap["same_category_and_path"] += (label, path) == (row["taxonomy_id"], row["path"])
            overlap["different_category"] += label != row["taxonomy_id"]
            overlap["same_category_different_path"] += label == row["taxonomy_id"] and path != row["path"]
    if positions.keys() != by_key.keys():
        raise ValueError("Incomplete title alignment")
    paper_paths = set(paper.values())
    capture_paths = {(r["taxonomy_id"], r["path"]) for r in rows}
    audit = dict(overlap=dict(overlap), paper_categories=len(paper_paths),
                 capture_categories=len(capture_paths), shared_category_paths=len(paper_paths & capture_paths),
                 matching_title_rows=len(positions),
                 rows_with_tag_in_title=sum(bool(v) for v in positions.values()),
                 title_positions_sha256=live.digest(positions))
    return positions, audit


def fit_fold(rows, positions, fold):
    c = live.context(rows, fold)
    vocabulary = set(c["vocabulary"])
    records = [runtime.ProductRecord(i, r["product_key"], [t for t in r["tags"] if t in vocabulary],
                r["target"], r["fold"], r["taxonomy_id"], len(r["path"].split(".")),
                tuple(v for v in positions[r["product_key"]] if v[0] in vocabulary))
               for i, r in enumerate(c["train"] + c["test"])]
    battles = runtime._derive_battles(records[:len(c["train"])])
    fits = []
    for strategy, seed in ORDERS:
        fitted = runtime.umllr_run_fold(fold, records, battles, live.P,
                                        tag_order_strategy=strategy, tag_order_seed=seed)
        predicted = [int(p.predicted_value) for p in fitted.predictions]
        fits.append(dict(strategy=strategy, seed=seed, predictions=predicted,
                         scores=headline.score(c["y_test"], predicted),
                         mean_active_tag_coefficients=fitted.mean_scoring_ops))
    return dict(fold=fold, test_sha256=live.digest([r["product_key"] for r in c["test"]]),
                vocabulary_sha256=live.digest(c["vocabulary"]), training_battles=len(battles), fits=fits)


def contrasts(losses, control):
    result = [dict(model=key, **corrected_test([a-b for a,b in zip(values, losses[control], strict=True)],
                                             headline.FOLD_SIZES))
              for key, values in losses.items() if key != control]
    for row, adjusted in zip(result, holm([r["p_value"] for r in result]), strict=True):
        row["holm_p_value"] = adjusted
    return result


def collect(rows, saved, reference, manifest):
    models = {s: [] for s, _ in ORDERS}
    for fold in range(5):
        evidence = saved[fold]
        c = live.context(rows, fold)
        if (evidence["test_sha256"] != reference[fold]["test_sha256"]
                or evidence["vocabulary_sha256"] != reference[fold]["vocabulary_sha256"]
                or [(r["strategy"], r["seed"]) for r in evidence["fits"]] != ORDERS):
            raise ValueError("Greedy fits have different folds, tags or seed coverage")
        for fit in evidence["fits"]:
            if headline.score(c["y_test"], fit["predictions"]) != fit["scores"]:
                raise ValueError("Saved greedy predictions fail score replay")
            if fit["strategy"] == "taxonomy_association" and fit["predictions"] != reference[fold]["predictions"]["greedy"]:
                raise ValueError("Taxonomy association does not reproduce the headline comparison")
        for key, folds in models.items():
            selected = [r for r in evidence["fits"] if r["strategy"] == key]
            folds.append(dict(fold=fold, seeds=[dict(seed=r["seed"], **r["scores"],
                              mean_active_tag_coefficients=r["mean_active_tag_coefficients"]) for r in selected],
                              loss_q2=mean(r["scores"]["loss_q2"] for r in selected),
                              exact_accuracy=mean(r["scores"]["exact_accuracy"] for r in selected),
                              mean_active_tag_coefficients=mean(r["mean_active_tag_coefficients"] for r in selected)))
    result = [dict(model=key, folds=folds, loss_q2=mean(r["loss_q2"] for r in folds),
                   loss_q2_sd=stdev(r["loss_q2"] for r in folds),
                   exact_accuracy=mean(r["exact_accuracy"] for r in folds),
                   mean_active_tag_coefficients=mean(r["mean_active_tag_coefficients"] for r in folds))
              for key, folds in models.items()]
    fold_losses = {key: [r["loss_q2"] for r in folds] for key, folds in models.items()}
    return dict(manifest=manifest, status="validated", models=sorted(result, key=lambda r: r["loss_q2"]),
                training_battles=[saved[f]["training_battles"] for f in range(5)],
                greedy_contrasts=contrasts(fold_losses, "taxonomy_association"))


def run(source_commit):
    with db.get_connection() as conn:
        if not conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (EXPERIMENT,)).fetchone()[0]:
            raise RuntimeError("This comparison is already running")
        rows, provenance, _ = headline.load_archive(conn)
        positions, audit = source_evidence(conn, rows, provenance)
        original = conn.execute("SELECT report FROM padjective.paper_low_q_reports WHERE experiment_key=%s",
                                (headline.EXPERIMENT,)).fetchone()[0]
        if (original["status"] != "validated" or original["manifest"]["cohort_sha256"] != provenance["cohort_sha256"]
                or original["manifest"]["evaluation_q"] != 2):
            raise ValueError("Headline scores come from a different cohort or metric")
        reference = dict(conn.execute("""SELECT fold,evidence FROM padjective.paper_store_held_out_runs
            WHERE experiment_key=%s AND model='greedy'""", (headline.EXPERIMENT,)).fetchall())
        saved = dict(conn.execute("""SELECT fold,evidence FROM padjective.paper_store_held_out_runs
            WHERE experiment_key=%s AND model='greedy-orders'""", (EXPERIMENT,)).fetchall())
        manifest = dict(experiment_key=EXPERIMENT, source_commit=source_commit, evaluation_q=2,
                        valuation_prime=live.P, fitting_q=live.P, fold_sizes=list(headline.FOLD_SIZES),
                        taxonomy_and_titles=audit, **provenance)
        fingerprint = live.digest(manifest)
        if any(item["manifest_sha256"] != fingerprint for item in saved.values()):
            raise ValueError("Cannot combine fits from different manifests")
        conn.commit()
        live.emit("greedy_orders_started", completed_folds=len(saved), audit=audit)
        pending = [f for f in range(5) if f not in saved]
        if pending:
            with ProcessPoolExecutor(max_workers=5, mp_context=multiprocessing.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(fit_fold, rows, positions, f) for f in pending]):
                    item = future.result()
                    fold = item["fold"]
                    item["manifest_sha256"] = fingerprint
                    conn.execute("""INSERT INTO padjective.paper_store_held_out_runs
                        (experiment_key,fold,model,evidence) VALUES (%s,%s,'greedy-orders',%s)""",
                        (EXPERIMENT, fold, Jsonb(item)))
                    conn.commit()
                    saved[fold] = item
                    live.emit("greedy_orders_fold_saved", fold=fold, completed=len(saved))
        report = collect(rows, saved, reference, manifest)
        losses = {m["model"]: [r["loss_q2"] for r in m["folds"]] for m in original["models"]}
        report["exploratory_headline_contrasts_vs_l1"] = contrasts(losses, "ulr")
        conn.execute("""INSERT INTO padjective.paper_low_q_reports(experiment_key,report)
            VALUES (%s,%s) ON CONFLICT (experiment_key) DO NOTHING""", (EXPERIMENT, Jsonb(report)))
        checked = conn.execute("SELECT report FROM padjective.paper_low_q_reports WHERE experiment_key=%s",
                               (EXPERIMENT,)).fetchone()[0]
        if checked != report:
            raise ValueError("Greedy order report failed Postgres readback")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.source_commit)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    live.emit("greedy_orders_complete", output=args.output, models=len(report["models"]))


if __name__ == "__main__":
    main()
