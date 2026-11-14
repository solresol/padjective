"""Rescore archived paper predictions at q=2 without fitting any models.

Product records and predictions are read from Shopify PostgreSQL. Only counts,
fold scores and provenance are exported; the report is also persisted in the
existing padjective.paper_low_q_reports table.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from psycopg.types.json import Jsonb

from . import db
from .paper_low_q_experiment import EXPERIMENT as NATIVE_EXPERIMENT, SOURCE_BATCH
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset, q_weighted_distance
from .q_loss_curves import FOLD_SIZES
from .q_loss_rescore import rescore

EXPERIMENT = "paper-q2-primary-2026-11-05"
Q_VALUES = {"2": 2., "71": 71.}
PILOT_RUNS = ("b5bff126-fb4a-492d-969b-b960ccc2220f", "be559584-51c3-45b2-a5a4-a05f10164683")
REFERENCE_FILES = ("subspace-voting-results.json", "tree-tradeoff-results.json",
                   "published-methods-results.json", "tag-order-significance-results.json",
                   "valid-path-ablation-results.json")


def close(actual, expected):
    if not math.isclose(actual, expected, abs_tol=2e-12, rel_tol=1e-10):
        raise ValueError(f"Archived score did not reproduce: {actual} != {expected}")


def check_scores(scores, archived, prediction_hash=None):
    if scores["n"] != archived["n"]:
        raise ValueError("Wrong held-out fold size")
    close(scores["losses"]["71"], archived["mean_padic_loss"])
    close(scores["exact_accuracy"], archived["exact_accuracy"])
    close(scores["root_accuracy"], archived["first_digit_accuracy"])
    if prediction_hash is not None and scores["prediction_sha256"] != prediction_hash:
        raise ValueError("Archived prediction hash did not reproduce")


def subspace_key(row):
    return tuple(row[k] for k in ("fold", "fraction", "roster", "members", "rule"))


def tree_key(fold, config):
    return fold, json.dumps(config, sort_keys=True, separators=(",", ":"))


def project(predictions, candidates):
    """Keep the original nearest-code decoder, including its integer tie-break."""
    valid = set(candidates)
    choices = sorted(valid)
    cache = {}
    for value in set(predictions) - valid:
        cache[value] = min(choices, key=lambda c: (
            q_weighted_distance(value, c, prime_base=71, q=71), c))
    return [value if value in valid else cache[value] for value in predictions]


def collect(conn, directory):
    sources = {name: json.loads((directory / name).read_text()) for name in REFERENCE_FILES}
    dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema="padjective")
    if _snapshot_digest(dataset) != SNAPSHOT_DIGEST or dataset.prime_base != 71:
        raise ValueError("The frozen benchmark changed")
    targets = [[r.encoded_path for r in dataset.records if r.cv_fold == f] for f in range(5)]
    if [len(y) for y in targets] != FOLD_SIZES:
        raise ValueError("The frozen fold populations changed")
    report = dict(experiment_key=EXPERIMENT, created_at=datetime.now(timezone.utc).isoformat(),
                  snapshot_id=PAPER_SNAPSHOT, snapshot_digest=SNAPSHOT_DIGEST,
                  products=len(dataset.records), fold_sizes=FOLD_SIZES, valuation_prime=71,
                  evaluation_weights=Q_VALUES, original_fitting_q=71,
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sources={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                           for name in REFERENCE_FILES},
                  subspace=[], tree=[], published=[], greedy=[], projection=[],
                  inference_ablation=[], pilots=[])

    old_subspace = {subspace_key(r): r for r in sources[REFERENCE_FILES[0]]["fold_results"]}
    with conn.cursor(name="q2_subspace") as cursor:
        cursor.execute("""SELECT cv_fold,fraction,roster,members,rule,evidence
            FROM padjective.paper_subspace_results WHERE batch_id=%s
            ORDER BY fraction,roster,members,rule,cv_fold""", (SOURCE_BATCH,))
        for fold, fraction, roster, members, rule, evidence in cursor:
            row = dict(fold=fold, fraction=fraction, roster=roster, members=members, rule=rule)
            old = old_subspace.pop(subspace_key(row))
            scores = rescore(targets[fold], evidence["predictions"], Q_VALUES)
            check_scores(scores, old["metrics"], old["prediction_sha256"])
            check_scores(scores, evidence["metrics"], evidence["prediction_sha256"])
            report["subspace"].append(dict(row, scores=scores))
    if old_subspace:
        raise ValueError("Incomplete saved subspace grid")
    print(f"Rescored {len(report['subspace'])} subspace prediction sets", flush=True)

    tree_source = sources[REFERENCE_FILES[1]]
    old_tree = {tree_key(r["fold"], r["config"]): r for r in tree_source["rows"]}
    with conn.cursor(name="q2_trees") as cursor:
        cursor.execute("""SELECT cv_fold,configuration,evidence
            FROM padjective.paper_tree_tradeoff_runs WHERE batch_id=%s
            ORDER BY config_key,cv_fold""", (tree_source["batch_id"],))
        for fold, config, evidence in cursor:
            old = old_tree.pop(tree_key(fold, config))
            scores = rescore(targets[fold], evidence["predictions"], Q_VALUES)
            check_scores(scores, old["metrics"], old["prediction_sha256"])
            check_scores(scores, evidence["metrics"], evidence["prediction_sha256"])
            report["tree"].append(dict(fold=fold, config=config, scores=scores))
    if old_tree:
        raise ValueError("Incomplete saved tree/forest grid")
    print(f"Rescored {len(report['tree'])} tree/forest prediction sets", flush=True)

    published = {r["run_id"]: r for r in sources[REFERENCE_FILES[2]]["runs"]
                 if r["method"] == "zubarev"}
    for run_id, fold, evidence, metrics in conn.execute("""SELECT run_id,cv_fold,evidence,metrics
        FROM padjective.paper_published_method_runs
        WHERE run_id=ANY(%s::uuid[]) ORDER BY run_id""", (list(published),)):
        old = published.pop(str(run_id))
        if evidence["snapshot_digest"] != SNAPSHOT_DIGEST:
            raise ValueError("Published-method snapshot differs")
        scores = rescore(targets[fold], evidence["predictions"], Q_VALUES)
        check_scores(scores, metrics["held_out"])
        check_scores(scores, old["metrics"]["held_out"])
        report["published"].append(dict(run_id=str(run_id), fold=fold,
                                         degree=old["degree"], seed=old["seed"], scores=scores))
    if published:
        raise ValueError("Incomplete published-method roster")

    controls = {(r["tag_order_strategy"], r["tag_order_seed"], f["cv_fold"]): f
                for r in sources[REFERENCE_FILES[3]]["runs"] for f in r["folds"]}
    for family, fold, seed, status, evidence in conn.execute("""SELECT family,cv_fold,
        member_index,status,evidence FROM padjective.paper_low_q_models
        WHERE experiment_key=%s AND q_label='71' AND family LIKE 'greedy_%%'
        ORDER BY family,member_index,cv_fold""", (NATIVE_EXPERIMENT,)):
        strategy = family.removeprefix("greedy_")
        seed = seed if strategy == "random" else None
        old = controls.pop((strategy, seed, fold))
        if status != "one_pass_complete" or evidence["snapshot_digest"] != SNAPSHOT_DIGEST:
            raise ValueError("Wrong greedy control")
        predictions = [int(v) for v in evidence["predictions"]]
        scores = rescore(targets[fold], predictions, Q_VALUES)
        close(scores["losses"]["71"], old["padic_loss_mean"])
        close(scores["exact_accuracy"], old["exact_accuracy"])
        if scores["prediction_sha256"] != evidence["scores"]["prediction_sha256"]:
            raise ValueError("Greedy prediction hash mismatch")
        report["greedy"].append(dict(fold=fold, strategy=strategy, seed=seed, scores=scores))
        if strategy == "taxonomy_association":
            candidates = {r.encoded_path for r in dataset.records if r.cv_fold != fold}
            projected = project(predictions, candidates)
            report["projection"].append(dict(fold=fold, raw=scores,
                invalid_count=sum(v not in candidates for v in predictions),
                projected=rescore(targets[fold], projected, Q_VALUES)))
    if controls or sum(r["invalid_count"] for r in report["projection"]) != 1046:
        raise ValueError("The original greedy controls/projection did not reproduce")

    ablation = sources[REFERENCE_FILES[4]]
    private, = conn.execute("SELECT private_evidence FROM padjective.paper_inference_ablations "
                            "WHERE analysis_id=%s", (ablation["analysis_id"],)).fetchone()
    old_ablation = {(r["fold"], r["members"]): r for r in ablation["rows"]}
    for row in private:
        fold, members = row["fold"], row["members"]
        old = old_ablation.pop((fold, members))
        scores = rescore(targets[fold], row["unrestricted_predictions"], Q_VALUES)
        check_scores(scores, old["unrestricted"], old["unrestricted_sha256"])
        report["inference_ablation"].append(dict(fold=fold, members=members, scores=scores))
    if old_ablation:
        raise ValueError("Incomplete saved inference ablation")

    # Failed attempts and unfinished Mihara runs have no held-out predictor.
    for run_id, config, evidence, metrics in conn.execute("""SELECT run_id,configuration,evidence,metrics
        FROM padjective.paper_salazar_single_runs WHERE snapshot_id=%s AND status='completed'
        AND run_id=ANY(%s::uuid[]) ORDER BY run_id""", (PAPER_SNAPSHOT, list(PILOT_RUNS))):
        if evidence["snapshot_digest"] != SNAPSHOT_DIGEST:
            raise ValueError("Pilot snapshot differs")
        fold = config.get("fold", 0)
        for variant, model in evidence["models"].items():
            scores = rescore(targets[fold], model["predictions"], Q_VALUES)
            check_scores(scores, metrics[variant]["held_out"], metrics[variant]["prediction_sha256"])
            report["pilots"].append(dict(run_id=str(run_id), fold=fold, variant=variant, scores=scores))

    expected_counts = dict(subspace=6125, tree=1690, published=45, greedy=45,
                           projection=5, inference_ablation=45, pilots=5)
    if {k:len(report[k]) for k in expected_counts} != expected_counts:
        raise ValueError("Incomplete original prediction roster; no report persisted")

    rows = [r["scores"] for family in ("subspace", "tree", "published", "greedy",
                                       "inference_ablation", "pilots") for r in report[family]]
    report["validation"] = dict(
        prediction_sets=len(rows), scored_predictions=sum(r["n"] for r in rows),
        source_q71_scores_reproduced=True,
        archived_prediction_hashes_verified_for=["subspace", "tree", "greedy", "inference_ablation", "pilots"],
        published_method_validation="Frozen run IDs, snapshot digest and all archived held-out metrics reproduced; prior prediction hashes unavailable",
        histogram_and_independent_row_distances_agree=True,
        models_refitted=0, individual_predictions_exported=False,
        family_counts={k: len(report[k]) for k in ("subspace", "tree", "published", "greedy",
                                                   "projection", "inference_ablation", "pilots")})
    return json.loads(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with db.get_connection() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        report = collect(conn, args.reference_dir)
        conn.execute("""INSERT INTO padjective.paper_low_q_reports(experiment_key,report)
            VALUES(%s,%s) ON CONFLICT(experiment_key) DO UPDATE
            SET report=EXCLUDED.report,updated_at=NOW()""", (EXPERIMENT, Jsonb(report)))
        saved = conn.execute("SELECT report FROM padjective.paper_low_q_reports WHERE experiment_key=%s",
                             (EXPERIMENT,)).fetchone()[0]
        if saved != report:
            raise ValueError("Persisted aggregate report did not reproduce")
        conn.commit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["validation"]), flush=True)


if __name__ == "__main__":
    main()
