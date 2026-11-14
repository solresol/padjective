"""Second metric implementation for the complete saved ensemble experiment."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import db
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_validate_published import compare_scores
from .paper_validate_randomised import fingerprint


def prefix_count_scores(targets, predictions, p=71, precision=7):
    """Loss from a nested prefix-agreement histogram, not pairwise valuations."""
    actual, predicted = np.asarray(targets, dtype=np.int64), np.asarray(predictions, dtype=np.int64)
    q = p**precision
    assert actual.shape == predicted.shape and actual.ndim == 1 and len(actual)
    assert np.all((actual >= 0) & (actual < q)) and np.all((predicted >= 0) & (predicted < q))
    common = [len(actual)]+[int(np.count_nonzero(actual % p**depth == predicted % p**depth))
                          for depth in range(1, precision+1)]
    units = sum((common[depth]-common[depth+1])*p**(precision-depth) for depth in range(precision))
    return dict(n=len(actual), mean_padic_loss=float(Fraction(units, len(actual)*q)),
        exact_accuracy=float(Fraction(common[-1], len(actual))),
        first_digit_accuracy=float(Fraction(common[1], len(actual))))


def audit(conn, aggregate):
    batch_id = aggregate["batch_id"]
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT report FROM padjective.paper_ensemble_scaling_reports WHERE batch_id=%s", (batch_id,))
        saved = cur.fetchone()
    assert saved and saved["report"] == aggregate and aggregate["validation"]["status"] == "passed"
    dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema="padjective")
    assert _snapshot_digest(dataset) == SNAPSHOT_DIGEST == aggregate["snapshot_digest"]
    folds = np.array([r.cv_fold for r in dataset.records])
    y = np.array([r.encoded_path for r in dataset.records], dtype=np.int64)
    examples = sets = 0
    for index, model in enumerate(aggregate["models"], start=1):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""SELECT evidence->'models'->'coordinate' AS model FROM
                padjective.paper_randomised_method_runs WHERE run_id=%s""", (model["run_id"],))
            saved_model = cur.fetchone()["model"]
        assert fingerprint(saved_model["predictions"]) == model["prediction_sha256"]
        for key, part, mask in (("training_raw_predictions", "training_raw", folds != model["fold"]),
            ("training_predictions", "training", folds != model["fold"]),
            ("held_out_raw_predictions", "held_out_raw", folds == model["fold"]),
            ("predictions", "held_out", folds == model["fold"])):
            actual = prefix_count_scores(y[mask], saved_model[key])
            compare_scores(actual, model["metrics"][part])
            sets += 1
            examples += actual["n"]
        if index % 100 == 0:
            print(json.dumps(dict(event="prefix_metric_models", checked=index)), flush=True)
    ensemble_examples = 0
    for index, row in enumerate(aggregate["ensembles"], start=1):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""SELECT evidence FROM padjective.paper_ensemble_scaling_results
                WHERE batch_id=%s AND cv_fold=%s AND roster=%s AND members=%s""",
                (batch_id, row["fold"], row["roster"], row["members"]))
            evidence = cur.fetchone()["evidence"]
        assert fingerprint(evidence["predictions"]) == row["prediction_sha256"]
        actual = prefix_count_scores(y[folds == row["fold"]], evidence["predictions"])
        compare_scores(actual, row["metrics"])
        ensemble_examples += actual["n"]
        sets += 1
    conn.commit()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""SELECT count(*) AS new_fits, min(started_at) AS first_started,
            max(finished_at) AS last_finished, sum((evidence->>'elapsed_seconds')::float) AS summed_job_seconds
            FROM padjective.paper_randomised_method_runs WHERE batch_id=%s""", (batch_id,))
        execution = dict(cur.fetchone())
        cur.execute("""SELECT DISTINCT configuration->>'python' AS python, configuration->>'numpy' AS numpy
            FROM padjective.paper_randomised_method_runs WHERE batch_id=ANY(%s::uuid[])
            AND method='linear_random' AND NOT pilot""", ([batch_id, aggregate["base_batch"]],))
        versions = [dict(r) for r in cur.fetchall()]
    assert execution["new_fits"] == 1170 and versions == [dict(python="3.11.11", numpy="2.2.5")]
    execution["wall_seconds"] = (execution["last_finished"]-execution["first_started"]).total_seconds()
    for key in ("first_started", "last_finished"):
        execution[key] = execution[key].isoformat()
    conn.commit()
    return dict(status="passed", batch_id=batch_id, metric_implementation="nested prefix-agreement histogram",
        execution=execution, numerical_versions=versions,
        checked_model_rows=len(aggregate["models"]), checked_ensemble_rows=len(aggregate["ensembles"]),
        score_sets=sets, model_prediction_scores_checked=examples, ensemble_prediction_scores_checked=ensemble_examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    aggregate = json.loads(args.results.read_text())
    with db.get_connection() as conn:
        report = audit(conn, aggregate)
        report.update(aggregate_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(),
            audit_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
        with conn.cursor() as cur:
            cur.execute("SET LOCAL default_tablespace='pg_default'")
            cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_ensemble_scaling_metric_audit (
                batch_id UUID PRIMARY KEY, audited_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                report JSONB NOT NULL) TABLESPACE pg_default""")
            cur.execute("INSERT INTO padjective.paper_ensemble_scaling_metric_audit (batch_id,report) VALUES (%s,%s)",
                        (report["batch_id"], Jsonb(report)))
        conn.commit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print(json.dumps(dict(event="prefix_metric_audit_complete", **report)), flush=True)


if __name__ == "__main__":
    main()
