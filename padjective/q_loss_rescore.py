"""Rescore the fixed paper predictions near Q=1, using PostgreSQL only."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from psycopg.types.json import Jsonb

from . import db
from .paper_low_q_experiment import EXPERIMENT as NATIVE_EXPERIMENT, SOURCE_BATCH, fingerprint
from .paper_classical_q_experiment import EXPERIMENT as CLASSICAL_EXPERIMENT
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset, q_weighted_distance
from .q_loss_curves import FOLD_SIZES, STYLES, canonical_sha256


EXPERIMENT = "q-loss-near-one-2026-11-02"
NEW_Q_VALUES = {"1.01": 1.01, "1.1": 1.1}


def rescore(actual, predicted, q_values):
    if not actual or len(actual) != len(predicted):
        raise ValueError("Nonempty matching held-out predictions are required")
    if any(not math.isfinite(q) or q <= 1 for q in q_values.values()):
        raise ValueError("Q must be finite and greater than one")
    histogram, exact = Counter(), 0
    for a, b in zip(actual, predicted, strict=True):
        delta = abs(int(a) - int(b))
        if not delta:
            exact += 1
            continue
        depth = 0
        while delta % 71 == 0:
            delta //= 71
            depth += 1
        histogram[depth] += 1
    n = len(actual)
    losses = {key: math.fsum(count * q**(-depth) for depth, count in histogram.items()) / n
              for key, q in q_values.items()}
    for key, q in q_values.items():
        # Independent row-by-row distance implementation, separate from the
        # first-disagreement histogram used to compute the exported scores.
        direct = math.fsum(q_weighted_distance(a, b, prime_base=71, q=q)
                           for a, b in zip(actual, predicted, strict=True)) / n
        assert abs(direct - losses[key]) < 1e-12
        assert histogram.get(0, 0) / n - 1e-12 <= direct <= 1 - exact / n + 1e-12
    return dict(n=n, losses=losses, exact_accuracy=exact / n,
                root_accuracy=1 - histogram.get(0, 0) / n,
                exact_correct=exact, first_disagreement_counts=dict(histogram),
                prediction_sha256=fingerprint(predicted))


def load_predictions(conn, dataset, key, fold):
    expected = [r for r in dataset.records if r.cv_fold == fold]
    if key == "taxonomy_association":
        row = conn.execute("""SELECT status,evidence FROM padjective.paper_low_q_models
            WHERE experiment_key=%s AND q_label='71' AND family='greedy_taxonomy_association'
              AND cv_fold=%s AND fraction=1 AND member_index=-1""", (NATIVE_EXPERIMENT, fold)).fetchone()
        assert row and row[0] == "one_pass_complete"
        e = row[1]
        assert e["snapshot_digest"] == SNAPSHOT_DIGEST
        predictions, digest = e["predictions"], e["scores"]["prediction_sha256"]
    elif key.startswith("ensemble_"):
        fraction = .75 if key == "ensemble_0.75_243" else 1.
        row = conn.execute("""SELECT evidence FROM padjective.paper_subspace_results
            WHERE batch_id=%s AND cv_fold=%s AND fraction=%s AND roster=0
              AND members=243 AND rule='raw_valid_medoid'""", (SOURCE_BATCH, fold, fraction)).fetchone()
        assert row
        e = row[0]
        predictions, digest = e["predictions"], e["prediction_sha256"]
    else:
        row = conn.execute("""SELECT status,evidence FROM padjective.paper_classical_q_runs
            WHERE experiment_key=%s AND model_key=%s AND cv_fold=%s""",
                           (CLASSICAL_EXPERIMENT, key, fold)).fetchone()
        assert row and row[0] == "complete"
        e = row[1]
        assert e["product_keys"] == [r.product_key for r in expected]
        assert e["actual_taxonomy_ids"] == [r.taxonomy_id for r in expected]
        predictions = [dataset.encoded_by_taxonomy[p] for p in e["predicted_taxonomy_ids"]]
        assert predictions == e["predicted_values"]
        digest = e["scores"]["prediction_sha256"]
    predictions = [int(p) for p in predictions]
    assert len(predictions) == FOLD_SIZES[fold] and fingerprint(predictions) == digest
    return [r.encoded_path for r in expected], predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    with db.get_connection() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        sources = [conn.execute("SELECT report FROM padjective.paper_low_q_reports WHERE experiment_key=%s",
                                (key,)).fetchone()[0] for key in (NATIVE_EXPERIMENT, CLASSICAL_EXPERIMENT)]
        dataset = _load_paper_dataset(conn, snapshot_ref=str(PAPER_SNAPSHOT), schema="padjective")
        assert _snapshot_digest(dataset) == SNAPSHOT_DIGEST
        assert all(r["snapshot_digest"] == SNAPSHOT_DIGEST for r in sources)
        assert sources[0]["q_values"] == sources[1]["q_values"]
        all_q = dict(NEW_Q_VALUES, **sources[0]["q_values"])
        reference = {s["configuration"]: s for s in sources[0]["series"] if s["training_q"] == "71"}
        reference.update({s["configuration"]: s for s in sources[1]["series"]})
        series, checked = [], 0
        for key in STYLES:
            folds = []
            for fold in range(5):
                actual, predicted = load_predictions(conn, dataset, key, fold)
                scores = rescore(actual, predicted, all_q)
                old = reference[key]["folds"][fold]
                assert scores["n"] == old["n"]
                assert abs(scores["exact_accuracy"] - old["exact_accuracy"]) < 1e-12
                assert abs(scores["root_accuracy"] - old["root_accuracy"]) < 1e-12
                assert all(abs(scores["losses"][q] - old["losses"][q]) < 1e-12
                           for q in sources[0]["q_values"])
                scores["losses"] = {q: scores["losses"][q] for q in NEW_Q_VALUES}
                folds.append(dict(cv_fold=fold, **scores))
                checked += len(actual)
            series.append(dict(
                configuration=key, training_q=reference[key]["training_q"], folds=folds,
                mean_losses={q: mean(f["losses"][q] for f in folds) for q in NEW_Q_VALUES},
                fold_sd={q: stdev(f["losses"][q] for f in folds) for q in NEW_Q_VALUES},
                exact_accuracy=mean(f["exact_accuracy"] for f in folds),
                root_accuracy=mean(f["root_accuracy"] for f in folds),
            ))
        report = dict(
            experiment_key=EXPERIMENT, created_at=datetime.now(timezone.utc).isoformat(),
            source_commit=args.source_commit,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            snapshot_id=str(PAPER_SNAPSHOT), snapshot_digest=SNAPSHOT_DIGEST,
            products=len(dataset.records), prime=71, q_values=NEW_Q_VALUES, series=series,
            sources=[dict(experiment_key=r["experiment_key"], canonical_sha256=canonical_sha256(r)) for r in sources],
            validation=dict(independently_checked_predictions=checked, checked_q_values=all_q,
                            existing_six_q_fold_scores_reproduce=True, prediction_hashes_match=True,
                            classical_product_ids_labels_and_fold_order_match=True,
                            histogram_and_row_distances_match=True, models_retrained=0),
        )
        conn.execute("""INSERT INTO padjective.paper_low_q_reports(experiment_key,report)
            VALUES(%s,%s) ON CONFLICT(experiment_key)
            DO UPDATE SET report=EXCLUDED.report,updated_at=NOW()""", (EXPERIMENT, Jsonb(report)))
        assert conn.execute("SELECT report FROM padjective.paper_low_q_reports WHERE experiment_key=%s",
                            (EXPERIMENT,)).fetchone()[0] == json.loads(json.dumps(report))
        conn.commit()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["validation"]))


if __name__ == "__main__":
    main()
