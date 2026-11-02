"""Independently validate stored classical predictions against frozen PG rows.

Run from an archived source root with the paper's Python environment, after
collecting results.json. Export no product-level data.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, stdev

from psycopg.types.json import Jsonb

from padjective import db
from padjective.paper_classical_q_experiment import EXPERIMENT, MODELS
from padjective.paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from padjective.paper_randomised_methods import SNAPSHOT_DIGEST
from padjective.paper_revision_experiments import _load_paper_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    with db.get_connection() as conn:
        dataset = _load_paper_dataset(conn, snapshot_ref=str(PAPER_SNAPSHOT), schema="padjective")
        assert _snapshot_digest(dataset) == SNAPSHOT_DIGEST
        with conn.cursor() as cur:
            cur.execute("SELECT report FROM padjective.paper_low_q_reports WHERE experiment_key=%s",
                        (EXPERIMENT,))
            assert cur.fetchone()[0] == report
            cur.execute("""SELECT model_key,cv_fold,status,evidence FROM padjective.paper_classical_q_runs
                           WHERE experiment_key=%s ORDER BY model_key,cv_fold""", (EXPERIMENT,))
            rows = cur.fetchall()
        assert len(rows) == 25
        assert {(m, f) for m, f, _, _ in rows} == {(m, f) for m in MODELS for f in range(5)}
        checked, direct = 0, {}
        for model, fold, status, evidence in rows:
            assert status == "complete"
            expected = [r for r in dataset.records if r.cv_fold == fold]
            assert evidence["product_keys"] == [r.product_key for r in expected]
            assert evidence["actual_taxonomy_ids"] == [r.taxonomy_id for r in expected]
            encoded = [dataset.encoded_by_taxonomy[p] for p in evidence["predicted_taxonomy_ids"]]
            assert encoded == evidence["predicted_values"]
            losses = {q: [] for q in report["q_values"]}
            histogram, exact = Counter(), 0
            for row, prediction in zip(expected, encoded, strict=True):
                delta = abs(int(row.encoded_path) - int(prediction))
                depth = 0
                if not delta:
                    exact += 1
                else:
                    while delta % 71 == 0:
                        delta //= 71
                        depth += 1
                    histogram[depth] += 1
                for label, q in report["q_values"].items():
                    losses[label].append(q ** (-depth) if delta else 0.)
            scores = evidence["scores"]
            assert exact == scores["exact_correct"]
            assert dict(histogram) == {int(k): v for k, v in scores["first_disagreement_counts"].items()}
            assert sum(histogram.values()) + exact == scores["n"] == len(expected)
            averages = {q: math.fsum(values) / len(expected) for q, values in losses.items()}
            assert all(abs(scores["losses"][q] - value) < 1e-12 for q, value in averages.items())
            exported = next(s for s in report["series"] if s["configuration"] == model)["folds"][fold]
            assert all(exported[k] == value for k, value in scores.items())
            assert evidence["reference_check"]["matches"]
            assert all(abs(v) < 1e-12 for k, v in evidence["reference_check"].items() if k != "matches")
            direct[model, fold] = averages
            checked += len(expected)
        for s in report["series"]:
            for q in report["q_values"]:
                values = [direct[s["configuration"], f][q] for f in range(5)]
                assert abs(mean(values) - s["mean_losses"][q]) < 1e-12
                assert abs(stdev(values) - s["fold_sd"][q]) < 1e-12
        summary = dict(
            postgres_report_matches_export=True, snapshot_digest=SNAPSHOT_DIGEST,
            snapshot_products=6693, model_folds=25, independently_checked_predictions=checked,
            product_ids_labels_and_fold_order_match=True, score_grid=report["q_values"],
            independent_scores_and_fold_aggregates_match=True,
            all_archived_q71_losses_and_exact_accuracies_match=True,
            private_predictions_exported=False,
        )
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO padjective.paper_low_q_reports(experiment_key,report)
                           VALUES (%s,%s) ON CONFLICT(experiment_key)
                           DO UPDATE SET report=EXCLUDED.report,updated_at=NOW()""",
                        (EXPERIMENT + "-validation", Jsonb(summary)))
        conn.commit()
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
