"""Audit valuation separation versus occupied powers in saved frozen fits.

This is inference-only. It never fits models or modifies their source tables.
The selected September first-pass checkpoints are NOT the published floating-
point greedy model. Only aggregate results are exported from PostgreSQL.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import uuid

SNAPSHOT = "244ddbe3-0a2c-4c04-9436-0a0253108a09"
SNAPSHOT_DIGEST = "039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222"
BATCH = "a2b0744f-175e-4899-a3d5-e6a8478a8d73"
FIT_COMMIT = "289f5498c498c2c56eb13e31f29fa9958e049df1"
P, E = 71, 7
GROUPS = ("all", "zero", "one", "multi", "multi_valuation_separated",
          "multi_not_valuation_separated", "multi_disjoint", "multi_not_disjoint",
          "multi_valuation_only")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def valuation(value: int, p: int):
    if not value:
        return None
    value = abs(value)
    exponent = 0
    while value % p == 0:
        exponent += 1
        value //= p
    return exponent


def occupied_powers(value: int, p: int, depth: int) -> set[int]:
    """Canonical Z/p**depth digits, including signed integers via reduction.

    For negative integers this is a truncation of their infinite p-adic
    expansion; using abs(value) would instead inspect the wrong number.
    """
    if depth < 1 or p < 2:
        raise ValueError("Positive depth and base >= 2 required")
    value %= p**depth
    support = set()
    for index in range(depth):
        value, digit = divmod(value, p)
        if digit:
            support.add(index)
    return support


def classify(contributions, p: int, depth: int) -> dict:
    active = [int(z) for z in contributions if z != 0]
    values = [valuation(z, p) for z in active]
    seen = set()
    disjoint = True
    hidden = 0
    for z in active:
        support = occupied_powers(z, p, depth)
        hidden += not support
        disjoint &= not bool(seen & support)
        seen |= support
    return dict(active_count=len(active), valuation_separated=len(set(values)) == len(values),
                disjoint=disjoint, hidden_contributions=hidden)


def has_carry(contributions, p: int, depth: int) -> bool:
    """Carries in addition of canonical nonnegative depth-digit representatives."""
    digits = [int(z) % p**depth for z in contributions if z != 0]
    carry = 0
    occurred = False
    for _ in range(depth):
        total = carry + sum(z % p for z in digits)
        carry = total // p
        occurred |= carry > 0
        digits = [z // p for z in digits]
    return occurred


def scalar_loss_units(target: int, prediction: int, p: int, depth: int) -> int:
    delta = (target - prediction) % p**depth
    return 0 if not delta else p**(depth - valuation(delta, p))


def score(rows, p=P, depth=E) -> dict:
    n = len(rows)
    exact = sum(r["target"] == r["prediction"] for r in rows)
    units = sum(r["loss_units"] for r in rows)
    wrong = n - exact
    return dict(n=n, exact=exact, incorrect=wrong, loss_units=units,
                exact_hit_rate=float(Fraction(exact, n)) if n else None,
                mean_padic_loss=float(Fraction(units, n*p**depth)) if n else None,
                incorrect_mean_padic_loss=float(Fraction(units, wrong*p**depth)) if wrong else None,
                taxonomies=len({r["taxonomy_id"] for r in rows}),
                raw_zero=sum(r["raw"] == 0 for r in rows),
                nonzero_integer_sum_but_modular_zero=sum(r["raw"] == 0 and r["integer_sum"] != 0 for r in rows),
                carry_predictions=sum(r["carry"] for r in rows),
                active_count_histogram=dict(sorted(Counter(r["active_count"] for r in rows).items())))


def groups(row) -> list[str]:
    active = row["active_count"]
    if active < 2:
        return ["all", "zero" if not active else "one"]
    answer = ["all", "multi",
              "multi_valuation_separated" if row["valuation_separated"] else "multi_not_valuation_separated",
              "multi_disjoint" if row["disjoint"] else "multi_not_disjoint"]
    if row["valuation_separated"] and not row["disjoint"]:
        answer.append("multi_valuation_only")
    return answer


def aggregate(rows) -> dict:
    buckets = {k: [] for k in GROUPS}
    for row in rows:
        for group in groups(row):
            buckets[group].append(row)
    return {key: score(values) for key, values in buckets.items()}


def summarise(rows, contributions) -> dict:
    if len(rows) != len(contributions):
        raise ValueError("Contributions must align with held-out rows")
    by_fold, by_taxonomy, by_active = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        by_fold[row["fold"]].append(row)
        by_taxonomy[row["taxonomy_id"]].append(row)
        by_active[row["active_count"]].append(row)
    sensitivity = []
    for depth in range(1, E+1):
        buckets = {k: [] for k in ("disjoint", "not_disjoint", "visible_disjoint", "visible_not_disjoint")}
        hidden_rows = hidden_contributions = 0
        for row, z in zip(rows, contributions, strict=True):
            if row["active_count"] < 2:
                continue
            c = classify(z, P, depth)
            key = "disjoint" if c["disjoint"] else "not_disjoint"
            buckets[key].append(row)
            hidden_rows += c["hidden_contributions"] > 0
            hidden_contributions += c["hidden_contributions"]
            if not c["hidden_contributions"]:
                buckets["visible_"+key].append(row)
        sensitivity.append(dict(depth=depth, hidden_rows=hidden_rows,
                                hidden_contributions=hidden_contributions,
                                groups={k: score(v) for k, v in buckets.items()}))
    return dict(pooled=aggregate(rows),
                by_fold={str(k): aggregate(v) for k, v in sorted(by_fold.items())},
                by_taxonomy={k: aggregate(v) for k, v in sorted(by_taxonomy.items())},
                by_active_count={str(k): aggregate(v) for k, v in sorted(by_active.items())},
                depth_sensitivity=sensitivity,
                aggregation="Pooled rows are product-weighted; by_fold also preserves each fold separately.",
                interpretation="Descriptive post-hoc association; neither predicate was imposed in fitting. No causal claim.")


def audit_checkpoint(records, feature_names, evidence, metrics, fold):
    """Verify saved predictions independently, using precisely their frozen features."""
    import numpy as np
    from .randomised_linear import fit_default

    if evidence["snapshot_digest"] != SNAPSHOT_DIGEST or evidence["feature_names"] != list(feature_names):
        raise ValueError("Saved fit does not match the frozen matrix/order")
    if (evidence["p"], evidence["precision"]) != (P, E):
        raise ValueError("Unexpected saved arithmetic")
    model = evidence["models"]["one_pass"]
    weights = [int(w) for w in model["coefficients"]]
    if not model["completed"] or len(weights) != len(feature_names) or not all(0 <= w < P**E for w in weights):
        raise ValueError("Incomplete or noncanonical saved checkpoint")
    lookup = dict(zip(feature_names, weights, strict=True))
    targets, training_raw = [], []
    for record in records:
        if record.tags != sorted(set(record.tags)):
            raise ValueError("Frozen feature rows are not binary/deduplicated")
        if record.cv_fold != fold:
            targets.append(int(record.encoded_path))
            training_raw.append(sum(lookup[t] for t in record.tags) % P**E)
    if training_raw != model["training_raw_predictions"]:
        raise ValueError("Saved training raw predictions do not reconstruct")
    default = fit_default(np.array(targets), np.array(training_raw), p=P, precision=E)
    if default != model["default"]:
        raise ValueError("Saved training-only default does not reconstruct")
    if [default if z == 0 else z for z in training_raw] != model["training_predictions"]:
        raise ValueError("Saved training predictions do not reconstruct")

    rows, contributions, testing_raw, predictions = [], [], [], []
    for record in records:
        if record.cv_fold != fold:
            continue
        z = [lookup[t] for t in record.tags if lookup[t] != 0]
        classification = classify(z, P, E)
        if classification["hidden_contributions"] or (classification["disjoint"] and not classification["valuation_separated"]):
            raise ValueError("Full-depth support/valuation implication failed")
        integer_sum = sum(z)
        raw = integer_sum % P**E
        prediction = default if raw == 0 else raw
        target = int(record.encoded_path)
        row = dict(product_key=record.product_key, taxonomy_id=record.taxonomy_id,
                   fold=fold, target=target, raw=raw, prediction=prediction,
                   integer_sum=integer_sum, carry=has_carry(z, P, E),
                   loss_units=scalar_loss_units(target, prediction, P, E), **classification)
        if classification["active_count"] and classification["valuation_separated"] and raw == 0:
            raise ValueError("Nonempty separated contributions cancelled")
        rows.append(row)
        contributions.append(z)
        testing_raw.append(raw)
        predictions.append(prediction)
    if testing_raw != model["held_out_raw_predictions"] or predictions != model["predictions"]:
        raise ValueError("Saved held-out predictions do not reconstruct")
    actual = score(rows)
    for key, value in (("n", actual["n"]), ("exact_accuracy", actual["exact_hit_rate"]),
                       ("mean_padic_loss", actual["mean_padic_loss"])):
        if abs(metrics["one_pass"]["held_out"][key] - value) > 1e-14:
            raise ValueError("Saved held-out metrics do not reconstruct")
    return rows, contributions


def analyse():
    import psycopg
    from . import db
    from .paper_published_methods import _snapshot_digest
    from .paper_revision_experiments import _load_paper_dataset

    started = time.monotonic()
    all_rows, all_contributions, models = [], [], []
    with db.get_connection() as conn:
        conn.read_only = True
        conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        conn.execute("SET LOCAL statement_timeout='20s'")
        conn.execute("SET LOCAL lock_timeout='1s'")
        data = _load_paper_dataset(conn, snapshot_ref=SNAPSHOT, schema="padjective")
        if _snapshot_digest(data) != SNAPSHOT_DIGEST or data.features.shape != (6693, 2542) or set(data.features.data) != {1}:
            raise ValueError("Frozen snapshot changed")
        feature_index = {tag: i for i, tag in enumerate(data.feature_names)}
        for i, record in enumerate(data.records):
            if set(data.features.getrow(i).indices) != {feature_index[t] for t in record.tags}:
                raise ValueError("Feature relation and predictor records disagree")
        saved = conn.execute("""SELECT run_id,cv_fold,configuration,status,evidence,metrics
            FROM padjective.paper_randomised_method_runs
            WHERE batch_id=%s AND method='linear_association' AND NOT pilot ORDER BY cv_fold""", (BATCH,)).fetchall()
        if [row[1] for row in saved] != list(range(5)):
            raise ValueError("Expected exactly one saved fit per fold")
        for run_id, fold, configuration, status, evidence, metrics in saved:
            if configuration["source_commit"] != FIT_COMMIT or status != "coordinate_optimum":
                raise ValueError("Saved source fit provenance changed")
            rows, contributions = audit_checkpoint(data.records, data.feature_names, evidence, metrics, fold)
            model = evidence["models"]["one_pass"]
            models.append(dict(run_id=str(run_id), fold=fold, source_commit=FIT_COMMIT,
                               model_sha256=fingerprint(model), configuration_sha256=fingerprint(configuration),
                               coefficient_sha256=fingerprint(model["coefficients"]),
                               held_out_prediction_sha256=fingerprint(model["predictions"]),
                               source_evidence_sha256=fingerprint(evidence),
                               verified_training_rows=len(model["training_predictions"]),
                               verified_held_out_rows=len(rows)))
            all_rows.extend(rows)
            all_contributions.extend(contributions)
    if len({r["product_key"] for r in all_rows}) != len(all_rows) or len(all_rows) != 6693:
        raise ValueError("Held-out population has omissions or repetitions")
    private = dict(rows=all_rows, contributions=all_contributions)
    report = dict(analysis_id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
                  analysis_version=1, snapshot_id=SNAPSHOT, snapshot_digest=SNAPSHOT_DIGEST,
                  batch_id=BATCH, source_table="padjective.paper_randomised_method_runs",
                  model_key="linear_association/one_pass", prime=P, precision=E,
                  model_description="Saved first complete association-ordered sweep of the exact finite-quotient coordinate fitter; NOT the published floating-point one-pass greedy fit.",
                  definitions=dict(valuation="Distinct valuations of nonzero active coefficients, counted once per binary feature.",
                                   powers="Pairwise-disjoint nonzero canonical digits at positions 0 through 6 of coefficients modulo 71**7.",
                                   negatives="Saved coefficients are canonical residues 0<=w<71**7. Signed equivalents must be reduced modulo 71**depth, never replaced with abs(w). No claim of disjoint infinite Z_p supports.",
                                   preprocessing="Use the frozen product-tag relation and exact feature order used by saved fits. Snapshot construction already uppercases, deduplicates and removes substring-nested tags before anonymisation; never reparse mutable raw tags or filter anonymous IDs as strings.",
                                   fallback="Sum contributions modulo 71**7; if that residue is zero use the independently verified training-only fitted default. Zero coefficients do not enter either predicate. Zero/one-active cases are reported separately.",
                                   sensitivity="Depths 1..7 change only the support predicate, never predictions, loss or the valuation predicate. Report hidden contributions and fully-visible subgroups to expose truncation vacuity."),
                  models=models, source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependency_source_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                            for name in ("paper_revision_experiments.py", "paper_published_methods.py", "randomised_linear.py", "product_taxonomy_bench.py")},
                  python=platform.python_version(), private_evidence_sha256=fingerprint(private),
                  validation=dict(status="passed", fits_verified=5, held_out_rows=6693,
                                  training_rows=5*6693-6693, predictions_reconstructed=True,
                                  defaults_recomputed_from_training_only=True, snapshot_matrix_and_records_matched=True,
                                  support_implies_valuation_at_full_depth=True),
                  results=summarise(all_rows, all_contributions), elapsed_seconds=time.monotonic()-started)
    return report, private


def persist(report, private):
    from psycopg.types.json import Jsonb
    from . import db
    with db.get_connection() as conn:
        conn.execute("SET LOCAL lock_timeout='1s'")
        conn.execute("SET LOCAL statement_timeout='20s'")
        conn.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_refinement_comparisons (
            analysis_id UUID PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            report JSONB NOT NULL, private_evidence JSONB NOT NULL
        ) TABLESPACE pg_default""")
        conn.execute("""INSERT INTO padjective.paper_refinement_comparisons
            (analysis_id,report,private_evidence) VALUES (%s,%s,%s)""",
                     (report["analysis_id"], Jsonb(report), Jsonb(private)))
        saved = conn.execute("SELECT report,private_evidence FROM padjective.paper_refinement_comparisons WHERE analysis_id=%s",
                             (report["analysis_id"],)).fetchone()
        if fingerprint(saved[0]) != fingerprint(report) or fingerprint(saved[1]) != fingerprint(private):
            raise ValueError("Persisted evidence did not round-trip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New aggregate JSON file; never overwrite existing results")
    parser.add_argument("--persist", action="store_true", help="Append report and private row evidence to the dedicated pg_default table")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    report, private = analyse()
    if args.persist:
        persist(report, private)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(dict(analysis_id=report["analysis_id"], elapsed_seconds=report["elapsed_seconds"],
                          pooled=report["results"]["pooled"]), indent=2))


if __name__ == "__main__":
    main()
