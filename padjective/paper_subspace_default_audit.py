"""Replay the paper's product-fold consensus with zero-score defaults abstaining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from . import db, paper_matched_store_folds as matched
from .paper_ensemble_scaling_batch import BANK_SEEDS
from .paper_low_q_experiment import SOURCE_BATCH
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_q2_primary import EXPERIMENT as REFERENCE
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_subspace_experiment import BASE_ENSEMBLE_BATCH
from .paper_sum_oracle import archived_report, save_report
from .paper_validate_randomised import distance_matrix, fingerprint
from .randomised_linear import consensus_predictions
from .subspace_voting import FRACTIONS

EXPERIMENT = "paper-subspace-default-audit-2026-11-10"
P, E = matched.PRIME, matched.PRECISION
N_MEMBERS = len(BANK_SEEDS)
SMALLEST_FRACTION = min(FRACTIONS)


def saved_votes(conn, fold, fraction, products):
    if fraction == 1.:
        rows = conn.execute("""
            SELECT (m->>'seed')::int, r.status, r.evidence #>> '{models,coordinate,default}',
                   r.evidence #> '{models,coordinate,held_out_raw_predictions}',
                   m->>'prediction_sha256'
            FROM padjective.paper_ensemble_scaling_reports e
            CROSS JOIN LATERAL jsonb_array_elements(e.report->'models') AS m
            JOIN padjective.paper_randomised_method_runs r ON r.run_id=(m->>'run_id')::uuid
            WHERE e.batch_id=%s AND (m->>'fold')::int=%s
        """, (BASE_ENSEMBLE_BATCH, fold))
    else:
        rows = conn.execute("""
            SELECT (evidence->>'fit_seed')::int, status, evidence->>'default',
                   evidence->'held_out_raw_predictions', evidence->>'prediction_sha256'
            FROM padjective.paper_subspace_runs
            WHERE batch_id=%s AND cv_fold=%s AND fraction=%s
        """, (SOURCE_BATCH, fold, fraction))
    rows = rows.fetchall()
    members = {seed: (status, int(default), raw, digest)
               for seed, status, default, raw, digest in rows}
    seeds = [base + fold for base in BANK_SEEDS]
    if len(rows) != N_MEMBERS or set(seeds) != members.keys():
        raise ValueError("Incomplete or unexpected member roster")

    raw_bank, defaults = [], []
    for seed in seeds:
        status, default, raw, digest = members[seed]
        if status != "coordinate_optimum" or len(raw) != products:
            raise ValueError("Incomplete fitted member or held-out predictions")
        if fingerprint([value if value else default for value in raw]) != digest:
            raise ValueError("Raw scores and default do not reproduce saved votes")
        raw_bank.append(raw)
        defaults.append(default)
    return np.asarray(raw_bank, dtype=np.int64).T, np.asarray(defaults, dtype=np.int64)


def consensus(votes, candidates):
    return consensus_predictions(votes, candidates, p=P, precision=E)[0]


def partial_consensus(votes, include, candidates, fallback):
    counts = include.sum(axis=1)
    result = fallback.copy()
    for count in np.unique(counts[counts > 0]):
        rows = counts == count
        kept = votes[rows][include[rows]].reshape(-1, count)
        result[rows] = consensus(kept, candidates)
    for i in np.unique(np.linspace(0, len(votes) - 1, 12, dtype=int)):
        kept = votes[i, include[i]]
        if len(kept):
            costs = distance_matrix(candidates, kept, P, E).sum(axis=1)
            if result[i] != candidates[np.argmin(costs)]:
                raise ValueError("Abstaining consensus differs from direct distance minimisation")
    return result


def endpoint(votes, is_default, candidates, targets, reference):
    original = consensus(votes, candidates)
    scores = matched.SCORE(targets, original.tolist())
    if (scores["prediction_sha256"] != reference["prediction_sha256"]
            or abs(scores["loss_q2"] - reference["losses"]["2"]) > 1e-12):
        raise ValueError("Original consensus did not reproduce the published endpoint")
    remaining = np.any(~is_default, axis=1)
    abstained = partial_consensus(votes, ~is_default, candidates, original)
    default_only = partial_consensus(votes, is_default, candidates, original)
    has_default = np.any(is_default, axis=1)
    return dict(
        original=scores,
        abstaining=matched.SCORE(targets, abstained.tolist()),
        all_abstain=int(np.count_nonzero(~remaining)),
        prediction_changes=int(np.count_nonzero(original != abstained)),
        products_with_default_vote=int(np.count_nonzero(has_default)),
        agrees_with_default_only=int(np.count_nonzero(has_default & (original == default_only))),
        root_agrees_with_default_only=int(
            np.count_nonzero(has_default & (original % P == default_only % P))
        ),
    )


def fold_audit(conn, fold, fraction, context, controls):
    targets = context["y_test"]
    candidates = np.unique(context["y_train"])
    raw, defaults = saved_votes(conn, fold, fraction, len(targets))
    is_default = raw == 0
    counts = is_default.sum(axis=1)
    default_values, codes = np.unique(defaults, return_inverse=True)
    default_frequency = np.bincount(codes, weights=is_default.sum(axis=0))
    most_frequent = int(np.argmax(default_frequency))
    result = dict(
        fold=fold, products=len(targets), members=N_MEMBERS,
        default_votes=int(counts.sum()), default_vote_rate=float(is_default.mean()),
        majority_default_products=int(np.count_nonzero(counts > N_MEMBERS / 2)),
        all_default_products=int(np.count_nonzero(counts == N_MEMBERS)),
        distinct_defaults=len(default_values),
        leading_default_code=int(default_values[most_frequent]),
        leading_default_votes=int(default_frequency[most_frequent]),
    )
    if fraction == SMALLEST_FRACTION:
        votes = np.where(is_default, defaults, raw)
        result["endpoints"] = {
            str(size): endpoint(votes[:, :size], is_default[:, :size], candidates, targets,
                                controls[fold, size])
            for size in (1, N_MEMBERS)
        }
    return result


def audit(conn):
    dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema="padjective")
    reference = archived_report(conn, REFERENCE)
    manifest, = conn.execute("""
        SELECT manifest FROM padjective.paper_subspace_batches
        WHERE batch_id=%s AND status='validated'
    """, (SOURCE_BATCH,)).fetchone()
    if (_snapshot_digest(dataset) != SNAPSHOT_DIGEST
            or reference["snapshot_digest"] != SNAPSHOT_DIGEST
            or manifest["snapshot_digest"] != SNAPSHOT_DIGEST):
        raise ValueError("The archives use different product folds")
    controls = {(r["fold"], r["members"]): r["scores"] for r in reference["subspace"]
                if r["roster"] == 0 and r["fraction"] == SMALLEST_FRACTION
                and r["rule"] == "raw_valid_medoid" and r["members"] in (1, N_MEMBERS)}
    contexts = matched.contexts(dataset)
    fractions = []
    for fraction in FRACTIONS:
        folds = [fold_audit(conn, f, fraction, c, controls) for f, c in contexts.items()]
        fractions.append(dict(
            fraction=fraction,
            mean_default_vote_rate=mean(f["default_vote_rate"] for f in folds),
            mean_majority_default_rate=mean(f["majority_default_products"] / f["products"] for f in folds),
            mean_all_default_rate=mean(f["all_default_products"] / f["products"] for f in folds),
            folds=folds,
        ))
        print(f"Audited {fraction:g} feature fraction", flush=True)

    return dict(
        experiment_key=EXPERIMENT, snapshot_id=PAPER_SNAPSHOT, snapshot_digest=SNAPSHOT_DIGEST,
        source_batch=SOURCE_BATCH, full_feature_batch=BASE_ENSEMBLE_BATCH, reference=REFERENCE,
        valuation_prime=P, consensus_q=P, evaluation_q=2,
        protocol="Use the archived ordered product-fold rosters. A zero raw score is a "
                 "default vote. Replay the smallest feature fraction at one and 243 members "
                 "with those votes excluded from the original consensus. Retain the original "
                 "prediction when every voter abstains. Rates and losses give folds equal weight.",
        fractions=fractions,
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
