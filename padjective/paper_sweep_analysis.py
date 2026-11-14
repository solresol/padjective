"""Summarise saved sweep histories for the three published ensemble cohorts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from psycopg.types.json import Jsonb

from . import db


REPORT_KEY = "paper-ensemble-sweeps-2026-11-04"


def load_histories(conn, evidence):
    original = json.loads((evidence / "ensemble-scaling-results.json").read_text())
    subspace = json.loads((evidence / "subspace-voting-results.json").read_text())
    low_q = json.loads((evidence / "low-q-retraining.json").read_text())
    live = json.loads((evidence / "manuscript-audit-results.json").read_text())
    published = {r["run_id"]: r for r in original["models"]}
    paper_rows = conn.execute("""
        SELECT run_id::text, cv_fold, status, evidence #> '{fit,history}'
        FROM padjective.paper_randomised_method_runs WHERE run_id = ANY(%s::uuid[])
    """, (list(published),)).fetchall()
    assert len(paper_rows) == len(published) == 1215
    for run, fold, status, history in paper_rows:
        assert (fold, len(history)) == (published[run]["fold"], published[run]["sweeps"])
        yield "paper", "71", 1., "paper", fold, published[run]["seed"], status, history
    for fold, member, fraction, status, history in conn.execute("""
        SELECT cv_fold, member_index, fraction, status, evidence #> '{fit,history}'
        FROM padjective.paper_subspace_runs WHERE batch_id = %s
    """, (subspace["batch_id"],)):
        yield "paper", "71", fraction, "paper", fold, member, status, history
    for fold, member, fraction, q, status, history in conn.execute("""
        SELECT cv_fold, member_index, fraction, q_label, status, evidence->'history'
        FROM padjective.paper_low_q_models
        WHERE experiment_key = %s AND family = 'coordinate'
    """, (low_q["experiment_key"],)):
        yield "low_q", q, fraction, "paper", fold, member, status, history
    live_ids = [row["batch_id"] for row in live["live"]["runs"]]
    assert len(set(live_ids)) == 9
    for batch, fold, member, fraction, status, history in conn.execute("""
        SELECT m.batch_id::text, m.fold, m.member, m.fraction,
               m.evidence #>> '{fit,status}', m.evidence #> '{fit,history}'
        FROM padjective.live_subspace_models m
        JOIN padjective.live_subspace_batches b USING (batch_id)
        WHERE m.batch_id = ANY(%s::uuid[]) AND b.status = 'validated'
    """, (live_ids,)):
        yield "store_held_out", "71", fraction, batch, fold, member, status, history


def summarise(histories):
    lengths = Counter(map(len, histories))
    sweeps = []
    for step in range(1, max(lengths) + 1):
        changes = [h[step - 1] for h in histories if len(h) >= step]
        productive = [c for c in changes if c > 0]
        sweeps.append(dict(
            sweep=step, attempted=len(changes), productive=len(productive),
            median_changes_if_productive=median(productive) if productive else None,
            changes_histogram=dict(sorted(Counter(changes).items())),
        ))
    return dict(fits=len(histories), median_sweeps=median(map(len, histories)),
                length_histogram=dict(sorted(lengths.items())), sweeps=sweeps)


def analyse(rows):
    cohorts = defaultdict(list)
    groups = defaultdict(list)
    seen = set()
    fingerprints = {}
    for cohort, q, fraction, batch, fold, member, status, history in rows:
        key = (cohort, q, fraction, batch, fold, member)
        assert key not in seen
        seen.add(key)
        assert status == "coordinate_optimum" and history
        assert [s["sweep"] for s in history] == list(range(1, len(history) + 1))
        assert all(s["complete"] for s in history)
        changes = [s["changes"] for s in history]
        assert changes[-1] == 0 and all(c > 0 for c in changes[:-1])
        objective_key = next(k for k in ("training_loss_units", "objective_units", "loss_units")
                             if k in history[0])
        objectives = [int(s[objective_key]) for s in history]
        assert all((a > b if n else a == b) for a, b, n in
                   zip(objectives, objectives[1:], changes[1:]))
        fingerprints[key] = changes
        cohorts[cohort].append(changes)
        groups[cohort, q, fraction, batch, fold].append(changes)
    assert {key: len(value) for key, value in cohorts.items()} == {
        "paper": 6075, "low_q": 4860, "store_held_out": 21870,
    }
    assert len(groups) == 135 and all(len(h) == 243 for h in groups.values())
    assert {key[1:3] for key in groups if key[0] == "paper"} == {
        ("71", f) for f in (.125, .25, .5, .75, 1.)
    }
    assert {key[1:3] for key in groups if key[0] == "low_q"} == {
        (q, f) for q in ("sqrt2", "2") for f in (.75, 1.)
    }
    # Stratify by the fourth productive sweep, near the centre of the paper
    # distribution. The fifth sweep was run for every member in the denominator.
    fourth = []
    for lower, upper in ((1, 1), (2, 5), (6, 20), (21, None)):
        eligible = [h for h in cohorts["paper"] if len(h) >= 5
                    and lower <= h[3] and (upper is None or h[3] <= upper)]
        fourth.append(dict(lower=lower, upper=upper, attempted=len(eligible),
                           productive=sum(h[4] > 0 for h in eligible)))
    assert sum(row["attempted"] for row in fourth) == sum(len(h) >= 5 for h in cohorts["paper"])
    return dict(
        report_key=REPORT_KEY,
        history_sha256=hashlib.sha256(json.dumps(sorted(fingerprints.items())).encode()).hexdigest(),
        cohorts={key: summarise(cohorts[key]) for key in ("paper", "low_q", "store_held_out")},
        groups=[dict(cohort=key[0], q=key[1], fraction=key[2], batch=key[3],
                     fold=key[4], fits=len(h), median_sweeps=median(map(len, h)),
                     length_histogram=dict(sorted(Counter(map(len, h)).items())))
                for key, h in sorted(groups.items())],
        paper_fifth_sweep_by_fourth_changes=fourth,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with db.get_connection() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        report = analyse(load_histories(conn, args.evidence_dir))
    report = json.loads(json.dumps(report))
    with db.get_connection() as conn:
        conn.execute("SET LOCAL default_tablespace = 'pg_default'")
        conn.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_sweep_reports (
            report_key TEXT PRIMARY KEY, report JSONB NOT NULL
        ) TABLESPACE pg_default""")
        conn.execute("""INSERT INTO padjective.paper_sweep_reports VALUES (%s, %s)
            ON CONFLICT (report_key) DO NOTHING""", (REPORT_KEY, Jsonb(report)))
        saved = conn.execute("SELECT report FROM padjective.paper_sweep_reports WHERE report_key=%s",
                             (REPORT_KEY,)).fetchone()[0]
        if saved != report:
            raise ValueError("The immutable report key already contains different evidence")
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
