"""Replay the frozen tag-order comparison and report four paired contrasts.

Reads only the pinned Shopify Postgres snapshot. No product-level output is
exported. Results are persisted in padjective with pg_default storage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t as student_t
from psycopg.types.json import Jsonb

from . import benchmark_runtime as runtime, db
from .paper_revision_experiments import _load_paper_dataset

SNAPSHOT = "244ddbe3-0a2c-4c04-9436-0a0253108a09"
FOLDS = (1300, 1337, 1348, 1372, 1336)
SEEDS = (7, 13, 23, 37, 101)
EXPECTED = {"taxonomy_association": .263237, "random": .312856,
            "frequency": .319671, "battle_elo": .324925,
            "mean_title_position": .342006}


def analyse(runs):
    expected = {(s, None) for s in EXPECTED if s != "random"}
    expected |= {("random", seed) for seed in SEEDS}
    keys = [(r["tag_order_strategy"], r["tag_order_seed"]) for r in runs]
    if len(keys) != 9 or set(keys) != expected:
        raise ValueError("Incomplete or duplicate strategy/seed grid")
    by_strategy = {}
    for strategy in EXPECTED:
        selected = [r for r in runs if r["tag_order_strategy"] == strategy]
        for run in selected:
            rows = run["folds"]
            if len(rows) != 5 or sorted(r["cv_fold"] for r in rows) != list(range(5)):
                raise ValueError("Incomplete or duplicate fold grid")
            for row in rows:
                if (row["num_test_samples"] != FOLDS[row["cv_fold"]]
                        or row["num_train_samples"] != 6693 - row["num_test_samples"]):
                    raise ValueError("Wrong frozen fold denominator")
        means = [statistics.mean(row["padic_loss_mean"] for run in selected
                 for row in run["folds"] if row["cv_fold"] == fold)
                 for fold in range(5)]
        if abs(statistics.mean(means) - EXPECTED[strategy]) > .0000005:
            raise ValueError(f"Published mean not reproduced: {strategy}")
        by_strategy[strategy] = means
    ratio = statistics.mean(n / (6693 - n) for n in FOLDS)
    contrasts = []
    for strategy in EXPECTED:
        if strategy == "taxonomy_association":
            continue
        differences = [a - b for a, b in zip(by_strategy[strategy],
                       by_strategy["taxonomy_association"])]
        mean = statistics.mean(differences)
        se = math.sqrt((1 / 5 + ratio) * statistics.variance(differences))
        p = float(2 * student_t.sf(abs(mean / se), 4)) if se else float(mean == 0)
        critical = float(student_t.ppf(.975, 4))
        contrasts.append(dict(strategy=strategy, fold_differences=differences,
                              mean_difference=mean, standard_error=se, df=4,
                              p_value=p, ci95=[mean-critical*se, mean+critical*se]))
    previous = 0.
    for rank, row in enumerate(sorted(contrasts, key=lambda r: r["p_value"])):
        previous = max(previous, min(1., (4-rank)*row["p_value"]))
        row["holm_p"] = previous
    return dict(fold_losses=by_strategy, test_train_ratio=ratio,
                contrasts=contrasts, family_size=4,
                method="Exploratory two-sided Nadeau-Bengio approximate corrected t; "
                       "five paired folds, df=4; random seeds averaged within fold; "
                       "Holm correction over four alternative orders")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-key", default="tag-orders-2026-10-21")
    args = parser.parse_args()
    with db.get_connection() as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        dataset = _load_paper_dataset(conn, snapshot_ref=SNAPSHOT, schema="padjective")
        assert dataset.features.shape == (6693, 2542) and dataset.prime_base == 71
        assert tuple(Counter(r.cv_fold for r in dataset.records)[i] for i in range(5)) == FOLDS
        conn.rollback()
        battles = runtime._derive_battles(dataset.records)
        runs = []
        for strategy in EXPECTED:
            for seed in (SEEDS if strategy == "random" else (None,)):
                runs.append(runtime._evaluate_umllr_strategy(
                    strategy=strategy, seed=seed, folds=range(5),
                    records=dataset.records, battles=battles, base=71))
                print(f"Completed {strategy}, seed={seed}", flush=True)
        report = dict(snapshot_id=SNAPSHOT, products=6693, features=2542,
                      created_at=datetime.now(timezone.utc).isoformat(),
                      analysis_key=args.analysis_key,
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      runtime_sha256=hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
                      runs=runs, **analyse(runs))
        db.ensure_table(conn, "padjective", "paper_order_significance", [
            "analysis_key TEXT PRIMARY KEY", "snapshot_id UUID NOT NULL",
            "report JSONB NOT NULL", "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"])
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO padjective.paper_order_significance "
                           "(analysis_key,snapshot_id,report) VALUES (%s,%s,%s) "
                           "ON CONFLICT (analysis_key) DO NOTHING",
                           (args.analysis_key, SNAPSHOT, Jsonb(report)))
            cursor.execute("SELECT report FROM padjective.paper_order_significance WHERE analysis_key=%s",
                           (args.analysis_key,))
            saved = cursor.fetchone()[0]
            if saved != report:
                raise ValueError("Analysis key already contains a different report")
        conn.commit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["contrasts"], indent=2))


if __name__ == "__main__":
    main()
