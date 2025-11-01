"""Utilities for managing cross-validation experiments on tag ordering."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import db, ranking


ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class Task:
    """Representation of a single cross-validation experiment request."""

    task_id: int
    seed: int
    folds: int


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS holdout_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed INTEGER NOT NULL UNIQUE,
            test_fraction REAL NOT NULL,
            folds INTEGER NOT NULL DEFAULT 5,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            total_pairs INTEGER,
            evaluated_pairs INTEGER,
            skipped_pairs INTEGER,
            correct_predictions REAL,
            accuracy REAL,
            coverage REAL,
            winning_tags TEXT,
            losing_tags TEXT,
            error_message TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS holdout_task_folds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            fold_index INTEGER NOT NULL,
            test_pairs INTEGER NOT NULL,
            evaluated_pairs INTEGER NOT NULL,
            skipped_pairs INTEGER NOT NULL,
            correct_predictions REAL NOT NULL,
            accuracy REAL,
            coverage REAL,
            tag_win_rates TEXT,
            UNIQUE(task_id, fold_index)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_holdout_tasks_status
            ON holdout_tasks(status)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_holdout_task_folds_task
            ON holdout_task_folds(task_id)
        """
    )

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(holdout_tasks)")}
    if "folds" not in columns:
        conn.execute("ALTER TABLE holdout_tasks ADD COLUMN folds INTEGER NOT NULL DEFAULT 5")
    if "winning_tags" not in columns:
        conn.execute("ALTER TABLE holdout_tasks ADD COLUMN winning_tags TEXT")
    if "losing_tags" not in columns:
        conn.execute("ALTER TABLE holdout_tasks ADD COLUMN losing_tags TEXT")


def _chunk_pairs(pairs: Sequence[Tuple[str, str]], folds: int) -> List[List[Tuple[str, str]]]:
    if folds <= 0:
        raise ValueError("folds must be positive")
    total = len(pairs)
    base, remainder = divmod(total, folds)
    chunks: List[List[Tuple[str, str]]] = []
    start = 0
    for idx in range(folds):
        size = base + (1 if idx < remainder else 0)
        chunks.append(list(pairs[start : start + size]))
        start += size
    return chunks


def ensure_tasks(
    tasks_db: Path,
    total_tasks: int,
    *,
    folds: int = 5,
    seed_offset: int = 0,
) -> None:
    """Ensure ``total_tasks`` experiment entries exist in ``tasks_db``."""

    now = datetime.utcnow().strftime(ISO_FORMAT)
    with _connect(tasks_db) as conn:
        _ensure_schema(conn)
        existing = conn.execute("SELECT COUNT(*) FROM holdout_tasks").fetchone()[0]
        missing = max(total_tasks - existing, 0)
        if not missing:
            return
        test_fraction = 1.0 / max(folds, 1)
        for i in range(existing, existing + missing):
            conn.execute(
                """
                INSERT OR IGNORE INTO holdout_tasks
                    (seed, test_fraction, folds, status, created_at)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (seed_offset + i, test_fraction, folds, now),
            )


def _select_pending_tasks(
    conn: sqlite3.Connection, limit: int
) -> List[Task]:
    rows = conn.execute(
        """
        SELECT id, seed, folds
        FROM holdout_tasks
        WHERE status = 'pending'
        ORDER BY id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [Task(task_id=row["id"], seed=row["seed"], folds=row["folds"]) for row in rows]


def _mark_running(conn: sqlite3.Connection, task_ids: Sequence[int]) -> None:
    if not task_ids:
        return
    now = datetime.utcnow().strftime(ISO_FORMAT)
    conn.executemany(
        """
        UPDATE holdout_tasks
        SET status = 'running', started_at = ?
        WHERE id = ? AND status = 'pending'
        """,
        [(now, task_id) for task_id in task_ids],
    )


def _finalise_task(
    conn: sqlite3.Connection,
    task_id: int,
    *,
    status: str,
    metrics: Dict[str, Any],
    error_message: Optional[str] = None,
) -> None:
    now = datetime.utcnow().strftime(ISO_FORMAT)
    winning_tags = json.dumps(metrics.get("winning_tags") or [])
    losing_tags = json.dumps(metrics.get("losing_tags") or [])
    conn.execute(
        """
        UPDATE holdout_tasks
        SET status = ?,
            completed_at = ?,
            total_pairs = ?,
            evaluated_pairs = ?,
            skipped_pairs = ?,
            correct_predictions = ?,
            accuracy = ?,
            coverage = ?,
            winning_tags = ?,
            losing_tags = ?,
            error_message = ?
        WHERE id = ?
        """,
        (
            status,
            now,
            metrics.get("total_pairs"),
            metrics.get("evaluated_pairs"),
            metrics.get("skipped_pairs"),
            metrics.get("correct_predictions"),
            metrics.get("accuracy"),
            metrics.get("coverage"),
            winning_tags,
            losing_tags,
            error_message,
            task_id,
        ),
    )

    conn.execute("DELETE FROM holdout_task_folds WHERE task_id = ?", (task_id,))
    fold_rows: Iterable[Dict[str, Any]] = metrics.get("folds") or []
    records = [
        (
            task_id,
            row["fold_index"],
            row["test_pairs"],
            row["evaluated_pairs"],
            row["skipped_pairs"],
            row["correct_predictions"],
            row.get("accuracy"),
            row.get("coverage"),
            json.dumps(row.get("tag_win_rates") or []),
        )
        for row in fold_rows
    ]
    if records:
        conn.executemany(
            """
            INSERT INTO holdout_task_folds (
                task_id,
                fold_index,
                test_pairs,
                evaluated_pairs,
                skipped_pairs,
                correct_predictions,
                accuracy,
                coverage,
                tag_win_rates
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )


def _evaluate_cross_validation(
    pairs: Sequence[Tuple[str, str]],
    *,
    seed: int,
    folds: int,
) -> Dict[str, Any]:
    """Compute k-fold cross-validation metrics for ``pairs``."""

    if not pairs:
        return {
            "total_pairs": 0,
            "evaluated_pairs": 0,
            "skipped_pairs": 0,
            "correct_predictions": 0.0,
            "accuracy": None,
            "coverage": 0.0,
            "folds": [],
            "winning_tags": [],
            "losing_tags": [],
        }

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    fold_pairs = _chunk_pairs(shuffled, folds)

    total_pairs = 0
    total_evaluated = 0
    total_skipped = 0
    total_correct = 0.0
    fold_results: List[Dict[str, Any]] = []
    tag_fold_rates: Dict[str, List[float]] = defaultdict(list)
    tag_total_appearances: Dict[str, int] = defaultdict(int)

    for fold_index, test_pairs in enumerate(fold_pairs):
        total_pairs += len(test_pairs)
        if not test_pairs:
            fold_results.append(
                {
                    "fold_index": fold_index,
                    "test_pairs": 0,
                    "evaluated_pairs": 0,
                    "skipped_pairs": 0,
                    "correct_predictions": 0.0,
                    "accuracy": None,
                    "coverage": 0.0,
                    "tag_win_rates": [],
                }
            )
            continue

        train_pairs = [
            pair
            for idx, fold in enumerate(fold_pairs)
            if idx != fold_index
            for pair in fold
        ]
        if not train_pairs:
            skipped_pairs = len(test_pairs)
            total_skipped += skipped_pairs
            fold_results.append(
                {
                    "fold_index": fold_index,
                    "test_pairs": len(test_pairs),
                    "evaluated_pairs": 0,
                    "skipped_pairs": skipped_pairs,
                    "correct_predictions": 0.0,
                    "accuracy": None,
                    "coverage": 0.0,
                    "tag_win_rates": [],
                }
            )
            continue

        leaderboard = ranking.compute_rankings(list(train_pairs))
        score_lookup = dict(zip(leaderboard["tag"], leaderboard["score"]))

        evaluated = 0
        skipped = 0
        correct = 0.0
        fold_tag_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"wins": 0.0, "appearances": 0})

        for winner, loser in test_pairs:
            winner_score = score_lookup.get(winner)
            loser_score = score_lookup.get(loser)
            if winner_score is None or loser_score is None:
                skipped += 1
                continue
            evaluated += 1
            fold_tag_stats[winner]["appearances"] += 1
            tag_total_appearances[winner] += 1
            if winner_score > loser_score:
                correct += 1.0
                fold_tag_stats[winner]["wins"] += 1.0
            elif winner_score == loser_score:
                correct += 0.5
                fold_tag_stats[winner]["wins"] += 0.5

        total_evaluated += evaluated
        total_skipped += skipped
        total_correct += correct

        accuracy = correct / evaluated if evaluated else None
        coverage = evaluated / len(test_pairs) if test_pairs else 0.0

        tag_win_rates = []
        for tag, stats in sorted(fold_tag_stats.items()):
            appearances = stats["appearances"]
            if not appearances:
                continue
            win_rate = stats["wins"] / appearances
            tag_win_rates.append(
                {
                    "tag": tag,
                    "win_rate": win_rate,
                    "appearances": appearances,
                }
            )
            tag_fold_rates[tag].append(win_rate)

        fold_results.append(
            {
                "fold_index": fold_index,
                "test_pairs": len(test_pairs),
                "evaluated_pairs": evaluated,
                "skipped_pairs": skipped,
                "correct_predictions": correct,
                "accuracy": accuracy,
                "coverage": coverage,
                "tag_win_rates": tag_win_rates,
            }
        )

    aggregated_accuracy = total_correct / total_evaluated if total_evaluated else None
    aggregated_coverage = total_evaluated / total_pairs if total_pairs else 0.0

    def _summarise_tags(reverse: bool) -> List[Dict[str, Any]]:
        items = []
        for tag, rates in tag_fold_rates.items():
            if not rates:
                continue
            avg_rate = sum(rates) / len(rates)
            items.append(
                {
                    "tag": tag,
                    "average_win_rate": avg_rate,
                    "folds_with_appearances": len(rates),
                    "total_appearances": tag_total_appearances.get(tag, 0),
                }
            )
        items.sort(key=lambda item: (item["average_win_rate"], item["tag"]), reverse=reverse)
        return items[:20]

    winning_tags = _summarise_tags(True)
    losing_tags = _summarise_tags(False)

    return {
        "total_pairs": total_pairs,
        "evaluated_pairs": total_evaluated,
        "skipped_pairs": total_skipped,
        "correct_predictions": total_correct,
        "accuracy": aggregated_accuracy,
        "coverage": aggregated_coverage,
        "folds": fold_results,
        "winning_tags": winning_tags,
        "losing_tags": losing_tags,
    }


def run_tasks(
    *,
    tasks_db: Path,
    database,
    schema: str = "padjective",
    take: int,
) -> List[Dict[str, Any]]:
    """Run up to ``take`` pending tasks against ``database``."""

    if take <= 0:
        return []

    pairs = ranking.load_pairs(database, schema)
    results: List[Dict[str, Any]] = []

    with _connect(tasks_db) as conn:
        _ensure_schema(conn)
        tasks = _select_pending_tasks(conn, take)
        if not tasks:
            return []
        _mark_running(conn, [task.task_id for task in tasks])

    for task in tasks:
        try:
            metrics = _evaluate_cross_validation(pairs, seed=task.seed, folds=task.folds)
        except Exception as exc:  # pragma: no cover - defensive programming
            with _connect(tasks_db) as conn:
                _finalise_task(
                    conn,
                    task.task_id,
                    status="error",
                    metrics={
                        "total_pairs": None,
                        "evaluated_pairs": None,
                        "skipped_pairs": None,
                        "correct_predictions": None,
                        "accuracy": None,
                        "coverage": None,
                        "folds": None,
                        "winning_tags": None,
                        "losing_tags": None,
                    },
                    error_message=str(exc),
                )
            continue

        with _connect(tasks_db) as conn:
            _finalise_task(conn, task.task_id, status="completed", metrics=metrics)
        results.append({"task": task, "metrics": metrics})

    return results


def task_status(tasks_db: Path) -> Dict[str, Any]:
    """Return aggregate statistics about tasks stored in ``tasks_db``."""

    with _connect(tasks_db) as conn:
        _ensure_schema(conn)
        counts = {row["status"]: row["count"] for row in conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM holdout_tasks
            GROUP BY status
            """
        )}

        completed_metrics = conn.execute(
            """
            SELECT AVG(accuracy) AS mean_accuracy,
                   AVG(coverage) AS mean_coverage,
                   MIN(created_at) AS first_created,
                   MAX(completed_at) AS last_completed,
                   COUNT(*) AS completed_tasks
            FROM holdout_tasks
            WHERE status = 'completed' AND accuracy IS NOT NULL
            """
        ).fetchone()

        folds = conn.execute(
            "SELECT folds FROM holdout_tasks LIMIT 1"
        ).fetchone()

        recent = conn.execute(
            """
            SELECT id, seed, accuracy, coverage, evaluated_pairs, completed_at
            FROM holdout_tasks
            WHERE status = 'completed' AND completed_at IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT 10
            """
        ).fetchall()

        tag_lists = conn.execute(
            """
            SELECT winning_tags, losing_tags
            FROM holdout_tasks
            WHERE status = 'completed' AND winning_tags IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT 1
            """
        ).fetchone()

    total = sum(counts.values())
    return {
        "total": total,
        "counts": counts,
        "folds": folds[0] if folds else None,
        "mean_accuracy": completed_metrics["mean_accuracy"] if completed_metrics else None,
        "mean_coverage": completed_metrics["mean_coverage"] if completed_metrics else None,
        "completed_tasks": completed_metrics["completed_tasks"] if completed_metrics else 0,
        "first_created": completed_metrics["first_created"] if completed_metrics else None,
        "last_completed": completed_metrics["last_completed"] if completed_metrics else None,
        "recent": [dict(row) for row in recent],
        "winning_tags": json.loads(tag_lists["winning_tags"])
        if tag_lists and tag_lists["winning_tags"]
        else [],
        "losing_tags": json.loads(tag_lists["losing_tags"])
        if tag_lists and tag_lists["losing_tags"]
        else [],
    }


def _format_status(status: Dict[str, Any]) -> str:
    if not status["total"]:
        return "No tasks found."
    counts = status["counts"]
    lines = [
        f"Total tasks: {status['total']}",
        f"  pending: {counts.get('pending', 0)}",
        f"  running: {counts.get('running', 0)}",
        f"  completed: {counts.get('completed', 0)}",
        f"  error: {counts.get('error', 0)}",
    ]
    folds = status.get("folds")
    if folds:
        lines.append(f"  folds per task: {folds}")
    if status["completed_tasks"]:
        mean_accuracy = status["mean_accuracy"]
        mean_coverage = status["mean_coverage"]
        lines.append(
            "  mean accuracy: {:.2%}".format(mean_accuracy) if mean_accuracy is not None else "  mean accuracy: n/a"
        )
        lines.append(
            "  mean coverage: {:.2%}".format(mean_coverage) if mean_coverage is not None else "  mean coverage: n/a"
        )
    winning_tags = status.get("winning_tags") or []
    if winning_tags:
        lines.append("Top winning tags:")
        for item in winning_tags:
            avg = item.get("average_win_rate")
            folds_seen = item.get("folds_with_appearances")
            appearances = item.get("total_appearances")
            if avg is not None:
                lines.append(
                    f"  {item['tag']}: {avg:.2%} avg wins ({folds_seen} folds, {appearances} appearances)"
                )
            else:
                lines.append(f"  {item['tag']}: n/a")
    losing_tags = status.get("losing_tags") or []
    if losing_tags:
        lines.append("Most losing tags:")
        for item in losing_tags:
            avg = item.get("average_win_rate")
            folds_seen = item.get("folds_with_appearances")
            appearances = item.get("total_appearances")
            if avg is not None:
                lines.append(
                    f"  {item['tag']}: {avg:.2%} avg wins ({folds_seen} folds, {appearances} appearances)"
                )
            else:
                lines.append(f"  {item['tag']}: n/a")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Manage cross-validation experiment tasks")
    parser.add_argument(
        "--tasks-db",
        type=Path,
        default=Path("holdout_tasks.sqlite"),
        help="Path to the SQLite database storing experiment tasks",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Ensure tasks exist")
    init_parser.add_argument("--total", type=int, default=1000, help="Total number of tasks to ensure")
    init_parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds to use for cross-validation",
    )
    init_parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset applied to generated random seeds",
    )

    status_parser = subparsers.add_parser("status", help="Display current task progress")

    run_parser = subparsers.add_parser("run", help="Execute pending tasks")
    run_parser.add_argument(
        "--dsn",
        help="Postgres DSN. Uses SHOPIFY_DB_DSN or DATABASE_URL if omitted.",
    )
    run_parser.add_argument(
        "--schema",
        default="padjective",
        help="Schema containing the battles table.",
    )
    run_parser.add_argument(
        "--take",
        type=int,
        default=100,
        help="Maximum number of pending tasks to execute",
    )
    run_parser.add_argument(
        "--ensure-total",
        type=int,
        default=None,
        help="Ensure at least this many tasks exist before running",
    )
    run_parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds to use when creating missing tasks",
    )

    args = parser.parse_args(argv)

    if args.command == "init":
        ensure_tasks(
            args.tasks_db,
            args.total,
            folds=args.folds,
            seed_offset=args.seed_offset,
        )
    elif args.command == "status":
        status = task_status(args.tasks_db)
        print(_format_status(status))
    elif args.command == "run":
        if args.ensure_total is not None:
            ensure_tasks(
                args.tasks_db,
                args.ensure_total,
                folds=args.folds,
            )
        conn = db.get_connection(args.dsn)
        try:
            results = run_tasks(
                tasks_db=args.tasks_db,
                database=conn,
                schema=args.schema,
                take=args.take,
            )
        finally:
            conn.close()
        if not results:
            print("No pending tasks executed.")
            return
        for result in results:
            task: Task = result["task"]
            metrics: Dict[str, Any] = result["metrics"]
            accuracy = metrics["accuracy"]
            coverage = metrics["coverage"]
            accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "n/a"
            coverage_text = f"{coverage:.2%}" if coverage is not None else "n/a"
            print(
                f"Task {task.task_id} (seed={task.seed}) accuracy={accuracy_text} coverage={coverage_text}"
            )
            winners = metrics.get("winning_tags") or []
            if winners:
                print("  Top winning tags:")
                for item in winners:
                    avg = item.get("average_win_rate")
                    folds_seen = item.get("folds_with_appearances")
                    appearances = item.get("total_appearances")
                    avg_text = f"{avg:.2%}" if avg is not None else "n/a"
                    print(
                        f"    {item['tag']}: {avg_text} avg wins ({folds_seen} folds, {appearances} appearances)"
                    )
            losers = metrics.get("losing_tags") or []
            if losers:
                print("  Most losing tags:")
                for item in losers:
                    avg = item.get("average_win_rate")
                    folds_seen = item.get("folds_with_appearances")
                    appearances = item.get("total_appearances")
                    avg_text = f"{avg:.2%}" if avg is not None else "n/a"
                    print(
                        f"    {item['tag']}: {avg_text} avg wins ({folds_seen} folds, {appearances} appearances)"
                    )


if __name__ == "__main__":
    main()
