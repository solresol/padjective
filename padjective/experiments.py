"""Utilities for managing hold-out experiments on tag ordering."""

from __future__ import annotations

import argparse
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import ranking


ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class Task:
    """Representation of a single hold-out experiment request."""

    task_id: int
    seed: int
    test_fraction: float


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
            error_message TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_holdout_tasks_status
            ON holdout_tasks(status)
        """
    )


def ensure_tasks(
    tasks_db: Path,
    total_tasks: int,
    *,
    test_fraction: float,
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
        for i in range(existing, existing + missing):
            conn.execute(
                """
                INSERT OR IGNORE INTO holdout_tasks
                    (seed, test_fraction, status, created_at)
                VALUES (?, ?, 'pending', ?)
                """,
                (seed_offset + i, test_fraction, now),
            )


def _select_pending_tasks(
    conn: sqlite3.Connection, limit: int
) -> List[Task]:
    rows = conn.execute(
        """
        SELECT id, seed, test_fraction
        FROM holdout_tasks
        WHERE status = 'pending'
        ORDER BY id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [Task(task_id=row["id"], seed=row["seed"], test_fraction=row["test_fraction"]) for row in rows]


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
            error_message,
            task_id,
        ),
    )


def _evaluate_holdout(
    pairs: Sequence[Tuple[str, str]],
    *,
    seed: int,
    test_fraction: float,
) -> Dict[str, Any]:
    """Compute hold-out accuracy metrics for a random split of ``pairs``."""

    if not pairs:
        return {
            "total_pairs": 0,
            "evaluated_pairs": 0,
            "skipped_pairs": 0,
            "correct_predictions": 0.0,
            "accuracy": None,
            "coverage": 0.0,
        }

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    test_size = max(int(len(shuffled) * test_fraction), 1)
    test_pairs = shuffled[:test_size]
    train_pairs = shuffled[test_size:]

    if not train_pairs:
        return {
            "total_pairs": len(test_pairs),
            "evaluated_pairs": 0,
            "skipped_pairs": len(test_pairs),
            "correct_predictions": 0.0,
            "accuracy": None,
            "coverage": 0.0,
        }

    leaderboard = ranking.compute_rankings(list(train_pairs))
    score_lookup = dict(zip(leaderboard["tag"], leaderboard["score"]))

    evaluated = 0
    skipped = 0
    correct = 0.0
    for winner, loser in test_pairs:
        winner_score = score_lookup.get(winner)
        loser_score = score_lookup.get(loser)
        if winner_score is None or loser_score is None:
            skipped += 1
            continue
        evaluated += 1
        if winner_score > loser_score:
            correct += 1.0
        elif winner_score == loser_score:
            correct += 0.5

    accuracy = correct / evaluated if evaluated else None
    coverage = evaluated / len(test_pairs) if test_pairs else 0.0

    return {
        "total_pairs": len(test_pairs),
        "evaluated_pairs": evaluated,
        "skipped_pairs": skipped,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "coverage": coverage,
    }


def run_tasks(
    *,
    tasks_db: Path,
    database: Path,
    take: int,
) -> List[Dict[str, Any]]:
    """Run up to ``take`` pending tasks against ``database``."""

    if take <= 0:
        return []

    pairs = ranking.load_pairs(database)
    results: List[Dict[str, Any]] = []

    with _connect(tasks_db) as conn:
        _ensure_schema(conn)
        tasks = _select_pending_tasks(conn, take)
        if not tasks:
            return []
        _mark_running(conn, [task.task_id for task in tasks])

    for task in tasks:
        try:
            metrics = _evaluate_holdout(pairs, seed=task.seed, test_fraction=task.test_fraction)
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

        test_fraction = conn.execute(
            "SELECT test_fraction FROM holdout_tasks LIMIT 1"
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

    total = sum(counts.values())
    return {
        "total": total,
        "counts": counts,
        "test_fraction": test_fraction[0] if test_fraction else None,
        "mean_accuracy": completed_metrics["mean_accuracy"] if completed_metrics else None,
        "mean_coverage": completed_metrics["mean_coverage"] if completed_metrics else None,
        "completed_tasks": completed_metrics["completed_tasks"] if completed_metrics else 0,
        "first_created": completed_metrics["first_created"] if completed_metrics else None,
        "last_completed": completed_metrics["last_completed"] if completed_metrics else None,
        "recent": [dict(row) for row in recent],
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
    if status["completed_tasks"]:
        mean_accuracy = status["mean_accuracy"]
        mean_coverage = status["mean_coverage"]
        lines.append(
            "  mean accuracy: {:.2%}".format(mean_accuracy) if mean_accuracy is not None else "  mean accuracy: n/a"
        )
        lines.append(
            "  mean coverage: {:.2%}".format(mean_coverage) if mean_coverage is not None else "  mean coverage: n/a"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Manage hold-out experiment tasks")
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
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of battles to reserve for testing",
    )
    init_parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset applied to generated random seeds",
    )

    status_parser = subparsers.add_parser("status", help="Display current task progress")

    run_parser = subparsers.add_parser("run", help="Execute pending tasks")
    run_parser.add_argument("--database", type=Path, required=True, help="SQLite battles database")
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
        "--test-fraction",
        type=float,
        default=0.2,
        help="Test fraction used when creating missing tasks",
    )

    args = parser.parse_args(argv)

    if args.command == "init":
        ensure_tasks(
            args.tasks_db,
            args.total,
            test_fraction=args.test_fraction,
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
                test_fraction=args.test_fraction,
            )
        results = run_tasks(tasks_db=args.tasks_db, database=args.database, take=args.take)
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


if __name__ == "__main__":
    main()
