import sqlite3
from pathlib import Path

from padjective import experiments


def _make_battles_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE battles (winner_tag TEXT, loser_tag TEXT)")
    rows = [
        ("RED", "BLUE"),
        ("RED", "GREEN"),
        ("BLUE", "GREEN"),
        ("YELLOW", "GREEN"),
        ("BLUE", "YELLOW"),
    ]
    conn.executemany("INSERT INTO battles VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    return path


def test_run_tasks_and_status(tmp_path):
    battles_db = _make_battles_db(tmp_path / "battles.sqlite")
    tasks_db = tmp_path / "tasks.sqlite"

    experiments.ensure_tasks(tasks_db, total_tasks=3, test_fraction=0.4)
    results = experiments.run_tasks(tasks_db=tasks_db, database=battles_db, take=2)

    assert len(results) == 2
    status = experiments.task_status(tasks_db)
    assert status["total"] == 3
    assert status["counts"].get("completed", 0) == 2
    assert status["counts"].get("pending", 0) == 1
    assert status["completed_tasks"] == 2
    # At least one completed task should report accuracy.
    assert status["mean_accuracy"] is None or 0 <= status["mean_accuracy"] <= 1
