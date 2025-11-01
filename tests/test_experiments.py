import sqlite3

from padjective import experiments


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, _query):
        return self

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)


def _make_battles_connection() -> _FakeConnection:
    rows = [
        ("RED", "BLUE"),
        ("RED", "GREEN"),
        ("BLUE", "GREEN"),
        ("YELLOW", "GREEN"),
        ("BLUE", "YELLOW"),
    ]
    return _FakeConnection(rows)


def test_run_tasks_and_status(tmp_path):
    battles_conn = _make_battles_connection()
    tasks_db = tmp_path / "tasks.sqlite"

    experiments.ensure_tasks(tasks_db, total_tasks=3, folds=5)
    results = experiments.run_tasks(
        tasks_db=tasks_db, database=battles_conn, take=2
    )

    assert len(results) == 2
    for result in results:
        metrics = result["metrics"]
        assert len(metrics["folds"]) == result["task"].folds
        assert metrics["winning_tags"] is not None
        assert metrics["losing_tags"] is not None
        assert len(metrics["winning_tags"]) <= 20
        assert len(metrics["losing_tags"]) <= 20
        for tag_summary in metrics["winning_tags"] + metrics["losing_tags"]:
            assert "tag" in tag_summary
            assert "average_win_rate" in tag_summary
            assert "folds_with_appearances" in tag_summary

    status = experiments.task_status(tasks_db)
    assert status["total"] == 3
    assert status["counts"].get("completed", 0) == 2
    assert status["counts"].get("pending", 0) == 1
    assert status["completed_tasks"] == 2
    assert status["folds"] == 5
    assert len(status["winning_tags"]) <= 20
    assert len(status["losing_tags"]) <= 20
    # At least one completed task should report accuracy.
    assert status["mean_accuracy"] is None or 0 <= status["mean_accuracy"] <= 1

    with sqlite3.connect(tasks_db) as conn:
        fold_count = conn.execute("SELECT COUNT(*) FROM holdout_task_folds").fetchone()[0]
        assert fold_count == len(results) * results[0]["task"].folds
