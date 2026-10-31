"""Exercise cron supervision using tiny subprocesses, never the nightly workload."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from padjective import cron_output


def invoke(tmp_path, source):
    logs = tmp_path / "logs"
    env = {**os.environ, "PADJECTIVE_CRON_LOG_DIR": str(logs)}
    result = subprocess.run(
        [sys.executable, cron_output.__file__, "--", sys.executable, "-c", source],
        capture_output=True, env=env,
    )
    receipts = list(logs.glob("*.json"))
    receipt = json.loads(receipts[0].read_text()) if receipts else None
    return result, logs, receipt


def test_success_keeps_all_output_and_is_silent(tmp_path):
    result, logs, receipt = invoke(tmp_path, "print('routine metrics saved to PostgreSQL')")
    assert result.returncode == 0
    assert result.stdout == result.stderr == b""
    assert receipt["status"] == "succeeded"
    assert receipt["exit_code"] == 0
    assert "routine metrics" in Path(receipt["log"]).read_text()
    assert stat.S_IMODE(logs.stat().st_mode) == 0o700
    for path in logs.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_success_keeps_fallback_warning_visible_and_in_log(tmp_path):
    result, _, receipt = invoke(tmp_path, "import sys; print('routine'); print('RuntimeWarning: falling back to KFold', file=sys.stderr)")
    assert result.returncode == 0
    assert result.stdout == b""
    assert b"RuntimeWarning: falling back to KFold" in result.stderr
    assert "RuntimeWarning: falling back to KFold" in Path(receipt["log"]).read_text()
    assert receipt["status"] == "succeeded"


@pytest.mark.parametrize("source, expected", [
    ("print('stdout-only publication failure'); raise SystemExit(23)", b"stdout-only publication failure"),
    ("raise RuntimeError('UniqueViolation: umllr_order_ablation_fold_metrics_pkey')", b"UniqueViolation"),
])
def test_failure_keeps_exit_and_diagnostics(tmp_path, source, expected):
    result, _, receipt = invoke(tmp_path, source)
    assert result.returncode != 0
    assert result.returncode == receipt["exit_code"]
    assert receipt["status"] == "failed"
    assert expected in result.stderr
    assert b"Padjective cron FAILED" in result.stderr
    assert expected in Path(receipt["log"]).read_bytes()
    assert result.stdout == b""


def test_logging_setup_failure_alerts_before_starting_workload(tmp_path):
    path = tmp_path / "not-a-directory"
    path.write_text("sentinel")
    marker = tmp_path / "must-not-exist"
    env = {**os.environ, "PADJECTIVE_CRON_LOG_DIR": str(path)}
    result = subprocess.run(
        [sys.executable, cron_output.__file__, "--", sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        capture_output=True, env=env,
    )
    assert result.returncode == 74
    assert b"workload not started" in result.stderr
    assert not marker.exists()


def test_missing_command_is_a_visible_failure(tmp_path):
    env = {**os.environ, "PADJECTIVE_CRON_LOG_DIR": str(tmp_path)}
    result = subprocess.run([sys.executable, cron_output.__file__, "--", str(tmp_path / "missing")], capture_output=True, env=env)
    assert result.returncode == 127
    assert b"cannot start workload" in result.stderr
    assert b"Padjective cron FAILED" in result.stderr


def test_environment_and_working_directory_are_preserved(tmp_path):
    source = "import os; assert os.environ['PADJECTIVE_CRON_LOGGED'] == '1'; print(os.getcwd())"
    result, _, receipt = invoke(tmp_path, source)
    assert result.returncode == 0
    assert Path(receipt["log"]).read_text().strip() == os.getcwd()


@pytest.mark.parametrize("git_exit", [0, 17])
def test_real_wrapper_with_stubbed_workload(tmp_path, git_exit):
    """Traverse the real shell entry point with no Git/network/training calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    for name in ["git", "uv", "rsync"]:
        stub = bin_dir / name
        stub.write_text(
            "#!/bin/sh\n"
            'echo "$(basename "$0") $*" >> "$STUB_CALLS"\n'
            'echo "routine $(basename "$0") output"\n'
            + (f"exit {git_exit}\n" if name == "git" else "exit 0\n")
        )
        stub.chmod(0o700)
    logs = tmp_path / "logs"
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "STUB_CALLS": str(calls), "PADJECTIVE_CRON_LOG_DIR": str(logs)}
    env.pop("PADJECTIVE_CRON_LOGGED", None)
    for key in ["SITE_DIR", "BENCHMARK_EXPORT_ROOT", "BENCHMARK_REPORT_ROOT", "PAPER_GENERATED_DIR",
                "TAXONOMY_CLASSIFIER_REPORT_DIR", "TAXONOMY_PCNN_CLASSIFIER_DB",
                "TAXONOMY_PCNN_CLASSIFIER_REPORT_DIR", "TAXONOMY_ULR_CLASSIFIER_REPORT_DIR"]:
        env["PADJECTIVE_" + key] = str(tmp_path / key)
    script = Path(cron_output.__file__).resolve().parent.parent / "cronscript.sh"
    result = subprocess.run(["bash", str(script)], capture_output=True, env=env, timeout=20)
    assert result.returncode == git_exit
    assert result.stdout == b""
    assert bool(result.stderr) == bool(git_exit)
    commands = calls.read_text().splitlines()
    assert commands[0] == "git pull -q"
    if git_exit:
        assert len(commands) == 1
        assert b"FAILED (exit 17)" in result.stderr
    else:
        assert any("uv run -m padjective.snapshot_metrics" in line for line in commands)
        assert any(line.startswith("rsync ") for line in commands)
    receipt = json.loads(next(logs.glob("*.json")).read_text())
    assert receipt["status"] == ("failed" if git_exit else "succeeded")


@pytest.mark.parametrize("table", ["taxonomy", "product_details"])
def test_optional_export_failure_goes_to_stderr(tmp_path, monkeypatch, capsys, table):
    from padjective import build_site

    monkeypatch.setattr(build_site, "_table_exists", lambda conn, schema, name: name == table)

    def fail(*args):
        raise OSError("export unavailable")

    monkeypatch.setattr(build_site, "_dump_table_to_sql", fail)
    build_site._create_comprehensive_dumps(object(), tmp_path, "padjective")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Warning: Failed to export" in captured.err


def test_final_receipt_failure_is_visible_after_workload_finishes(tmp_path, monkeypatch, capsys):
    original = cron_output._receipt
    writes = []

    def fail_on_completion(path, state):
        writes.append(state["status"])
        if state["status"] != "running":
            raise OSError("simulated receipt write failure")
        original(path, state)

    monkeypatch.setattr(cron_output, "_receipt", fail_on_completion)
    code = cron_output.run([sys.executable, "-c", "print('completed')"], tmp_path)
    assert code == 74
    assert writes == ["running", "succeeded"]
    assert "log/receipt could not be completed" in capsys.readouterr().err
    assert "completed" in next(tmp_path.glob("*.log")).read_text()
