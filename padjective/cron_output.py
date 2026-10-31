"""Keep nightly stdout privately; leave warnings and failures visible to cron.

This uses only the standard library so logging also covers git/uv setup failures.
It never reads a database, credentials, or the command's environment into a log.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _receipt(path: Path, state: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(command: list[str], log_dir: Path) -> int:
    """Run once, without changing the child's cwd, timing, umask or arguments."""
    state = {"started_at": _now(), "status": "running", "supervisor_pid": os.getpid()}
    try:
        log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix=datetime.now(timezone.utc).strftime("nightly-%Y%m%dT%H%M%SZ-"),
            suffix=".log", dir=log_dir,
        )
        log_path = Path(name)
        receipt_path = log_path.with_suffix(".json")
        state["log"] = str(log_path)
        _receipt(receipt_path, state)
    except OSError as exc:
        if "fd" in locals():
            os.close(fd)
        print(f"Padjective cron: cannot create private run log ({exc}); workload not started.", file=sys.stderr)
        return 74

    logging_failed = False
    returncode = 1
    with os.fdopen(fd, "wb", buffering=0) as log:
        env = os.environ.copy()
        env["PADJECTIVE_CRON_LOGGED"] = "1"
        try:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.PIPE, env=env)
            assert child.stderr is not None
            # stdout goes straight to disk; stderr is also delivered to cron immediately.
            # Keep draining on a logging error so an existing workload is not killed.
            while chunk := child.stderr.read1(8192):
                sys.stderr.buffer.write(chunk)
                sys.stderr.buffer.flush()
                try:
                    log.write(chunk)
                except OSError:
                    logging_failed = True
            child.stderr.close()
            returncode = child.wait()
            returncode = returncode if returncode >= 0 else 128 - returncode
        except OSError as exc:
            print(f"Padjective cron: cannot start workload ({exc}).", file=sys.stderr)
            returncode = 127
        try:
            log.flush()
            os.fsync(log.fileno())
        except OSError:
            logging_failed = True

    state.update(finished_at=_now(), exit_code=returncode,
                 status="failed" if returncode else "logging_failed" if logging_failed else "succeeded")
    try:
        _receipt(receipt_path, state)
    except OSError:
        logging_failed = True
    if logging_failed:
        print(f"Padjective cron: run log/receipt could not be completed; inspect {log_path}.", file=sys.stderr)
    if returncode:
        print(f"Padjective cron FAILED (exit {returncode}); full output: {log_path}", file=sys.stderr)
        # A failing program may explain itself only on stdout. Keep alert context bounded.
        try:
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 4096))
                tail = b"\n".join(stream.read().splitlines()[-20:])
            if tail:
                sys.stderr.buffer.write(tail + b"\n")
                sys.stderr.buffer.flush()
        except OSError:
            print("Padjective cron: failed to read diagnostic log tail.", file=sys.stderr)
    return returncode or (74 if logging_failed else 0)


def main() -> int:
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        print("Usage: cron_output.py -- command [args ...]", file=sys.stderr)
        return 2
    log_dir = Path(os.environ.get("PADJECTIVE_CRON_LOG_DIR", str(Path.home() / "logs/padjective-nightly")))
    return run(command, log_dir)


if __name__ == "__main__":
    raise SystemExit(main())
