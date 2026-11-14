"""Run the predeclared randomised-method experiment with durable job logs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from . import db
from .paper_randomised_methods import SEED_BASES, ensure_storage


def jobs(pilot: bool = False) -> list[tuple[str, list[str]]]:
    result = []
    for fold in range(1 if pilot else 5):
        for method in ("linear_random", "linear_association"):
            for base in (SEED_BASES[:1] if pilot or method == "linear_association" else SEED_BASES):
                name = f"{method}-f{fold}-s{base}"
                result.append((name, ["--method", method, "--fold", str(fold),
                    "--seed", str(base+fold), "--seconds", "300", "--max-sweeps", "100"]))
        for degree in (357910, 357911):
            bases = SEED_BASES[:1] if pilot else (SEED_BASES if degree == 357910 else SEED_BASES[:3])
            for base in bases:
                name = f"zubarev_random-k{degree}-f{fold}-s{base}"
                result.append((name, ["--method", "zubarev_random", "--fold", str(fold),
                    "--degree", str(degree), "--seed", str(base+fold), "--seconds", "300",
                    "--draws-per-beta", "8" if pilot else "64"]))
    if pilot:
        result = [(name, [*args, "--pilot"]) for name, args in result]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--log-root", type=Path, default=Path("experiment-logs"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("Use one to three workers on the shared host")
    with db.get_connection() as conn:
        ensure_storage(conn)
    batch_id = str(uuid.uuid4())
    directory = args.log_root / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    commands = jobs(args.pilot)
    manifest = dict(batch_id=batch_id, pilot=args.pilot, workers=args.workers,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        jobs=[dict(name=name, arguments=arguments) for name, arguments in commands])
    (directory / "batch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(event="batch_started", batch_id=batch_id,
                         jobs=len(commands), directory=str(directory))), flush=True)

    def execute(name, arguments):
        print(json.dumps(dict(event="job_started", name=name)), flush=True)
        path = directory / f"{name}.log"
        with path.open("x") as output:
            process = subprocess.run([sys.executable, "-m", "padjective.paper_randomised_methods",
                "--batch-id", batch_id, "--job-key", name, *arguments], env=environment,
                stdout=output, stderr=subprocess.STDOUT)
        events = []
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        finished = next((e for e in reversed(events) if e.get("event") == "finished"), {})
        return dict(name=name, exit_code=process.returncode, run_id=finished.get("run_id"),
                    status=finished.get("status", "missing_completion"))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, name, arguments) for name, arguments in commands]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            # Save after every job, so interrupted batches remain auditable.
            (directory / "completion.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(dict(event="job_finished", completed=len(results), **result)), flush=True)
    print(json.dumps(dict(event="batch_finished", batch_id=batch_id, completed=len(results))), flush=True)
    if any(r["exit_code"] or r["status"] in {"failed", "missing_completion"} for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
