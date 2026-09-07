"""Run the predeclared published-method comparison with bounded concurrency."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def jobs() -> list[tuple[str, list[str]]]:
    result = []
    # Interleave expensive jobs across folds. Small-degree sensitivity runs and
    # exact certificates follow; never use their held-out scores for selection.
    for fold in range(5):
        result.append((f"mihara-independent-f{fold}", ["--method", "mihara", "--fold", str(fold),
            "--rank-reduce", "--seed", str(42+fold), "--seconds", "180", "--max-draws", "1000000"]))
        for base in (42, 1729, 20260907):
            result.append((f"zubarev-k357910-f{fold}-s{base}", ["--method", "zubarev", "--fold", str(fold),
                "--degree", "357910", "--seed", str(base+fold), "--initialisation", "random",
                "--seconds", "300", "--draws-per-beta", "64", "--proposals-per-beta", "200000"]))
    for fold in range(5):
        for degree in (70, 5040):
            for base in (42, 1729, 20260907):
                result.append((f"zubarev-k{degree}-f{fold}-s{base}", ["--method", "zubarev", "--fold", str(fold),
                    "--degree", str(degree), "--seed", str(base+fold), "--initialisation", "random",
                    "--seconds", "300", "--draws-per-beta", "64", "--proposals-per-beta", "200000"]))
        for cap in (0, 32, 128):
            result.append((f"mihara-raw{cap}-f{fold}", ["--method", "mihara", "--fold", str(fold),
                "--max-tags", str(cap), "--seed", str(42+fold), "--seconds", "180", "--max-draws", "1000000"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--log-root", type=Path, default=Path("experiment-logs"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four workers on the shared host")
    batch_id = str(uuid.uuid4())
    directory = args.log_root / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    commands = jobs()
    (directory / "batch.json").write_text(json.dumps(dict(batch_id=batch_id, workers=args.workers,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        jobs=[dict(name=name, arguments=arguments) for name, arguments in commands]), indent=2) + "\n")
    print(json.dumps(dict(event="batch_started", batch_id=batch_id, jobs=len(commands), directory=str(directory))), flush=True)

    def execute(name: str, arguments: list[str]) -> dict:
        print(json.dumps(dict(event="job_started", name=name)), flush=True)
        path = directory / f"{name}.log"
        with path.open("x") as output:
            process = subprocess.run([sys.executable, "-m", "padjective.paper_published_methods", *arguments],
                                     env=environment, stdout=output, stderr=subprocess.STDOUT)
        events = []
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        finished = next((event for event in reversed(events) if event.get("event") == "finished"), {})
        return dict(name=name, exit_code=process.returncode, run_id=finished.get("run_id"),
                    status=finished.get("status", "missing_completion"))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, name, arguments) for name, arguments in commands]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(dict(event="job_finished", completed=len(results), **result)), flush=True)
    (directory / "completion.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(dict(event="batch_finished", batch_id=batch_id, completed=len(results))), flush=True)
    if any(result["exit_code"] or result["status"] in {"failed", "missing_completion"} for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
