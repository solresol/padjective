"""Extend the validated nine-member linear bank to 243 fixed seed bases."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import uuid

from psycopg.rows import dict_row

from . import db
from .paper_randomised_methods import SEED_BASES, ensure_storage

BASE_BATCH = "a2b0744f-175e-4899-a3d5-e6a8478a8d73"
BASE_COMMIT = "289f5498c498c2c56eb13e31f29fa9958e049df1"
SIZES = (1, 3, 9, 15, 27, 45, 81, 135, 243)
BANK_SEEDS = SEED_BASES + tuple(random.Random(20260910).sample(range(100000000, 2000000000), 234))


def jobs():
    return [(f"linear_random-f{fold}-s{base}", ["--method", "linear_random", "--fold", str(fold),
        "--seed", str(base+fold), "--seconds", "300", "--max-sweeps", "100"])
        for base in BANK_SEEDS[9:] for fold in range(5)]


def fitting_fingerprints():
    output = {}
    for name in ("randomised_linear.py", "paper_randomised_methods.py"):
        path = f"padjective/{name}"
        before = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{path}"])
        current = Path(path).read_bytes()
        if before != current:
            raise ValueError(f"Numerical fitting file changed: {path}")
        output[path] = hashlib.sha256(current).hexdigest()
    return output


def baseline_members(conn):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""SELECT run_id,cv_fold,configuration,status FROM
            padjective.paper_randomised_method_runs WHERE batch_id=%s
            AND method='linear_random' AND NOT pilot""", (BASE_BATCH,))
        rows = cur.fetchall()
    assert len(rows) == 45
    assert {(r["cv_fold"], r["configuration"]["seed"]) for r in rows} == {
        (fold, base+fold) for fold in range(5) for base in SEED_BASES}
    for row in rows:
        c = row["configuration"]
        assert row["status"] == "coordinate_optimum" and c["source_commit"] == BASE_COMMIT
        assert c["max_sweeps"] == 100 and c["seconds"] == 300 and c["precision"] == 7
    return [dict(run_id=str(r["run_id"]), fold=r["cv_fold"], seed=r["configuration"]["seed"]) for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-root", type=Path, default=Path("experiment-logs"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four workers")
    fingerprints = fitting_fingerprints()
    with db.get_connection() as conn:
        ensure_storage(conn)
        reused = baseline_members(conn)
    batch_id = str(uuid.uuid4())
    directory = args.log_root / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    commands = jobs()
    manifest = dict(batch_id=batch_id, base_batch=BASE_BATCH, workers=args.workers,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        fitting_fingerprints=fingerprints, seed_bases=BANK_SEEDS, ensemble_sizes=SIZES,
        reused_members=reused, jobs=[dict(name=n, arguments=a) for n, a in commands])
    (directory / "batch.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(event="batch_started", batch_id=batch_id, jobs=len(commands),
        reused=len(reused), directory=str(directory))), flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")

    def execute(name, arguments):
        path = directory / f"{name}.log"
        with path.open("x") as output:
            result = subprocess.run([sys.executable, "-m", "padjective.paper_randomised_methods",
                "--batch-id", batch_id, "--job-key", name, *arguments], env=env,
                stdout=output, stderr=subprocess.STDOUT)
        finished = {}
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "finished":
                finished = event
        return dict(job_key=name, exit_code=result.returncode, run_id=finished.get("run_id"),
                    status=finished.get("status", "missing_completion"))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute, n, a) for n, a in commands]):
            results.append(future.result())
            (directory / "completion.json").write_text(json.dumps(results, indent=2)+"\n")
            print(json.dumps(dict(event="job_finished", completed=len(results), **results[-1])), flush=True)
    print(json.dumps(dict(event="batch_finished", batch_id=batch_id, completed=len(results))), flush=True)
    if any(r["exit_code"] or r["status"] != "coordinate_optimum" for r in results):
        raise SystemExit("Incomplete grid; do not construct survivor ensembles")


if __name__ == "__main__":
    main()
