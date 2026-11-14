"""Bounded rerun of only the five published greedy fits, with signed supports.

No PostgreSQL connection or writes. Inputs are the checksum-pinned published
anonymous snapshot/runtime. The published signed integer predictor is retained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

from .paper_refinement_comparison import (
    SNAPSHOT, SNAPSHOT_DIGEST, classify, fingerprint, occupied_powers, valuation,
)

P = 71
RUNTIME_SHA = "4618605fcdf466a840c19f9de783bd003cf6aad996ea6d4de46dd7070f595aa1"
GROUPS = ("all", "zero", "one", "multi", "multi_valuation_separated",
          "multi_not_valuation_separated", "multi_disjoint_infinite",
          "multi_not_disjoint_infinite", "multi_valuation_only",
          "multi_disjoint_finite7", "multi_finite7_only",
          "multi_disjoint_canonical_residue_lifts")


def digit_length(n, p=P):
    n, width = abs(int(n)), 0
    while n:
        n //= p
        width += 1
    return width


def infinite_support_disjoint(contributions, p=P):
    """Exact canonical Z_p supports of signed integers, without infinite loops.

    Two negative integers share an eventual all-(p-1) tail. With at most one
    negative integer, only the finite occupied positions of positive terms
    need checking against each other and the negative term's canonical digits.
    Python floor division/modulo supplies those signed canonical digits.
    """
    active = [int(z) for z in contributions if z]
    negative = [z for z in active if z < 0]
    if len(negative) > 1:
        return False
    occupied = set()
    for z in active:
        if z < 0:
            continue
        support = occupied_powers(z, p, digit_length(z, p))
        if occupied & support:
            return False
        occupied |= support
    return not negative or all((negative[0] // p**i) % p == 0 for i in occupied)


def integer_distance(target, prediction, p=P):
    """Untruncated distance: a nonzero multiple of p**7 still has positive loss."""
    delta = int(target) - int(prediction)
    return Fraction(0) if delta == 0 else Fraction(1, p**valuation(delta, p))


def statistics(rows):
    n = len(rows)
    exact = sum(r["target"] == r["prediction"] for r in rows)
    loss = sum((r["loss"] for r in rows), Fraction(0))
    wrong = n - exact
    return dict(n=n, exact=exact, incorrect=wrong,
                exact_hit_rate=exact/n if n else None,
                loss_sum_exact=[loss.numerator, loss.denominator],
                mean_padic_loss=float(loss/n) if n else None,
                incorrect_mean_padic_loss=float(loss/wrong) if wrong else None,
                taxonomies=len({r["taxonomy_id"] for r in rows}),
                raw_zero=sum(r["raw"] == 0 for r in rows),
                nonempty_cancellation=sum(r["raw"] == 0 and r["active_count"] > 0 for r in rows),
                negative_contribution_count_histogram=dict(sorted(Counter(r["negative_count"] for r in rows).items())),
                active_count_histogram=dict(sorted(Counter(r["active_count"] for r in rows).items())),
                incorrect_matching_entire_reference_path=sum(r["target"] != r["prediction"] and
                    valuation(r["target"]-r["prediction"], P) >= r["target_depth"] for r in rows),
                predictions_outside_canonical_quotient=sum(not 0 <= r["prediction"] < P**7 for r in rows))


def aggregate(rows):
    buckets = {k: [] for k in GROUPS}
    for row in rows:
        keys = ["all"]
        if row["active_count"] < 2:
            keys.append("zero" if row["active_count"] == 0 else "one")
        else:
            keys.extend(["multi", "multi_valuation_separated" if row["valuation_separated"] else "multi_not_valuation_separated",
                         "multi_disjoint_infinite" if row["disjoint_infinite"] else "multi_not_disjoint_infinite"])
            if row["valuation_separated"] and not row["disjoint_infinite"]:
                keys.append("multi_valuation_only")
            if row["disjoint_finite7"]:
                keys.append("multi_disjoint_finite7")
                if not row["disjoint_infinite"]:
                    keys.append("multi_finite7_only")
            if row["disjoint_canonical_lifts"]:
                keys.append("multi_disjoint_canonical_residue_lifts")
        for key in keys:
            buckets[key].append(row)
    return {key: statistics(values) for key, values in buckets.items()}


def summarise(rows, contributions):
    by_fold, by_taxonomy, by_active = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        by_fold[row["fold"]].append(row)
        by_taxonomy[row["taxonomy_id"]].append(row)
        by_active[row["active_count"]].append(row)
    bound = max((digit_length(z) for terms in contributions for z in terms), default=0) + 1
    sensitivity = []
    for depth in sorted(set(range(1, 8)) | {8, 10, 14, bound}):
        chosen, rejected, visible_chosen = [], [], []
        hidden_rows = finite_only = 0
        for row, terms in zip(rows, contributions, strict=True):
            if row["active_count"] < 2:
                continue
            result = classify(terms, P, depth)
            (chosen if result["disjoint"] else rejected).append(row)
            hidden_rows += result["hidden_contributions"] > 0
            finite_only += result["disjoint"] and not row["disjoint_infinite"]
            if result["disjoint"] and not result["hidden_contributions"]:
                visible_chosen.append(row)
            if depth >= bound and result["disjoint"] != row["disjoint_infinite"]:
                raise ValueError("Finite certifying depth disagrees with exact infinite support")
        sensitivity.append(dict(depth=depth, hidden_rows=hidden_rows, finite_only_rows=finite_only,
                                disjoint=statistics(chosen), not_disjoint=statistics(rejected),
                                visible_disjoint=statistics(visible_chosen)))
    return dict(pooled=aggregate(rows), by_fold={str(k): aggregate(v) for k, v in sorted(by_fold.items())},
                by_taxonomy={k: aggregate(v) for k, v in sorted(by_taxonomy.items())},
                by_active_count={str(k): aggregate(v) for k, v in sorted(by_active.items())},
                finite_depth_sensitivity=sensitivity, certifying_depth=bound,
                exact_infinite_support_equals_certifying_finite_depth=True,
                signed_vs_canonical_lift_disjointness_changes=sum(r["disjoint_infinite"] != r["disjoint_canonical_lifts"] for r in rows),
                raw_integer_vs_quotient_scores_differ=sum(r["raw"] != r["raw"] % P**7 for r in rows),
                nonzero_integer_raw_becomes_modular_zero=sum(r["raw"] != 0 and r["raw"] % P**7 == 0 for r in rows),
                aggregation="Pooled summaries weight products; individual folds are retained.",
                interpretation="Descriptive post-hoc associations, not causal effects. All support variants group the same signed-integer predictions.")


def load_inputs(root):
    manifest = json.loads((root/"manifest.json").read_text())
    hashes = {}
    for name in ("benchmark_runtime.py", "paper/products-00000.jsonl.gz", "paper/tags.jsonl.gz", "paper/snapshot.json"):
        hashes[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if hashes[name] != manifest["sha256"][name]:
            raise ValueError("Frozen input checksum changed: "+name)
    if hashes["benchmark_runtime.py"] != RUNTIME_SHA or platform.python_version() != manifest["python_version"]:
        raise ValueError("Wrong published runtime/Python")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}
    if any(value != manifest["versions"][name] for name, value in versions.items()):
        raise ValueError("Wrong published numerical dependencies")
    spec = importlib.util.spec_from_file_location("_pinned_final_greedy_runtime", root/"benchmark_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)
    tables = runtime.load_snapshot_tables(root/"paper", snapshot_label="paper")
    encoded = {str(tid): runtime.encode_path(runtime.parse_taxonomy_digits(str(path)), P)
               for tid, path in tables.products[["taxonomy_id", "taxonomy_path"]].drop_duplicates().itertuples(index=False, name=None)}
    records = runtime._build_umllr_records(tables, encoded)
    digest = hashlib.sha256()
    for row in records:
        if len(row.tags) != len(set(row.tags)):
            raise ValueError("Frozen predictor has duplicate features")
        digest.update(json.dumps([row.product_key, row.cv_fold, row.encoded_path, sorted(row.tags)], separators=(",", ":")).encode()+b"\n")
    if digest.hexdigest() != SNAPSHOT_DIGEST or len(records) != 6693 or len(encoded) != 363:
        raise ValueError("Frozen predictor population/order drift")
    expected_path = root/"validation-single-thread-models.json"
    expected = next(row for row in json.loads(expected_path.read_text())["models"]["rows"] if row["model_key"] == "umllr")
    hashes[expected_path.name] = hashlib.sha256(expected_path.read_bytes()).hexdigest()
    return runtime, records, expected, dict(input_sha256=hashes, versions=versions,
                                          manifest_sha256=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
                                          published_source_commit=manifest["source_commit"])


def evaluate_fold(runtime, records, saved, expected):
    fold = saved["fold"]
    coefficients = {tag: int(value) for tag, value, _sequence in saved["coefficients"]}
    training = [r for r in records if r.cv_fold != fold]
    train_raw = [sum(coefficients.get(tag, 0) for tag in r.tags) for r in training]
    default = runtime._select_default_prediction([r.encoded_path for r, z in zip(training, train_raw, strict=True) if z == 0],
                                                  [r.encoded_path for r in training], P)
    if default != saved["default"]:
        raise ValueError("Training-only signed-integer default does not reconstruct")
    rows, contributions, predictions = [], [], []
    for record in records:
        if record.cv_fold != fold:
            continue
        terms = [coefficients.get(tag, 0) for tag in record.tags if coefficients.get(tag, 0)]
        finite = classify(terms, P, 7)
        disjoint = infinite_support_disjoint(terms, P)
        if disjoint and not finite["valuation_separated"]:
            raise ValueError("Infinite-support disjointness must imply valuation separation")
        raw = sum(terms)
        prediction = saved["default"] if raw == 0 else raw
        loss = integer_distance(record.encoded_path, prediction, P)
        if abs(float(loss)-runtime.p_adic_distance(record.encoded_path, prediction, P)) > 1e-15:
            raise ValueError("Exact rational and published distance disagree")
        row = dict(fold=fold, taxonomy_id=record.taxonomy_id, target=int(record.encoded_path),
                   target_depth=record.taxonomy_depth, prediction=prediction, raw=raw,
                   active_count=len(terms), negative_count=sum(z < 0 for z in terms), loss=loss,
                   valuation_separated=finite["valuation_separated"], disjoint_infinite=disjoint,
                   disjoint_finite7=finite["disjoint"],
                   disjoint_canonical_lifts=infinite_support_disjoint([z % P**7 for z in terms], P))
        if terms and row["valuation_separated"] and raw == 0:
            raise ValueError("Nonempty valuation-separated terms cancelled")
        rows.append(row)
        contributions.append(terms)
        predictions.append(prediction)
    if predictions != saved["predictions"]:
        raise ValueError("Signed-integer predictions do not reconstruct from saved fit")
    actual = statistics(rows)
    for key, value in (("num_test_samples", len(rows)), ("num_params", sum(value != 0 for value in coefficients.values())),
                       ("exact_accuracy", actual["exact_hit_rate"]), ("padic_loss_mean", actual["mean_padic_loss"]),
                       ("mean_scoring_ops", sum(r["active_count"] for r in rows)/len(rows))):
        if abs(value-expected[key]) > 1e-12:
            raise ValueError(f"Published fold {fold} parity failed: {key}: {value} != {expected[key]}")
    return rows, contributions


def dump_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(root, output, replay=None):
    started = time.monotonic()
    runtime, records, expected, provenance = load_inputs(root)
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((replay/"summary.json").read_text()) if replay else None
    rows, contributions, models = [], [], []
    for fold in range(5):
        fold_started = time.monotonic()
        if replay:
            source = replay/f"fold-{fold}.json.gz"
            if hashlib.sha256(source.read_bytes()).hexdigest() != previous["models"][fold]["file_sha256"]:
                raise ValueError("Replay fit checksum changed")
            saved = json.loads(gzip.decompress(source.read_bytes()))
        else:
            fit = runtime.umllr_run_fold(fold, records, [], P, tag_order_strategy="taxonomy_association")
            saved = dict(fold=fold, default=fit.default_prediction,
                         coefficients=[[c.tag, c.coefficient, c.sequence] for c in fit.coefficients],
                         predictions=[p.predicted_value for p in fit.predictions])
        model_bytes = gzip.compress(json.dumps(saved, separators=(",", ":"), sort_keys=True).encode(), mtime=0)
        model_path = output/f"fold-{fold}.json.gz"
        model_path.write_bytes(model_bytes)
        fold_rows, terms = evaluate_fold(runtime, records, saved, expected["folds"][fold])
        rows.extend(fold_rows)
        contributions.extend(terms)
        models.append(dict(fold=fold, file=model_path.name, file_sha256=hashlib.sha256(model_bytes).hexdigest(),
                           coefficient_sha256=fingerprint(saved["coefficients"]), prediction_sha256=fingerprint(saved["predictions"]),
                           coefficient_count=len(saved["coefficients"]), negative_coefficients=sum(c[1] < 0 for c in saved["coefficients"]),
                           elapsed_seconds=time.monotonic()-fold_started))
        print(json.dumps(dict(event="fold_verified", fold=fold, elapsed_seconds=time.monotonic()-started)), flush=True)
    results = summarise(rows, contributions)
    if replay and results != previous["results"]:
        # JSON converts histogram integer keys to strings; normalise before comparison.
        if json.loads(json.dumps(results)) != previous["results"]:
            raise ValueError("Replay aggregates differ")
    report = dict(status="passed", analysis="published_final_greedy_signed_refinement", analysis_version=1,
                  created_at=datetime.now(timezone.utc).isoformat(), snapshot_id=SNAPSHOT, snapshot_digest=SNAPSHOT_DIGEST,
                  python=platform.python_version(), prime=P, models=models, provenance=provenance,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_helper_sha256=hashlib.sha256((Path(__file__).parent/"paper_refinement_comparison.py").read_bytes()).hexdigest(),
                  mode="replay_without_fitting" if replay else "five_frozen_published_greedy_fits",
                  protocol=dict(predictor="Signed integer sum; fitted training-only default iff the integer sum is zero. No reduction modulo 71**7.",
                                support="Exact infinite canonical p-adic support for the actual signed integer contributions; zeros omitted.",
                                finite_sensitivity="Canonical digits of each actual signed contribution modulo 71**D. Never use abs for digit support. Predictions and loss stay signed/untruncated.",
                                residue_comparison="Canonical nonnegative lifts z mod 71**7 are used only for a structural counterfactual on the SAME predictions, not as a replacement predictor or refitted checkpoint model.",
                                inputs="Published, anonymous, checksum-pinned release. No raw catalogue access or database connection."),
                  validation=dict(folds=5, held_out_rows=len(rows), published_fold_metrics_matched=True,
                                  predictions_reconstructed=True, defaults_recomputed_from_training_only=True,
                                  integer_losses_checked_as_exact_rationals=True, replay_aggregates_matched=bool(replay)),
                  results=results, resource_use=dict(one_cpu_affinity=sorted(os.sched_getaffinity(0)),
                    address_space_cap_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
                    wall_time_cap_seconds=120, elapsed_seconds=time.monotonic()-started,
                    peak_resident_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    dump_json(output/"summary.json", report)
    print(json.dumps(dict(status=report["status"], resource_use=report["resource_use"], pooled=results["pooled"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-source", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory exists; choose a new path")
    # Apply the authorised cap before importing numerical dependencies or fitting.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    signal.alarm(120)
    run(args.release_root, args.output_dir, args.replay_source)


if __name__ == "__main__":
    main()
