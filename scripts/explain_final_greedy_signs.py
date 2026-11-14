"""Read-only replay of sign strata and a fixed-default quotient counterfactual."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import signal

from padjective.paper_final_greedy_refinement import (
    P, aggregate, evaluate_fold, integer_distance, load_inputs, statistics,
)
from padjective.paper_refinement_comparison import valuation


def audit(root, run_dir):
    report_path = run_dir/'summary.json'
    report = json.loads(report_path.read_text())
    runtime, records, expected, _ = load_inputs(root)
    rows, defaults = [], {}
    for meta in report['models']:
        path = run_dir/meta['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['file_sha256']
        saved = json.loads(gzip.decompress(path.read_bytes()))
        defaults[meta['fold']] = saved['default']
        fold_rows, _ = evaluate_fold(runtime, records, saved, expected['folds'][meta['fold']])
        rows.extend(fold_rows)
    changes = dict(predictions=0, exact_hit_indicators=0, losses=0, fallback_triggers=0)
    for row in rows:
        raw = row['raw'] % P**7
        prediction = defaults[row['fold']] if raw == 0 else raw
        changes['predictions'] += prediction != row['prediction']
        changes['exact_hit_indicators'] += (prediction == row['target']) != (row['prediction'] == row['target'])
        changes['losses'] += integer_distance(row['target'], prediction) != row['loss']
        changes['fallback_triggers'] += (raw == 0) != (row['raw'] == 0)
        if row['active_count'] >= 2 and row['disjoint_infinite'] and row['negative_count']:
            assert row['prediction'] < 0 and row['prediction'] != row['target']
    disjoint = [r for r in rows if r['active_count'] >= 2 and r['disjoint_infinite']]
    return dict(status='passed', created_at=datetime.now(timezone.utc).isoformat(),
                summary_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                disjoint_positive_only=statistics([r for r in disjoint if not r['negative_count']]),
                disjoint_with_negative=statistics([r for r in disjoint if r['negative_count']]),
                by_negative_count={str(k): aggregate([r for r in rows if r['negative_count'] == k])
                                   for k in sorted({r['negative_count'] for r in rows})},
                disjoint_negative_forces_negative_output_verified=True,
                maximum_nonzero_error_valuation=max(valuation(r['target']-r['prediction'], P)
                                                   for r in rows if r['target'] != r['prediction']),
                quotient_counterfactual_changes=changes,
                counterfactual_definition='Reduce raw signed sums modulo 71**7 and retain the SAME fitted defaults; no refitting. This is not the earlier checkpoint model.',
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    signal.alarm(120)
    result = audit(args.release_root, args.run_dir)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'by_negative_count'}, indent=2))


if __name__ == '__main__':
    main()
