"""Replay signed final-greedy fits and check the published strict-refinement table."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal

from padjective.paper_final_greedy_refinement import (
    aggregate, evaluate_fold, load_inputs, statistics, summarise,
)

EXPECTED_TABLE = [(32, .095, 220, .850, 8.8e-17), (18, .112, 234, .881, 2.0e-15),
                  (26, .001, 209, .814, 1.4e-15), (12, .083, 234, .838, 2.0e-9),
                  (14, .144, 242, .814, 2.9e-8), (102, .079, 1139, .840, 4.5e-61)]


def wilson(exact, n):
    z = 1.959963984540054
    rate = exact/n
    denominator = 1+z*z/n
    center = (rate+z*z/(2*n))/denominator
    radius = z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/denominator
    return [center-radius, center+radius]


def audit(root, run_dir):
    from scipy import stats
    report_path = run_dir/'summary.json'
    report = json.loads(report_path.read_text())
    runtime, records, expected, provenance = load_inputs(root)
    rows, contributions, source_checks = [], [], []
    for meta in report['models']:
        path = run_dir/meta['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['file_sha256']
        saved = json.loads(gzip.decompress(path.read_bytes()))
        fold_rows, terms = evaluate_fold(runtime, records, saved, expected['folds'][meta['fold']])
        rows.extend(fold_rows)
        contributions.extend(terms)
        source_checks.append(dict(fold=meta['fold'], saved_fit_sha256=meta['file_sha256'],
                                  held_out_rows=len(fold_rows)))
    recomputed = json.loads(json.dumps(summarise(rows, contributions)))
    assert recomputed == report['results'], 'Saved aggregates do not replay'
    table = []
    for fold in [0, 1, 2, 3, 4, None]:
        selected = [r for r in rows if r['active_count'] >= 2 and (fold is None or r['fold'] == fold)]
        v = [r for r in selected if r['valuation_separated']]
        nv = [r for r in selected if not r['valuation_separated']]
        first, second = statistics(v), statistics(nv)
        u = stats.mannwhitneyu([float(r['loss']) for r in v], [float(r['loss']) for r in nv], alternative='two-sided')
        chi = stats.chi2_contingency([[first['exact'], first['incorrect']], [second['exact'], second['incorrect']]])
        target = EXPECTED_TABLE[5 if fold is None else fold]
        matched = (first['n'], round(first['mean_padic_loss'], 3), second['n'], round(second['mean_padic_loss'], 3), f'{u.pvalue:.1e}') == (*target[:4], f'{target[4]:.1e}')
        table.append(dict(fold=fold, strict=first, non_strict=second, mann_whitney_p=float(u.pvalue),
                          chi_squared_p=float(chi.pvalue), published_rounded_row_matches=matched))
    assert all(row['published_rounded_row_matches'] for row in table), 'Published table does not reproduce'
    mean = sum(recomputed['by_fold'][str(f)]['all']['mean_padic_loss'] for f in range(5))/5
    assert abs(mean-expected['mean_padic_loss']) < 1e-12
    pooled = recomputed['pooled']
    disjoint, overlap = pooled['multi_disjoint_infinite'], pooled['multi_valuation_only']
    fisher = stats.fisher_exact([[disjoint['exact'], disjoint['incorrect']], [overlap['exact'], overlap['incorrect']]])
    return dict(status='passed', created_at=datetime.now(timezone.utc).isoformat(),
                summary_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                source_models=source_checks, rows_reconstructed=len(rows), all_aggregates_replayed=True,
                published_fold_metrics_reproduced=True, equal_fold_mean_loss=mean,
                published_strict_table_reproduced=True, published_strict_table=table,
                illustrative_binomial_intervals={key: wilson(pooled[key]['exact'], pooled[key]['n'])
                    for key in ['multi_valuation_separated', 'multi_not_valuation_separated',
                                'multi_disjoint_infinite', 'multi_valuation_only']},
                exploratory_disjoint_vs_overlap_fisher_p=float(fisher.pvalue),
                uncertainty_note='Wilson intervals and the unadjusted Fisher test treat rows as independent; they do not account for merchant/taxonomy dependence or overlapping training folds, and are not confirmatory evidence.',
                source_input_hashes=provenance['input_sha256'],
                audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new filename')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    signal.alarm(120)
    result = audit(args.release_root, args.run_dir)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
