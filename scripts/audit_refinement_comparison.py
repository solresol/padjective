"""Recheck the aggregate report against private saved rows; no fitting or writes to Postgres."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from padjective import db
from padjective.paper_refinement_comparison import (
    E, P, classify, fingerprint, has_carry, scalar_loss_units, summarise, valuation,
)


def audit(report):
    with db.get_connection() as conn:
        conn.read_only = True
        conn.execute("SET LOCAL statement_timeout='20s'")
        conn.execute("SET LOCAL lock_timeout='1s'")
        stored_report, private = conn.execute("""SELECT report,private_evidence
            FROM padjective.paper_refinement_comparisons WHERE analysis_id=%s""",
                                              (report['analysis_id'],)).fetchone()
        assert stored_report == report, 'Public and private reports differ'
        assert fingerprint(private) == report['private_evidence_sha256']
        defaults = {}
        for source in report['models']:
            saved = conn.execute("""SELECT evidence->'models'->'one_pass'
                FROM padjective.paper_randomised_method_runs WHERE run_id=%s""",
                                 (source['run_id'],)).fetchone()[0]
            assert fingerprint(saved) == source['model_sha256'], 'Original saved model changed'
            defaults[source['fold']] = saved['default']
    rows, contributions = private['rows'], private['contributions']
    assert len(rows) == len(contributions) == 6693
    assert len({r['product_key'] for r in rows}) == 6693
    errors = Counter()
    label_codes = {r['target'] for r in rows}
    for row, z in zip(rows, contributions, strict=True):
        assert all(0 < value < P**E for value in z)
        for key, value in classify(z, P, E).items():
            assert row[key] == value, 'Stored classification differs'
        assert sum(z) == row['integer_sum']
        assert sum(z) % P**E == row['raw']
        expected = defaults[row['fold']] if row['raw'] == 0 else row['raw']
        assert row['prediction'] == expected, 'Saved fallback/output differs'
        assert row['loss_units'] == scalar_loss_units(row['target'], expected, P, E)
        assert row['carry'] == has_carry(z, P, E)
        if row['active_count'] >= 2 and row['disjoint'] and row['target'] != expected:
            value, depth = row['target'], 0
            while value:
                value //= P
                depth += 1
            error_depth = valuation(expected-row['target'], P)
            errors[(row['taxonomy_id'], depth, error_depth, expected in label_codes)] += 1
    reconstructed = json.loads(json.dumps(summarise(rows, contributions)))
    assert reconstructed == report['results'], 'Fold/taxonomy/depth aggregates differ'
    return dict(status='passed', source_analysis_id=report['analysis_id'], rows_checked=len(rows),
                source_models_checked=len(defaults), all_aggregates_reconstructed=True,
                private_evidence_sha256=report['private_evidence_sha256'],
                disjoint_errors=[dict(taxonomy_id=t, target_depth=d, error_valuation=v,
                                       prediction_is_frozen_label=valid, n=n)
                                 for (t, d, v, valid), n in sorted(errors.items())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new filename')
    report = json.loads(args.report.read_text())
    result = audit(report)
    result.update(checked_at=datetime.now(timezone.utc).isoformat(),
                  report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                  audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
