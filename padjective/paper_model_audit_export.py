"""Export compact, product-free manuscript evidence from a Postgres audit."""
import argparse
import json
from pathlib import Path

from .paper_model_audit import fingerprint
from .paper_tree_tradeoff_figures import analyse


def compact(report):
    analysis = analyse(report['tree'])
    evidence = {k:v for k,v in report.items() if k != 'tree'}
    evidence['postgres_report_sha256'] = fingerprint(report)
    tree = report['tree']
    evidence['tree'] = dict(batch_id=tree['batch_id'], validation=tree['validation'],
        source_commit=tree['manifest']['source_commit'],
        protocol_sha256=tree['manifest']['protocol_sha256'],
        aggregate_rows_sha256=fingerprint(tree['rows']),
        configurations=len(analysis['rows']), ensemble_sizes=[1,3,9,15,27,45,81,135,243],
        budget_tables=analysis['budget_tables'], small_ensemble_tradeoff=analysis['small_ensemble_tradeoff'],
        stored_slot_dominated_configurations=len(analysis['storage_dominated_by_unweighted_tree']))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('aggregate_report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = json.loads(args.aggregate_report.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact(report), indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
