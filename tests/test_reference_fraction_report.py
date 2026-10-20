import copy

import pytest

from padjective.reference_fraction_report import METHODS, METHOD_RATE, markdown, reference_count, summarise
from padjective.residual_ensemble_experiment import TRAJECTORIES


def synthetic_report():
    rows = []
    for roster, limit in TRAJECTORIES.items():
        for method in METHODS:
            p = METHOD_RATE.get(method)
            for n in range(1, limit+1):
                refs = 0 if p is None else (p*(n-1)+99)//100
                for fold in range(5):
                    loss = .1+.01*fold if p is None else .12+p/1000+.01*fold+.01*(refs % 2)
                    score = dict(n=10+10*fold, mean_padic_loss=loss,
                        exact_accuracy=.8 if p is None else .65-p/1000, first_digit_accuracy=.95)
                    row = dict(roster=roster, method=method, members=n, fitted_members=n, fold=fold,
                        training_rows=score['n']//2, forgotten_correct_rows=1,
                        stored_nonzero_coefficients=7, mean_active_terms=2, training=score, held_out=score)
                    if p is not None and p != 100:
                        row['reference_members_count'] = refs
                    rows.append(row)
    return dict(sweep_id='synthetic', rows=rows, batches={'10': 'a', '25': 'b', '75': 'c', '90': 'd'},
        manifest=dict(source_commit='synthetic', snapshot_digest='digest'),
        validation=dict(status='passed', trajectories=60, certified_coordinates=10,
                        reference_checkpoint_checks=400, control_rows_verified=6075))


def test_equal_fold_means_and_reference_parity_are_distinct_from_ensemble_parity():
    summary = summarise(synthetic_report())
    row = next(r for r in summary['summaries'] if r['roster'] == 0 and
               r['method'] == 'reference_25' and r['members'] == 243)
    assert row['held_out']['mean_padic_loss'] == pytest.approx(.175)
    assert row['training_fraction'] == .5
    assert row['reference_members_count'] == 61
    actual = {(r['percent'], r['kind']): r for r in summary['parity']}
    assert actual[(25, 'reference')]['odd_minus_even'] == pytest.approx(.01)
    assert actual[(100, 'ensemble')]['odd_minus_even'] == pytest.approx(-.01)
    delta = next(r for r in summary['comparisons'] if r['percent'] == 25 and r['roster'] == 0
                 and r['members'] == 243 and r['comparator'] == 'independent')
    assert delta['loss_difference'] == pytest.approx(.055)
    assert delta['folds_better'] == 0


def test_complete_grid_and_unique_folds_are_required():
    report = synthetic_report()
    broken = copy.deepcopy(report)
    broken['rows'].pop()
    with pytest.raises(AssertionError):
        summarise(broken)
    report['rows'][-1]['fold'] = 0
    with pytest.raises(AssertionError):
        summarise(report)


def test_legacy_reference_count_and_fixed_budget_table_columns(tmp_path):
    assert reference_count(dict(method='error_only', members=243, fitted_members=243)) == 242
    assert reference_count(dict(method='error_only', members=11, fitted_members=10)) == 10
    report = synthetic_report()
    markdown(report, summarise(report), tmp_path)
    assert '| 25% | 0.165000 | 0.175000 | 62.50% |' in (tmp_path/'README.md').read_text()
