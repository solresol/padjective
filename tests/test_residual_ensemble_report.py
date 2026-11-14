import copy

import pytest

from padjective.residual_ensemble_experiment import TRAJECTORIES
from padjective.residual_ensemble_report import summarise


def synthetic_report():
    rows = []
    for roster, limit in TRAJECTORIES.items():
        for method in ('error_only', 'independent'):
            for size in range(1, limit+1):
                for fold in range(5):
                    # Unequal fold sizes must still receive equal weight.
                    loss = .1 + .01*fold + (.02 if method == 'error_only' else 0)
                    score = dict(n=10+10*fold, mean_padic_loss=loss,
                                 exact_accuracy=.6, first_digit_accuracy=.9)
                    rows.append(dict(roster=roster, method=method, members=size, fold=fold,
                        training_rows=score['n']//2, forgotten_correct_rows=1,
                        stored_nonzero_coefficients=7, mean_active_terms=2,
                        training=score, held_out=score))
    return dict(batch_id='synthetic', validation=dict(status='passed'), rows=rows)


def test_equal_fold_means_and_paired_differences():
    report = synthetic_report()
    actual = summarise(report)
    row = next(r for r in actual['summaries'] if r['roster'] == 0 and
               r['method'] == 'error_only' and r['members'] == 81)
    assert row['held_out']['mean_padic_loss'] == pytest.approx(.14)
    assert row['training_fraction'] == .5
    assert row['held_out_fold_losses'] == pytest.approx([.12, .13, .14, .15, .16])
    delta = next(r for r in actual['paired_comparisons'] if r['roster'] == 0 and r['members'] == 81)
    assert delta['loss_difference'] == pytest.approx(.02)
    assert delta['fold_loss_differences'] == pytest.approx([.02]*5)
    assert delta['folds_better'] == 0


def test_partial_or_duplicate_fold_grid_is_rejected():
    report = synthetic_report()
    broken = copy.deepcopy(report)
    broken['rows'].pop()
    with pytest.raises(AssertionError):
        summarise(broken)
    report['rows'][-1]['fold'] = 0
    with pytest.raises(AssertionError):
        summarise(report)
