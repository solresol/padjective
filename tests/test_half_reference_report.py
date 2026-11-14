import pytest

from padjective.half_reference_report import markdown, summarise
from padjective.residual_ensemble_experiment import TRAJECTORIES


def synthetic_report():
    rows = []
    for roster, limit in TRAJECTORIES.items():
        for method, offset, accuracy in (('independent', 0, .7), ('error_only', .03, .5),
                                          ('half_reference', .01, .6)):
            for size in range(1, limit+1):
                for fold in range(5):
                    score = dict(n=10+10*fold, mean_padic_loss=.1+.01*fold+offset,
                                 exact_accuracy=accuracy, first_digit_accuracy=.9)
                    rows.append(dict(roster=roster, method=method, members=size, fold=fold,
                        training_rows=score['n']//2, forgotten_correct_rows=1,
                        stored_nonzero_coefficients=7, mean_active_terms=2,
                        training=score, held_out=score))
    return dict(batch_id='synthetic', rows=rows,
        manifest=dict(source_commit='synthetic', control_batch='control', snapshot_digest='digest'),
        validation=dict(status='passed', certified_coordinates=10, reference_checkpoint_checks=15))


def test_half_sample_is_compared_with_each_correct_control_and_equal_fold_weight():
    summary = summarise(synthetic_report())
    row = next(r for r in summary['summaries'] if r['roster'] == 0 and
               r['method'] == 'half_reference' and r['members'] == 81)
    assert row['held_out']['mean_padic_loss'] == pytest.approx(.13)
    comparisons = [r for r in summary['comparisons'] if r['roster'] == 0 and r['members'] == 81]
    by_control = {r['comparator']: r for r in comparisons}
    assert by_control['independent']['loss_difference'] == pytest.approx(.01)
    assert by_control['independent']['exact_accuracy_difference'] == pytest.approx(-.1)
    assert by_control['independent']['folds_better'] == 0
    assert by_control['error_only']['loss_difference'] == pytest.approx(-.02)
    assert by_control['error_only']['folds_better'] == 5


def test_sensitivity_table_control_columns_and_incomplete_report(tmp_path):
    report = synthetic_report()
    markdown(report, summarise(report), tmp_path)
    assert '| 0 | 0.130000 | 0.120000 | 0.150000 | 60.00% |' in (tmp_path/'README.md').read_text()
    report['rows'].pop()
    with pytest.raises(AssertionError):
        summarise(report)
