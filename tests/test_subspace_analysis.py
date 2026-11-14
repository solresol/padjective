import copy
import numpy as np
import pytest
from padjective.paper_ensemble_scaling_batch import SIZES
from padjective.paper_subspace_analysis import analyse
from padjective.paper_subspace_experiment import jobs
from padjective.subspace_voting import FRACTIONS, RULES


def synthetic_grid():
    rows=[]
    sizes=[1339,1339,1339,1338,1338]
    for f in FRACTIONS:
        for roster in range(21):
            for m in (SIZES if roster==0 else (9,81)):
                for a in RULES:
                    for k in range(5):
                        loss=.2+(.02+.001*k)*(1-f)+.001*RULES.index(a)+.1/m
                        rows.append(dict(fraction=f,roster=roster,members=m,rule=a,fold=k,
                            metrics=dict(n=sizes[k],mean_padic_loss=loss,exact_accuracy=.5,first_digit_accuracy=.8),
                            mean_depth=2,selected_features_total=m*100,feature_union_count=100,
                            eligible_features=200,stored_nonzero_coefficients=m*20,mean_active_terms=m*2))
    return rows


def test_grid_and_test_families():
    assert len(jobs())==len(set(jobs()))==4860
    result=analyse(synthetic_grid())
    assert len(result['cells'])==225
    assert len(result['primary_contrasts'])==12
    assert len(result['secondary_grid_contrasts'])==216
    assert len(result['secondary_projection_control_contrasts'])==10
    assert len(result['roster_sensitivity'])==50
    first=result['primary_contrasts'][0]
    assert first['mean_difference']==pytest.approx(.022*(1-.125))
    assert first['family_size']==12
    assert first['bonferroni_simultaneous_ci95'][1]>first['ci95'][1]
    for c in result['primary_contrasts'][8:]:
        assert c['mean_difference']==pytest.approx(0,abs=1e-15)


def test_missing_or_duplicate_folds_are_rejected():
    rows=synthetic_grid()
    with pytest.raises(AssertionError):
        analyse(rows[1:])
    with pytest.raises(AssertionError):
        analyse(rows+[copy.deepcopy(rows[0])])


def test_model_readback_reconstruction():
    from scipy.sparse import csr_matrix
    from padjective.paper_subspace_evaluate import check_model
    from padjective.paper_randomised_methods import model_evidence
    from padjective.paper_validate_randomised import fingerprint
    from padjective.randomised_linear import fit_coordinates
    x=csr_matrix(np.array([[1,0],[0,1],[1,1],[0,0]],dtype=np.int64))
    y=np.array([1,2,72,1])
    fit=fit_coordinates(x,y,p=71,precision=7,seed=42)
    e,metrics=model_evidence(fit.coefficients,x,x,y,y,p=71,precision=7,pilot=False,completed=True)
    e.update(metrics={k:metrics[k] for k in ('training_raw','training','held_out_raw','held_out')},
        coefficient_sha256=fingerprint(e['coefficients']),prediction_sha256=fingerprint(e['predictions']),
        stored_nonzero_coefficients=metrics['stored_nonzero_coefficients'],
        mean_active_terms=metrics['mean_nonzero_terms_consulted'])
    c=dict(x_train=x,x_test=x,y_train=y,y_test=y)
    check_model(e,c,np.arange(2))
    broken=copy.deepcopy(e)
    broken['predictions'][0]=2
    with pytest.raises(AssertionError):
        check_model(broken,c,np.arange(2))


def test_figure_exports_require_validated_complete_grid(tmp_path):
    from padjective.paper_subspace_figures import export
    report=dict(batch_id='synthetic-test-only',validation=dict(status='passed'),analysis=analyse(synthetic_grid()))
    export(report,tmp_path)
    assert len(list(tmp_path.iterdir()))==4
    assert all(p.stat().st_size>1000 for p in tmp_path.iterdir())
    with pytest.raises(AssertionError):
        export(report,tmp_path)
