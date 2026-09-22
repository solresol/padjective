import numpy as np
import pytest
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from padjective import paper_tree_ensemble_tradeoff as study


def test_stump_and_impure_leaf_storage_and_decisions():
    x=sparse.csr_matrix(np.array([[0],[0],[1],[1]],dtype=np.float32))
    y=np.array(['a','b','b','b'])
    model=DecisionTreeClassifier(max_depth=1,random_state=42).fit(x,y)
    hard,paths,_=study.tree_counts(model,x)
    soft,_,contrib=study.tree_counts(model,x,probabilities=True)
    assert hard['internal_nodes']==1 and hard['leaves']==2
    assert hard['stored_slots']==6
    assert soft['leaf_nonzero_probabilities']==3
    assert soft['stored_slots']==11
    assert paths.tolist()==[1,1,1,1]
    assert contrib.tolist()==[2,2,1,1]


def test_constant_predictor_has_zero_decisions_and_one_label_slot():
    x=sparse.csr_matrix(np.zeros((5,3),dtype=np.float32))
    model=DecisionTreeClassifier().fit(x,['a']*5)
    counts,paths,_=study.tree_counts(model,x)
    assert not paths.any()
    assert counts['stored_slots']==1


def test_grid_is_fixed_unique_and_contains_paper_reference():
    trees=list(study.tree_configs())
    forests=list(study.forest_configs())
    assert len(trees)==80 and len(forests)==48
    assert len({study.config_key(c) for c in trees})==80
    assert len({study.config_key(c) for c in forests})==48
    assert dict(family='tree',class_weight='balanced',seed=42,sweep='depth',max_depth=None) in trees
    assert {c['class_weight'] for c in trees}=={'balanced',None}


def test_forest_job_preserves_probability_average_and_prefixes(monkeypatch):
    # Unequal leaf probabilities expose hard-vote substitutes and class-index errors.
    rng=np.random.default_rng(19)
    x=sparse.csr_matrix(rng.integers(0,2,(60,5)).astype(np.float32))
    labels=np.array(['z','a','m']*20,dtype=object)
    codes={'z':1,'a':72,'m':2}
    y=study.encode(labels,codes)
    context=dict(x_train=x[:40],x_test=x[40:],labels_train=labels[:40],
                 y_train=y[:40],y_test=y[40:],codes=codes,row_sha256='test')
    monkeypatch.setattr(study,'CONTEXT',{0:context})
    monkeypatch.setattr(study,'FOREST_SIZES',(1,3))
    class Connection:
        def __enter__(self): return self
        def __exit__(self,*args): pass
    monkeypatch.setattr(study.db,'get_connection',Connection)
    saved=[]
    monkeypatch.setattr(study,'save_result',lambda conn,b,f,c,e:saved.append((c,e)))
    config=dict(family='forest',class_weight=None,seed=42,max_depth=2)
    study.fit_job('test',0,config)
    assert [c['members'] for c,e in saved]==[1,3]
    for c,e in saved:
        model=RandomForestClassifier(n_estimators=c['members'],max_depth=2,
              random_state=42,n_jobs=1).fit(x[:40],labels[:40])
        pred=study.encode(model.predict(x[40:]),codes)
        assert e['predictions']==pred.tolist()
        assert e['metrics']['exact_accuracy']==pytest.approx(np.mean(pred==y[40:]))
        expected=sum(np.asarray(t.decision_path(x[40:]).sum(axis=1)).ravel()-1 for t in model.estimators_)
        assert e['active']['mean']==pytest.approx(expected.mean())


def test_padic_metric_distinguishes_root_and_exact_correctness():
    score=study.score(np.array([1,1,1]),np.array([2,72,1]))
    assert score['mean_padic_loss']==pytest.approx((1+1/71)/3)
    assert score['first_digit_accuracy']==pytest.approx(2/3)
    assert score['exact_accuracy']==pytest.approx(1/3)
