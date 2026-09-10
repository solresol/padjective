from itertools import combinations_with_replacement, product
import numpy as np
import pytest
from scipy.stats import t
from padjective.randomised_linear import medoid
from padjective.subspace_voting import (FRACTIONS, aggregate_rosters, aggregate_sizes, corrected_test, feature_mask,
    holm, path_digits, plurality, project_codes, survivor_vote)


def tree_reference(values,p):
    paths=[path_digits(v,p,7) for v in values]
    root=min(set(v[0] for v in paths),key=lambda c:(-sum(v[0]==c for v in paths),c))
    keep=[v for v in paths if v[0]==root]
    prefix=(root,)
    while True:
        stop=sum(v==prefix for v in keep)
        if stop>=len(keep)-stop:
            return sum(d*p**i for i,d in enumerate(prefix))
        keep=[v for v in keep if len(v)>len(prefix)]
        children={v[len(prefix)] for v in keep}
        child=min(children,key=lambda c:(-sum(v[len(prefix)]==c for v in keep),c))
        prefix+=child,
        keep=[v for v in keep if v[:len(prefix)]==prefix]


def test_nested_masks_and_independent_seeds():
    features=np.arange(2,51,2)
    masks=[feature_mask(features,f,2,1729) for f in FRACTIONS]
    for i,m in enumerate(masks):
        assert len(m)==int(np.ceil(len(features)*FRACTIONS[i]))
        assert np.array_equal(m,feature_mask(features,FRACTIONS[i],2,1729))
    assert all(set(a)<=set(b) for a,b in zip(masks,masks[1:]))
    assert set(masks[-1])==set(features)
    assert not np.array_equal(masks[2],feature_mask(features,.5,2,42))


def test_three_rules_are_distinct():
    values=[1]*3+[2]*2+[73]*2+[144]*2
    assert plurality(values)==1
    assert medoid(values,71,7)[0]==2
    assert survivor_vote(values)==73


def test_stop_continue_is_not_child_plurality():
    values=[1]*3+[72]*2+[143]*2
    assert plurality(values)==medoid(values,71,7)[0]==1
    assert survivor_vote(values)==72
    assert survivor_vote([1,72])==1


def test_ultrametric_can_choose_minority_root_for_large_ensemble():
    root_one=[1+71*k for k in range(71)]+[1,72,143]
    values=root_one+[2]*73
    assert len(root_one)==74 and len(values)==147
    assert medoid(values,71,7)[0]==2
    assert survivor_vote(values)%71==1


def test_exhaustive_tree_vote_and_majority_property():
    options=[1,2,4,7,5,8,13]
    for size in range(1,6):
        for values in combinations_with_replacement(options,size):
            winner=survivor_vote(values,p=3)
            assert winner==tree_reference(values,3) and winner in values
            mode=plurality(values)
            if values.count(mode)>size/2:
                assert winner==mode==medoid(values,3,7)[0]


def test_all_one_member_rules_equal_projection():
    bank=np.array([[0,1,2],[7,7,4],[26,5,8]])
    candidates=[1,2,4,5,8,13]
    results=list(aggregate_sizes(bank,candidates,[1,3],p=3,precision=3))
    assert np.array_equal(results[0][1]['raw_valid_medoid'],project_codes(bank[:,0],candidates,3,3))


def test_corrected_formula_and_holm():
    d=np.array([.1,.2,.3,.2,.1])
    result=corrected_test(d,[20]*5)
    se=np.sqrt((.2+.25)*np.var(d,ddof=1))
    assert result['standard_error']==pytest.approx(se)
    assert result['p_value']==pytest.approx(2*t.sf(abs(d.mean()/se),4))
    assert holm([.03,.01,.04])==pytest.approx([.06,.03,.06])
    assert corrected_test([0]*5,[20]*5)['p_value']==1


def test_singleton_components_create_non_affine_interaction():
    outputs={}
    for x in product([0,1],repeat=3):
        # Raw g_j=x_j, zero default=2: each decoded component is affine 2-x_j.
        voters=[v if v else 2 for v in x]
        outputs[x]=plurality(voters)
        assert outputs[x]==survivor_vote(voters)==medoid(voters,71,7)[0]
    assert outputs[(0,0,0)]==outputs[(1,0,0)]==outputs[(0,1,0)]==2
    assert outputs[(1,1,0)]==1  # Violates affine parallelogram identity mod 71.


def test_rosters_against_original_consensus():
    from padjective.randomised_linear import consensus_predictions
    rng=np.random.default_rng(20260911)
    bank=rng.integers(0,3**3,size=(12,9))
    candidates=[1,2,4,5,7,8,13,14,16,17,22,23,25,26]
    requests=[(0,np.arange(9),(1,3,9)),(1,rng.permutation(9),(3,9))]
    outputs={}
    for roster,size,pred in aggregate_rosters(bank,candidates,requests,p=3,precision=3):
        voters=bank[:,requests[roster][1][:size]]
        projected=project_codes(voters,candidates,3,3)
        for rule,values in (('raw_valid_medoid',voters),('projected_medoid',projected)):
            expected,_=consensus_predictions(values,candidates,p=3,precision=3)
            assert np.array_equal(expected,pred[rule])
        assert pred['projected_survivor'].tolist()==[tree_reference(row,3) for row in projected]
        outputs[roster,size]=pred
    assert all(np.array_equal(outputs[0,9][key],outputs[1,9][key]) for key in outputs[0,9])


def test_raw_plurality_retains_latent_codes_and_represents_xnor():
    # g1=143*x1, g2=143*x2, defaults 72; g3=0, default 2.
    # Both 72 and 143 project to 72, but raw-code collisions carry information.
    bank=np.array([[143*x if x else 72,143*y if y else 72,2]
                   for x,y in product([0,1],repeat=2)])
    result=dict(aggregate_sizes(bank,[2,72],[3]))[3]
    assert result['raw_plurality_project'].tolist()==[72,2,2,72]
    for rule in ('raw_valid_medoid','projected_medoid','projected_plurality','projected_survivor'):
        assert result[rule].tolist()==[72]*4
