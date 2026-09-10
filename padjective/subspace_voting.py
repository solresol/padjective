"""Deterministic random subspaces and taxonomy voting; no fitting-state changes."""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import t

from .paper_validate_randomised import distance_matrix
from .paper_validate_ensemble_scaling import nested_consensus

FRACTIONS = (0.125, 0.25, 0.5, 0.75, 1.0)
RULES = ('raw_valid_medoid', 'projected_medoid', 'projected_plurality',
         'projected_survivor', 'raw_plurality_project')


def feature_mask(supported, fraction, fold, seed_base):
    supported = np.asarray(supported, dtype=np.int64)
    if fraction not in FRACTIONS or not len(supported) or len(np.unique(supported)) != len(supported):
        raise ValueError('Nonempty distinct supported features and protocol fraction required')
    order = np.random.default_rng(np.random.SeedSequence([20260911, fold, seed_base])).permutation(supported)
    return np.sort(order[:math.ceil(fraction*len(supported))])


def project_codes(values, candidates, p=71, precision=7):
    data=np.asarray(values,dtype=np.int64)
    options=np.unique(np.asarray(candidates,dtype=np.int64))
    if not len(options) or np.any(data<0) or np.any(data>=p**precision):
        raise ValueError('Canonical predictions and nonempty candidates required')
    unique,inverse=np.unique(data,return_inverse=True)
    decoded=np.empty(len(unique),dtype=np.int64)
    for start in range(0,len(unique),1024):
        block=unique[start:start+1024]
        decoded[start:start+len(block)]=options[np.argmin(distance_matrix(block,options,p,precision),axis=1)]
    return decoded[inverse].reshape(data.shape)


def plurality(values):
    codes,counts=np.unique(values,return_counts=True)
    if not len(codes):
        raise ValueError('At least one voter required')
    return int(codes[int(np.argmax(counts))])


def path_digits(code,p=71,precision=7):
    code=int(code)
    if not 0<code<p**precision:
        raise ValueError('A nonempty canonical path is required')
    digits=[]
    while code:
        digit=code%p
        if not digit:
            raise ValueError('Zero is reserved for termination, not an internal branch')
        digits.append(digit)
        code//=p
    return tuple(digits)


def survivor_vote(values,p=71,precision=7):
    """Root plurality, then stop/continue majority, then child plurality."""
    remaining=np.asarray(values,dtype=np.int64)
    if not len(remaining):
        raise ValueError('At least one voter required')
    # Callers validate their candidate vocabulary once; values must be paths.
    root=plurality(remaining%p)
    remaining=remaining[remaining%p==root]
    prefix=root
    for depth in range(1,precision+1):
        stopped=remaining==prefix
        if 2*int(stopped.sum())>=len(remaining):
            return int(prefix)
        remaining=remaining[~stopped]
        branch=plurality((remaining//p**depth)%p)
        if not branch:
            raise ValueError('Non-contiguous input path')
        prefix+=branch*p**depth
        remaining=remaining[(remaining//p**depth)%p==branch]
    raise AssertionError('A finite valid path vote must terminate')


def aggregate_sizes(bank,candidates,sizes,p=71,precision=7):
    bank=np.asarray(bank,dtype=np.int64)
    candidates=np.unique(candidates)
    for candidate in candidates:
        path_digits(candidate,p,precision)
    projected=project_codes(bank,candidates,p,precision)
    order=[np.arange(bank.shape[1])]
    raw_results={m:y for _,m,y in nested_consensus(bank,candidates,sizes,order,p,precision)}
    projected_results={m:y for _,m,y in nested_consensus(projected,candidates,sizes,order,p,precision)}
    for size in sizes:
        voters=projected[:,:size]
        output=dict(raw_valid_medoid=raw_results[size],projected_medoid=projected_results[size],
            projected_plurality=np.array([plurality(row) for row in voters],dtype=np.int64),
            projected_survivor=np.array([survivor_vote(row,p,precision) for row in voters],dtype=np.int64),
            raw_plurality_project=project_codes([plurality(row) for row in bank[:,:size]],candidates,p,precision))
        assert all(np.all(np.isin(y,candidates)) for y in output.values())
        for rule in ('projected_medoid','projected_plurality','projected_survivor'):
            assert np.all(np.any(voters==output[rule][:,None],axis=1))
        if size==1:
            assert all(np.array_equal(y,output['raw_valid_medoid']) for y in output.values())
        yield size,output


def corrected_test(differences,fold_sizes):
    """Approximate Nadeau-Bengio-style overlap correction; paired fold unit."""
    values=np.asarray(differences,dtype=float)
    sizes=np.asarray(fold_sizes,dtype=int)
    if len(values)!=len(sizes) or len(values)<2 or np.any(sizes<=0):
        raise ValueError('At least two aligned nonempty folds required')
    n=len(values)
    ratio=float(np.mean(sizes/(sizes.sum()-sizes)))
    mean=float(values.mean())
    se=float(np.sqrt((1/n+ratio)*values.var(ddof=1)))
    if se==0:
        p=1.0 if mean==0 else 0.0
        statistic=0.0 if mean==0 else None
    else:
        statistic=mean/se
        p=float(2*t.sf(abs(statistic),n-1))
    return dict(mean_difference=mean,fold_differences=values.tolist(),standard_error=se,
                df=n-1,t_statistic=statistic,p_value=p,test_train_ratio=ratio,
                ci95=[mean-float(t.ppf(.975,n-1))*se,mean+float(t.ppf(.975,n-1))*se],
                zero_variance=se==0)


def holm(p_values):
    p=np.asarray(p_values,dtype=float)
    if p.ndim!=1 or np.any((p<0)|(p>1)) or not np.all(np.isfinite(p)):
        raise ValueError('Finite probabilities required')
    order=np.argsort(p,kind='stable')
    adjusted=np.empty(len(p))
    adjusted[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted.tolist()
