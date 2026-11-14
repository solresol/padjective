"""Prespecified fold-paired analysis of the random-subspace/voting grid."""
from __future__ import annotations

import numpy as np
from scipy.stats import t

from .paper_ensemble_scaling_batch import SIZES
from .subspace_voting import FRACTIONS, RULES, corrected_test, holm


def multiplicity(contrasts, simultaneous=False):
    adjusted=holm([c['p_value'] for c in contrasts])
    for c,p in zip(contrasts,adjusted):
        c['holm_p_value']=p
        c['holm_reject_05']=p<.05
        c['family_size']=len(contrasts)
        if simultaneous:
            radius=float(t.ppf(1-.05/(2*len(contrasts)),c['df']))*c['standard_error']
            c['bonferroni_simultaneous_ci95']=[c['mean_difference']-radius,c['mean_difference']+radius]
    return contrasts


def analyse(rows):
    """Rows are aggregate fold evidence only; never individual product records."""
    primary=[r for r in rows if r['roster']==0]
    index={(r['fraction'],r['members'],r['rule'],r['fold']):r for r in primary}
    expected={(f,m,a,k) for f in FRACTIONS for m in SIZES for a in RULES for k in range(5)}
    assert len(primary)==len(index)==1125 and set(index)==expected
    fold_sizes=[index[(1.,1,RULES[0],k)]['metrics']['n'] for k in range(5)]
    assert sum(fold_sizes)==6693
    assert all(r['metrics']['n']==fold_sizes[r['fold']] for r in rows)

    def losses(f,m,a):
        return np.array([index[(f,m,a,k)]['metrics']['mean_padic_loss'] for k in range(5)])

    def contrast(name,difference,**labels):
        return dict(name=name,**labels,**corrected_test(difference,fold_sizes))

    tests=[]
    for f in FRACTIONS[:-1]:
        tests.append(contrast(f'fraction_{f}_vs_all_at_243',
            losses(f,243,RULES[0])-losses(1.,243,RULES[0]),kind='subset',fraction=f))
    for a in RULES[1:]:
        tests.append(contrast(f'{a}_vs_raw_valid_medoid_all_at_243',
            losses(1.,243,a)-losses(1.,243,RULES[0]),kind='aggregation',rule=a))
    for f in FRACTIONS[:-1]:
        tests.append(contrast(f'fraction_{f}_interaction_243_vs_1',
            (losses(f,243,RULES[0])-losses(1.,243,RULES[0]))-
            (losses(f,1,RULES[0])-losses(1.,1,RULES[0])),kind='interaction',fraction=f))
    assert len(tests)==12
    tests=multiplicity(tests,simultaneous=True)
    screening=[]
    for f in FRACTIONS:
        for m in SIZES:
            for a in RULES:
                if f==1. and a==RULES[0]:
                    continue
                screening.append(contrast(f'{f}_{m}_{a}_vs_reference_same_size',
                    losses(f,m,a)-losses(1.,m,RULES[0]),fraction=f,members=m,rule=a))
    assert len(screening)==216
    screening=multiplicity(screening)
    voting=[]
    for f in FRACTIONS:
        for a in ('projected_plurality','projected_survivor'):
            voting.append(contrast(f'{f}_{a}_vs_projected_medoid_at_243',
                losses(f,243,a)-losses(f,243,'projected_medoid'),fraction=f,rule=a))
    voting=multiplicity(voting)

    cells=[]
    for f in FRACTIONS:
        for m in SIZES:
            for a in RULES:
                points=[index[(f,m,a,k)] for k in range(5)]
                metrics={key:float(np.mean([r['metrics'][key] for r in points]))
                         for key in ('mean_padic_loss','exact_accuracy','first_digit_accuracy')}
                costs={key:float(np.mean([r[key] for r in points])) for key in
                       ('mean_depth','selected_features_total','feature_union_count',
                        'eligible_features','stored_nonzero_coefficients','mean_active_terms')}
                cells.append(dict(fraction=f,members=m,rule=a,fold_losses=losses(f,m,a).tolist(),
                    mean_metrics=metrics,mean_costs=costs,
                    fold_loss_sd=float(np.std(losses(f,m,a),ddof=1))))
    sensitivity=[r for r in rows if r['roster']!=0]
    key=lambda r:(r['fraction'],r['members'],r['rule'],r['fold'],r['roster'])
    roster_index={key(r):r for r in sensitivity}
    assert len(roster_index)==len(sensitivity)==5000
    assert set(roster_index)=={(f,m,a,k,r) for f in FRACTIONS for m in (9,81)
        for a in RULES for k in range(5) for r in range(1,21)}
    roster_summaries=[]
    for f in FRACTIONS:
        for m in (9,81):
            for a in RULES:
                means=[float(np.mean([roster_index[(f,m,a,k,r)]['metrics']['mean_padic_loss']
                       for k in range(5)])) for r in range(1,21)]
                roster_summaries.append(dict(fraction=f,members=m,rule=a,roster_mean_losses=means,
                    mean=float(np.mean(means)),minimum=min(means),maximum=max(means),
                    conditional_sd=float(np.std(means,ddof=1))))
    return dict(fold_sizes=fold_sizes,cells=cells,primary_contrasts=tests,
        secondary_grid_contrasts=screening,secondary_projection_control_contrasts=voting,
        roster_sensitivity=roster_summaries,
        statistical_method='Two-sided paired fold t with approximate Nadeau-Bengio-style overlap correction',
        interpretation='Negative differences favour the named alternative. Reused folds and exploratory tests; '
            'no claim of equivalence or independent test-set confirmation. Roster SD is conditional, not a test SE.')
