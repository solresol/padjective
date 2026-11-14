"""Export the fixed side-experiment comparisons as scholarly PDF figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .paper_ensemble_scaling_batch import SIZES
from .subspace_voting import FRACTIONS, RULES

LABELS=dict(raw_valid_medoid='Raw → valid medoid',projected_medoid='Project → medoid',
    projected_plurality='Project → plurality',projected_survivor='Project → survivor',
    raw_plurality_project='Raw plurality → project')
COLOURS=('#93adc9','#688fbc','#3569a8','#25496f','#d18a3c')
MARKERS=('o','s','^','D','x')
STYLES=(':','--','-.','-',(0,(5,2)))


def export(report,directory):
    assert report['validation']['status']=='passed'
    cells=report['analysis']['cells']
    assert len(cells)==225
    lookup={(c['fraction'],c['members'],c['rule']):c for c in cells}
    assert len(lookup)==225
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True)
    for i,f in enumerate(FRACTIONS):
        axes[0].plot(SIZES,[lookup[f,m,RULES[0]]['mean_metrics']['mean_padic_loss'] for m in SIZES],
            color=COLOURS[i],marker=MARKERS[i],linestyle=STYLES[i],linewidth=1.5,
            markersize=4,label=f'{100*f:g}% of eligible features')
    for i,rule in enumerate(RULES):
        axes[1].plot(SIZES,[lookup[1.,m,rule]['mean_metrics']['mean_padic_loss'] for m in SIZES],
            color=COLOURS[i],marker=MARKERS[i],linestyle=STYLES[i],linewidth=1.5,
            markersize=4,label=LABELS[rule])
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks([1,3,9,27,81,243],labels=['1','3','9','27','81','243'])
        ax.set_xlabel('Ensemble members (log scale)')
        ax.grid(axis='y',color='#dddddd',linewidth=.6)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False,fontsize=8)
    axes[0].set_ylabel('Mean p-adic loss (lower is better)')
    axes[0].set_title('Feature fraction: current valid-path consensus',loc='left',y=1.32)
    axes[1].set_title('Aggregation rule: all-feature components',loc='left',y=1.32)
    fig.suptitle('Random feature subsets and ensemble aggregation',x=.07,ha='left',fontsize=15)
    fig.text(.07,.925,'Five fold means; 6,693 products; fixed nested member banks; no extrapolation.',fontsize=10)
    fig.subplots_adjust(left=.07,right=.985,top=.66,bottom=.16,wspace=.12)
    fig.text(.07,.035,'Exploratory extension on reused folds. Feature eligibility is determined from the fitting fold only.\n'
        f'Source: Postgres experiment {report["batch_id"]}.',fontsize=8,color='#444444')
    save(fig,directory/'subspace-size-comparisons')

    tests=report['analysis']['primary_contrasts']
    assert len(tests)==12 and all(c['family_size']==12 for c in tests)
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    titles=('Feature subsets at 243 members: current consensus',
            'Alternative aggregation at 243 members: all features',
            'Interaction: subset penalty at 243 members minus subset penalty at 1')
    for group,ax in enumerate(axes):
        selected=tests[group*4:(group+1)*4]
        labels=[(f'{c["fraction"]*100:g}% versus 100% features' if 'fraction' in c else LABELS[c['rule']]) for c in selected]
        for i,c in enumerate(selected):
            lo,hi=c['bonferroni_simultaneous_ci95']
            x=c['mean_difference']
            ax.errorbar(x,i,xerr=np.array([[x-lo],[hi-x]]),fmt='o',color='#3569a8',
                        capsize=3,markersize=5,linewidth=1.5)
        ax.set_yticks(range(4),labels=labels)
        ax.invert_yaxis()
        ax.set_ylim(3.7,-.7)
        ax.axvline(0,color='#333333',linewidth=.9)
        ax.grid(axis='x',color='#dddddd',linewidth=.6)
        ax.set_title(titles[group],loc='left',fontsize=11)
    axes[-1].set_xlabel('Difference in mean p-adic loss (negative favours the named change)')
    fig.suptitle('Prespecified primary comparisons',x=.04,ha='left',fontsize=15)
    fig.text(.04,.935,'Five paired folds; approximate overlap correction; simultaneous 95% intervals for 12 contrasts.',fontsize=10)
    fig.subplots_adjust(left=.29,right=.97,top=.87,bottom=.16,hspace=.4)
    fig.text(.04,.035,'Intervals use Bonferroni correction; the accompanying table reports Holm-adjusted two-sided tests.\n'
        'Interactions compare the subset-minus-all-feature difference at 243 versus 1 member.\n'
        'These reused-fold tests are approximate, exploratory evidence, not independent test-set confirmation.',fontsize=9,color='#444444')
    save(fig,directory/'subspace-primary-contrasts')


def save(fig,stem):
    for suffix in ('.pdf','.png'):
        path=stem.with_suffix(suffix)
        assert not path.exists(),f'Do not overwrite {path}'
        fig.savefig(path,dpi=180,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    args.output_directory.mkdir(parents=True,exist_ok=True)
    export(json.loads(args.results.read_text()),args.output_directory)
    print(json.dumps(dict(event='subspace_figures_exported',source_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(),
                          output_directory=str(args.output_directory))))


if __name__=='__main__':
    main()
