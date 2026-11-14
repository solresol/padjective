"""Reproduce aggregate-only trade-off figures without reading product data."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
import numpy as np


def config_key(config):
    # Keep the aggregate plotting module usable without the database runner.
    return json.dumps(config, sort_keys=True, separators=(',', ':'))

METRICS = ('mean_padic_loss', 'first_digit_accuracy', 'exact_accuracy')
STYLES = {
    'padic100': dict(label='p-adic · all features',color='#2166AC',marker='o',linestyle='-'),
    'padic75': dict(label='p-adic · 75% features',color='#67A9CF',marker='D',linestyle='--'),
    'tree': dict(label='Decision tree',color='#D97920',marker='s',linestyle='-'),
    'forest': dict(label='Random forest',color='#555555',marker='^',linestyle='--'),
}


def series(config):
    if config['family']=='padic':
        return 'padic100' if config['fraction']==1 else 'padic75'
    return config['family']


def group_rows(rows):
    """Average forest seeds inside each fold before the equal-fold mean."""
    groups=defaultdict(list)
    for row in rows:
        config=dict(row['config'])
        if config['family']=='forest':
            config.pop('seed')
        groups[config_key(config)].append(row)
    output=[]
    for key, members in groups.items():
        config=json.loads(key)
        folds=[]
        for f in range(5):
            fold=[r for r in members if r['fold']==f]
            assert len(fold)==(3 if config['family']=='forest' else 1)
            assert len({r['metrics']['n'] for r in fold})==1
            item=dict(fold=f,n=fold[0]['metrics']['n'])
            for metric in METRICS:
                item[metric]=float(np.mean([r['metrics'][metric] for r in fold]))
                item[metric+'_seed_min']=min(r['metrics'][metric] for r in fold)
                item[metric+'_seed_max']=max(r['metrics'][metric] for r in fold)
            item['active']=float(np.mean([r['active']['mean'] for r in fold]))
            item['stored_slots']=float(np.mean([r['counts']['stored_slots'] for r in fold]))
            item['broader_work_lower']=float(np.mean([r.get('broader_work',r.get('broader_work_lower')) for r in fold]))
            item['broader_work_upper']=float(np.mean([r.get('broader_work',r.get('broader_work_upper')) for r in fold]))
            if all('training' in r for r in fold):
                item['training_loss']=float(np.mean([r['training']['mean_padic_loss'] for r in fold]))
            folds.append(item)
        mean_keys=[*METRICS,'active','stored_slots','broader_work_lower','broader_work_upper']
        if 'training_loss' in folds[0]:
            mean_keys.append('training_loss')
        output.append(dict(config=config,series=series(config),folds=folds,
            **{k:float(np.mean([f[k] for f in folds])) for k in mean_keys}))
    return output


def frontier(rows,x='active',metric='mean_padic_loss'):
    """Discrete held-out envelope; do not interpret it as a tuned test score."""
    minimise=metric=='mean_padic_loss'
    ordered=sorted(rows,key=lambda r:(r[x],r[metric] if minimise else -r[metric]))
    result=[]
    best=float('inf') if minimise else -float('inf')
    for row in ordered:
        value=row[metric]
        if (value<best-1e-14 if minimise else value>best+1e-14):
            result.append(row)
            best=value
    return result


def weight_rows(rows,weight):
    return [r for r in rows if r['config']['family']=='padic' or r['config']['class_weight']==weight]


def description(c):
    if c['family']=='padic':
        return f"p-adic {c['fraction']:.0%}, {c['members']} members"
    weight='balanced' if c['class_weight']=='balanced' else 'unweighted'
    if c['family']=='forest':
        return f"forest ({weight}), {c['members']} trees, depth {c['max_depth'] or 'unlimited'}"
    setting=next((f'{k}={c[k]}' for k in ('max_depth','max_leaf_nodes','ccp_alpha') if k in c),'')
    return f'tree ({weight}), {setting}'


def budget_table(rows,x,budgets):
    output=[]
    for budget in budgets:
        point=dict(budget=budget)
        for name in STYLES:
            eligible=[r for r in rows if r['series']==name and r[x]<=budget]
            if eligible:
                best=min(eligible,key=lambda r:(r['mean_padic_loss'],r[x]))
                point[name]=dict(config=best['config'],loss=best['mean_padic_loss'],
                    active=best['active'],stored_slots=best['stored_slots'],
                    root_accuracy=best['first_digit_accuracy'],exact_accuracy=best['exact_accuracy'])
        output.append(point)
    return output


def setup_style():
    os.environ.setdefault('SOURCE_DATE_EPOCH', '1790035200')
    from matplotlib.font_manager import findfont
    family=None
    for candidate in ('Courier', 'Courier New', 'Nimbus Mono PS'):
        try:
            findfont(candidate, fallback_to_default=False)
            family=candidate
            break
        except ValueError:
            pass
    if family is None:
        raise RuntimeError('Install Courier or its URW equivalent Nimbus Mono PS for journal figures.')
    plt.rcParams.update({'font.family':family,'font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#AAAAAA',
        'text.color':'#252525','axes.labelcolor':'#252525','xtick.color':'#555555',
        'ytick.color':'#555555','svg.fonttype':'path','svg.hashsalt':'padjective-tradeoff',
        'pdf.fonttype':42,'ps.fonttype':42,
        'axes.titleweight':'semibold','savefig.facecolor':'white'})


def save(fig,output,name):
    for extension in ('png','svg','pdf','eps'):
        fig.savefig(output/f'{name}.{extension}',dpi=180,bbox_inches='tight',
                    metadata={'Creator':'Padjective trade-off figures'})
    plt.close(fig)


def axis(ax,x,metric,linear_loss=False):
    ax.set_xscale('symlog',linthresh=1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:
        f'{v/1e6:g}M' if v>=1e6 else f'{v/1e3:g}k' if v>=1e3 else f'{v:g}'))
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.set_xlabel({'active':'Mean decisions or coefficient consultations / product',
        'stored_slots':'Stored inference slots (declared representation)',
        'broader_work_lower':'Broader scoring proxy / product'}[x])
    if metric=='mean_padic_loss':
        ax.set_yscale('linear' if linear_loss else 'log')
        ax.set_ylim((0,1.05) if linear_loss else (.06,1.05))
        if not linear_loss:
            ticks=[.06,.08,.1,.15,.2,.3,.5,.7,1.]
            ax.set_yticks(ticks,labels=[str(t) for t in ticks])
        ax.set_ylabel('Held-out p-adic loss · lower is better')
    else:
        ax.set_ylim(0,1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel('Root accuracy' if metric=='first_digit_accuracy' else 'Exact-path accuracy')


def curves(ax,rows,x,metric='mean_padic_loss',label=True):
    for name,style in STYLES.items():
        selected=[r for r in rows if r['series']==name]
        ax.scatter([r[x] for r in selected],[r[metric] for r in selected],s=20,
            color=tuple(.75+.25*c for c in matplotlib.colors.to_rgb(style['color'])),
            marker=style['marker'],linewidths=0)
        if name.startswith('padic'):
            points=sorted(selected,key=lambda r:r['config']['members'])
        else:
            points=frontier(selected,x,metric)
        ax.plot([r[x] for r in points],[r[metric] for r in points],
            color=style['color'],marker=style['marker'],markersize=4,
            linestyle=style['linestyle'],linewidth=1.7,label=style['label'] if label else None)


def annotate_point(ax,row,x,text,offset,metric='mean_padic_loss',bold=False):
    ax.annotate(text,(row[x],row[metric]),xytext=offset,textcoords='offset points',
                fontsize=8,ha='left' if offset[0]>=0 else 'right',va='center',
                color='#202020',fontweight='bold' if bold else 'normal',
                bbox=dict(facecolor='white',edgecolor='none',pad=1),
                arrowprops=dict(arrowstyle='-',color='#555555',lw=.65),zorder=8)


def annotate_members(ax,rows,x,linear_loss=False,metric='mean_padic_loss'):
    # Sparse labels: member counts on two banks and tree counts on the forest.
    specs=[('padic100',1,'1 member',(-8,19)),
           ('padic100',9,'9 members',(-28,-25)),
           ('padic75',15,'15 members',(6,24)),
           ('padic75',243,'243 members',(9,33)),
           ('forest',81,'81 trees',(-8,19))]
    for name,m,label,offset in specs:
        candidates=[r for r in rows if r['series']==name and r['config']['members']==m]
        row=min(candidates,key=lambda r:r['mean_padic_loss'])
        if x!='active' and name=='padic100':
            continue
        if x=='broader_work_lower' and name=='padic75':
            offset=(-22,28) if m==15 else (28,57)
        if metric!='mean_padic_loss':
            offset=(offset[0],-offset[1])
        annotate_point(ax,row,x,label,offset,metric,bold=name=='padic75' and m==15)


def landscape(rows,output,x,name,linear_loss=False):
    fig,axes=plt.subplots(1,2,figsize=(14,5.4))
    for ax,weight,title in zip(axes,('balanced',None),('Class-balanced trees and forests','Unweighted trees and forests')):
        curves(ax,weight_rows(rows,weight),x)
        axis(ax,x,'mean_padic_loss',linear_loss)
        annotate_members(ax,weight_rows(rows,weight),x,linear_loss)
        ax.set_title(title,fontsize=12,pad=12)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.94),
               ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Model complexity and held-out hierarchical loss',fontsize=17,x=.08,ha='left',y=1.03)
    fig.text(.08,.965,'6,693 products · five frozen folds · forest scores averaged over three seeds per fold',fontsize=10)
    fig.text(.08,-.045,'Faint marks: all settings. Tree/forest lines: descriptive held-out frontiers; p-adic lines: fixed member sequence.\n'
        + ({'active':'Active counters exclude aggregation and do not imply equal runtime.',
            'stored_slots':'Slots count stored inference values and indices; they are not bytes or statistical degrees of freedom.',
            'broader_work_lower':'Bookkeeping sensitivity, not runtime. The p-adic lower bound omits defaults and thus favours p-adic models.'}[x]),fontsize=9,color='#555555')
    fig.subplots_adjust(top=.84,bottom=.17,wspace=.25)
    save(fig,output,name)


def accuracy_plot(rows,output):
    fig,axes=plt.subplots(1,2,figsize=(14,5.4))
    selected=weight_rows(rows,None)
    for ax,metric,title in zip(axes,METRICS[1:],('Correct taxonomy root','Correct complete taxonomy path')):
        curves(ax,selected,'active',metric)
        axis(ax,'active',metric)
        annotate_members(ax,selected,'active',metric=metric)
        ax.set_title(title,fontsize=12,pad=12)
    axes[0].legend(loc='lower right',frameon=False,fontsize=9)
    fig.suptitle('Accuracy and prediction work',fontsize=17,x=.08,ha='left',y=1.03)
    fig.text(.08,.965,'Unweighted classical fits · 6,693 products · equal-fold means; forests averaged across three seeds',fontsize=10)
    fig.text(.08,-.045,'Each classical line is the descriptive frontier for that panel’s accuracy metric; it is not a separately validated model.\n'
        'Root accuracy and exact-path accuracy answer different questions. Aggregation work is excluded.',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.84,bottom=.17,wspace=.25)
    save(fig,output,'accuracy-vs-work')


def fold_plot(rows,output,linear_loss=False):
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    selected=weight_rows(rows,None)
    for f,ax in enumerate(axes.flat[:5]):
        per_fold=[dict(r,**r['folds'][f]) for r in selected]
        curves(ax,per_fold,'active')
        axis(ax,'active','mean_padic_loss',linear_loss)
        ax.set_title(f"Fold {f+1} · {per_fold[0]['n']:,} held-out products",fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_xlabel('Mean active work / product',fontsize=10)
        ax.set_ylabel('Held-out p-adic loss',fontsize=10)
    axes[1,2].axis('off')
    handles,labels=axes[0,0].get_legend_handles_labels()
    axes[1,2].legend(handles,labels,loc='center left',frameon=False)
    fig.suptitle('Variation across the five held-out folds',fontsize=17,x=.07,ha='left')
    fig.text(.07,.93,'Unweighted trees and forests · the same p-adic banks in every comparison · lines are descriptive',fontsize=10)
    fig.text(.07,.012,'Folds share training products; their spread is not five independent replications. No seeds or ensemble members were selected on held-out labels.',fontsize=9)
    fig.subplots_adjust(top=.87,bottom=.085,hspace=.34,wspace=.25)
    save(fig,output,'fold-comparisons-linear' if linear_loss else 'fold-comparisons')


def depth_plot(rows,output):
    fig,axes=plt.subplots(1,2,figsize=(14,5.4))
    for ax,weight,title in zip(axes,('balanced',None),('Class-balanced tree','Unweighted tree')):
        trees=[r for r in rows if r['series']=='tree' and r['config']['class_weight']==weight and
               r['config']['sweep']=='depth' and r['config']['max_depth'] is not None]
        trees.sort(key=lambda r:r['config']['max_depth'])
        x=[r['config']['max_depth'] for r in trees]
        ax.plot(x,[r['mean_padic_loss'] for r in trees],'-s',color='#D97920',markersize=4,label='Held-out loss')
        ax.plot(x,[r['training_loss'] for r in trees],':',color='#777777',label='Training loss')
        for key in ('padic100','padic75'):
            p=next(r for r in rows if r['series']==key and r['config']['members']==9)
            ax.axhline(p['mean_padic_loss'],color=STYLES[key]['color'],linestyle=STYLES[key]['linestyle'],
                linewidth=1.3,label=STYLES[key]['label']+' · 9 members')
        ax.set_xscale('log',base=2)
        ax.set_xticks([1,4,16,64,256,512],labels=['1','4','16','64','256','512'])
        ax.set_ylim(0,1)
        ax.set_xlabel('Maximum permitted tree depth')
        ax.set_ylabel('p-adic loss · lower is better')
        ax.set_title(title,fontsize=12,pad=12)
        ax.grid(False)
    axes[0].legend(loc='upper right',frameon=False,fontsize=9)
    fig.suptitle('Tree depth, training fit and held-out performance',fontsize=17,x=.08,ha='left',y=1.03)
    fig.text(.08,.965,'Five frozen folds · fixed seed 42 · all 2,542 binary tag features · fixed 9-member references',fontsize=10)
    fig.text(.08,-.025,'Depth is a maximum; the main trade-off charts use actual mean decisions. Leaf-count and post-pruning sweeps are included there.',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.84,bottom=.14,wspace=.25)
    save(fig,output,'tree-depth')


def small_ensemble_plot(rows,output):
    fig,ax=plt.subplots(figsize=(8,4.8))
    for name in ('padic100','padic75'):
        bank=sorted((r for r in rows if r['series']==name),key=lambda r:r['config']['members'])
        ax.plot([r['config']['members'] for r in bank],[r['mean_padic_loss'] for r in bank],
                **STYLES[name],markersize=5,linewidth=1.5)
    for name,m,offset in [('padic100',9,(-40,15)),('padic75',15,(-10,-31)),
                          ('padic75',81,(-2,22)),('padic75',243,(-8,-29))]:
        row=next(r for r in rows if r['series']==name and r['config']['members']==m)
        annotate_point(ax,dict(row,members=m),'members',f"{m} members: {row['mean_padic_loss']:.4f}",
                       offset,bold=m==15)
    ax.set_xscale('log')
    ax.set_xticks([1,3,9,15,27,81,243],labels=['1','3','9','15','27','81','243'])
    ax.set_xlim(.85,310)
    ax.set_ylim(.105,.315)
    ax.set_xlabel('Ensemble members (log scale)')
    ax.set_ylabel('Held-out p-adic loss (linear scale)')
    ax.set_title('Most of the measured gain arrives in a small ensemble',fontsize=13,pad=14)
    ax.legend(frameon=False,loc='upper right',fontsize=10)
    fig.text(.125,-.04,'15 members at 75% features: 91.8% of the 1-to-243 loss reduction at 6.2% of the member work.\n'
             'Descriptive choice on the existing five folds; aggregation is additional work.',fontsize=9)
    save(fig,output,'small-ensemble')


def journal_figures(rows,output):
    """17.5 cm vector artwork; use line/marker differences for monochrome print."""
    selected=weight_rows(rows,None)
    with plt.rc_context({'font.size':9}):
        for name,panels in (
            ('tree_tradeoff_work',[('active',False,'(a) Logarithmic loss'),
                                  ('active',True,'(b) Linear loss')]),
            ('tree_tradeoff_costs',[('stored_slots',True,'(a) Stored inference slots'),
                                   ('broader_work_lower',True,'(b) Broader scoring proxy')])):
            fig,axes=plt.subplots(2,1,figsize=(17.5/2.54,7.3))
            for ax,(x,linear,title) in zip(axes,panels):
                curves(ax,selected,x)
                axis(ax,x,'mean_padic_loss',linear)
                annotate_members(ax,selected,x,linear)
                ax.set_title(title,loc='left',fontsize=10,pad=9)
                ax.set_ylabel('Mean p-adic loss')
                ax.tick_params(labelsize=8)
                ax.set_xlabel({'active':'Mean branch decisions or coefficient consultations / product',
                    'stored_slots':'Stored inference slots',
                    'broader_work_lower':'Broader scoring proxy / product'}[x],fontsize=9)
            axes[0].legend(frameon=False,loc='upper right',fontsize=8)
            fig.subplots_adjust(top=.96,bottom=.08,left=.12,right=.98,hspace=.42)
            save(fig,output,name)


def analyse(report):
    rows=group_rows(report['rows'])
    tables={}
    for weight in ('balanced',None):
        label=weight or 'unweighted'
        for x,budgets in (('active',(3,10,30,100,300,1000,3000,10000)),
                ('stored_slots',(100,1000,3000,10000,30000,100000,300000,1000000))):
            tables[f'{label}_{x}']=budget_table(weight_rows(rows,weight),x,budgets)
    crossovers=[]
    for weight in ('balanced',None):
        for linear in ('padic100','padic75'):
            bank=[r for r in rows if r['series']==linear]
            for family in ('tree','forest'):
                classical=frontier([r for r in weight_rows(rows,weight) if r['series']==family])
                previous=None
                for point in classical:
                    available=[r for r in bank if r['active']<=point['active']]
                    if available:
                        best=min(available,key=lambda r:r['mean_padic_loss'])
                        if point['mean_padic_loss']<best['mean_padic_loss']:
                            crossovers.append(dict(weight=weight,linear=linear,family=family,
                                previous_frontier_point=previous,first_overtaking_point=point,
                                best_affordable_linear=best))
                            break
                    previous=point
    dominated=[]
    trees=[r for r in weight_rows(rows,None) if r['series']=='tree']
    for p in [r for r in rows if r['config']['family']=='padic']:
        options=[r for r in trees if r['stored_slots']<=p['stored_slots'] and r['mean_padic_loss']<p['mean_padic_loss']]
        if options:
            t=min(options,key=lambda r:r['mean_padic_loss'])
            dominated.append(dict(padic=p['config'],tree=t['config']))
    bank={r['config']['members']:r for r in rows if r['series']=='padic75'}
    small=bank[15]
    tradeoff=dict(config=small['config'],active=small['active'],loss=small['mean_padic_loss'],
        root_accuracy=small['first_digit_accuracy'],exact_accuracy=small['exact_accuracy'],
        fraction_of_1_to_243_loss_reduction=(bank[1]['mean_padic_loss']-small['mean_padic_loss'])/
            (bank[1]['mean_padic_loss']-bank[243]['mean_padic_loss']),
        fraction_of_243_member_work=small['active']/bank[243]['active'],
        same_budget_comparisons=budget_table(weight_rows(rows,None),'active',[small['active']])[0],
        selection='Illustrative choice after inspecting the fixed-fold results; not a validated optimum.')
    return dict(batch_id=report['batch_id'],rows=rows,budget_tables=tables,small_ensemble_tradeoff=tradeoff,
                active_crossovers=crossovers,storage_dominated_by_unweighted_tree=dominated,
                note='All frontiers and budget winners are descriptive selections on the same held-out folds.')


def persist_review(output):
    """Attach the reviewed aggregate bundle to its own experiment record."""
    from psycopg.types.json import Jsonb
    from . import db
    raw=(output/'results.json').read_bytes()
    result=json.loads(raw)
    analysis=json.loads((output/'analysis.json').read_text())
    assert analysis['results_sha256']==hashlib.sha256(raw).hexdigest()
    assert result['batch_id']==analysis['batch_id']
    assert analysis['reference_check']['status']=='passed'
    bundle=dict(analysis=analysis,writeup=(output/'results.md').read_text(),
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())
                   if p.suffix in ('.json','.md','.png','.svg','.pdf','.eps')})
    with db.get_connection() as conn:
        prior=conn.execute('SELECT report FROM padjective.paper_tree_tradeoff_batches WHERE batch_id=%s',
                           (result['batch_id'],)).fetchone()[0]
        assert prior['rows']==result['rows'] and prior['validation']==result['validation']
        conn.execute('''UPDATE padjective.paper_tree_tradeoff_batches
            SET report=report || %s WHERE batch_id=%s''',
            (Jsonb(dict(review_bundle=bundle)),result['batch_id']))
        conn.commit()
        saved=conn.execute("SELECT report->'review_bundle' FROM padjective.paper_tree_tradeoff_batches WHERE batch_id=%s",
                           (result['batch_id'],)).fetchone()[0]
        assert saved==bundle
    print(json.dumps(dict(review_persisted=True,batch_id=result['batch_id'],artifacts=len(bundle['artifacts']))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--persist-review',action='store_true',
        help='Persist an already generated and reviewed bundle to Postgres; do not redraw.')
    args=parser.parse_args()
    if args.persist_review:
        persist_review(args.output_directory)
        return
    raw=args.results.read_bytes()
    report=json.loads(raw)
    assert report['validation']['status']=='passed'
    analysis=analyse(report)
    analysis['results_sha256']=hashlib.sha256(raw).hexdigest()
    analysis['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    reference_path=Path('docs/ensemble-scaling/analysis.json')
    reference=next(r for r in json.loads(reference_path.read_text())['reference_points'] if r['key']=='dt')
    tree=next(r for r in analysis['rows'] if r['config']==dict(family='tree',
        class_weight='balanced',seed=42,sweep='depth',max_depth=None))
    np.testing.assert_allclose([r['mean_padic_loss'] for r in tree['folds']],reference['fold_losses'],rtol=0,atol=2e-14)
    np.testing.assert_allclose([tree['active'],tree['mean_padic_loss'],tree['exact_accuracy']],
        [reference['active'],reference['loss'],reference['exact_accuracy']],rtol=0,atol=2e-12)
    analysis['reference_check']=dict(status='passed',path=str(reference_path),
        sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),reference=reference)
    output=args.output_directory
    output.mkdir(parents=True,exist_ok=True)
    (output/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    setup_style()
    landscape(analysis['rows'],output,'active','loss-vs-work')
    landscape(analysis['rows'],output,'stored_slots','loss-vs-storage')
    landscape(analysis['rows'],output,'broader_work_lower','loss-vs-broader-work')
    for x,name in [('active','loss-vs-work'),('stored_slots','loss-vs-storage'),
                   ('broader_work_lower','loss-vs-broader-work')]:
        landscape(analysis['rows'],output,x,name+'-linear',linear_loss=True)
    accuracy_plot(analysis['rows'],output)
    fold_plot(analysis['rows'],output)
    fold_plot(analysis['rows'],output,linear_loss=True)
    depth_plot(analysis['rows'],output)
    small_ensemble_plot(analysis['rows'],output)
    journal_figures(analysis['rows'],output)
    print(json.dumps(dict(configurations=len(analysis['rows']),output=str(output))))


if __name__=='__main__':
    main()
