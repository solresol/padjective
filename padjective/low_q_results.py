"""Summarise aggregate low-Q results without accessing private predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


LABELS = {
    'taxonomy_association': 'Greedy: taxonomy association',
    'random': 'Greedy: random order', 'frequency': 'Greedy: frequency',
    'battle_elo': 'Greedy: Battle Elo', 'mean_title_position': 'Greedy: title position',
    'ensemble_1_1': '1 member: all features',
    'ensemble_0.75_1': '1 member: 75% features',
    'ensemble_1_15': '15 members: all features',
    'ensemble_0.75_15': '15 members: 75% features',
    'ensemble_1_243': '243 members: all features',
    'ensemble_0.75_243': '243 members: 75% features',
}


def summarise(report, output):
    index = {(s['training_q'], s['configuration']): s for s in report['series']}
    configs = sorted(k for q, k in index if q == 'sqrt2')
    assert len(configs) == (9 if report['members_retrained'] == 15 else 11)
    assert all((q, k) in index for q in ('71', 'sqrt2', '2') for k in configs)
    configs.sort(key=lambda k: index['71', k]['mean_losses']['71'])
    columns = [('71', '71'), ('71', 'sqrt2'), ('sqrt2', 'sqrt2')]
    ranks = [dict(zip(configs, rankdata([index[t, k]['mean_losses'][q] for k in configs])))
             for t, q in columns]
    q2_ranks = [dict(zip(configs, rankdata([index[t, k]['mean_losses']['2'] for k in configs])))
                for t in ('71', '2')]
    prefixes = []
    for r in (ranks[2], q2_ranks[1]):
        count = 0
        for k in configs:
            if r[k] != ranks[0][k]:
                break
            count += 1
        prefixes.append(count)
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    fig.subplots_adjust(left=.02, right=.94, top=.79, bottom=.13)
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h', '*']
    for i, k in enumerate(configs):
        y = [r[k] for r in ranks]
        changed = len(set(y)) > 1
        ax.plot([0, 1, 2], y, color='black' if changed else '0.68',
                linewidth=1.7 if changed else 1., marker=markers[i], markersize=5)
        ax.text(-.07, y[0], LABELS[k], ha='right', va='center', fontsize=12.5)
        for x, value in enumerate(y):
            ax.annotate(f'{value:g}', (x, value), xytext=(5, -10),
                        textcoords='offset points', fontsize=12)
    ax.set_xlim(-1.15, 2.2)
    ax.set_ylim(len(configs) + .5, .35)
    ax.set_xticks([0, 1, 2], ['Original fits\nscored at Q=71',
                             'Original fits\nscored at Q=√2',
                             'Retrained at Q=√2\nscored at Q=√2'])
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', length=0, pad=13, labelsize=12)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in [0, 1, 2]:
        ax.axvline(x, color='0.85', linewidth=.7, zorder=0)
    fig.suptitle('Model ranking under a lower hierarchy depth weight', fontsize=15)
    fig.text(.025, .025, 'Matched greedy/ensemble configurations; lower rank is better.\n'
             'Five frozen folds; classical baselines excluded.', fontsize=12)
    fig.savefig(output / 'ranking.png', dpi=180)
    plt.close(fig)

    lines = ['# Low-Q retraining and ranking results', '',
             f"Completed experiment `{report['experiment_key']}` on the fixed 6,693-product benchmark.", '',
             f"New ensemble endpoints use {report['members_retrained']} members, both full and 75% feature masks.",
             'The original 243-member controls are rescored separately. The classical prediction tables',
             'use different populations and are excluded from this matched comparison.', '',
             f"Fixed-fit rescoring preserves {sum(ranks[0][k] == ranks[1][k] for k in configs)} of {len(configs)} ranks at sqrt(2), and "
             f"{sum(ranks[0][k] == q2_ranks[0][k] for k in configs)} of {len(configs)} at Q=2.",
             f"After retraining, the first {prefixes[0]} positions remain in order at sqrt(2), and the first {prefixes[1]} at Q=2.", '',
             '![Ranking comparison](ranking.png)', '',
             '## Q=sqrt(2)', '',
             '| Configuration | Original loss at Q=71 | Original fits scored at sqrt(2) | Retrained loss at sqrt(2) | Ranks: original / rescored / retrained |',
             '|---|---:|---:|---:|---:|']
    for k in configs:
        a, b = index['71', k], index['sqrt2', k]
        lines.append(f"| {LABELS[k]} | {a['mean_losses']['71']:.6f} | {a['mean_losses']['sqrt2']:.6f} | "
                     f"{b['mean_losses']['sqrt2']:.6f} ± {b['fold_sd']['sqrt2']:.6f} | "
                     + ' / '.join(f'{r[k]:g}' for r in ranks) + ' |')
    lines += ['', '## Q=2', '',
              '| Configuration | Original fits rescored | Retrained | Exact accuracy: original → retrained |',
              '|---|---:|---:|---:|']
    for k in configs:
        a, b = index['71', k], index['2', k]
        lines.append(f"| {LABELS[k]} | {a['mean_losses']['2']:.6f} | {b['mean_losses']['2']:.6f} ± "
                     f"{b['fold_sd']['2']:.6f} | {100*a['exact_accuracy']:.3f}% → {100*b['exact_accuracy']:.3f}% |")
    for q in ('sqrt2', '2'):
        if report['members_retrained'] != 243:
            break
        if q == 'sqrt2':
            lines += ['', '## The two 243-member endpoints', '']
        a, b = index[q, 'ensemble_0.75_243'], index[q, 'ensemble_1_243']
        differences = np.array([x['losses'][q] - y['losses'][q] for x, y in zip(a['folds'], b['folds'])])
        display_q = 'sqrt(2)' if q == 'sqrt2' else q
        lines.append(f"At Q={display_q}, the 75% minus 100% feature mean loss difference is {differences.mean():.6f}; "
                     f"the paired fold SD is {differences.std(ddof=1):.6f}. The 75% model has lower loss in "
                     f"{int((differences < 0).sum())} of five folds. These describe observed fold variation, not a confidence interval.")
        lines.append('')
    crossing = output / 'fixed-crossover.json'
    if crossing.exists():
        c = json.loads(crossing.read_text())
        lines += ['', '## Secondary frozen-model check', '',
                  f"The original 243-member full-feature and 75%-feature loss curves cross near Q={c['crossing_q']:.4f}.",
                  'This holds those original fits fixed and does not add another training setting.',
                  'See [crossing calculation](fixed-crossover.json).']
    lines += ['', '## Scope and reproducibility', '',
              'Each score is an equal-weight mean of the five stored fold means. Random-order seeds',
              'are averaged within fold. The ± values describe sample fold SD, not confidence intervals.',
              'Training and scoring weights are separate: the middle and right columns use the same',
              'evaluation metric, so their difference measures the effect of retraining and its matching',
              'consensus rule. This is exploratory reuse of existing folds, with no new independent test set.', '',
              f"Algorithm SHA-256: `{report['algorithm_sha256']}`.",
              f"Snapshot digest: `{report['snapshot_digest']}`.",
              'Private fitted coefficients and individual predictions remain in Shopify PostgreSQL.',
              'See [protocol](protocol.md) and [aggregate results](results.json).', '',
              f"Run `python -m padjective.paper_low_q_experiment --members {report['members_retrained']} --workers {report.get('workers', 4)} --output results.json`",
              'against Shopify PostgreSQL, then `python -m padjective.low_q_results results.json`.', '']
    (output / 'README.md').write_text('\n'.join(lines))
    summary = {key: [dict(configuration=k, rank=float(r[k])) for k in configs]
               for key, r in zip(('original', 'rescored_sqrt2', 'retrained_sqrt2', 'rescored_q2', 'retrained_q2'),
                                 ranks + q2_ranks)}
    (output / 'rankings.json').write_text(json.dumps(summary, indent=2) + '\n')
    for label, r in zip(('Original Q=71', 'Original rescored at sqrt2', 'Retrained at sqrt2'), ranks):
        print(label, [k for k in sorted(configs, key=lambda k: r[k])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    summarise(json.loads(args.results.read_text()), args.results.parent)
