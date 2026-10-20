"""Matched three-method summaries and figures for half-reference error training."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .paper_ensemble_scaling_batch import SIZES
from .residual_ensemble_experiment import TRAJECTORIES
from .residual_ensemble_report import summarise as two_method_summary

METHODS = ('half_reference', 'error_only', 'independent')


def summarise(report):
    assert report['validation']['status'] == 'passed'
    assert len(report['rows']) == 6075
    original = dict(report, rows=[r for r in report['rows'] if r['method'] != 'half_reference'])
    substituted = copy.deepcopy(dict(report,
        rows=[r for r in report['rows'] if r['method'] != 'error_only']))
    for row in substituted['rows']:
        if row['method'] == 'half_reference':
            row['method'] = 'error_only'
    old, new = two_method_summary(original), two_method_summary(substituted)
    summaries = old['summaries']
    for row in new['summaries']:
        if row['method'] == 'error_only':
            row['method'] = 'half_reference'
            summaries.append(row)
    lookup = {(r['roster'], r['method'], r['members']): r for r in summaries}
    comparisons = []
    for roster, limit in TRAJECTORIES.items():
        for size in range(1, limit+1):
            trial = lookup[(roster, 'half_reference', size)]
            for method in ('error_only', 'independent'):
                control = lookup[(roster, method, size)]
                differences = [a-b for a, b in zip(trial['held_out_fold_losses'],
                    control['held_out_fold_losses'], strict=True)]
                comparisons.append(dict(roster=roster, members=size, comparator=method,
                    loss_difference=statistics.fmean(differences), fold_loss_differences=differences,
                    folds_better=sum(d < 0 for d in differences),
                    exact_accuracy_difference=trial['held_out']['exact_accuracy']-control['held_out']['exact_accuracy']))
    return dict(batch_id=report['batch_id'], validation=report['validation'],
        summaries=sorted(summaries, key=lambda r: (r['roster'], r['method'], r['members'])),
        comparisons=comparisons)


def figures(report, summary, directory):
    blue, red, grey = '#1266a0', '#a65d4d', '#555555'
    labels = dict(half_reference='Random half-reference', error_only='All-previous errors',
                  independent='Independent full-data fits')
    colors = dict(half_reference=blue, error_only=red, independent=grey)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), layout='constrained')
    for method in METHODS:
        rows = [r for r in summary['summaries'] if r['roster'] == 0 and r['method'] == method]
        color = colors[method]
        for parity in ((0, 1) if method == 'error_only' else (None,)):
            part = [r for r in rows if parity is None or r['members'] % 2 == parity]
            label = labels[method] if parity != 1 else None
            style = '--' if method == 'error_only' else '-'
            for ax, metric, scale in ((axes[0, 0], 'mean_padic_loss', 1),
                                       (axes[0, 1], 'exact_accuracy', 100)):
                ax.plot([r['members'] for r in part], [scale*r['held_out'][metric] for r in part],
                        color=color, linestyle=style, label=label, linewidth=1.8)
        axes[1, 0].plot([r['members'] for r in rows],
            [100*(1-r['training']['exact_accuracy']) for r in rows], color=color,
            alpha=.65 if method == 'error_only' else 1, label=labels[method], linewidth=1.4)
        if method == 'half_reference':
            axes[1, 0].plot([r['members'] for r in rows], [100*r['training_fraction'] for r in rows],
                color=blue, linestyle=':', label='Rows used for new half-reference fit', linewidth=1.5)
    for fold in range(5):
        rows = sorted([r for r in report['rows'] if r['roster'] == 0 and
            r['method'] == 'half_reference' and r['fold'] == fold], key=lambda r: r['members'])
        axes[0, 0].plot([r['members'] for r in rows], [r['held_out']['mean_padic_loss'] for r in rows],
                        color=blue, alpha=.1, linewidth=.65)
    for roster, style in ((0, '-'), (9, '--'), (18, ':')):
        for method, color, short in (('half_reference', blue, 'Half-reference'),
                                     ('independent', grey, 'Independent')):
            rows = [r for r in summary['summaries'] if r['roster'] == roster and
                    r['method'] == method and r['members'] <= 81]
            axes[1, 1].plot([r['members'] for r in rows], [r['held_out']['mean_padic_loss'] for r in rows],
                color=color, linestyle=style, label=f'{short}, order {roster}', linewidth=1.6)
    for ax, title, ylabel in (
        (axes[0, 0], 'Held-out loss: matched primary trajectories', 'Mean p-adic loss (lower is better)'),
        (axes[0, 1], 'Whole taxonomy-path accuracy', 'Held-out exact accuracy (%)'),
        (axes[1, 0], 'Training errors and sampled-reference fit sizes', 'Training rows (%)'),
        (axes[1, 1], 'Starting-order and sampling sensitivity', 'Held-out mean p-adic loss'),
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xscale('log')
        ticks = [1, 3, 9, 27, 81]+([] if ax is axes[1, 1] else [243])
        ax.set_xticks(ticks, [str(n) for n in ticks])
        ax.set_xlabel('Ensemble members (log scale)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('Train on errors made by a fresh random half of previous members', fontsize=14)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(directory / f'ensemble-curves.{extension}', dpi=180)
    svg = directory / 'ensemble-curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def markdown(report, summary, directory):
    lookup = {(r['roster'], r['method'], r['members']): r for r in summary['summaries']}
    primary = [lookup[(0, 'half_reference', n)] for n in range(1, 244)]
    best = min(primary, key=lambda r: r['held_out']['mean_padic_loss'])
    final = primary[-1]
    independent = lookup[(0, 'independent', 243)]
    old = lookup[(0, 'error_only', 243)]
    lines = ['# Random half-reference ensemble results', '',
        'The paper is unchanged. This experiment tests a fresh random half of previous members '
        'as the reference for identifying each next member’s training errors.', '',
        f"At 243 members, the new method has mean held-out p-adic loss "
        f"**{final['held_out']['mean_padic_loss']:.6f}**, versus "
        f"{old['held_out']['mean_padic_loss']:.6f} for all-previous error training and "
        f"{independent['held_out']['mean_padic_loss']:.6f} for independent fits. Exact-path accuracy is "
        f"{final['held_out']['exact_accuracy']:.2%}, {old['held_out']['exact_accuracy']:.2%} and "
        f"{independent['held_out']['exact_accuracy']:.2%}, respectively.", '',
        f"Its best observed primary prefix is **{best['members']} members, loss "
        f"{best['held_out']['mean_padic_loss']:.6f}**. This was selected after scoring, so it is "
        'an exploratory observation, not a validated stopping rule.', '',
        '## Matched primary curves', '',
        'Numbers are equal-fold means over the same five folds. Loss is p-adic distance, not an '
        'error percentage. Smaller loss is better.', '',
        '| Members | Independent loss | All-previous error loss | Half-reference loss | Half-reference exact accuracy |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for n in SIZES:
        values = [lookup[(0, method, n)]['held_out']['mean_padic_loss'] for method in
                  ('independent', 'error_only', 'half_reference')]
        lines.append(f"| {n} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | "
                     f"{lookup[(0, 'half_reference', n)]['held_out']['exact_accuracy']:.2%} |")
    lines += ['', '![Measured curves](ensemble-curves.png)', '',
        '## Comparisons across all tested sizes', '']
    for method, label in (('independent', 'independent fits'), ('error_only', 'all-previous error training')):
        rows = [r for r in summary['comparisons'] if r['comparator'] == method and r['members'] > 1]
        count = sum(r['loss_difference'] < -1e-12 for r in rows)
        first = [r['members'] for r in rows if r['roster'] == 0 and r['loss_difference'] < -1e-12]
        lines += [f"The new method has lower mean loss than {label} in **{count}/{len(rows)}** "
            'multi-member configurations across the three fixed orders. '
            +(f"The first primary size with lower loss is {min(first)}." if first else
              'No primary size has lower loss.'), '']
    lines += ['These are correlated size/order configurations on the same data, not independent trials '
        'or a significance test.', '', '## Starting-order and sampling sensitivity', '',
        '| Order | Half-reference loss at 81 | Independent loss at 81 | All-previous error loss at 81 | Half-reference accuracy |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for roster in TRAJECTORIES:
        half, all_errors, independent = [lookup[(roster, method, 81)] for method in METHODS]
        lines.append(f"| {roster} | {half['held_out']['mean_padic_loss']:.6f} | "
            f"{independent['held_out']['mean_padic_loss']:.6f} | {all_errors['held_out']['mean_padic_loss']:.6f} | "
            f"{half['held_out']['exact_accuracy']:.2%} |")
    lines += ['', 'Each order uses its matched seed sequence. The subset-sampling stream is keyed by '
        'order and fold, so these checks vary both starting order and reference samples. Beyond 81 '
        'only the primary order was extended.', '', '## Late-size behaviour', '',
        '| Members | Mean loss | Root accuracy | Exact training errors | Rows used for this member | Stored coefficients | Active terms per prediction |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in (27, 45, 81, 135, 243):
        r = lookup[(0, 'half_reference', n)]
        lines.append(f"| {n} | {r['held_out']['mean_padic_loss']:.6f} | "
            f"{r['held_out']['first_digit_accuracy']:.2%} | {1-r['training']['exact_accuracy']:.2%} | "
            f"{r['training_fraction']:.2%} | {r['stored_nonzero_coefficients']:.1f} | {r['mean_active_terms']:.2f} |")
    for method, label in (('half_reference', 'new half-reference'), ('error_only', 'all-previous')):
        means = [statistics.fmean(lookup[(0, method, n)]['held_out']['mean_padic_loss']
                   for n in range(135, 244) if n % 2 == parity) for parity in (0, 1)]
        lines += ['', f"Across sizes 135–243, the {label} mean over even sizes is {means[0]:.6f}; "
            f"the mean over odd sizes is {means[1]:.6f}; the absolute difference is {abs(means[1]-means[0]):.6f}."]
    lines += ['', 'This parity comparison describes the measured curves. It was not used to tune '
        'the sampling rate, select members or stop fitting. Finite curves cannot establish an '
        'asymptotic floor. Active terms exclude defaults and consensus search; member counts '
        'do not represent a fixed storage or computation budget.', '', '## Algorithm and verification', '',
        'Before member t, sample ceil((t−1)/2) earlier members uniformly without replacement. '
        'Use the same fresh subset for every training row at that step, decode its equal-weight '
        'p-adic consensus, and fit only whole-path errors. Defaults use those selected rows only. '
        'All fitted members still vote on all products at prediction time. The original training-fold '
        'candidate codebook, coordinate fitting and inference rule are unchanged. Held-out labels '
        'never choose masks, random samples, seeds or stopping points.', '',
        f"The new batch is `{report['batch_id']}`, source `{report['manifest']['source_commit']}`. "
        f"Controls come from validated batch `{report['manifest']['control_batch']}`. The snapshot "
        f"digest is `{report['manifest']['snapshot_digest']}` (6,693 products, 2,542 binary features, 363 paths).", '',
        f"All 15 trajectories passed replay and independent fit audits, including "
        f"{report['validation']['certified_coordinates']:,} coordinate checks and "
        f"{report['validation']['reference_checkpoint_checks']} reference-vote checkpoints against the existing decoder. "
        'Samples, masks, integer predictions and defaults were reconstructed. Full votes were '
        'also checked against the existing decoder and scores against two integer implementations. '
        'The first two prefix predictions reproduce the previous error-only controls exactly. '
        'All 4,050 old control rows matched their saved database evidence.', '',
        'Private samples, row indices, fitted coefficients and product predictions remain in Shopify '
        'Postgres under the new batch in `padjective.residual_ensemble_*`; tables and indexes use '
        '`pg_default`. Exported files contain aggregate scores, fingerprints and provenance only. '
        'The five folds have been reused for earlier experiments, so this is exploratory evidence '
        'on the established dataset rather than independent confirmation.', '',
        '[Protocol](../half-reference-ensemble-protocol.md) · [Aggregate evidence](results.json) · '
        '[Summaries and paired differences](summary.json)', '',
        'Rebuild this report:', '', '```sh', 'uv run -m padjective.half_reference_report \\',
        '  --results docs/half-reference-ensemble/results.json \\',
        '  --output-dir docs/half-reference-ensemble', '```', '']
    (directory / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    summary = summarise(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    figures(report, summary, args.output_dir)
    markdown(report, summary, args.output_dir)
    print(json.dumps(dict(batch_id=report['batch_id'], directory=str(args.output_dir))))


if __name__ == '__main__':
    main()
