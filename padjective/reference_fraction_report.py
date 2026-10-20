"""Aggregate analysis and static research figures for reference fraction sweeps."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .residual_ensemble_experiment import TRAJECTORIES

RATE_METHODS = {10: 'reference_10', 25: 'reference_25', 50: 'half_reference',
                75: 'reference_75', 90: 'reference_90', 100: 'error_only'}
METHOD_RATE = {method: p for p, method in RATE_METHODS.items()}
COLORS = {10: '#0072b2', 25: '#009e73', 50: '#e69f00', 75: '#cc79a7',
          90: '#d55e00', 100: '#777777'}
METHODS = tuple(RATE_METHODS.values()) + ('independent',)


def reference_count(row):
    if row['method'] == 'independent':
        return None
    if 'reference_members_count' in row:
        return row['reference_members_count']
    fitted = row.get('fitted_members', row['members'])
    previous = fitted if fitted < row['members'] else max(0, fitted-1)
    return (METHOD_RATE[row['method']]*previous+99)//100


def summarise(report):
    assert report['validation']['status'] == 'passed' and len(report['rows']) == 14175
    groups = defaultdict(list)
    for row in report['rows']:
        assert row['method'] in METHODS
        groups[(row['roster'], row['method'], row['members'])].append(row)
    assert set(groups) == {(roster, method, n) for roster, limit in TRAJECTORIES.items()
                           for method in METHODS for n in range(1, limit+1)}
    output = []
    for (roster, method, size), rows in sorted(groups.items()):
        assert len(rows) == 5 and {r['fold'] for r in rows} == set(range(5))
        item = dict(roster=roster, method=method, percent=METHOD_RATE.get(method), members=size,
            training_fraction=statistics.fmean(r['training_rows']/r['training']['n'] for r in rows),
            forgotten_correct_rows=statistics.fmean(r['forgotten_correct_rows'] for r in rows),
            stored_nonzero_coefficients=statistics.fmean(r['stored_nonzero_coefficients'] for r in rows),
            mean_active_terms=statistics.fmean(r['mean_active_terms'] for r in rows),
            reference_members_count=None if method == 'independent' else
                statistics.fmean(reference_count(r) for r in rows))
        for part in ('training', 'held_out'):
            item[part] = {key: statistics.fmean(r[part][key] for r in rows)
                          for key in ('mean_padic_loss', 'exact_accuracy', 'first_digit_accuracy')}
        item['fold_losses'] = [r['held_out']['mean_padic_loss'] for r in sorted(rows, key=lambda r: r['fold'])]
        output.append(item)
    lookup = {(r['roster'], r['method'], r['members']): r for r in output}
    comparisons = []
    for percent, method in RATE_METHODS.items():
        for roster, limit in TRAJECTORIES.items():
            for size in range(1, limit+1):
                actual = lookup[(roster, method, size)]
                for comparator in ('independent', 'half_reference'):
                    control = lookup[(roster, comparator, size)]
                    delta = [a-b for a, b in zip(actual['fold_losses'], control['fold_losses'], strict=True)]
                    comparisons.append(dict(percent=percent, roster=roster, members=size,
                        comparator=comparator, loss_difference=statistics.fmean(delta),
                        fold_differences=delta, folds_better=sum(d < 0 for d in delta),
                        exact_accuracy_difference=actual['held_out']['exact_accuracy']-control['held_out']['exact_accuracy']))
    windows = []
    for method in METHODS:
        for lower, upper in ((27, 81), (81, 135), (135, 243)):
            rows = [lookup[(0, method, n)] for n in range(lower, upper+1)]
            windows.append(dict(method=method, percent=METHOD_RATE.get(method), lower=lower, upper=upper,
                mean_loss=statistics.fmean(r['held_out']['mean_padic_loss'] for r in rows),
                mean_exact_accuracy=statistics.fmean(r['held_out']['exact_accuracy'] for r in rows)))
    parity = []
    for percent, method in RATE_METHODS.items():
        for kind in ('ensemble', 'reference'):
            means, counts = [], []
            for remainder in (0, 1):
                fold_means, fold_counts = [], []
                for fold in range(5):
                    rows = [r for r in report['rows'] if r['roster'] == 0 and r['fold'] == fold
                        and r['method'] == method and 135 <= r['members'] <= 243
                        and (r['members'] if kind == 'ensemble' else reference_count(r)) % 2 == remainder]
                    assert rows
                    fold_means.append(statistics.fmean(r['held_out']['mean_padic_loss'] for r in rows))
                    fold_counts.append(len(rows))
                means.append(statistics.fmean(fold_means))
                counts.append(fold_counts)
            parity.append(dict(percent=percent, kind=kind, even_mean=means[0], odd_mean=means[1],
                               odd_minus_even=means[1]-means[0], prefix_counts_per_fold=counts))
    return dict(sweep_id=report['sweep_id'], validation=report['validation'], summaries=output,
                comparisons=comparisons, windows=windows, parity=parity)


def save_figure(fig, directory, name):
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(directory/f'{name}.{extension}', dpi=180)
    svg = directory/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def configure_axis(ax, *, max_size=243, title='', ylabel='Mean p-adic loss'):
    ax.set_title(title, fontsize=11)
    ax.set_xscale('log')
    ticks = [1, 3, 9, 27, 81]+([243] if max_size > 81 else [])
    ax.set_xticks(ticks, [str(n) for n in ticks])
    ax.set_xlabel('Ensemble members (log scale)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.17)


def figures(report, summary, directory):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    lookup = {(r['roster'], r['method'], r['members']): r for r in summary['summaries']}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), layout='constrained')
    for percent, method in RATE_METHODS.items():
        rows = [lookup[(0, method, n)] for n in range(1, 244)]
        style = '--' if percent == 100 else '-'
        for ax, metric, scale in ((axes[0, 0], 'mean_padic_loss', 1), (axes[0, 1], 'exact_accuracy', 100)):
            ax.plot(range(1, 244), [scale*r['held_out'][metric] for r in rows],
                    color=COLORS[percent], linestyle=style, label=f'{percent}%', linewidth=1.4)
    for ax, metric, scale in ((axes[0, 0], 'mean_padic_loss', 1), (axes[0, 1], 'exact_accuracy', 100)):
        ax.plot(range(1, 244), [scale*lookup[(0, 'independent', n)]['held_out'][metric] for n in range(1, 244)],
                color='#111111', linewidth=2.1, label='Independent')
        ax.legend(ncol=2, fontsize=8)
    configure_axis(axes[0, 0], title='Held-out loss by reference fraction')
    configure_axis(axes[0, 1], title='Whole-path accuracy', ylabel='Exact accuracy (%)')
    for lower, upper, marker, color in ((81, 135, 'o', '#1266a0'), (135, 243, 's', '#a35f24')):
        rows = [r for r in summary['windows'] if r['lower'] == lower and r['upper'] == upper]
        window = {r['percent']: r['mean_loss'] for r in rows}
        axes[1, 0].plot(list(RATE_METHODS), [window[p] for p in RATE_METHODS], marker=marker,
                        color=color, label=f'Mean over sizes {lower}–{upper}')
        axes[1, 0].axhline(window[None], color=color, linestyle='--', alpha=.65,
                           label=f'Independent, sizes {lower}–{upper}')
    axes[1, 0].set(title='Late-size averages across fractions', xlabel='Previous members sampled (%)',
                   ylabel='Mean p-adic loss')
    axes[1, 0].set_xticks(list(RATE_METHODS))
    axes[1, 0].grid(alpha=.17)
    axes[1, 0].legend(fontsize=8)
    matrix = np.array([[lookup[(0, method, n)]['held_out']['mean_padic_loss'] for n in range(1, 244)]
                       for method in METHODS])
    heat = axes[1, 1].imshow(matrix, aspect='auto', interpolation='nearest', cmap='viridis_r')
    axes[1, 1].set_yticks(range(7), [f'{p}%' for p in RATE_METHODS]+['Independent'])
    ticks = [1, 27, 81, 135, 243]
    axes[1, 1].set_xticks([n-1 for n in ticks], [str(n) for n in ticks])
    axes[1, 1].set(title='Every measured primary prefix', xlabel='Members (linear scale)')
    fig.colorbar(heat, ax=axes[1, 1], label='Mean p-adic loss', shrink=.85)
    fig.suptitle('Which fraction of previous members should identify the next training errors?', fontsize=14)
    save_figure(fig, directory, 'fraction-patterns')

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), layout='constrained')
    for ax, roster in zip(axes.flat, TRAJECTORIES, strict=False):
        for percent, method in RATE_METHODS.items():
            ax.plot(range(1, 82), [lookup[(roster, method, n)]['held_out']['mean_padic_loss'] for n in range(1, 82)],
                    color=COLORS[percent], label=f'{percent}%', linewidth=1.2)
        ax.plot(range(1, 82), [lookup[(roster, 'independent', n)]['held_out']['mean_padic_loss'] for n in range(1, 82)],
                color='#111111', linewidth=2, label='Independent')
        configure_axis(ax, max_size=81, title=f'Starting order {roster}, fixed sampling stream')
        ax.legend(ncol=2, fontsize=8)
    ax = axes[1, 1]
    for kind, marker, label in (('ensemble', 'o', 'Full ensemble size'),
                               ('reference', 's', 'Actual sampled committee size')):
        values = {r['percent']: r['odd_minus_even'] for r in summary['parity'] if r['kind'] == kind}
        ax.plot(list(RATE_METHODS), [values[p] for p in RATE_METHODS], marker=marker, label=label)
    ax.axhline(0, color='#777777', linewidth=.7)
    ax.set(title='Odd minus even loss, primary sizes 135–243', xlabel='Previous members sampled (%)',
           ylabel='Conditional mean loss difference')
    ax.set_xticks(list(RATE_METHODS))
    ax.grid(alpha=.17)
    ax.legend(fontsize=8)
    fig.suptitle('Starting-order sensitivity and committee-size parity', fontsize=14)
    save_figure(fig, directory, 'fraction-sensitivity')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), layout='constrained')
    for ax, (percent, method) in zip(axes.flat, RATE_METHODS.items(), strict=True):
        for fold in range(5):
            for model, color in ((method, COLORS[percent]), ('independent', '#555555')):
                rows = sorted([r for r in report['rows'] if r['roster'] == 0 and r['method'] == model
                               and r['fold'] == fold], key=lambda r: r['members'])
                ax.plot(range(1, 244), [r['held_out']['mean_padic_loss'] for r in rows],
                        color=color, alpha=.13, linewidth=.65)
        for model, color, label in ((method, COLORS[percent], f'{percent}% mean'),
                                    ('independent', '#111111', 'Independent mean')):
            ax.plot(range(1, 244), [lookup[(0, model, n)]['held_out']['mean_padic_loss'] for n in range(1, 244)],
                    color=color, linewidth=1.5, label=label)
        configure_axis(ax, title=f'{percent}% of previous members')
        ax.legend(fontsize=8)
    fig.suptitle('Five fold curves and equal-fold means, primary order', fontsize=14)
    save_figure(fig, directory, 'fraction-folds')


def markdown(report, summary, directory):
    lookup = {(r['roster'], r['method'], r['members']): r for r in summary['summaries']}
    windows = {(r['method'], r['lower'], r['upper']): r for r in summary['windows']}
    parity = {(r['percent'], r['kind']): r for r in summary['parity']}
    baseline = lookup[(0, 'independent', 243)]
    lines = ['# Reference committee fraction sweep', '',
        'The paper and earlier reports are unchanged. New 10%, 25%, 75% and 90% '
        'experiments are compared with the validated 50%, 100% and independent controls. '
        'All values use the same 6,693 products and five frozen folds.', '',
        '## Fixed-budget and late-size comparison', '',
        'Lower mean p-adic loss is better. Loss is hierarchical distance, not an error percentage. '
        'Each prefix score is an equal-fold mean. Window means average these prefix scores and '
        'describe the curve; they are not independent repetitions.', '',
        '| Fraction | Loss at 81 | Loss at 243 | Exact accuracy at 243 | Mean loss, sizes 81–135 | Mean loss, sizes 135–243 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for percent, method in RATE_METHODS.items():
        at81, final = lookup[(0, method, 81)], lookup[(0, method, 243)]
        lines.append(f"| {percent}% | {at81['held_out']['mean_padic_loss']:.6f} | "
            f"{final['held_out']['mean_padic_loss']:.6f} | {final['held_out']['exact_accuracy']:.2%} | "
            f"{windows[(method, 81, 135)]['mean_loss']:.6f} | {windows[(method, 135, 243)]['mean_loss']:.6f} |")
    lines.append(f"| Independent | {lookup[(0, 'independent', 81)]['held_out']['mean_padic_loss']:.6f} | "
        f"{baseline['held_out']['mean_padic_loss']:.6f} | {baseline['held_out']['exact_accuracy']:.2%} | "
        f"{windows[('independent', 81, 135)]['mean_loss']:.6f} | {windows[('independent', 135, 243)]['mean_loss']:.6f} |")
    lines += ['', '![Fraction curves and late-size patterns](fraction-patterns.png)', '',
        '## Loss across ensemble sizes', '',
        '| Fraction | 9 members | 27 members | 81 members | 135 members | 243 members |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for percent, method in list(RATE_METHODS.items())+[(None, 'independent')]:
        label = f'{percent}%' if percent is not None else 'Independent'
        values = ' | '.join(f"{lookup[(0, method, n)]['held_out']['mean_padic_loss']:.6f}"
                            for n in (9, 27, 81, 135, 243))
        lines.append(f'| {label} | {values} |')
    lines += ['', 'The 100% curve has a strong odd/even oscillation. All-prefix figures and '
        'window averages retain both branches; the 243-member endpoint is an odd member count. '
        'A single endpoint can therefore misrepresent the whole curve.', '',
        '## Matched comparisons with independent fits', '',
        '| Fraction | Lower-loss configurations / 402 | Primary loss difference at 243 | Folds with lower loss at 243 | Exact accuracy difference at 243 (points) |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for percent in RATE_METHODS:
        comparisons = [r for r in summary['comparisons'] if r['percent'] == percent and
                       r['comparator'] == 'independent' and r['members'] > 1]
        wins = sum(r['loss_difference'] < -1e-12 for r in comparisons)
        final = next(r for r in comparisons if r['roster'] == 0 and r['members'] == 243)
        lines.append(f"| {percent}% | {wins}/{len(comparisons)} | {final['loss_difference']:+.6f} | "
            f"{final['folds_better']}/5 | {100*final['exact_accuracy_difference']:+.2f} |")
    lines += ['', 'These size/order configurations reuse the same observations and are correlated. '
        'Counts are descriptive comparisons, not binomial trials or significance tests. '
        'Per-fold paired differences against independent and 50% fits are retained in summary.json.', '',
        '## Starting-order and sampling sensitivity at 81 members', '',
        '| Fraction | Primary order 0 | Order 9 | Order 18 | Orders below matched independent loss |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for percent, method in list(RATE_METHODS.items())+[(None, 'independent')]:
        values = [lookup[(roster, method, 81)]['held_out']['mean_padic_loss'] for roster in TRAJECTORIES]
        wins = sum(value < lookup[(roster, 'independent', 81)]['held_out']['mean_padic_loss']-1e-12
                   for roster, value in zip(TRAJECTORIES, values, strict=True))
        label = f'{percent}%' if percent is not None else 'Independent'
        lines.append(f"| {label} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | "
                     +(f'{wins}/3 |' if percent is not None else '— |'))
    lines += ['', 'The three orders vary both the initial member order and its sampling stream. '
        'They are sensitivity checks on the same five folds. Only the primary order was extended '
        'beyond 81; no unbalanced mean across orders is used at larger sizes.', '',
        '![Order sensitivity and parity](fraction-sensitivity.png)', '',
        '## Ensemble parity and actual committee parity', '',
        'The table conditions on sizes 135–243 of the primary ensemble. Positive odd-minus-even '
        'differences mean odd counts had higher loss. Each conditional mean is first computed '
        'within fold, then averaged over folds. The raw counts per fold are retained in summary.json.', '',
        '| Fraction | Full ensemble odd-minus-even loss | Sampled committee odd-minus-even loss | Committee members used to fit member 243 |',
        '| ---: | ---: | ---: | ---: |']
    for percent, method in RATE_METHODS.items():
        lines.append(f"| {percent}% | {parity[(percent, 'ensemble')]['odd_minus_even']:+.6f} | "
            f"{parity[(percent, 'reference')]['odd_minus_even']:+.6f} | "
            f"{lookup[(0, method, 243)]['reference_members_count']:.0f} |")
    lines += ['', 'Fraction, rounded committee size, parity and ensemble age change together. '
        'These groupings reveal associations in the measured sequences; they do not isolate '
        'a causal parity effect. The same label-free RNG key is used at every fraction; '
        'sample counts change, and sampled sets need not be nested.', '',
        '## Observed primary minima', '',
        'Every prefix was scored. The following minima are selected after looking at the same '
        'held-out folds and can be optimistic. They are not validated rate choices or stopping '
        'rules. Compare them with the fixed-budget and window tables above.', '',
        '| Fraction | Members at observed minimum | Loss | Matched independent loss | Exact accuracy |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for percent, method in list(RATE_METHODS.items())+[(None, 'independent')]:
        best = min([lookup[(0, method, n)] for n in range(1, 244)], key=lambda r: r['held_out']['mean_padic_loss'])
        control = lookup[(0, 'independent', best['members'])]
        label = f'{percent}%' if percent is not None else 'Independent'
        lines.append(f"| {label} | {best['members']} | {best['held_out']['mean_padic_loss']:.6f} | "
            f"{control['held_out']['mean_padic_loss']:.6f} | {best['held_out']['exact_accuracy']:.2%} |")
    lines += ['', '## Training mechanism and storage at 243', '',
        '| Fraction | Reference errors used for fit | Full ensemble exact training errors | Correct rows overturned by last member | Stored coefficients | Active terms per held-out prediction |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for percent, method in RATE_METHODS.items():
        r = lookup[(0, method, 243)]
        lines.append(f"| {percent}% | {r['training_fraction']:.2%} | {1-r['training']['exact_accuracy']:.2%} | "
            f"{r['forgotten_correct_rows']:.1f} | {r['stored_nonzero_coefficients']:.1f} | {r['mean_active_terms']:.2f} |")
    lines += ['', 'Stored coefficients and active consultations are different counters. '
        'Active terms exclude defaults, consensus search and the training-time reference-vote work. '
        'A member is not a fixed unit of storage, latency or training cost.', '',
        '![Individual fold curves](fraction-folds.png)', '', '## Protocol, validation and scope', '',
        'For each new member, sample ceil(percent × previous_count / 100) previous models '
        'uniformly without replacement. One fresh committee is shared by all training rows at '
        'that step. Fit only its whole-path errors, starting coefficients at zero and fitting '
        'the default on selected rows alone. Keep all original feature columns and the original '
        'training-fold candidate paths. All fitted members vote on every product at inference. '
        'Held-out labels never determine samples, masks, seeds, weights or stopping.', '',
        'The 50% sampling stream is unchanged byte for byte. Fractions use the same seed keys '
        'and coordinate fit seeds. The first member trains normally. Each member must complete '
        'a no-improvement sweep under the p=71, precision-7 integer objective; limits are 100 '
        'sweeps and 300 seconds. New fits finish before any new held-out scoring.', '',
        f"Sweep `{report['sweep_id']}`; fitting source `{report['manifest']['source_commit']}`; "
        f"snapshot digest `{report['manifest']['snapshot_digest']}`. The snapshot contains "
        '6,693 products, 2,542 binary features and 363 paths, drawn directly from Shopify Postgres.', '',
        f"All {report['validation']['trajectories']} new trajectories passed, including "
        f"{report['validation']['certified_coordinates']:,} supported-coordinate checks on actual training subsets, "
        f"{report['validation']['reference_checkpoint_checks']} independent reference-decoder checkpoints and "
        f"{report['validation']['control_rows_verified']:,} unchanged control rows. All samples/masks were replayed; "
        'integer predictions and defaults reconstructed; full votes checked against the existing '
        'prefix decoder; and scores matched under two integer implementations. Initial prefixes '
        'matched the 100% control through sizes 2, 2, 4 and 10 for 10%, 25%, 75% and 90%.', '',
        'The original 50% and 100% evidence was reused from validated batches. New private '
        'coefficients, samples, selected row indices, predictions and histories remain in the '
        '`padjective.residual_ensemble_*` child batches. The master manifest/report is in '
        '`padjective.reference_fraction_sweeps`. All tables and indexes use `pg_default`. '
        'Only aggregate evidence and fingerprints are exported.', '',
        'The six fractions and many nested prefixes reuse the established folds. This is an '
        'exploratory extension, not new independent test data. Finite late-size behaviour cannot '
        'prove an asymptotic floor. No manuscript files were changed.', '',
        '[Protocol](../reference-fraction-sweep-protocol.md) · [Full aggregate evidence](results.json) · '
        '[Summaries, paired differences and parity groups](summary.json)', '',
        'Child batches:', '', '| Fraction | Batch |', '| ---: | --- |']
    for percent, batch in sorted(report['batches'].items(), key=lambda item: int(item[0])):
        lines.append(f'| {percent}% | `{batch}` |')
    lines += ['', 'Rebuild the report and figures:', '', '```sh',
        'uv run -m padjective.reference_fraction_report \\',
        '  --results docs/reference-fraction-sweep/results.json \\',
        '  --output-dir docs/reference-fraction-sweep', '```', '']
    (directory/'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    summary = summarise(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    figures(report, summary, args.output_dir)
    markdown(report, summary, args.output_dir)
    print(json.dumps(dict(sweep_id=report['sweep_id'], directory=str(args.output_dir))))


if __name__ == '__main__':
    main()
