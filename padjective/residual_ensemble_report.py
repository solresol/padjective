"""Aggregate-only report and research figures for the error-only ensemble study."""
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

from .paper_ensemble_scaling_batch import SIZES
from .residual_ensemble_experiment import TRAJECTORIES


def summarise(report):
    assert report['validation']['status'] == 'passed'
    groups = defaultdict(list)
    for row in report['rows']:
        groups[(row['roster'], row['method'], row['members'])].append(row)
    assert len(groups) == 2*sum(TRAJECTORIES.values())
    output = []
    for (roster, method, members), rows in sorted(groups.items()):
        assert len(rows) == 5 and {r['fold'] for r in rows} == set(range(5))
        mean = lambda values: statistics.fmean(values)
        item = dict(roster=roster, method=method, members=members,
            training_rows=mean(r['training_rows'] for r in rows),
            training_fraction=mean(r['training_rows']/r['training']['n'] for r in rows),
            forgotten_correct_rows=mean(r['forgotten_correct_rows'] for r in rows),
            stored_nonzero_coefficients=mean(r['stored_nonzero_coefficients'] for r in rows),
            mean_active_terms=mean(r['mean_active_terms'] for r in rows))
        for part in ('training', 'held_out'):
            item[part] = {key: mean(r[part][key] for r in rows)
                          for key in ('mean_padic_loss', 'exact_accuracy', 'first_digit_accuracy')}
        item['held_out_fold_losses'] = [r['held_out']['mean_padic_loss'] for r in sorted(rows, key=lambda r: r['fold'])]
        output.append(item)
    lookup = {(r['roster'], r['method'], r['members']): r for r in output}
    comparisons = []
    for roster, limit in TRAJECTORIES.items():
        for size in range(1, limit+1):
            adaptive = lookup[(roster, 'error_only', size)]
            independent = lookup[(roster, 'independent', size)]
            differences = [a-b for a, b in zip(adaptive['held_out_fold_losses'],
                                             independent['held_out_fold_losses'], strict=True)]
            comparisons.append(dict(roster=roster, members=size,
                loss_difference=adaptive['held_out']['mean_padic_loss']-independent['held_out']['mean_padic_loss'],
                fold_loss_differences=differences, folds_better=sum(d < 0 for d in differences),
                exact_accuracy_difference=adaptive['held_out']['exact_accuracy']-independent['held_out']['exact_accuracy'],
                root_accuracy_difference=adaptive['held_out']['first_digit_accuracy']-independent['held_out']['first_digit_accuracy']))
    return dict(batch_id=report['batch_id'], validation=report['validation'],
                summaries=output, paired_comparisons=comparisons)


def make_figures(report, summary, directory):
    blue, grey = '#1266a0', '#666666'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), layout='constrained')
    for method, color, label in (('error_only', blue, 'Train on current errors'),
                                  ('independent', grey, 'Independent full-data fits')):
        rows = [r for r in summary['summaries'] if r['roster'] == 0 and r['method'] == method]
        sizes = [r['members'] for r in rows]
        for fold in range(5):
            fold_rows = sorted([r for r in report['rows'] if r['roster'] == 0 and
                                r['method'] == method and r['fold'] == fold], key=lambda r: r['members'])
            for parity in ((0, 1) if method == 'error_only' else (None,)):
                part = [r for r in fold_rows if parity is None or r['members'] % 2 == parity]
                axes[0, 0].plot([r['members'] for r in part],
                    [r['held_out']['mean_padic_loss'] for r in part], color=color, alpha=.13, linewidth=.7)
        for parity in ((0, 1) if method == 'error_only' else (None,)):
            part = [r for r in rows if parity is None or r['members'] % 2 == parity]
            part_label = label if parity is None else f"Error-only, {'even' if parity == 0 else 'odd'} sizes"
            style = '--' if parity == 1 else '-'
            axes[0, 0].plot([r['members'] for r in part],
                [r['held_out']['mean_padic_loss'] for r in part], color=color,
                label=part_label, linestyle=style, linewidth=2)
            axes[0, 1].plot([r['members'] for r in part],
                [100*r['held_out']['exact_accuracy'] for r in part], color=color,
                label=part_label, linestyle=style, linewidth=2)
        axes[1, 0].plot(sizes, [100*(1-r['training']['exact_accuracy']) for r in rows],
                        color=color, label=label+': wrong after vote', linewidth=1.6)
        if method == 'error_only':
            axes[1, 0].plot(sizes, [100*r['training_fraction'] for r in rows], color=blue,
                            linestyle=':', label='Rows used to train this member', linewidth=1.5)
    for roster, style in ((0, '-'), (9, '--'), (18, ':')):
        for method, color, short in (('error_only', blue, 'Errors'), ('independent', grey, 'Independent')):
            rows = [r for r in summary['summaries'] if r['roster'] == roster and
                    r['method'] == method and r['members'] <= 81]
            for parity in ((0, 1) if method == 'error_only' else (None,)):
                part = [r for r in rows if parity is None or r['members'] % 2 == parity]
                axes[1, 1].plot([r['members'] for r in part], [r['held_out']['mean_padic_loss'] for r in part],
                    color=color, linestyle=style, label=f'{short}, start {roster}' if parity != 1 else None,
                    linewidth=1.6)
    for ax, title, ylabel in (
        (axes[0, 0], 'Held-out loss: five-fold mean and fold curves', 'Mean p-adic loss (lower is better)'),
        (axes[0, 1], 'Exact taxonomy-path accuracy', 'Held-out exact accuracy (%)'),
        (axes[1, 0], 'Training errors and the next fit', 'Training rows (%)'),
        (axes[1, 1], 'Three starting orders (even / odd branches shown)', 'Held-out mean p-adic loss'),
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xscale('log')
        ticks = [1, 3, 9, 27, 81] + ([] if ax is axes[1, 1] else [243])
        ax.set_xticks(ticks, [str(n) for n in ticks])
        ax.set_xlabel('Ensemble members (log scale)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('Training each new member only on the preceding ensemble’s mistakes', fontsize=14)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(directory / f'ensemble-curves.{extension}', dpi=180)
    svg = directory / 'ensemble-curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def write_markdown(report, summary, directory):
    lookup = {(r['roster'], r['method'], r['members']): r for r in summary['summaries']}
    paired = {(r['roster'], r['members']): r for r in summary['paired_comparisons']}
    primary = [lookup[(0, 'error_only', n)] for n in range(1, 244)]
    best = min(primary, key=lambda r: r['held_out']['mean_padic_loss'])
    baseline_best = min([lookup[(0, 'independent', n)] for n in range(1, 244)],
                        key=lambda r: r['held_out']['mean_padic_loss'])
    final, independent_final = lookup[(0, 'error_only', 243)], lookup[(0, 'independent', 243)]
    bscore, fscore = best['held_out'], final['held_out']
    compared = [r for r in summary['paired_comparisons'] if r['members'] > 1]
    wins = sum(r['loss_difference'] < -1e-12 for r in compared)
    lines = [
        '# Error-only training ensemble results', '',
        'The paper has not been changed. This is an exploratory comparison on the existing five folds.', '',
        f"**Error-only training beats the matched independent ensemble in {wins} of {len(compared)} "
        'tested multi-member configurations.** The one-member controls are identical. '
        'Each configuration is a starting order and member count, scored over the same five folds; '
        'the configurations are correlated, not independent statistical trials.', '',
        f"The primary error-only run has its lowest measured mean p-adic loss at **{best['members']} members: "
        f"{bscore['mean_padic_loss']:.6f}**, with {bscore['exact_accuracy']:.2%} exact-path accuracy. "
        f"At 243 members its loss is {fscore['mean_padic_loss']:.6f}, compared with "
        f"{independent_final['held_out']['mean_padic_loss']:.6f} for independent fits at the same size.", '',
        f"The best independent prefix among the 243 measured sizes is {baseline_best['members']} members "
        f"(loss {baseline_best['held_out']['mean_padic_loss']:.6f}). These best sizes were observed after "
        'scoring and are not validated stopping rules.', '',
        '## Matched primary results', '',
        'Loss is mean p-adic distance, not an error percentage. Accuracy is whole-path exact match. '
        'Each number is an equal-weight mean across five folds.', '',
        '| Members | Independent loss | Error-only loss | Difference | Independent accuracy | Error-only accuracy | Folds with lower loss |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for size in SIZES:
        adaptive, base = lookup[(0, 'error_only', size)], lookup[(0, 'independent', size)]
        delta = paired[(0, size)]
        lines.append(f"| {size} | {base['held_out']['mean_padic_loss']:.6f} | "
            f"{adaptive['held_out']['mean_padic_loss']:.6f} | {delta['loss_difference']:+.6f} | "
            f"{base['held_out']['exact_accuracy']:.2%} | {adaptive['held_out']['exact_accuracy']:.2%} | "
            f"{delta['folds_better']}/5 |")
    lines += ['', '![Measured ensemble curves](ensemble-curves.png)', '',
        '## What was trained', '',
        'The first member used all training rows. Each subsequent member fitted only rows whose '
        'entire path was wrong under the current ensemble consensus. The mask was recomputed '
        'before every fit; rows could re-enter. Defaults used the selected rows only. All fitted '
        'members voted on every held-out product, using the original training-fold candidate paths '
        'and the established equal-weight p-adic consensus rule. Held-out answers never chose '
        'training rows, seeds, weights or stopping points.', '',
        'The models used p=71, precision 7, zero initial coefficients, random coordinate orders, '
        'strict improvements, and a completed no-improvement sweep. The primary run went through '
        '243 members. Two additional starting orders went through 81. Their matched controls '
        'used the same seeds with independent full-data fits from the validated bank.', '',
        '## Starting-order sensitivity', '',
        '| Starting bank index | Error-only loss at 81 | Independent loss at 81 | Difference | Error-only exact accuracy | Best error-only prefix through 81 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for roster in TRAJECTORIES:
        adaptive, base = lookup[(roster, 'error_only', 81)], lookup[(roster, 'independent', 81)]
        best81 = min([lookup[(roster, 'error_only', n)] for n in range(1, 82)],
                     key=lambda r: r['held_out']['mean_padic_loss'])
        lines.append(f"| {roster} | {adaptive['held_out']['mean_padic_loss']:.6f} | "
            f"{base['held_out']['mean_padic_loss']:.6f} | {paired[(roster, 81)]['loss_difference']:+.6f} | "
            f"{adaptive['held_out']['exact_accuracy']:.2%} | {best81['members']} ({best81['held_out']['mean_padic_loss']:.6f}) |")
    lines += ['', 'These are different starting orders on the same folds and seed bank. They are '
        'sensitivity checks, not independent datasets. Beyond 81, only the predeclared primary '
        'order was extended.', '', '## Late-size behaviour and training mechanism', '',
        '| Members | Error-only loss | Root accuracy | Exact training errors | Rows used for this member | Previously correct training rows made wrong | Stored coefficients | Active terms per prediction |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for size in SIZES:
        r = lookup[(0, 'error_only', size)]
        lines.append(f"| {size} | {r['held_out']['mean_padic_loss']:.6f} | "
            f"{r['held_out']['first_digit_accuracy']:.2%} | {1-r['training']['exact_accuracy']:.2%} | "
            f"{r['training_fraction']:.2%} | {r['forgotten_correct_rows']:.1f} | "
            f"{r['stored_nonzero_coefficients']:.1f} | {r['mean_active_terms']:.2f} |")
    late81, late135 = lookup[(0, 'error_only', 81)], lookup[(0, 'error_only', 135)]
    best_early = min(primary[:81], key=lambda r: r['held_out']['mean_padic_loss'])
    parity_loss = {parity: statistics.fmean(r['held_out']['mean_padic_loss'] for r in primary
                   if r['members'] >= 135 and r['members'] % 2 == parity) for parity in (0, 1)}
    lines += ['', f"The measured loss change is {late135['held_out']['mean_padic_loss']-late81['held_out']['mean_padic_loss']:+.6f} "
        f"from 81 to 135 and {fscore['mean_padic_loss']-late135['held_out']['mean_padic_loss']:+.6f} from 135 to 243. "
        'Finite curves do not establish an asymptotic floor. Both member-budget comparisons and '
        'the coefficient counts matter: a member is not a fixed unit of storage or prediction work. '
        'Active terms exclude defaults and consensus search.', '',
        'The every-prefix evaluation reveals a persistent **even/odd oscillation**. '
        f"Over sizes 135–243, mean loss across even sizes is {parity_loss[0]:.6f}; across odd sizes "
        f"it is {parity_loss[1]:.6f}. The figures join each parity separately to make this visible. "
        f"The best prefix through 81 was {best_early['members']} members "
        f"(loss {best_early['held_out']['mean_padic_loss']:.6f}), only "
        f"{best_early['held_out']['mean_padic_loss']-bscore['mean_padic_loss']:.6f} above the best through 243. "
        'This is a descriptive pattern found after scoring, not a new selection or stopping rule. '
        'New members repeatedly overturn correct training predictions: the last member makes '
        f"{final['forgotten_correct_rows']:.1f} previously correct rows wrong per fold on average. "
        'The measured behaviour is a fluctuating plateau rather than steady convergence.', '',
        '## Evidence and limitations', '',
        f"Batch: `{report['batch_id']}`. Fitting source: `{report['manifest']['source_commit']}`. "
        f"Snapshot: `{report['manifest']['snapshot_id']}`; digest `{report['manifest']['snapshot_digest']}`.", '',
        f"Audit passed for all 15 trajectories, including {report['validation']['certified_coordinates']:,} "
        'supported coordinate checks on the actual selected training subsets. Every mask, '
        'integer prediction and training-only default was reconstructed; consensus checkpoints '
        'matched the prefix decoder, and all scores agreed under two integer metric implementations. '
        'First-member predictions matched their independent controls exactly.', '',
        'The dataset has 6,693 products, 2,542 binary features and 363 paths. This remains an '
        'exploratory reuse of the established five folds. No new independent test dataset was '
        'collected. Comparing many prefixes can make the observed best size optimistic. Fold '
        'differences and starting-order variation are descriptive; no nominal significance test '
        'is applied to the correlated size points.', '',
        'Private products, selected-row indices, coefficients and predictions remain in Shopify '
        'Postgres under the `padjective.residual_ensemble_*` tables. The exported JSON contains '
        'aggregate scores, fingerprints and provenance only. All new tables use `pg_default`.', '',
        'Protocol: [residual-ensemble-protocol.md](../residual-ensemble-protocol.md). '
        'Raw aggregate evidence: [results.json](results.json). '
        'Fold means and paired differences: [summary.json](summary.json).', '',
        'To reproduce the figures from validated aggregate evidence:', '',
        '```sh', 'uv run -m padjective.residual_ensemble_report \\',
        '  --results docs/residual-ensemble/results.json --output-dir docs/residual-ensemble', '```', '',
    ]
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
    make_figures(report, summary, args.output_dir)
    write_markdown(report, summary, args.output_dir)
    print(json.dumps(dict(batch_id=report['batch_id'], directory=str(args.output_dir))))


if __name__ == '__main__':
    main()
