# Final bounded diagnostic protocol, 2 October 2026

Run only on raksasa, using the existing isolated checkout, one nice-15 process,
and one numerical thread. Replay the original warm Adam member for at most
100 updates and 20 fitting seconds. Fit timing includes diagnostic callbacks;
the optimizer checks its limit between updates. Apply an outer 30-second
timeout including startup and data loading. Do not repeat a partial replay.

Read original run b5bff126-fb4a-492d-969b-b960ccc2220f and snapshot
244ddbe3-0a2c-4c04-9436-0a0253108a09 in a read-only database transaction.
Reuse its exact-fitted coefficients, feature order, training mask and fold-0
split. Verify the snapshot, mask, feature and coefficient fingerprints. No
exact refit, new snapshot, held-out prediction or original result write.

Fixed settings: p=71, precision 7, seed 42, batch 32, Adam, initial radius
71^-7, zero optimizer histories, kappa 1, beta 0.9, beta2 0.999 and epsilon
1e-8. Keep the optimizer and sampler unchanged. Existing diagnostics are
observational; compare update 1/100 counters, training losses and radius
extrema with the stored original history, excluding elapsed time.

Log each completed update's before/after centers and radii, batch indices and
present feature IDs, raw/used slopes, moments, second moments, clocks, speeds,
vertex depths, group/tie/selection data, requested/applied common times,
limiting boundaries, actual radius changes and numerical no-ops. Flush every
trace record and persist progress every ten updates. Private traces and
initial coefficients stay on the host and out of Git/Library.

Audit stored records without further fitting: exact integer training score,
unchanged coefficient hashes, common clipping and radius arithmetic, actual
motions/no-ops, negative feasible slope suppression, current-slope-opposed
motion, missing-minibatch support, useful child/vertex access and tie effects.
Report distributions and observations separately from causal claims. Compare
available checkpoints and retain the actual count if the time cap is reached.

Finish this initial experiment with a concise scientific conclusion and one
recommended next experiment, without launching it. Preserve all old product
scores, synthetic results, manuscript work and existing private report/patch
identities. No ensembles, wider searches, pushes or publication.
