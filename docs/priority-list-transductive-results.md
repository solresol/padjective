# Passing values to unseen tags inside a held-out store, 23 November 2026

On store holdouts, 38% of products carry no fitted tag and take the
global default at a loss of 0.813. This run lets such a product borrow a
prediction from the other products of its own store, through the tags they
share. It uses the held-out store's tags and membership but not its labels.
It does not help: the backfilled list moves from 0.6420 to 0.6433 and
the pooled list from 0.6259 to 0.6263. Diagnostics that read held-out labels
show why. The store's tag graph connects the right products, but the fitted
list's predictions on the store's covered products are too poor to seed it.

Report: `docs/priority-list-transductive-2026-11-23.json`, experiment
`paper-priority-list-transductive-2026-11-23`, source commit `850c13c`, status
`exploratory`. It was not inserted into `padjective.paper_low_q_reports`.
The priority-list models are described in `docs/priority-list-results.md`.

## Method

A taxonomy-association priority list is fitted at weight 71 on the other
stores, with backfilled cell medoids and with pooled values. Its predictions
on a held-out fold reproduce `docs/priority-list-graph-2026-11-23.json`
exactly. Store membership comes from the same query as the archived store
folds and matches their assignment digest. The rest works inside one
held-out store at a time.

- **Tag passing.** Each tag with no fitted coefficient takes the medoid of
  the predictions of the store's products that carry it. A product with no
  fitted tag takes the value of its tightest valued tag: lowest mean
  distance from those predictions to their medoid, ties to the tag on more
  products, then the tag name. One round uses only products the fitted list
  covers. The full version repeats from the newly predicted products until
  nothing changes. The vote variant takes the 71-adic medoid of a product's
  valued tags.
- **Store default.** Products still without a prediction take the medoid of
  the predictions on the store's covered products.
- **Store tags first.** One round of tag passing in which a covered product
  also takes the value of its tightest unfitted tag.
- **Store consensus.** Every product of the store takes the store default.

## Held-out stores

Means over the five held-out folds. A fold holds 79 stores with a mean of
16.9 products and a median of 5.3; 26 stores have a single product.
The uncovered products sit in 29 of the stores, and all of them have at
least one tag. 8.8 stores have no covered product at all; they hold
31% of the uncovered products, which nothing here can reach. The fold
has 417 unfitted tags, and only 12 of them occur in more than one
store. Tag passing reaches 38% of the uncovered products in the first
round and 8% more in later rounds; it stops after about three rounds.

## Results

Mean d71,2 loss on the five store holdouts. The change is against the
fitted list, with the unadjusted corrected paired p-value in brackets.
The last three columns are for the backfilled list: the share of uncovered
products given a prediction, the loss on those products, and the loss the
global default has on the same products.

| Variant | Backfilled | Change | Pooled | Change | Reached | Loss | Default loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fitted list only | 0.6420 |  | 0.6259 |  | 0% |  |  |
| Store default | 0.6338 | -0.0082 (0.60) | 0.6155 | -0.0104 (0.54) | 69% | 0.764 | 0.796 |
| Tag passing, one round | 0.6443 | +0.0023 (0.55) | 0.6306 | +0.0047 (0.64) | 38% | 0.746 | 0.733 |
| Tag passing, to convergence | 0.6433 | +0.0013 (0.80) | 0.6263 | +0.0004 (0.94) | 46% | 0.759 | 0.770 |
| Tag passing, vote | 0.6434 | +0.0014 (0.77) | 0.6271 | +0.0012 (0.80) | 46% | 0.760 | 0.770 |
| Tag passing, then store default | 0.6355 | -0.0065 (0.65) | 0.6177 | -0.0082 (0.60) | 69% | 0.765 | 0.796 |
| Tag passing across the whole fold | 0.6423 | +0.0003 (0.96) | 0.6260 | +0.0001 (0.99) | 47% | 0.751 | 0.770 |
| Store tags first | 0.6377 | -0.0043 (0.64) | 0.6239 | -0.0020 (0.89) | 38% | 0.746 | 0.733 |
| Store consensus for every product | 0.6420 | +0.0000 (1.00) | 0.6312 | +0.0054 (0.73) | 69% | 0.764 | 0.796 |

No comparison is significant, before or after Holm adjustment. Every
adjusted p-value is 1.

- Tag passing changes nothing. On the products it reaches, the loss is
  0.759 against 0.770 for the global default.
- The store default gives the largest change, -0.0082 for the backfilled
  list, but the fold differences run from -0.036 to +0.015.
- Pooling the held-out stores of a fold adds almost nothing, because
  unfitted tags are rarely shared between stores.
- Store consensus leaves the loss where it was and cuts exact accuracy from
  0.151 to 0.104. The loss on covered products goes from 0.540 to
  0.553, so averaging a store's predictions does not cancel their errors.

## Diagnostics that read held-out labels

These rows are not predictors. They replace one ingredient with the truth to
show where the loss comes from. The first three change only the uncovered
products.

| Row | Backfilled | Pooled | Covered products | Uncovered products | Reached | Loss | Default loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fitted list only | 0.6420 | 0.6259 | 0.540 | 0.813 | 0% | | |
| Tag passing from true covered labels | 0.5643 | 0.5482 | 0.540 | 0.603 | 46% | 0.266 | 0.770 |
| Store default from true covered labels | 0.5343 | 0.5181 | 0.540 | 0.531 | 69% | 0.371 | 0.796 |
| Best single code per store | 0.4006 | 0.3844 | 0.540 | 0.170 | 100% | 0.170 | 0.813 |
| Medoid of the store's other labels | 0.2383 | 0.2378 | 0.260 | 0.207 | 100% | 0.206 | 0.813 |

- **The graph is a good conduit.** Passing the true labels of a store's
  covered products through the same tags gives 0.266 on the products
  reached, against 0.759 when predictions are passed.
- **The seed is the problem.** The store default from true covered labels
  gives 0.371 on the products it reaches, against 0.764 from predictions.
- **A held-out store is close to one category.** Predicting each product
  from the medoid of the labels of its store's other products gives
  0.2383, with root accuracy 0.879 and exact accuracy 0.460. The fitted
  list has 0.6420, 0.465 and 0.151. This row sees every other label in
  the store, so it is a ceiling and not a few-shot result.
- The best single code per store is chosen on the products it is scored on,
  so it is optimistic for stores with few uncovered products.

## Reading

The loss on store holdouts is not mainly a coverage problem. Covered
products already have a loss of 0.540, and the medoid of a store's
predictions is no closer to the truth than the predictions themselves. This
run does not separate two explanations for that: a store's predictions may
agree on a wrong category, or most stores may have too few covered products
to average.

What a new store lacks is a correct anchor. The leave-one-out row suggests
that a few labelled products per store would be worth more than any
reordering or propagation of tag values, and the oracle tag-passing row
shows the store's tags could carry those labels. That has not been tested
with a small number of labels.

The classical models were not given the held-out stores' tags or membership,
so these rows are not comparable with them.
