"""Gating checks behind PAPER_PLAN.md.

These verify the load-bearing empirical claims in the recommended paper direction
against the LIVE padjective tables (taxonomy_association UMLLR model). Run 2026-07-12.

IMPORTANT -- prime base: the live model uses ``prime_base = 79`` (from
``padjective.umllr_fold_metrics``), because the live taxonomy has a larger maximum
fan-out than the frozen paper snapshot (which uses p = 71). All valuation-based checks
below therefore use p = 79. Paper-specific reruns must instead use the frozen snapshot
at p = 71.

Reproduce the input CSVs from raksasa (Postgres ``shopifystores``)::

    psql shopifystores -c "\\copy (select cv_fold, tag, coefficient, sequence \
        from padjective.umllr_tag_coefficients) to stdout with csv header" > coefficients.csv
    psql shopifystores -c "\\copy (select cv_fold, product_id, true_value, \
        predicted_value, loss from padjective.umllr_predictions) to stdout with csv header" \
        > predictions.csv
    psql shopifystores -c "\\copy (SELECT p.id, p.product_title AS title, \
        pd.product_detail->'product'->>'tags' AS tags FROM cantbuymelove.product p \
        JOIN public.product_details pd ON p.myshopify_domain = pd.myshopify_domain \
        AND p.run_name = pd.run_name AND p.product_handle = pd.product_handle \
        JOIN cantbuymelove.product_taxonomy pt ON pt.product_id = p.id \
        WHERE p.product_title IS NOT NULL) to stdout with csv header" > products.csv

Then ``uv run reference/paper_plan_gating_checks.py --csv-dir <dir>``.

Checks (all confirmed on 2026-07-12):
  * Reconstruction: summing each held-out product's nested-filtered tag coefficients
    reproduces 100% of stored predicted_values -> the active-coefficient sets are exactly
    the fitter's.
  * Check A: pure powers of p among nonzero coefficients = 0.09% (retires the README
    "collapse onto pure powers of p" claim for the taxonomy model).
  * Check B: coefficient valuation vs mean relative title position is null
    (pooled Spearman rho = -0.018, permutation p = 0.75) -> no linguistics headline.
  * Headroom: 11.9% of products are multi-active; of those 91.5% are non-strict;
    strict vs non-strict mean loss 0.120 vs 0.903.
  * Ceiling: a perfect valuation-disjoint intervention moves overall loss only
    0.381 -> 0.296 (bounded by the 11.9% multi-active tail).
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict

PRIME = 79  # live model prime_base; frozen paper snapshot is 71
DEFAULT_PREDICTION: dict[int, int] = defaultdict(int)  # all folds default to 0 in the live model


def valuation(x: int, p: int = PRIME) -> float:
    if x == 0:
        return math.inf
    x = abs(x)
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


def unit_part(x: int, p: int = PRIME) -> int:
    if x == 0:
        return 0
    x = abs(x)
    while x % p == 0:
        x //= p
    return x


def filter_nested(tags):
    """Port of tagbattle.filter_nested_tags (case-insensitive substring pruning)."""
    unique, seen = [], set()
    for tag in tags:
        tag = tag.strip()
        if not tag or tag.lower() in seen:
            continue
        unique.append(tag)
        seen.add(tag.lower())
    out = []
    for tag in unique:
        tl = tag.lower()
        if any(tl != o.lower() and tl in o.lower() for o in unique):
            continue
        out.append(tag)
    return out


def spearman(pairs):
    n = len(pairs)
    if n < 5:
        return None, n

    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return None, n
    return cov / (vx * vy), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True, help="dir with coefficients.csv, predictions.csv, products.csv")
    args = ap.parse_args()
    d = args.csv_dir.rstrip("/")

    coef = {}
    for row in csv.DictReader(open(f"{d}/coefficients.csv")):
        coef[(int(row["cv_fold"]), row["tag"])] = int(row["coefficient"])

    titles, tags_raw = {}, {}
    for row in csv.DictReader(open(f"{d}/products.csv")):
        pid = int(row["id"])
        titles[pid] = row["title"] or ""
        tags_raw[pid] = row["tags"] or ""

    def active_upper(pid):
        return [t.upper() for t in filter_nested(tags_raw.get(pid, "").split(","))]

    # --- reconstruction + active-count distribution + strict/non-strict records ---
    recs, losses = [], {}
    n = match = 0
    active_counts: dict[int, int] = defaultdict(int)
    for row in csv.DictReader(open(f"{d}/predictions.csv")):
        f = int(row["cv_fold"])
        pid = int(row["product_id"])
        pred = int(row["predicted_value"])
        loss = float(row["loss"])
        if pid not in tags_raw:
            continue
        act = [coef.get((f, t), 0) for t in active_upper(pid) if coef.get((f, t), 0) != 0]
        recon = sum(act) or DEFAULT_PREDICTION[f]
        n += 1
        match += recon == pred
        active_counts[len(act)] += 1
        losses[pid] = loss
        if len(act) >= 2:
            recs.append((f, pid, loss, act))
    print(f"reconstruction: {match}/{n} stored predictions reproduced ({100 * match / n:.2f}%)")
    print("active-nonzero counts:", dict(sorted(active_counts.items())))

    nonzero = [c for c in coef.values() if c != 0]
    pure = sum(1 for c in nonzero if c > 0 and unit_part(c) == 1)
    print(f"Check A: pure powers of {PRIME} = {pure}/{len(nonzero)} ({100 * pure / len(nonzero):.2f}%)")
    vdist: dict[float, int] = defaultdict(int)
    for c in nonzero:
        vdist[valuation(c)] += 1
    print("  valuation distribution:",
          {("inf" if k == math.inf else int(k)): v
           for k, v in sorted(vdist.items(), key=lambda kv: (kv[0] == math.inf, kv[0]))})

    strict = [l for _, _, l, act in recs if len({valuation(c) for c in act}) == len(act)]
    nonstrict = [l for _, _, l, act in recs if len({valuation(c) for c in act}) != len(act)]
    tot = len(strict) + len(nonstrict)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"headroom: multi-active={tot} ({100 * tot / n:.1f}% of {n}); "
          f"strict={len(strict)} ({100 * len(strict) / tot:.1f}%, loss {mean(strict):.3f}) "
          f"non-strict={len(nonstrict)} ({100 * len(nonstrict) / tot:.1f}%, loss {mean(nonstrict):.3f})")
    overall = mean(list(losses.values()))
    ceiling = (sum(losses.values()) - (sum(nonstrict) - mean(strict) * len(nonstrict))) / len(losses)
    print(f"ceiling: overall loss {overall:.4f} -> {ceiling:.4f} "
          f"if every non-strict multi-active product hit strict loss")

    # --- Check B: valuation vs title position ---
    pos: dict[str, list[float]] = defaultdict(list)
    for pid, title in titles.items():
        T = title.upper()
        if not T:
            continue
        for t in active_upper(pid):
            i = T.find(t)
            if i >= 0:
                pos[t].append(i / max(1, len(T)))
    meanpos = {t: sum(v) / len(v) for t, v in pos.items() if len(v) >= 3}
    tagval: dict[str, list[float]] = defaultdict(list)
    for (f, t), c in coef.items():
        if c != 0:
            tagval[t].append(valuation(c))
    meanval = {t: sum(v) / len(v) for t, v in tagval.items()}
    pairs = [(meanval[t], meanpos[t]) for t in meanval if t in meanpos]
    rho, nn = spearman(pairs)
    random.seed(1)
    vals = [p[0] for p in pairs]
    poss = [p[1] for p in pairs]
    cnt, nperm = 0, 3000
    for _ in range(nperm):
        random.shuffle(poss)
        r, _ = spearman(list(zip(vals, poss)))
        if r is not None and abs(r) >= abs(rho):
            cnt += 1
    print(f"Check B: valuation~title_pos rho={rho:+.3f} (n={nn}), permutation p={cnt / nperm:.3f}")


if __name__ == "__main__":
    main()
