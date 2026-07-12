# Gating checks for the padjective paper plan.
# Check A: what fraction of learned UMLLR coefficients are pure powers of p (README conjecture)?
#          And what structural form do they actually take (taxonomy codes / differences of codes)?
# Check B: does the p-adic valuation of a tag's coefficient correlate with where the tag
#          appears in product titles (the linguistics-paper gating scatter)?
import csv
import math
import sys
from collections import defaultdict

P = 71
BASE = "/private/tmp/claude-501/-Users-gregb-Documents-devel-padjective/232188ed-e5b2-40f1-a36b-480372ac1fc8/scratchpad/checks"

# ---------- load coefficients ----------
coeffs = []  # (fold, tag, int coefficient, sequence)
with open(f"{BASE}/coefficients.csv") as f:
    for row in csv.DictReader(f):
        c = int(row["coefficient"])
        coeffs.append((int(row["cv_fold"]), row["tag"], c, int(row["sequence"])))

def valuation(x, p=P):
    if x == 0:
        return None  # infinite
    v = 0
    x = abs(x)
    while x % p == 0:
        x //= p
        v += 1
    return v

def unit_part(x, p=P):
    if x == 0:
        return 0
    x_abs = abs(x)
    while x_abs % p == 0:
        x_abs //= p
    return x_abs

def is_pure_power(x, p=P):
    return x != 0 and unit_part(x, p) == 1 and x > 0

# ---------- Check A ----------
nonzero = [c for c in coeffs if c[2] != 0]
zero_count = len(coeffs) - len(nonzero)
pure = [c for c in nonzero if is_pure_power(c[2])]
print("=" * 70)
print("CHECK A: coefficient structure (live tables, taxonomy_association, p=71)")
print(f"total coefficients: {len(coeffs)}  (zero: {zero_count}, nonzero: {len(nonzero)})")
print(f"pure powers of {P} (coefficient == {P}^k): {len(pure)} = {100*len(pure)/len(nonzero):.2f}% of nonzero")

# digit count distribution of nonzero coefficients (how many nonzero base-71 digits)
def digits(x, p=P):
    x = abs(x)
    out = []
    while x:
        out.append(x % p)
        x //= p
    return out

ndig_dist = defaultdict(int)
for _, _, c, _ in nonzero:
    nd = sum(1 for d in digits(c) if d != 0)
    ndig_dist[nd] += 1
print("nonzero base-71 digit count distribution:",
      dict(sorted(ndig_dist.items())))

# are coefficients (or their negations) exact taxonomy codes?
tax_codes = set()
with open(f"{BASE}/taxonomy.csv") as f:
    for row in csv.DictReader(f):
        path = row["taxonomy_path"]
        if path and all(part.isdigit() for part in path.split(".")):
            code = 0
            for i, part in enumerate(path.split(".")):
                code += int(part) * (P ** i)
            tax_codes.add(code)
print(f"taxonomy codes loaded: {len(tax_codes)}")

exact_code = sum(1 for _, _, c, _ in nonzero if c in tax_codes)
neg_code = sum(1 for _, _, c, _ in nonzero if -c in tax_codes)
# differences of taxonomy codes: check on a sample (full cross product is 2.6k^2=6.7M -> fine as a set)
code_list = sorted(tax_codes)
code_set = tax_codes
diff_hits = 0
sample = nonzero[:: max(1, len(nonzero) // 2000)]  # ~2000 sampled coefficients
for _, _, c, _ in sample:
    if c in code_set or -c in code_set:
        diff_hits += 1
        continue
    if any((c + t) in code_set for t in code_list):  # c = code1 - code2
        diff_hits += 1
print(f"coefficients that ARE a taxonomy code: {exact_code} ({100*exact_code/len(nonzero):.2f}%)")
print(f"coefficients whose NEGATION is a taxonomy code: {neg_code} ({100*neg_code/len(nonzero):.2f}%)")
print(f"coefficients expressible as difference of two current taxonomy codes "
      f"(sampled n={len(sample)}): {diff_hits} ({100*diff_hits/len(sample):.2f}%)")

# valuation distribution
val_dist = defaultdict(int)
for _, _, c, _ in nonzero:
    val_dist[valuation(c)] += 1
print("valuation distribution of nonzero coefficients:", dict(sorted(val_dist.items())))

# ---------- Check B ----------
print()
print("=" * 70)
print("CHECK B: coefficient valuation vs title position (linguistics gating check)")

# per-tag mean relative title position, computed like tagbattle: case-insensitive
# substring match of tag in title; relative position = start / len(title)
tag_positions = defaultdict(list)
tagset_by_upper = {}
all_tags = {t.upper() for _, t, _, _ in coeffs}
with open(f"{BASE}/products.csv") as f:
    for row in csv.DictReader(f):
        title = (row["title"] or "").upper()
        raw = row["tags"] or ""
        if not title or not raw:
            continue
        tags = [t.strip().upper() for t in raw.split(",") if t.strip()]
        # drop nested tags (tag that is substring of another tag on same product)
        kept = [t for t in tags if not any(t != u and t in u for u in tags)]
        L = len(title)
        for t in kept:
            pos = title.find(t)
            if pos >= 0:
                tag_positions[t].append(pos / max(1, L))

mean_pos = {t: sum(v) / len(v) for t, v in tag_positions.items() if len(v) >= 3}
print(f"tags with >=3 in-title occurrences: {len(mean_pos)}")

# battle win rate per tag (winner = appears LATER in title)
wins = defaultdict(int)
losses = defaultdict(int)
with open(f"{BASE}/battles.csv") as f:
    for row in csv.DictReader(f):
        wins[row["winner_tag"].upper()] += 1
        losses[row["loser_tag"].upper()] += 1
win_rate = {}
for t in set(wins) | set(losses):
    n = wins[t] + losses[t]
    if n >= 5:
        win_rate[t] = wins[t] / n

def spearman(pairs):
    # simple Spearman rho with average ranks
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
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return None, n
    return cov / (vx * vy), n

# per fold: valuation vs mean title position, and valuation vs battle win rate
for fold in sorted({c[0] for c in coeffs}):
    fold_val = {t.upper(): valuation(c) for f, t, c, _ in coeffs
                if f == fold and c != 0}
    pairs_pos = [(fold_val[t], mean_pos[t]) for t in fold_val if t in mean_pos]
    pairs_win = [(fold_val[t], win_rate[t]) for t in fold_val if t in win_rate]
    rho_pos, n_pos = spearman(pairs_pos)
    rho_win, n_win = spearman(pairs_win)
    msg = f"fold {fold}: valuation~title_pos rho={rho_pos:+.3f} (n={n_pos})" if rho_pos is not None else f"fold {fold}: title_pos n={n_pos} (too few)"
    if rho_win is not None:
        msg += f" | valuation~battle_winrate rho={rho_win:+.3f} (n={n_win})"
    print(msg)

# pooled across folds (mean valuation per tag)
tagval = defaultdict(list)
for f, t, c, _ in coeffs:
    if c != 0:
        tagval[t.upper()].append(valuation(c))
mean_val = {t: sum(v) / len(v) for t, v in tagval.items()}
pairs_pos = [(mean_val[t], mean_pos[t]) for t in mean_val if t in mean_pos]
pairs_win = [(mean_val[t], win_rate[t]) for t in mean_val if t in win_rate]
rho_pos, n_pos = spearman(pairs_pos)
rho_win, n_win = spearman(pairs_win)
print(f"POOLED: valuation~title_pos rho={rho_pos:+.3f} (n={n_pos}) | "
      f"valuation~battle_winrate rho={rho_win:+.3f} (n={n_win})")

# permutation p-value for the pooled title-position correlation
import random
random.seed(42)
obs = rho_pos
vals = [p[0] for p in pairs_pos]
poss = [p[1] for p in pairs_pos]
count = 0
NPERM = 2000
for _ in range(NPERM):
    random.shuffle(poss)
    r, _ = spearman(list(zip(vals, poss)))
    if r is not None and abs(r) >= abs(obs):
        count += 1
print(f"permutation p-value (two-sided, {NPERM} perms): {count / NPERM:.4f}")
