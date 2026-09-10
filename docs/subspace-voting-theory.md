# What masked linear ensembles can represent

These are elementary derivations for the constructions in the 11 September
2026 experiment. They concern the hypothesis class, not the functions that the
coordinate optimiser will find, and not generalisation error. They have not
been checked for novelty. The accompanying counterexamples are executable in
`tests/test_subspace_voting.py`.

Let the input be binary, x in {0,1}^D, and let q=p^E. Component j has a mask
S_j and raw output

\[
g_j(x)=\sum_{i\in S_j}w_{ji}x_i\pmod q.
\]

Its zero output is replaced by a fixed default d_j. Write this default-adjusted
output as h_j(x). Fix a nonempty candidate vocabulary C of valid paths, and
write pi_C for nearest-path projection with the specified deterministic ties.
Defaults and C are held fixed after fitting. The current experiment uses p=71,
E=7, and C equal to paths observed in the fitting fold, not the complete taxonomy.

## 1. Masks restrict the class; they do not enlarge it

Any masked member can be embedded in a full-feature member by setting all
omitted coefficients to zero. Its raw output, default-adjusted output and every
deterministic ensemble output are unchanged. Therefore, for a fixed number of
members and fixed decoding rules, the masked class is contained in the
unrestricted class.

This does not imply that the fitted all-feature model is better. Different
masks change optimisation, estimation variance and dependence between members.
An improvement from random subspaces would be a fitting/generalisation effect,
not evidence that removing available coefficients increased the unrestricted
hypothesis class.

## 2. Feature-union and common-kernel invariance

Let W be the matrix of component weights, extended by zeros outside each mask.
If Wx=Wx' modulo q, then all component outputs, defaults, projections and
deterministic votes agree. In particular, the ensemble cannot distinguish two
inputs that agree on the union of the masks.

The raw signature has at most min(2^D,q^M) distinct values on the binary domain.
No downstream voting rule can recover information lost at this stage.

The congruence must be stated modulo q, not merely modulo p: later digits can
affect ties or total p-adic costs and thereby alter an aggregate root decision.

If each member independently receives a uniform k-subset of D eligible
features, then

\[
\mathbb E[\text{unseen features}]=D(1-k/D)^M.
\]

For a particular r-feature set, the probability that at least one member sees
all of it is

\[
1-\left(1-\frac{(k)_r}{(D)_r}\right)^M,
\]

where the probability is zero if k<r. These are coverage statements. As the
next example shows, co-occurrence in one member is not necessary for every
kind of interaction.

## 3. Voting can create interactions across disjoint masks

Use three binary features and three singleton masks. Set g_j(x)=x_j and
default d_j=2, with C={1,2}. Thus a member predicts 1 when its feature is present
and 2 otherwise. Whole-path plurality, p-adic medoid and survivor voting all
predict 1 exactly when at least two features are present.

This is not an affine function into Z/qZ. Its values at 000, 100 and 010 are
all 2, but its value at 110 is 1. An affine function would satisfy

\[
F(110)-F(100)-F(010)+F(000)=0\pmod q,
\]

whereas the displayed function gives -1. No component saw two features
together, yet their aggregate has a cross-feature interaction.

The claim is about non-affinity, not arbitrary interaction capacity: majority
of singleton voters remains a threshold function, and cannot express XOR.

## 4. A parity obstruction for four of the aggregation rules

Suppose C contains exactly two paths with different roots, each component sees
at most k features, and ties are resolved deterministically. Then the following
rules produce a binary decision with real polynomial threshold degree at most k:

- valid-candidate p-adic consensus of raw outputs;
- p-adic medoid of projected outputs;
- plurality of projected outputs;
- survivor voting of projected outputs.

For raw consensus, the difference between the two candidate costs is

\[
T(x)=\sum_j\{d_p(h_j(x),a)-d_p(h_j(x),b)\}.
\]

Every summand depends on at most k binary variables, so it has an exact real
multilinear expansion of degree at most k. The choice is a threshold on T.
For the other three rules, two different-root labels reduce the decision to
majority of the projected binary votes. Its signed vote count is likewise a
sum of functions on at most k variables. A fixed tie can be represented by an
arbitrarily small constant shift because the domain is finite.

Consequently these rules cannot represent parity or its complement on r>k
relevant features, irrespective of the number of members. To prove this, encode
those features as z_i in {-1,1}, and let chi(z)=product_i z_i. If a degree-k
polynomial P strictly separated the parity classes, chi(z)P(z) would be positive
at every input (reverse its sign for the complement). Its uniform expectation
would therefore be positive. But every monomial of P omits at least one of
the r variables, so orthogonality gives E[chi P]=0, a contradiction. Fixing
additional irrelevant inputs does not change the argument.

This statement is deliberately binary. It is not a general degree bound for
multiclass survivor voting, and it does not apply to raw-code plurality before
projection.

The threshold-degree language and orthogonality-witness method are standard;
see Section 2.1 and Theorem 2.2 of Alexander Sherstov's
[*The Intersection of Two Halfspaces Has High Threshold Degree*](https://web.cs.ucla.edu/~sherstov/pdf/hshs.pdf).
The argument above applies that framework to the present masks and decoding
rules. It is not a new general lower-bound technique.

## 5. Raw-code plurality can escape that obstruction

Here is an explicit XNOR construction with two singleton members and one
constant member. Take p=71, E>=2, C={2,72}, and

\[
g_1(x)=143x_1,\quad g_2(x)=143x_2,\quad g_3(x)=0,
\qquad d_1=d_2=72,\ d_3=2.
\]

All defaults are in C. Both 72 and 143 project to 72: their p-adic distance is
1/71, whereas their distance from 2 is 1.

| Input | Default-adjusted raw votes | Raw plurality, then projection |
|---|---|---|
| 00 | 72, 72, 2 | 72 |
| 01 | 72, 143, 2 | 2 |
| 10 | 143, 72, 2 | 2 |
| 11 | 143, 143, 2 | 72 |

The mixed inputs have three different raw codes. Their plurality tie goes to
the smallest code, 2. Equal inputs have a strict raw-code majority. The result
is XNOR: 72 for equality, 2 for inequality.

If projection happens first, both singleton voters become the constant 72,
so all three projected aggregation rules return 72 everywhere. Raw
valid-candidate consensus also returns 72 everywhere for these parameters.
More strongly, Proposition 4 rules out XNOR for any singleton parameters of
those four rules with this binary candidate vocabulary. Thus raw-code plurality
followed by projection can represent a function those rules cannot, under the
same singleton-mask restriction. This is a separation example, not a claim that
one whole hypothesis class contains the other in all settings.

The example depends on the stated raw-code tie rule. It demonstrates that
collisions between intermediate codes can carry information that early
projection discards; it does not say that this mechanism will be learnt or
will help on product classification.

## 6. A strict path majority makes the three projected rules agree

If a path a occurs more than M/2 times, plurality and survivor voting return a.
For survivor voting, its voters remain a strict majority in every survivor
set containing them, choose their branch, and eventually choose stop.

The p-adic medoid also returns a, and this part holds in any metric space.
For any other candidate c, the triangle inequality gives

\[
\sum_j d(c,v_j)-\sum_j d(a,v_j)
\ge (2n_a-M)d(a,c)>0.
\]

The same statement holds for raw consensus if its default-adjusted raw votes
have a strict majority at a valid path. Without a strict majority, the rules
need not agree.

## 7. The three path rules optimise different things

Whole-path plurality minimises empirical 0/1 disagreement with the voters.
The p-adic medoid minimises empirical p-adic disagreement. Survivor voting
makes a sequence of local decisions, with continuing voters pooled before a
branch is chosen. It does not generally minimise either global disagreement.

For p=71, use nine votes: three at 1, two at 2, two at 73, and two at 144.

- Whole-path plurality returns 1 (three exact votes).
- The p-adic medoid returns 2. Root 2 has six votes, and its three paths tie in
  p-adic cost; the smallest code wins.
- Survivor voting returns 73. Root 2 wins; stop has two votes versus four
  continuing; its two children tie and the smaller branch wins.

Pooling continue votes matters even within one root. With three votes at 1,
two at 72 and two at 143, plurality and medoid return 1, but survivor voting
returns 72: four continuing votes defeat three stopping votes.

Global p-adic consensus is not always greedy root plurality either. Give root
1 seventy-four votes distributed as the 71 codes 1+71k, k=0,...,70, plus one
extra vote each at 1, 72 and 143. Give root 2 seventy-three identical votes at
2. Candidate 2 costs 74. Even the best root-1 candidate costs at least
73+72/71>74. The medoid therefore selects the minority root 2, whereas survivor
voting selects root 1. This distinction is relevant for large ensembles; it
would be missed by replacing metric minimisation with greedy prefix voting.

A useful sufficient agreement condition is available for valid-path voters.
Suppose the most popular root a has n_a votes and every other root b has n_b
votes. If

\[
n_a-n_b>(n_a-1)/p\quad\text{for every }b\ne a,
\]

then every p-adic medoid lies in root a. To see this, choose any observed vote
in root a as a candidate. Its cost is at most M-n_a+(n_a-1)/p: other-root votes
cost one, its own vote costs zero, and each remaining same-root vote costs at
most 1/p. Every candidate in root b costs at least M-n_b. The displayed
condition makes the first bound strictly smaller. In particular, with M<=p,
a unique root-plurality winner must also be the medoid's root. This is a
sufficient condition, not a sharp threshold for the first possible disagreement.

## 8. Fixed precision gives a finite capacity bound

For fixed masks, each member has q^{|S_j|} possible coefficient vectors and at
most q defaults. With fixed C and a fixed aggregation/tie rule, the number of
representable functions is at most

\[
q^{\sum_j|S_j|+M}.
\]

Allowing masks of sizes k_j multiplies this upper bound by
product_j binomial(D,k_j). Defaults restricted to C only reduce the count.
There are |C|^{2^D} arbitrary labelings of the binary cube. A necessary condition
for representing all of them is therefore

\[
(\sum_j k_j+M)\log q+\sum_j\log\binom{D}{k_j}
\ge 2^D\log|C|.
\]

For fixed precision and a polynomially bounded number of parameters/members,
the left side grows at most polynomially in D, while the right side is
exponential for |C|>=2. Such a family cannot be universal on arbitrary binary
inputs. This counting obstruction is independent of which of the five fixed
aggregation rules is used. It neither supplies a useful numerical
generalisation bound for the present data nor prevents good performance on
structured taxonomies.
