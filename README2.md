# padjective

Calculate p-adic adjective embeddings

# Purpose

Document and reproduce experiments that learn adjective ordering rules with p-adic techniques.

# Current experiment: p-adic adjective embeddings

## Motivation

Across multiple projects we have noticed that when we train models with p-adic losses the learned coefficients almost always collapse onto pure powers of the prime \(p\). Even when starting from arbitrary integer weights the optimiser quickly drives them toward \(p^{b_n}\) with coefficient one. Mixed terms of the form \(a_n p^{b_n}\) with \(a_n \neq 1\) are rare in practice. We suspect there is a tropicalisation argument lurking here that would explain the collapse analytically.

## Embedding strategies under test

1. **Byte/character encodings.** Interpret the UTF-8 (or ASCII) byte sequence of a word directly as a p-adic expansion. Words that are 2-adically close therefore share suffixes, which aligns with how Indo-European inflectional endings behave. The hope is that grammatical shifts correspond to simple linear operations in this space.
2. **Lexical hierarchy encodings.** Place each word inside a lightly pruned WordNet-like tree and turn the branch decisions into digits of the p-adic number. We and others have published variants of this approach. The embeddings themselves span a larger set of coefficients, yet downstream supervised learners still favour pure powers of \(p\).
3. **Sequential encodings for adjective order.** Focus on sequences of adjectives that precede a noun. Each adjective receives a p-adic integer that is a single power of \(p\), encoding where the adjective tends to appear relative to its neighbours. This representation is agnostic to meaning but accurately predicts which adjective should come first or next—essentially mirroring the behaviour required of an autoregressive language model.

## What works so far

With these encodings in place we fit linear models using p-adic losses. Compared to brute-force p-adic linear regression the supervised optimisation converges much faster and achieves strong accuracy on held-out adjective sequences. The workflow demonstrates that we can efficiently learn ordering preferences without leaning on semantic information.

## Open questions

- Can we formalise why p-adic losses prefer pure powers via tropical geometry or another analytic argument?
- How does training time compare against conventional (real-valued) optimisation baselines for the same tasks?
- Which qualitative examples best illustrate the ordering predictions to a reader who is unfamiliar with p-adic methods?

Answering those questions should make the experiment more compelling while keeping the focus on adjective ordering.
