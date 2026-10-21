# Discretisation bias

Causal Canvas fits a discrete Bayesian network on top of the learned DAG. Continuous variables are binned beforehand. The choice of bin boundaries influences every downstream do-calculus quantity.

## What discretisation does, formally

For continuous $X$, define a partition $\\{B_1, \\ldots, B_K\\}$ of its support and a deterministic map $\\tilde X = k$ iff $X \\in B_k$. The BN is fit on $\\tilde X$. Any subsequent intervention $\\mathrm{do}(\\tilde X = k)$ targets the *bin*, not a specific value.

The implication is that an intervention reported as "raise $\\tilde X$ from bin 0 to bin 1" averages over the within-bin distribution of $X$ as it appeared in the training data — which itself depends on the bin boundaries.

## The three discretisers

- **`simple`** — equal-quantile partition. `argument` is the number of bins. Stable, easy to reason about; loses information for skewed distributions where one quantile may span an order of magnitude.
- **`tree`** — supervised partition by single-target decision tree of depth `argument`. Bin boundaries are placed where the target is most predicted, so the BN has more signal to learn from. Output is target-aware, which is sometimes useful and sometimes a leakage concern (see below).
- **`mdlp`** — Fayyad–Irani entropy-based partition (Fayyad and Irani, 1993). Same target-aware concern as `tree` but uses minimum-description-length to choose the number of bins automatically.

## Sources of bias

### B1. Within-bin pooling

Two users with very different values of $X$ in the same bin become indistinguishable. If $Y$ depends non-monotonically on $X$ within a bin, the BN cannot capture it.

### B2. Boundary sensitivity

Coarsened estimates of $P(Y \\mid X = x)$ are sensitive to where the bin edges fall. Two valid discretisations can produce ATE estimates differing by 5–15 percentage points on small datasets.

### B3. Target-aware leakage

Tree- and MDLP-based discretisers use the target $Y$ to pick bin boundaries on $X$. Subsequent ATE estimates of $\\mathrm{do}(X)$ on $Y$ are biased toward the splits that *already* segment $Y$, inflating apparent effect sizes. This is known but not always disclosed in the BN literature.

**Mitigation in Causal Canvas**: when `discretiser.method` is `tree` or `mdlp`, run a sensitivity check by re-fitting with `simple` discretisation and comparing rank order of effects. Causal Canvas does not perform this comparison automatically.

### B4. Sparse-cell noise

A finely discretised feature with $K$ bins multiplies the conditional-probability table size by $K$ for every parent. With $d$ parents each at $K$ bins, the table has $K^{d+1}$ cells. With $n$ training rows, the average effective sample size per cell is $n / K^{d+1}$, often $\\ll 30$ in practice. Sparse-cell estimates are noisy.

**Mitigation**: `proportion_threshold` merges bins with too few observations into the previous one; `max_categories` caps the number of bins post-discretisation.

## Recommended workflow

1. Fit with the default `tree` discretiser to get an initial DAG and ATE estimates.
2. Re-fit with `simple` (5–10 quantile bins) and compare. Direction stability of major effects is the relevant check.
3. If the two disagree on rank order of top drivers, treat the result as inconclusive and consider whether the underlying continuous relationships are highly non-monotonic.
4. Document the discretiser choice in any reported result.

## See also

- [Estimands](estimands.md).
- [Assumptions](assumptions.md) — Markov compatibility (A4) intersects with discretisation choice.
- Fayyad and Irani (1993) and Chickering, Meek, Rounthwaite (2001) in [References](references.md).
