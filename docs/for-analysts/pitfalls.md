# Pitfalls

Causal Canvas estimates causation from observational data. That's strictly weaker than an A/B test, and there are several specific ways the output can mislead you. The list below is the short version of "what could be wrong".

## Direction errors

The DAG can flip cause and effect when two variables are tightly correlated and you didn't constrain the direction. Examples:

- "Churn → tenure" instead of "tenure → churn" — your data has *days observed before churn* mixed in with churn itself.
- "G3 → G2" instead of "G2 → G3" — the second-period grade can't cause the first-period grade.

**Mitigation**: list every direction you know is wrong in `tabu_edges` (specific edges) or `tabu_edge_features` (variables that no other feature can affect, like demographics).

## Hidden confounders

If a variable affects both your supposed cause and your supposed effect, and you didn't measure it, the estimated effect is biased — sometimes flipped. Causal Canvas can't detect this; it can only assume *causal sufficiency* (no unmeasured confounders).

**Mitigation**: before running, write down the top 3 variables you think could confound the relationship of interest. If any aren't in the data, add a caveat to the conclusions.

## Discretisation distortion

Numeric features are discretised before the BN fit. The choice of bins can change which interventions look effective.

- Coarse bins (`tree` discretiser, depth 2): three values per feature, fast, low noise, but loses gradients.
- Fine bins (`simple` discretiser, 10 quantiles): captures gradients but every cell of the resulting CPT has fewer observations, increasing noise.

**Mitigation**: re-run with at least two `discretiser.method` choices and check if the rank order of drivers is stable.

## Non-deterministic structure

Linear NOTEARS is deterministic given its hyper-parameters. Non-linear NOTEARS-MLP is *not* — different random seeds give slightly different graphs.

**Mitigation**: run 5+ times with different `non_linear_args.seed` (or set it explicitly), keep edges that appear in ≥ 80% of runs. Treat any edge with lower stability as a hypothesis, not a finding.

## NOTEARS' equal-variance trap

The NOTEARS algorithm has a known weakness on Gaussian data with equal-variance noise: it can latch onto edge directions that are mathematically indistinguishable from their reverse. See Reisach et al. (2021), "Beware of the Simpson's paradox of NOTEARS".

**Mitigation**: standardise features before running, and if you get a result that contradicts strong prior knowledge, run it under permuted variable order to check stability.

## Acting on small effects

The BN's estimated ATE has uncertainty. If the post-intervention churn estimate is 14% and the pre-intervention is 13%, that's well within model noise — not a finding.

**Mitigation**: run `evaluate-experiments` to get bootstrap intervals for predictions; treat any intervention effect smaller than the bootstrap CI as null.

## Treating it as a substitute for an A/B test

Causal Canvas is a *prioritisation tool*, not a substitute for an experiment. The recommended workflow is:

1. Run Causal Canvas to rank candidate interventions.
2. Pick the top 1–3.
3. **A/B test them.**
4. Update product strategy based on the test, not on the model's estimate.

## See also

- [Interpreting results](interpreting-results.md) — what to look at first.
- [Discretisation bias (deeper, statistician version)](../for-statisticians/discretisation-bias.md).
- [Assumptions and identifiability](../for-statisticians/assumptions.md) — the formal version of "what could be wrong".
