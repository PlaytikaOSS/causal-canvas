# Interventions and counterfactuals

Once the DAG is learned and the Bayesian network's CPDs are fit, Causal Canvas can answer two qualitatively different questions: "what happens if we *force* X to value x?" (intervention) and "what would have happened if X had been different?" (counterfactual).

## Intervention: $\\mathrm{do}(X = x)$

The do-operator simulates the effect of an external mechanism that fixes $X$ to the value $x$, regardless of what would normally cause $X$ to take that value. In the DAG, this severs all edges *incoming* to $X$ — but leaves edges *outgoing* from $X$ intact, so the effect on downstream variables is preserved.

Formally:

$$
P(Y \\mid \\mathrm{do}(X = x)) = \\sum_{\\mathbf{Z}} \\prod_{i: i \\neq X} P(X_i \\mid \\mathrm{Pa}(X_i)) \\cdot \\mathbf{1}[X = x]
$$

Read this as: "in the modified world where X is held at x, marginalise over all other variables to get the distribution of Y".

This is **not the same** as conditioning on $X = x$. Conditioning $P(Y \\mid X = x)$ also tells you something about *the kind of unit that has X = x* — their other characteristics, why they ended up there. The do-operator tries to isolate the causal effect of *X itself*.

### Soft / probabilistic interventions

Causal Canvas's `interventions` config lets you specify a target distribution $q(x)$ rather than a point value. The estimand becomes

$$
P(Y \\mid \\mathrm{do}(X \\sim q)) = \\sum_x q(x) \\cdot P(Y \\mid \\mathrm{do}(X = x))
$$

So "shift the population to 70% bin 0, 30% bin 1" is a weighted average of the two point interventions.

## Counterfactual: "what if X had been different?"

A counterfactual answers a question about a specific observed outcome:

> "User U was given the high-popup variant and churned. Would they have stayed if they had been given the low-popup variant?"

In Pearl's framework, this is computed in three steps:

1. **Abduction** — update beliefs about the noise terms given the factual observation.
2. **Action** — apply the do-operator to set $X$ to the counterfactual value.
3. **Prediction** — derive the distribution of the outcome under the modified model.

Causal Canvas's `counterfactuals` config produces *population-level* counterfactual distributions: the abduction step is approximated by the BN's marginals over unobserved variables. For *unit-level* counterfactuals (questions about specific users), you'd need a structural causal model with explicit noise variables, which this codebase does not provide.

## When does the answer mean something causal?

The output of `infer-dowhy` is causally meaningful only if:

1. The learned DAG is approximately correct.
2. There are no unmeasured confounders (causal sufficiency).
3. The Markov compatibility holds for the discretised variables.
4. The intervention target has the back-door criterion satisfied through the modelled parents.

When any of these fail, the numbers are still computed but stop being causally interpretable. See [Assumptions](../for-statisticians/assumptions.md).

## Going deeper

- [Estimands](../for-statisticians/estimands.md) — formal definitions.
- [Bayesian networks](bayesian-networks.md) — what's being intervened on.
- [Pitfalls](../for-analysts/pitfalls.md) — common ways interventions get over-trusted.
