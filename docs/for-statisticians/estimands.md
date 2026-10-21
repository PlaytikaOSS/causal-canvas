# Estimands

What Causal Canvas is estimating, and the operational difference between the three quantities a typical run produces.

## Observational marginals: $P(Y)$ and $P(Y \\mid X = x)$

Given the fitted Bayesian network on the learned DAG, observational queries factor as

$$
P(Y) = \\sum_{\\mathbf{Z}} \\prod_i P(X_i \\mid \\mathrm{Pa}(X_i))
$$

Variable elimination is performed by `causal_canvas.inference_engine.InferenceEngine` (pgmpy `VariableElimination`). These are not interventional quantities — they describe the data as it was observed.

## Interventional: $P(Y \\mid \\mathrm{do}(X = x))$

The do-operator severs all incoming edges to $X$ and fixes its value to $x$:

$$
P_M(Y \\mid \\mathrm{do}(X = x)) = \\sum_{\\mathbf{Z}} \\prod_{i: i \\neq X} P(X_i \\mid \\mathrm{Pa}(X_i)) \\cdot \\mathbf{1}[X = x]
$$

In Causal Canvas, this is implemented in `causal_canvas.inference_utils` as repeated calls to the `do_intervention` API of the `InferenceEngine`, one per category of $X$. The output is the marginal distribution of $Y$ under the modified graph.

**For probabilistic interventions** (e.g. shifting 70% of users to bin 0, 30% to bin 1), the intervention is a *soft* one:

$$
P(Y \\mid \\mathrm{do}(X \\sim q)) = \\sum_x q(x) \\cdot P(Y \\mid \\mathrm{do}(X = x))
$$

The config's `interventions` field expresses $q$ directly as a categorical distribution.

## Average Treatment Effect (ATE)

For a binary or coarsened categorical lever $X$ with reference value $x_0$ and treatment value $x_1$:

$$
\\mathrm{ATE}(X; x_0 \\to x_1) = \\mathbb{E}[Y \\mid \\mathrm{do}(X = x_1)] - \\mathbb{E}[Y \\mid \\mathrm{do}(X = x_0)]
$$

In the discrete setting, the expectation reduces to a probability difference for binary $Y$. `infer_do_why.py` writes per-feature ATE plots.

## Counterfactuals: "given what we saw, what would have happened?"

A counterfactual conditions on the *factual* observation and queries the same individual under a different value of $X$. Pearl's three-step procedure (abduction–action–prediction) is what `do_intervention('X', 'x')` approximates here, with abduction implicit in the BN's fitted CPTs.

In the codebase, counterfactual quantities are derived under the simplification that the population's noise terms are summarised by the marginals — i.e. these are *population-level* counterfactuals, not unit-level ones. For unit-level counterfactuals you would need a structural causal model with explicit noise variables, which Causal Canvas does not currently provide.

## What is *not* an estimand here

- Conditional ATE (CATE) — the tool does not estimate effect heterogeneity by sub-population. You would obtain CATE by conditioning the BN on each segment manually and re-running `infer-dowhy`.
- Mediation effects (natural direct, natural indirect). Pearl's mediation formula is identifiable from the BN but not currently implemented.
- Time-varying effects. The BN is a snapshot.

## See also

- [Assumptions](assumptions.md).
- [Discretisation bias](discretisation-bias.md) — how the choice of bins affects the do-calculus output.
- [References](references.md).
