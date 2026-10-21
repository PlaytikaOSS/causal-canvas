# For Statisticians

This section names the assumptions, the estimands, and the limitations Causal Canvas operates under. The shape of the problem: structural causal model learned from observational data via continuous-optimisation NOTEARS, fitted with a discrete Bayesian network for downstream do-calculus inference.

- [Assumptions and identifiability](assumptions.md) — what must hold for the output to be causally interpretable.
- [Estimands](estimands.md) — what is being estimated, and the difference between observational, interventional, and counterfactual quantities here.
- [Non-linear NOTEARS](nonlinear-notears.md) — extension to continuous structural equations beyond linear, with distribution-type customisation.
- [Discretisation bias](discretisation-bias.md) — the consequence of discretising the BN's nodes for the do-calculus that follows.
- [References](references.md) — BibTeX bibliography.

## What Causal Canvas does, formally

Given an observational sample $\\{X_i\\}_{i=1}^n$ over $d$ variables, Causal Canvas:

1. **Learns** a DAG $G$ over the variables by solving the NOTEARS continuous-optimisation problem of Zheng et al. (2018), optionally with a non-linear MLP parameterisation (Zheng et al., 2020).
2. **Discretises** continuous variables using one of `simple` (quantile), `tree` (decision-tree split), or `mdlp` (Fayyad–Irani entropy-based split) discretisers.
3. **Fits** conditional probability tables on the discretised variables to obtain a discrete Bayesian network compatible with $G$. Maximum-likelihood or Bayesian (K2 / BDeu prior) estimators are supported.
4. **Performs** do-calculus inference (marginals, interventions, counterfactuals, ATE) on the fitted BN via `causal_canvas.inference_engine.InferenceEngine` (pgmpy variable elimination).

## What it does *not* do

- Continuous-time causal inference.
- Latent variable modelling (no hidden confounders inferred).
- Non-tabular data.
- Sensitivity analysis to unmeasured confounding (e.g. E-values, Rosenbaum bounds) — these would need to be added by the analyst.
- Identifiability checks beyond DAG acyclicity. The user is responsible for verifying that the assumptions in [Assumptions](assumptions.md) plausibly hold.
