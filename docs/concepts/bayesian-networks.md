# Bayesian networks

A Bayesian network is a directed acyclic graph (DAG) plus a conditional probability distribution attached to each node. Together they describe how random variables relate, and let you ask questions like "given these inputs, what is the chance of churn?".

## Structure + numbers

The DAG is the *qualitative* part of the model: which variables influence which. The conditional probability distributions (CPDs) are the *quantitative* part: how strongly. For a node $X$ with parents $\\mathrm{Pa}(X)$, the CPD specifies $P(X \\mid \\mathrm{Pa}(X))$ — one row per combination of parent values.

The joint distribution factorises along the DAG:

$$
P(X_1, \\ldots, X_d) = \\prod_i P(X_i \\mid \\mathrm{Pa}(X_i))
$$

This factorisation is what makes inference tractable. Without it, a joint distribution over $d$ binary variables would require $2^d - 1$ free parameters; with the BN factorisation, the parameter count is the sum of CPD sizes, which scales much more gently when each variable has few parents.

## How Causal Canvas fits the CPDs

After NOTEARS produces the DAG and the discretiser bins continuous variables, the CPDs are estimated by `causal_canvas.bayesian_network.BayesianNetwork.fit_cpds` (pgmpy `DiscreteMLE` / `DiscreteBayesianEstimator`) from the discretised training data. Two estimators are supported:

- **Maximum-likelihood estimator (`MaximumLikelihoodEstimator`)** — counts and divides. Unbiased but high-variance for sparse cells.
- **Bayesian estimator (`BayesianEstimator`)** — adds a Dirichlet prior. Causal Canvas exposes two flavours:
    - **K2 prior** — uniform pseudocount of 1 across cells. Light regularisation.
    - **BDeu prior** — equivalent-sample-size pseudocount split across cells. Heavier regularisation; tune via `equivalent_sample_size`.

Bayesian estimation is the default in the example configs because BNs in this codebase often see sparse cells from finely discretised features.

## What you can do with a fitted BN

- **Marginal queries** — $P(Y)$ and $P(Y \\mid X = x)$ via variable elimination.
- **Interventions** — $P(Y \\mid \\mathrm{do}(X = x))$ via the do-operator severing $X$'s incoming edges. See [Interventions and counterfactuals](interventions-counterfactuals.md).
- **Counterfactuals** — population-level, given the BN's CPDs. Unit-level counterfactuals would need an explicit structural equation model with noise variables.

## What a BN is *not*

- **A classifier.** The BN is a generative model of the joint distribution; using it for prediction is fine, but the goal is to model relationships, not maximise classification accuracy.
- **A neural network.** It is a probabilistic graphical model. The non-linear NOTEARS-MLP variant uses MLPs *during structure learning*, but the fitted BN itself is purely tabular.
- **Time-aware.** Each variable is observed once per row. For temporal data, you would need a dynamic BN — not implemented here.

## Going deeper

- [Causal discovery](causal-discovery.md) — how the DAG is learned.
- [Interventions and counterfactuals](interventions-counterfactuals.md).
- [Estimands](../for-statisticians/estimands.md).
