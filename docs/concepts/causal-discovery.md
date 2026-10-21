# Causal discovery

Causal discovery is the problem of recovering, from observational data alone, the directed acyclic graph that encodes which variables cause which.

## Why it's hard

Statistical correlation is symmetric: if $X$ and $Y$ are correlated, observation alone cannot distinguish "X causes Y" from "Y causes X" from "both share a common cause". The classical solutions all add constraints:

- **Randomised experiments** force one variable's value, breaking the symmetry. Gold standard but often infeasible.
- **Natural experiments / instrumental variables** rely on an exogenous source of variation. Powerful but require subject-matter ingenuity.
- **Constraint-based discovery** (PC, FCI) tests conditional independencies and rules out edge directions inconsistent with them. Robust but combinatorial.
- **Score-based discovery** (GES, NOTEARS, BIC-based) treats DAG search as optimising a goodness-of-fit score. NOTEARS is the score-based method Causal Canvas uses.

## What NOTEARS does

NOTEARS (Zheng et al., 2018) recasts the discrete combinatorial problem of "which DAG fits best" as a continuous optimisation over real-valued weight matrices, using a clever acyclicity constraint $h(W) = \\mathrm{tr}(e^{W \\circ W}) - d = 0$. The optimisation can be solved with standard gradient methods, and the resulting matrix is thresholded to recover a discrete DAG.

The non-linear extension (Zheng et al., 2020) replaces the linear weight model with a small per-variable MLP, allowing structural equations like "log churn rate is a non-linear function of tech crashes and engagement minutes" rather than just a linear combination.

## What Causal Canvas adds on top

- **Tabu edges** — the analyst can list edges (or whole "downstream from" rules) that the search must respect. Encodes prior knowledge cheaply.
- **Distribution-type loss functions** — see [Non-linear NOTEARS](../for-statisticians/nonlinear-notears.md). Each variable can have its own NLL, so churn (binary) is fit with cross-entropy and revenue (Tweedie) with the appropriate scale-1.5 loss, instead of MSE everywhere.
- **Pruning to the largest DAG** — after threshold pruning, Causal Canvas extracts the largest connected sub-DAG so downstream BN inference works on a single component.

## What it does not do

- Distinguish "A causes B" from "A and B share an unmeasured confounder C". This requires either an instrumental variable or assuming causal sufficiency. See [Assumptions](../for-statisticians/assumptions.md).
- Time-lagged causation or feedback loops. The output is a snapshot DAG.
- Latent variable discovery. There is no FCI-style PAG output.

## Going deeper

- [Bayesian networks](bayesian-networks.md) — what comes after the DAG.
- [Assumptions and identifiability](../for-statisticians/assumptions.md).
- [Non-linear NOTEARS](../for-statisticians/nonlinear-notears.md).
- [References](../for-statisticians/references.md).
