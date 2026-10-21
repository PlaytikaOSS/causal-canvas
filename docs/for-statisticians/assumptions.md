# Assumptions and identifiability

Every causal-inference output rests on assumptions the data alone cannot verify. Below, the assumptions Causal Canvas relies on, the consequences when they fail, and what the tool can and cannot detect.

## A1. Acyclicity

The structural causal model is a directed *acyclic* graph. NOTEARS enforces this through the smooth acyclicity penalty $h(W) = \\mathrm{tr}(e^{W \\circ W}) - d$ being driven to zero (Zheng et al., 2018, eq. 2).

**When it fails**: feedback loops (X → Y → X) that genuinely exist in the data-generating process cannot be modelled. The fit can still converge, but with arbitrary direction choices on the cycle.

**Detectable**: yes — NOTEARS reports the final $h(W)$; non-zero values indicate failed projection. Causal Canvas surfaces this in logs.

## A2. Causal sufficiency

All common causes of any pair of measured variables are themselves measured. No unmeasured confounders.

**When it fails**: estimated effects on downstream interventions are biased. The DAG can still be learned consistently if the unmeasured confounder is not too influential, but ATE estimates are not interpretable as causal.

**Detectable**: no — the tool cannot detect missing confounders. Sensitivity analysis (E-values, Rosenbaum bounds) is the analyst's responsibility.

## A3. Faithfulness

The conditional independencies in the data exactly match the DAG separations (no fortuitous cancellations).

**When it fails**: edges that genuinely exist appear conditionally independent in finite samples and get pruned.

**Detectable**: indirectly, via stability across random seeds and bootstrap re-fits.

## A4. Markov compatibility

The fitted Bayesian network is Markov-compatible with the learned DAG: the joint distribution factorises as $\\prod_i P(X_i \\mid \\text{Pa}(X_i))$.

**When it fails**: rare in well-discretised data; failure mode is sparse cells in the CPT producing unreliable conditional probabilities.

**Detectable**: yes — `BayesianNetworkEstimator` issues warnings on cells with effective sample size below a threshold.

## A5. Stable structural equations

The data-generating mechanism is stationary over the training window. Interventions are imagined at the same moment as the training period.

**When it fails**: deploying a recommendation based on a Causal Canvas output trained on Q1 data into Q3 may not reproduce the estimated effect, even if all assumptions held in Q1.

**Detectable**: indirectly, via re-running on rolling windows.

## A6. NOTEARS-specific: equal-variance trap

Reisach et al. (2021) showed that on Gaussian data with equal noise variance, NOTEARS can prefer reversed edge directions that fit the data equivalently. The non-linear MLP variant (Zheng et al., 2020) mitigates this for non-Gaussian variables but does not eliminate it.

**Mitigation in this codebase**: features are standardised in `Preprocessor`, and `tabu_edges` / `tabu_edge_features` let the analyst encode known direction constraints up-front.

## Identifiability of interventions

Given the DAG and a fitted BN, $P(Y \\mid \\mathrm{do}(X = x))$ is identifiable by Pearl's do-calculus iff $X$ is "sufficient" for the back-door criterion against $Y$. Causal Canvas does not currently emit a back-door check; the analyst should verify that each intervention target $X$ is unconfounded given the modelled parents of $X$ in the learned DAG.

## See also

- [Estimands](estimands.md) — what's being estimated under these assumptions.
- [Non-linear NOTEARS](nonlinear-notears.md).
- [References](references.md).
