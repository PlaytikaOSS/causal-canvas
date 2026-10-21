# Interpreting results

Causal Canvas produces a lot of files. This page tells you which ones matter most and what each one says.

## The three artefacts you should always look at

### 1. `causal_discovery/04_threshold_DAG.html`

The DAG after pruning. This is the model's hypothesis about which variable causes which. Open it in a browser.

**What to check:**

- Are the arrow directions sensible? An arrow from "churn" to "tenure" means the model thinks churn caused tenure to be what it is — that's almost always wrong (tenure is upstream). When you see this, your `tabu_edges` config didn't capture the constraint.
- Is the target (the variable you set as `event_column`) at the *bottom* of most chains? It usually should be — features cause the outcome, not the other way around.
- Is the graph too dense? More than ~3 incoming edges per node usually means `w_threshold` is too low; raise it and re-run.
- Is the graph too sparse? If your target has no parents, the discovery picked nothing — usually a sample-size or signal-strength issue, not a model bug.

### 2. `bayesian_network/dependency_estimates/<target>.csv`

The fitted conditional probability table for the target node. For binary churn with two parents, this is a table of `P(churn | parent1, parent2)` rows.

**What to check:**

- Does the probability move monotonically with the parents you'd expect? (E.g. as `tech_crashes` goes up, `P(churn)` should go up.)
- Are there cells with very few observations? Those rows are noisy; the BN's inference will be too. Usually addressed by coarser discretisation (`max_categories` or `proportion_threshold` in the config).

### 3. `causal_discovery/threshold_weights_mlp.csv` (non-linear runs only)

The pruned edge-weight matrix from MLP-NOTEARS. Sort by absolute value and you have a ranked list of "most likely drivers" of every variable.

**What to check:**

- Is the top-ranked driver of the target a feature you can actually intervene on? If it's a slow-moving demographic, the ranking is informative but not actionable.
- Does the rank-1 vs rank-2 driver gap look meaningful or are they within noise? Small gaps mean the ranking is unstable.

## When you also need to look at

### `causal_discovery/01_fully_connected_graph.html`

The pre-pruning graph. Useful for sanity-checking that *every* variable was considered, not just the ones surviving `w_threshold`.

### `causal_discovery/all_weights_heatmap.png`

A visual of the full weight matrix. The diagonal should be empty (no self-loops); large off-diagonal cells outside the threshold are candidates that just barely missed the cut.

### `bayesian_network/model_fit/bayesian_network.pkl`

The serialised BN. Fed into `infer-dowhy` to compute interventions and counterfactuals. **Only load .pkl files you produced or trust** — see the [Security section](../reference/security.md).

### `bayesian_network/model_fit/condensed_splits.json`

The discretisation thresholds the model used. Read this if you want to translate the BN's bin labels back to raw numeric ranges.

## When *not* to read further

If `tox -e test` flagged warnings about model convergence, or `causal_discovery/score_edges_full_structure.yaml` shows weights all in the same narrow range, the run did not learn structure cleanly. Re-fit before interpreting.
