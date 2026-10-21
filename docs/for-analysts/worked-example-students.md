# Worked example: UCI Portuguese-grades

Annotated read-through of a real Causal Canvas run. The dataset is the UCI "Student Performance" Portuguese-language exam dataset — 33 variables covering demographics, family background, lifestyle, and prior grades; the target `G3` is the final period grade.

## Why this dataset

It's small (≤ 650 rows), has a mix of categorical and numeric features, has plausible causal structure most readers will recognise (parental education affects student behaviour affects grades), and ships with the repo. Good for sanity-checking your install.

## 1. The config

`examples/students/config.yaml` declares:

- `event_column: G3` — the target.
- `connections_type: non_linear` — use MLP-NOTEARS, since some relationships (e.g. study time → grade) are clearly non-linear.
- `discretiser.method: tree`, `argument: 4` — tree-based discretisation up to depth 4.
- `discretisation_cutoffs_target: [10]` — make `G3` binary at the 10-out-of-20 pass mark.
- `tabu_edges`: a few obvious ones, including `[G2, G1]` (G1 was measured before G2, so G2 can't cause G1).

## 2. Running it

```bash
poetry run infer-bn --config_file examples/students/config.yaml
```

A pass takes ~30 seconds on a laptop. Logs end with `Saving Bayesian Network` and the output directory under `dummy_example_results/.../tree_4/`.

## 3. What to read first

Open `04_threshold_DAG.html`. You should see (approximately):

- `G2 → G3` — the second-period grade strongly drives the final grade. Expected.
- `failures → G3` (negative weight) — past failures predict lower final grade. Expected.
- `studytime → G3` (positive) — more study time, better grade. Expected.
- `Mjob`, `Fjob` upstream of `studytime`. Plausible: parental occupation correlates with study habits.
- Demographics (age, sex) only weakly connected — they're in `tabu_edge_features`, so they can't be effects.

If any of these are missing or reversed, the run had stability or threshold issues — see [Pitfalls](pitfalls.md).

## 4. The CPT for the target

Open `bayesian_network/dependency_estimates/G3.csv`. The columns are the parent variables surviving the DAG prune; the rows enumerate every parent value combination. The two columns at the end are `P(G3 = "[-inf, 10)")` (failed the exam) and `P(G3 = "[10, inf)")` (passed).

Read off a few rows manually:

- `G2 = "[-inf, 10)"`, no other parents change → `P(G3 < 10)` should be ≥ 80%. Failing the second period strongly predicts failing the final.
- `G2 = "[10, inf)"`, `failures = 0` → `P(G3 ≥ 10)` should be ≥ 90%. Passing G2 with no past failures is a near-certain pass.

If those numbers look wrong (e.g. the conditional probability moves the wrong way as `G2` increases), the discretisation went wrong — usually fixed by raising `proportion_threshold` to suppress noisy small bins.

## 5. Running an intervention

```bash
poetry run infer-dowhy --config_file examples/students/config_inference.yaml
```

The example inference config defines an intervention `low` and `high` on `G2`, with respective probability shifts to the lower / higher bin. Output:

- `interventions/intervention_plot_G2.png` — distribution of `G3` under each intervention.
- `interventions/intervention_ate_plot_G2.png` — average treatment effect on `G3`.
- `counterfactuals/counterfactuals_*_plot_G2.png` — counterfactual plots if `counterfactuals` is set.

Expected: shifting `G2` toward the high bin raises the marginal probability of passing `G3` substantially. The ATE quantifies "how much".

## 6. Where this becomes useful

The students dataset is too small for production-grade decisions, but the pattern transfers. On a real player-360 dataset with churn as the target, the same workflow gives you:

- A ranked list of features that drive churn (pruned DAG + edge weights).
- Estimated effect of pushing the population toward "low ad-load" or "high reward-frequency".
- Confidence intervals via `evaluate-experiments`.

## See also

- [Quickstart](quickstart.md)
- [Interpreting results](interpreting-results.md)
- [Pitfalls](pitfalls.md)
- The student-grades example folder: `examples/students/`.
