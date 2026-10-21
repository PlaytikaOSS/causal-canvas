# For Data Analysts

Causal Canvas is built around a YAML config and five console scripts. If you can run pandas you can run this. The pages below get you from a fresh clone to a defensible interpretation in under an hour.

- [Quickstart](quickstart.md) — run the bundled students example end-to-end.
- [Interpreting results](interpreting-results.md) — which output files matter and what each one says.
- [Pitfalls](pitfalls.md) — when to mistrust the output, and what to do about it.
- [Worked example: students](worked-example-students.md) — annotated walkthrough of a real run on the UCI Portuguese-grades dataset.

## Mental model

Causal Canvas answers two related questions:

1. **What drives the outcome?** It learns a directed acyclic graph (DAG) over your features and ranks edges by estimated weight.
2. **What happens if we change a driver?** It fits a Bayesian network on the discovered DAG and lets you simulate interventions and counterfactuals.

It is **not** a classifier (the BN is fitted as a generative model, not for ML accuracy), and it is **not** an A/B test (effects are estimated from observational data under stated assumptions). Use it to *prioritise* hypotheses; validate them with experiments.

## Prerequisites

- Python 3.9 or 3.10, Poetry, the cloned repo (`poetry install`).
- A tabular dataset (Parquet or CSV) with one row per unit (user, session, etc.) and a binary or categorical target column.
- A list of features that *cannot* cause each other (e.g. demographics can't be caused by behaviour). These go in `tabu_edges` and `tabu_edge_features` in the config.
