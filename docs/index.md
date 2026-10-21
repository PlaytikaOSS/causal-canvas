# Causal Canvas

Find what drives an outcome in your data — and estimate what happens if you change it.

Causal Canvas learns a cause-and-effect graph from observational data, fits a probabilistic model on top of it, and lets you simulate interventions ("what if we raised daily rewards?") and counterfactuals ("what would have happened if this user hadn't seen the popup?").

## Pick your path

<div class="grid cards" markdown>

- :material-clipboard-text:{ .lg .middle } **Product Manager**

    ---

    Outcomes, decisions, and limits — without statistical jargon.

    [:octicons-arrow-right-24: For PMs](for-pms/index.md)

- :material-chart-line:{ .lg .middle } **Data Analyst**

    ---

    Run the bundled example, interpret the graph, avoid the obvious pitfalls.

    [:octicons-arrow-right-24: For Analysts](for-analysts/index.md)

- :material-function:{ .lg .middle } **Statistician**

    ---

    Assumptions, estimands, identifiability — and references.

    [:octicons-arrow-right-24: For Statisticians](for-statisticians/index.md)

</div>

## When to use it

- You have observational tabular data and want to rank likely drivers of an outcome.
- You can enumerate variables that *cannot* cause each other (tabu edges).
- You will validate any acted-upon intervention with an experiment.

## When not to use it

- You already have an A/B test for the question — use the test.
- You suspect major unmeasured confounders — the output will be biased.
- The decision is already made; this won't change it.
