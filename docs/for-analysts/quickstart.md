# Analyst quickstart (5 minutes)

End-to-end run on the bundled UCI Portuguese-grades example.

## 1. Clone and install

```bash
git clone https://github.com/PlaytikaOSS/causal-canvas.git
cd causal_canvas
poetry config virtualenvs.in-project true
poetry install
```

## 2. Run causal discovery + Bayesian-network fit

```bash
poetry run infer-bn --config_file examples/students/config.yaml
```

What this does:

- Loads `data/student-por_train.csv` (the path is in the config's `data_input_path`).
- Preprocesses (drops `drop_columns`, encodes categoricals, validation split).
- Runs non-linear NOTEARS structure learning (because the example config sets `connections_type: non_linear`). The output is a DAG.
- Discretises numeric features (the example uses tree-based discretisation with depth 4).
- Fits a Bayesian network on the discretised features.
- Writes everything to `output_path` (default: `dummy_example_results/students_portuguese_grades/non_linear_causal_discovery/k2_prior/tree_4/`).

A pass takes ~30-60 seconds on a laptop.

## 3. Look at the output

The output directory contains:

```text
causal_discovery/
  01_fully_connected_graph.html       <- pre-pruning graph (open in browser)
  04_threshold_DAG.html                <- the DAG you should usually read
  05_DAG_subgraph.html                 <- largest connected sub-DAG
  DAG_for_inference.dot                <- text serialisation
  all_weights_mlp.csv                  <- raw edge weights from MLP-NOTEARS
  threshold_weights_mlp.csv            <- weights surviving `w_threshold`
  all_weights_heatmap.png              <- visual edge map
bayesian_network/
  dependency_estimates/<node>.csv      <- learned conditional probability table per node
  model_fit/bayesian_network.pkl       <- the trained BN (used by infer-dowhy)
config.yaml                            <- copy of the config used for the run
```

Open `04_threshold_DAG.html` first. That's the DAG the rest of the pipeline reasons about. Then look at `dependency_estimates/G3.csv` to see the conditional probability of the final-grade target across its parents.

## 4. Run do-calculus inference (interventions)

```bash
poetry run infer-dowhy --config_file examples/students/config_inference.yaml
```

This loads the trained BN, computes marginal distributions, applies the interventions defined in the config (e.g. shift `G2` toward higher values), and writes intervention/counterfactual plots and ATE estimates to the inference output directory.

## 5. Read more

- [Interpreting results](interpreting-results.md) — what each output file is telling you.
- [Pitfalls](pitfalls.md) — common ways to misread a Causal Canvas output.
- [Worked example: students](worked-example-students.md) — an annotated reading of the students-example outputs.
- [CLI reference](../reference/cli.md) — every console script and flag.
