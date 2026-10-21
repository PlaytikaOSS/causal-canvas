# Outputs

Each pipeline writes a deterministic file tree under the `output_path` declared in its config.

## `infer-bn` outputs

```text
└── experiment/
    ├── causal_discovery/
    │   ├── 01_fully_connected_graph.html       Pre-pruning interactive DAG.
    │   ├── 02_largest_subgraph.html             Largest connected component, pre-prune.
    │   ├── 03_threshold_graph.html              After w_threshold pruning.
    │   ├── 04_threshold_DAG.html                Post-prune DAG (open this first).
    │   ├── 05_DAG_subgraph.html                 Largest connected sub-DAG.
    │   ├── DAG_for_inference.dot                NetworkX DAG used by infer-dowhy.
    │   ├── label_encoder.pkl                    Fitted categorical encoder.
    │   ├── scaler.pkl                           Fitted numeric scaler.
    │   ├── all_weights_mlp.csv                  Raw MLP-NOTEARS edge weights (non-linear runs).
    │   ├── threshold_weights_mlp.csv            Edges surviving w_threshold.
    │   ├── all_std_error_weights_mlp.csv        Standard errors on raw weights.
    │   ├── threshold_std_error_weights_mlp.csv  Standard errors on surviving weights.
    │   ├── all_weights_heatmap.png              Visual edge map (full).
    │   ├── all_weights_w_threshold_heatmap.png  Visual edge map (after threshold).
    │   ├── score_edges_full_structure.yaml      Edge weights, full structure.
    │   └── score_edges_sparser_structure.yaml   Edge weights, pruned structure.
    ├── bayesian_network/
    │   ├── dependency_estimates/
    │   │   └── <node>.csv                       Per-node CPT.
    │   └── model_fit/
    │       ├── bayesian_network.pkl             Fitted BN (consumed by infer-dowhy).
    │       ├── quantile_splits.json             Discretiser thresholds (numeric features).
    │       ├── discretiser.pkl                  Fitted discretiser (tree / mdlp methods).
    │       └── condensed_splits.json            Post-merge splits when proportion_threshold is set.
    └── config.yaml                              Copy of the run's config.
```

Open `causal_discovery/04_threshold_DAG.html` first for the structural result, then `bayesian_network/dependency_estimates/<target>.csv` for the conditional probability of the outcome.

## `evaluate-experiments` outputs

```text
└── error_analysis/
    ├── cpd_plots/                              Per-model CPT-variability plots.
    ├── optim_set_predictions/                  Per-method predictions on the optimisation set.
    │   ├── <method>/
    │   │   └── predictions_with_discretiser_*.csv
    │   ├── bootstrap_scores.csv                Bootstrap intervals for F1/precision/recall.
    │   ├── bootstrap_scores.png
    │   └── prediction_times.png
    ├── test_set_predictions/                   Same shape as optim_set_predictions.
    └── config.yaml                             Copy of the evaluation config.
```

Use `bootstrap_scores.csv` to assess whether differences between models are within noise.

## `infer-dowhy` outputs

Per the `interventions`, `counterfactuals`, and `conditionals` sections of the config, `infer-dowhy` writes:

```text
└── interventions_dir/
    ├── interventions/
    │   ├── intervention_plot_<feature>.png             Y distribution under each intervention.
    │   ├── intervention_ate_plot_<feature>.png         ATE per intervention category.
    │   └── intervention_uplift_plot_<feature>.png      Uplift relative to baseline.
    ├── counterfactuals/
    │   ├── counterfactuals_plot_<feature>.png
    │   ├── counterfactuals_ate_plot_<feature>.png
    │   └── counterfactuals_uplift_plot_<feature>.png
    ├── marginals.json                                  Conditional marginals from `conditionals`.
    └── config.yaml                                     Copy of the inference config.
```

## See also

- [Interpreting results (analyst-focused)](../for-analysts/interpreting-results.md).
