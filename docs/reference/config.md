# Configuration

Each console script takes a `--config_file PATH` argument pointing at a YAML file. The schema differs per command. This page is the canonical reference, grouped by pipeline stage.

!!! note "Auto-generation coming"
    PR-11 will replace the hand-written tables on this page with auto-generated content from the pydantic [`ScriptConfig`](api/script_config.md) schema. Until that lands, treat any drift between this page and the schema as a bug — please open an issue.

## `infer-bn` configuration

The `infer-bn` console script (entry point: `infer_bayesian_network.py`) reads a single YAML file. Bundled example: [`examples/students/config.yaml`](https://github.com/PlaytikaOSS/causal-canvas/blob/main/examples/students/config.yaml).

### Data and I/O

| Key | Type | Required | Description |
|---|---|---|---|
| `data_input_path` | path | yes | Input dataset. Either `.parquet` or `.csv`. |
| `structure_path` | path | no | Path to a previously learned `DAG.dot`. If set, causal discovery is skipped and only the BN fit runs. If absent, runs discovery and writes a `DAG_for_inference.dot`. |
| `id_column` | str | no | Name of the row-identifier column (e.g. `user_id`). |
| `date_column` | str | no | Name of the timestamp column (e.g. `fs_calc_date`). Used to split a validation set. |
| `event_column` | str | yes | Name of the target column (e.g. `churn`, `G3`). |
| `drop_columns` | list[str] | no | Columns to drop before modelling. Default `[]`. |
| `features_select` | list[str] | no | Allow-list of features to model. If unset, uses all non-dropped non-id non-date columns. |
| `categorical_features` | list[str] | no | Subset of `features_select` to encode as categorical before structure learning. |
| `output_path` | path | yes | Directory to write outputs to. Created if missing. |
| `create_new_folder` | bool | no | If `true`, refuse to overwrite an existing output directory. Default `false`. |
| `add_datetime_to_folder` | bool | no | If `true`, append `YYYYMMDD_HHMMSS` to `output_path`. Useful for multi-run experiments. Default `false`. |

### Causal-discovery hyperparameters

| Key | Type | Default | Description |
|---|---|---|---|
| `connections_type` | `"linear"` \| `"non_linear"` | `"linear"` | Linear NOTEARS or NOTEARS-MLP. |
| `tabu_edges` | list[[from, to]] | `[]` | Forbidden edges (cause-effect pairs the algorithm must not draw). E.g. `[["G2", "Fedu"]]` forbids `G2 → Fedu`. |
| `tabu_edge_features` | list[str] | `[]` | Variables that can never *be caused* by other features (typically demographics). |
| `h_tol` | float | `1e-4` | NOTEARS convergence tolerance: exit when $h(W) <$ `h_tol`. |
| `max_iter` | int | `10000` | Max dual-ascent steps in NOTEARS. |
| `w_threshold` | float | `0.8` | Absolute-weight threshold for edge pruning. Edges with $|W_{ij}| <$ `w_threshold` are dropped. |
| `lasso_multiplier` | float | `0.0` | Coefficient on the $\ell_1$ regularisation term in *linear* NOTEARS. Increase to encourage sparser graphs. |
| `sample_frac` | float, $(0, 1]$ | `1.0` | Fraction of rows to use for modelling. |
| `invert_signs` | bool | `false` | If `true`, the visualised edge signs are reversed (cosmetic only). |

### Non-linear NOTEARS arguments (`non_linear_args`)

Only consulted when `connections_type: non_linear`.

| Key | Type | Default | Description |
|---|---|---|---|
| `feature_to_distribution_map` | dict[str, code] | *required* | Per-feature output distribution. See codes below. |
| `lasso_multiplier` | float | `0.0` | $\ell_1$ on the MLP first-layer weights. |
| `ridge_multiplier` | float | `0.0` | $\ell_2$ on the MLP first-layer weights. |
| `use_bias` | bool | `false` | Add intercept terms. |
| `hidden_layer_units` | list[int] | `[2]` | MLP hidden-layer widths. Length determines depth. |
| `use_gpu` | bool | `false` | Use CUDA if available. |
| `verbose` | bool | `false` | Log loss / `h` / `rho` during dual ascent. |

#### Distribution-type codes

| Code | Distribution | Use for |
|---|---|---|
| `bin` | Bernoulli | Binary features. |
| `cat` | Categorical / Multinomial | Multi-class categorical features. |
| `cont` | Gaussian | General continuous, MSE loss. |
| `pos_cont` | Non-negative Gaussian | Strictly positive continuous. |
| `bimodal_pos_cont` | Mixture of two non-negative Gaussians | Bimodal positive continuous (engagement-like). |
| `ord` | Ordinal | Ordered discrete features (e.g. Likert). |
| `poiss` | Poisson | Counts. |
| `gamma` | Gamma | Strictly-positive heavy-tailed continuous. |
| `tweedie` | Tweedie (scale 1.5) | Mass-at-zero plus continuous tail (e.g. revenue). |

### Discretisation (`discretiser`)

Applied to numeric features before BN fit.

| Key | Type | Default | Description |
|---|---|---|---|
| `method` | `"simple"` \| `"tree"` \| `"mdlp"` | `"simple"` | Quantile bins (`simple`), supervised tree splits (`tree`), or Fayyad–Irani MDLP (`mdlp`). |
| `argument` | int | varies | `simple`: number of quantile bins. `tree`: max tree depth. `mdlp`: minimum split value. |
| `proportion_threshold` | float, $[0, 1]$ \| null | `null` | Merge bins below this proportion into the previous bin. |
| `max_categories` | int \| null | `null` | Cap the number of bins post-merge. |
| `discretisation_cutoffs_target` | list[float] \| null | `null` | Manual cut points for a numeric target (e.g. `[10]` makes `G3` binary at 10). |

### BN inference (`inference_method`)

| Key | Type | Default | Description |
|---|---|---|---|
| `method` | `"MaximumLikelihoodEstimator"` \| `"BayesianEstimator"` | `"MaximumLikelihoodEstimator"` | CPT estimator. Bayesian estimator regularises sparse cells. |
| `bayes_prior` | `"K2"` \| `"BDeu"` | — | Required when `method = BayesianEstimator`. K2 = uniform pseudocount of 1; BDeu = equivalent-sample-size pseudocount. |
| `equivalent_sample_size` | int | — | Required for BDeu. Scales the prior strength: `equivalent_sample_size / (node_cardinality * prod(parents_cardinalities))` per cell. |
| `conditional_dependency_estimates` | list[str] | `[]` | Nodes whose CPT to write out as a CSV. Typically the target plus a few key drivers. |

### Visualisation

| Key | Type | Description |
|---|---|---|
| `event_graph_label` | str | Display label for the target node in the rendered HTML graph. |
| `event_color` | str | CSS colour or named colour for the target node. |
| `higher_contribution_feature_color` | str | Colour for the feature with the highest causal weight on the target. |

## `evaluate-experiments` configuration

The `evaluate-experiments` console script reads a separate YAML schema. Bundled example: [`examples/students/config_evaluation.yaml`](https://github.com/PlaytikaOSS/causal-canvas/blob/main/examples/students/config_evaluation.yaml).

| Key | Type | Description |
|---|---|---|
| `data_train_input_path` | path \| list[path] | Train dataset(s); list form supports pre-segmented data sharing a prefix path. |
| `data_test_input_path` | path \| list[path] | Test dataset(s). If unset, train data is split via `optim_test_dates` / `test_dates`. |
| `event_column.name` | str | Target column name. |
| `event_column.target_type` | str | If `"cont"`, no categorical transformation is applied before discretisation. |
| `event_column.target_column_names_cpds` | list[str] | Bin labels created by the BN, e.g. `["[-inf, 10)", "[10, inf)"]`. |
| `event_column.target_class` | str \| bool | Minority/positive class label, e.g. `"[10, inf)"`, `True`, `1`. |
| `date_column` | str | Date column name (when train and test are split by dates). |
| `optim_test_dates` | list[date] | Dates for the threshold-optimisation split. |
| `test_dates` | list[date] | Dates for the held-out test split. |
| `features_select` | list[str] | Features to use (must match the trained models' inputs). |
| `all_models_path` | path | Directory containing one subdirectory per inference method. |
| `inference_methods` | list[str] | Subdirectory names under `all_models_path` to evaluate (e.g. `["mle", "k2_prior"]`). |
| `models_combinations` | dict[str, list[int]] | Per-discretiser argument grid. E.g. `{"tree": [1, 2]}` evaluates `tree_1` and `tree_2` subdirs. |
| `score_to_optimise` | `"f1-score"` \| `"precision"` \| `"recall"` \| null | Score to optimise threshold against. If null, uses the default 50% threshold. |
| `boot_iterations` | int | Bootstrap iterations for score CIs. |
| `alpha` | float | Quantile for the bootstrap CI (e.g. 0.05 for 95% CI). |
| `use_multiprocessing` | bool | Split the test set across processes for prediction. |
| `output_path` | path | Output directory. |

## `infer-dowhy` configuration

The `infer-dowhy` console script reads a third schema. Bundled example: [`examples/students/config_inference.yaml`](https://github.com/PlaytikaOSS/causal-canvas/blob/main/examples/students/config_inference.yaml).

| Key | Type | Description |
|---|---|---|
| `event_column` | str | Target column. |
| `target_class` | str \| bool | Class label to score against (e.g. `"[10, inf)"` for the pass-grade target). |
| `model_path` | path | Path to the fitted `bayesian_network.pkl` from a previous `infer-bn` run. **Loaded via `safe_pickle_load`** — see [Security](security.md). |
| `conditionals` | dict | Conditional marginals to compute, e.g. `{churn: [is_active, level_end]}` produces $P(\text{churn} \mid \text{is\_active}, \text{level\_end})$. |
| `interventions` | dict | Soft interventions per feature, with named scenarios. Each scenario maps category indices to probabilities, e.g. `{level_end: {low_leveling: {0: 0.7, 1: 0.2, 2: 0.1}}}`. Interventions reset between scenarios; combination interventions are not yet supported. |
| `counterfactuals` | list[str] | Features for which to compute counterfactuals + ATE. Equivalent to a hard `do_intervention('X', 'x')` per category. |
| `output_path` | path | Output directory. |

## See also

- [CLI reference](cli.md) — command-line surface and flag reference.
- [Outputs](outputs.md) — what each pipeline writes to `output_path`.
- [Security](security.md) — guarantees on pickle loading and shell invocation.
- [API: ScriptConfig](api/script_config.md) — the pydantic schema (will become this page's source in PR-11).
